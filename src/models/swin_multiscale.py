"""
SwinMultiScale: Multi-Scale Feature Fusion Classifier using Swin Transformer

Key architectural innovations:
1. Extracts features from all 4 Swin stages (not just final output)
2. Adaptive pooling to unify spatial dimensions across scales
3. Attention-weighted multi-scale fusion with learnable weights
4. Optional deep supervision with auxiliary classification heads
5. SE-style channel attention on fused features

Swin-Tiny stage outputs:
- Stage 1: H/4 × W/4, 96 channels  (56×56 for 224 input)
- Stage 2: H/8 × W/8, 192 channels (28×28)
- Stage 3: H/16 × W/16, 384 channels (14×14)
- Stage 4: H/32 × W/32, 768 channels (7×7)
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .train_utils import TrainingConfig, add_common_cli, run_training


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        scale = self.fc(x)
        return x * scale


class ScaleProjection(nn.Module):
    """Project features from one scale to a common dimension."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiScaleFusion(nn.Module):
    """Adaptive fusion of multi-scale features with attention weighting.
    
    Similar to AdaptiveFusion in DenseNet, but designed for varying channel
    dimensions from different Swin stages.
    """
    
    def __init__(
        self,
        stage_channels: List[int],
        fusion_dim: int = 512,
        num_stages: int = 4,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.fusion_dim = fusion_dim
        
        # Project each stage to common dimension
        self.projections = nn.ModuleList([
            ScaleProjection(ch, fusion_dim) for ch in stage_channels
        ])
        
        # Attention weight generator from concatenated context
        total_input = fusion_dim * num_stages
        self.weight_net = nn.Sequential(
            nn.Linear(total_input, total_input // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_input // 4, num_stages),
            nn.Softmax(dim=1)
        )
        
        # SE attention on fused features
        self.se = SEBlock(fusion_dim, reduction=8)
        
        # Final layer norm
        self.norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of (B, C_i) pooled features from each stage
        Returns:
            Fused features (B, fusion_dim)
        """
        # Project all features to common dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Concatenate for context
        context = torch.cat(projected, dim=1)  # (B, fusion_dim * num_stages)
        
        # Generate attention weights
        weights = self.weight_net(context)  # (B, num_stages)
        
        # Weighted sum of projected features
        stacked = torch.stack(projected, dim=1)  # (B, num_stages, fusion_dim)
        weights = weights.unsqueeze(-1)  # (B, num_stages, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, fusion_dim)
        
        # Apply SE attention and normalization
        fused = self.se(fused)
        fused = self.norm(fused)
        
        return fused


class AuxiliaryHead(nn.Module):
    """Auxiliary classification head for deep supervision."""
    
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SwinMultiScale(nn.Module):
    """Swin Transformer with Multi-Scale Feature Fusion for Classification.
    
    Extracts features from all 4 Swin stages and fuses them with attention
    weighting for improved classification performance.
    
    Args:
        num_classes: Number of output classes
        fusion_dim: Dimension for fused features (default: 512)
        use_aux_heads: Whether to use auxiliary heads for deep supervision
        aux_weight: Weight for auxiliary losses during training
        pretrained: Whether to use pretrained Swin weights
    """
    
    def __init__(
        self,
        num_classes: int = 11,
        fusion_dim: int = 512,
        use_aux_heads: bool = True,
        aux_weight: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_aux_heads = use_aux_heads
        self.aux_weight = aux_weight
        
        # Swin-Tiny backbone with feature extraction from all stages
        # Swin-Tiny: [96, 192, 384, 768] channels at 4 stages
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            in_chans=1,  # Grayscale input
            num_classes=0,  # Remove classification head
            features_only=True,  # Extract intermediate features
        )
        
        # Get stage output channels from model config
        self.stage_channels = self.backbone.feature_info.channels()
        # Typically: [96, 192, 384, 768] for Swin-Tiny
        
        # Multi-scale fusion module
        self.fusion = MultiScaleFusion(
            stage_channels=self.stage_channels,
            fusion_dim=fusion_dim,
            num_stages=len(self.stage_channels),
        )
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(fusion_dim, num_classes),
        )
        
        # Auxiliary heads for deep supervision (one per stage)
        if use_aux_heads:
            self.aux_heads = nn.ModuleList([
                AuxiliaryHead(ch, num_classes, dropout=0.1)
                for ch in self.stage_channels
            ])
        else:
            self.aux_heads = None
        
        # Initialize new modules
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract and fuse multi-scale features.
        
        Returns:
            pooled_features: List of (B, C_i) pooled features per stage
            fused: (B, fusion_dim) fused feature vector
        """
        # Extract features from all stages
        stage_features = self.backbone(x)
        # Each is (B, C_i, H_i, W_i)
        
        # Global average pooling per stage
        pooled_features = []
        for feat in stage_features:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C_i)
            pooled_features.append(pooled)
        
        # Fuse multi-scale features
        fused = self.fusion(pooled_features)
        
        return pooled_features, fused
    
    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        During training with aux_heads: returns (main_logits, aux_logits_list)
        During eval or without aux_heads: returns main_logits only
        """
        pooled_features, fused = self.forward_features(x)
        
        # Main classification
        logits = self.classifier(fused)
        
        # Auxiliary outputs for deep supervision
        if self.training and self.use_aux_heads and self.aux_heads is not None:
            aux_logits = [
                head(feat) for head, feat in zip(self.aux_heads, pooled_features)
            ]
            return logits, aux_logits
        
        return logits


def build_swin_multiscale(
    num_classes: int = 11,
    pretrained: bool = True,
    use_aux_heads: bool = True,
    fusion_dim: int = 512,
) -> nn.Module:
    """Build SwinMultiScale classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained Swin weights
        use_aux_heads: Enable auxiliary heads for deep supervision
        fusion_dim: Dimension for fused features
    
    Returns:
        SwinMultiScale model
    """
    model = SwinMultiScale(
        num_classes=num_classes,
        fusion_dim=fusion_dim,
        use_aux_heads=use_aux_heads,
        pretrained=pretrained,
    )
    return model


class MultiScaleLoss(nn.Module):
    """Combined loss with main + auxiliary heads for deep supervision."""
    
    def __init__(self, aux_weight: float = 0.4, weight: torch.Tensor | None = None):
        super().__init__()
        self.aux_weight = aux_weight
        self.main_criterion = nn.CrossEntropyLoss(weight=weight)
        self.aux_criterion = nn.CrossEntropyLoss(weight=weight)
    
    def forward(
        self,
        outputs: torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, tuple):
            main_logits, aux_logits_list = outputs
            
            # Main loss
            main_loss = self.main_criterion(main_logits, targets)
            
            # Auxiliary losses (weighted average)
            aux_loss = sum(
                self.aux_criterion(aux, targets) for aux in aux_logits_list
            ) / len(aux_logits_list)
            
            # Combined loss
            return main_loss + self.aux_weight * aux_loss
        else:
            return self.main_criterion(outputs, targets)


def run_training_multiscale(
    build_model_fn,
    defaults: TrainingConfig,
    use_aux_heads: bool = True,
    aux_weight: float = 0.4,
) -> None:
    """Custom training loop that supports deep supervision with auxiliary heads.
    
    Similar to run_training but handles tuple outputs from SwinMultiScale.
    """
    import json
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    
    from .train_utils import (
        TrainingConfig,
        prepare_datasets,
        _load_class_weights,
        _set_seed,
        evaluate,
        collect_predictions,
    )
    from ..analysis.config import OUTPUT_CONFIG
    from ..analysis.utils import ensure_output_directories
    
    ensure_output_directories()
    config = TrainingConfig.from_env(defaults)
    _set_seed(config.seed)
    
    train_dataset, val_dataset, class_labels = prepare_datasets(config)
    num_classes = len(class_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Device selection
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "mps" if use_mps else ("cuda" if use_cuda else "cpu")
    )
    
    model = build_model_fn(num_classes)
    model = model.to(device)
    
    # Load class weights if available
    class_weights = _load_class_weights(num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Use MultiScaleLoss for deep supervision
    if use_aux_heads:
        criterion = MultiScaleLoss(aux_weight=aux_weight, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        running_loss = 0.0
        sample_count = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            sample_count += targets.size(0)
        
        train_loss = running_loss / sample_count
        scheduler.step()
        
        # Validation (model returns only main logits in eval mode)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(
                model.state_dict(),
                OUTPUT_CONFIG.models_root / f"{config.model_name}_best.pth"
            )
        
        print(
            f"[{config.model_name}] Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )
    
    # Save final model and artifacts
    OUTPUT_CONFIG.models_root.mkdir(parents=True, exist_ok=True)
    weights_path = OUTPUT_CONFIG.models_root / f"{config.model_name}_weights.pth"
    torch.save(model.state_dict(), weights_path)
    
    # Load best model for final evaluation
    best_path = OUTPUT_CONFIG.models_root / f"{config.model_name}_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    
    preds, targets_arr = collect_predictions(model, val_loader, device)
    cm = confusion_matrix(targets_arr, preds, labels=class_labels)
    np.save(OUTPUT_CONFIG.models_root / f"confusion_matrix_{config.model_name}.npy", cm)
    
    # Per-class accuracy
    per_class_acc = {}
    for label in class_labels:
        mask = targets_arr == label
        per_class_acc[label] = float((preds[mask] == label).mean()) if mask.any() else None
    per_class_df = pd.DataFrame({"label": list(per_class_acc.keys()), "accuracy": list(per_class_acc.values())})
    per_class_df.to_json(
        OUTPUT_CONFIG.models_root / f"per_class_accuracy_{config.model_name}.json",
        orient="records", indent=2,
    )
    
    # Training curves
    plt.figure(figsize=(8, 5))
    epochs_arr = np.arange(1, config.epochs + 1)
    plt.plot(epochs_arr, history["train_loss"], label="Train Loss")
    plt.plot(epochs_arr, history["val_loss"], label="Val Loss")
    plt.plot(epochs_arr, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{config.model_name} Training (Best Val Acc: {best_acc:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFIG.models_root / f"training_curves_{config.model_name}.png")
    plt.close()
    
    # Summary
    summary = {
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_val_accuracy": history["val_accuracy"][-1],
        "best_val_accuracy": best_acc,
        "use_aux_heads": use_aux_heads,
        "aux_weight": aux_weight,
        "input_size": config.input_size,
        "input_channels": config.input_channels,
    }
    (OUTPUT_CONFIG.models_root / f"{config.model_name}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    
    print(f"\n[{config.model_name}] Training complete! Best val accuracy: {best_acc:.4f}")


def main() -> None:
    """Training entry point with deep supervision support."""
    defaults = TrainingConfig(
        model_name="swin_multiscale",
        input_channels=1,
        input_size=224,
        epochs=50,
        batch_size=48,  # Slightly reduced due to multi-scale overhead
        lr=5e-4,
        momentum=0.9,
        weight_decay=5e-2,
        step_size=15,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )
    
    parser = argparse.ArgumentParser(
        description="Train SwinMultiScale (Multi-Scale Feature Fusion) on OrganAMNIST"
    )
    add_common_cli(parser, defaults)
    parser.add_argument(
        "--no-aux-heads",
        action="store_true",
        help="Disable auxiliary heads (deep supervision)",
    )
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.4,
        help="Weight for auxiliary losses (default: 0.4)",
    )
    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=512,
        help="Dimension for fused features (default: 512)",
    )
    args = parser.parse_args()
    
    # Build model factory with parsed arguments
    use_aux = not args.no_aux_heads
    fusion_dim = args.fusion_dim
    aux_weight = args.aux_weight
    
    def build_fn(num_classes: int) -> nn.Module:
        return build_swin_multiscale(
            num_classes=num_classes,
            pretrained=True,
            use_aux_heads=use_aux,
            fusion_dim=fusion_dim,
        )
    
    # Use custom training loop that supports deep supervision
    run_training_multiscale(build_fn, defaults, use_aux_heads=use_aux, aux_weight=aux_weight)


if __name__ == "__main__":
    main()

