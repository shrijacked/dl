"""
Training script for ConvTransGFusion model.

ConvTransGFusion combines ConvNeXt and Swin Transformer branches with
Attention-Guided Feature Fusion (AGFF) for hybrid CNN-Transformer classification.

Usage:
    python -m src.training.train_convtransgfusion --epochs 50 --batch-size 32 --lr 5e-4
    
Environment Variables:
    MODEL_NAME: Model name for logging (default: convtransgfusion)
    EPOCHS: Number of training epochs
    BATCH_SIZE: Batch size for training
    LEARNING_RATE: Initial learning rate
    WEIGHT_DECAY: Weight decay for AdamW optimizer
    NUM_WORKERS: Number of data loading workers
    SEED: Random seed for reproducibility
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..analysis.config import DATASET_CONFIG, OUTPUT_CONFIG
from ..analysis.utils import ensure_output_directories, load_labels
from ..models.convtransgfusion import build_convtransgfusion
from ..models.train_utils import (
    TrainingConfig,
    add_common_cli,
    OrganDataset,
    prepare_datasets,
    _load_class_weights,
    _set_seed,
    evaluate,
    collect_predictions,
)


def run_training_convtransgfusion(
    defaults: TrainingConfig,
    drop_path_rate: float = 0.1,
    dropout: float = 0.1,
    use_cosine: bool = True,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.0,
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict:
    """
    Training loop for ConvTransGFusion with advanced training features.
    
    Args:
        defaults: Training configuration
        drop_path_rate: Drop path rate for ConvNeXt branch
        dropout: Dropout rate for classification head
        use_cosine: Use cosine annealing scheduler
        warmup_epochs: Number of warmup epochs
        label_smoothing: Label smoothing factor
        mixup_alpha: Mixup alpha (0 to disable)
        grad_clip_norm: Gradient clipping norm (None to disable)
    
    Returns:
        Training results dict with history and best metrics
    """
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
        drop_last=True,  # Better for batch normalization
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
    print(f"[{config.model_name}] Using device: {device}")
    
    # Build model
    model = build_convtransgfusion(
        num_classes=num_classes,
        pretrained=True,
        drop_path_rate=drop_path_rate,
        dropout=dropout,
    )
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{config.model_name}] Total parameters: {total_params:,}")
    print(f"[{config.model_name}] Trainable parameters: {trainable_params:,}")
    
    # Load class weights if available
    class_weights = _load_class_weights(num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"[{config.model_name}] Using class weights for imbalanced training")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    # AdamW optimizer (better for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler
    if use_cosine:
        if warmup_epochs > 0:
            # Linear warmup + cosine annealing
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(1, config.epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs, eta_min=1e-6
            )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "learning_rate": []}
    best_acc = 0.0
    best_epoch = 0
    
    print(f"\n[{config.model_name}] Starting training for {config.epochs} epochs...")
    print(f"[{config.model_name}] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        sample_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            sample_count += targets.size(0)
            
            pbar.set_postfix({"loss": f"{running_loss / sample_count:.4f}"})
        
        train_loss = running_loss / sample_count
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Validation phase
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)
        
        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                OUTPUT_CONFIG.models_root / f"{config.model_name}_best.pth"
            )
        
        print(
            f"[{config.model_name}] Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} lr={current_lr:.2e}"
        )
    
    # Save final model and artifacts
    OUTPUT_CONFIG.models_root.mkdir(parents=True, exist_ok=True)
    weights_path = OUTPUT_CONFIG.models_root / f"{config.model_name}_weights.pth"
    torch.save(model.state_dict(), weights_path)
    
    # Load best model for final evaluation
    best_path = OUTPUT_CONFIG.models_root / f"{config.model_name}_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    
    # Collect predictions for confusion matrix
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
    
    # Training curves plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs_arr = np.arange(1, config.epochs + 1)
    
    # Loss curves
    axes[0].plot(epochs_arr, history["train_loss"], label="Train Loss", color='#2563eb')
    axes[0].plot(epochs_arr, history["val_loss"], label="Val Loss", color='#dc2626')
    axes[0].axvline(x=best_epoch, color='#059669', linestyle='--', alpha=0.7, label=f'Best @ {best_epoch}')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs_arr, history["val_accuracy"], label="Val Accuracy", color='#059669')
    axes[1].axhline(y=best_acc, color='#dc2626', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.4f}')
    axes[1].axvline(x=best_epoch, color='#059669', linestyle='--', alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[2].plot(epochs_arr, history["learning_rate"], color='#7c3aed')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f"{config.model_name} Training (Best Val Acc: {best_acc:.4f} @ Epoch {best_epoch})")
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFIG.models_root / f"training_curves_{config.model_name}.png", dpi=150)
    plt.close()
    
    # Summary
    summary = {
        "model_name": config.model_name,
        "architecture": "ConvTransGFusion (ConvNeXt + Swin + AGFF)",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "drop_path_rate": drop_path_rate,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "use_cosine_scheduler": use_cosine,
        "warmup_epochs": warmup_epochs,
        "grad_clip_norm": grad_clip_norm,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "final_val_accuracy": history["val_accuracy"][-1],
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "input_size": config.input_size,
        "input_channels": config.input_channels,
    }
    (OUTPUT_CONFIG.models_root / f"{config.model_name}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    
    print(f"\n[{config.model_name}] Training complete!")
    print(f"[{config.model_name}] Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"[{config.model_name}] Weights saved to: {weights_path}")
    
    return {"history": history, "best_accuracy": best_acc, "best_epoch": best_epoch}


def main() -> None:
    """Training entry point with CLI arguments."""
    defaults = TrainingConfig(
        model_name="convtransgfusion",
        input_channels=1,
        input_size=224,
        epochs=50,
        batch_size=32,
        lr=5e-4,
        momentum=0.9,
        weight_decay=5e-2,
        step_size=15,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )
    
    parser = argparse.ArgumentParser(
        description="Train ConvTransGFusion (Hybrid CNN-Transformer) on OrganAMNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_cli(parser, defaults)
    
    # Model-specific arguments
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.1,
        help="Drop path rate for ConvNeXt branch",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for classification head",
    )
    parser.add_argument(
        "--no-cosine",
        action="store_true",
        help="Disable cosine annealing (use step LR instead)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for LR schedule",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross-entropy",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )
    
    args = parser.parse_args()
    
    # Update defaults from args where applicable
    defaults.epochs = args.epochs
    defaults.batch_size = args.batch_size
    defaults.lr = args.lr
    defaults.weight_decay = args.weight_decay
    defaults.num_workers = args.num_workers
    defaults.seed = args.seed
    defaults.input_size = args.input_size
    
    run_training_convtransgfusion(
        defaults=defaults,
        drop_path_rate=args.drop_path_rate,
        dropout=args.dropout,
        use_cosine=not args.no_cosine,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
    )


if __name__ == "__main__":
    main()

