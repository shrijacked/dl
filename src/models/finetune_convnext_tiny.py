"""
Fine-tuning script for ConvNeXt-Tiny to improve competition score.

Advanced techniques:
- AdamW optimizer with cosine annealing + linear warmup
- Label smoothing CrossEntropyLoss
- Mixup / CutMix data augmentation
- RandAugment-like augmentation for grayscale
- Model EMA (Exponential Moving Average)
- Gradient clipping
- Layer-wise learning rates (discriminative fine-tuning)
- Test-time augmentation (TTA) for predictions

Usage (single command):
    python -m src.models.finetune_convnext_tiny
    
    # Or with custom settings:
    python -m src.models.finetune_convnext_tiny --epochs 30 --lr 5e-5 --mixup 0.4 --cutmix 1.0

Expected improvement: 0.91 -> ~0.93-0.95+ with TTA
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import ssl
import certifi
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Fix SSL certificate verification for macOS
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler

from ..analysis.config import DATASET_CONFIG, OUTPUT_CONFIG
from ..analysis.utils import ensure_output_directories, load_labels
from .train_utils import EnsureNumChannels


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning ConvNeXt-Tiny."""
    model_name: str = "convnext_tiny_finetuned"
    base_model_name: str = "convnext_tiny"  # For loading pretrained weights
    input_channels: int = 1
    input_size: int = 224
    
    # Training hyperparameters - optimized for ConvNeXt
    epochs: int = 30
    batch_size: int = 32  # Smaller batch for better generalization
    lr: float = 5e-5  # Lower LR for fine-tuning pretrained model
    min_lr: float = 1e-7
    weight_decay: float = 0.05  # ConvNeXt benefits from higher weight decay
    warmup_epochs: int = 3
    warmup_lr: float = 1e-7
    
    # Regularization
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.2  # Stochastic depth - slightly higher for regularization
    
    # Data augmentation
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    cutmix_prob: float = 0.5
    mixup_switch_prob: float = 0.5
    use_randaug: bool = True
    
    # Model EMA
    use_ema: bool = True
    ema_decay: float = 0.9998
    
    # Other
    gradient_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42
    
    # Layer-wise LR multipliers for ConvNeXt stages
    # stages.0, stages.1 (early) -> lower LR
    # stages.2, stages.3 (later) -> higher LR
    # head -> full LR
    lr_mult_stages_0_1: float = 0.1
    lr_mult_stages_2: float = 0.3
    lr_mult_stages_3: float = 0.6


class OrganDatasetFinetune(Dataset):
    """Dataset with enhanced augmentation for fine-tuning."""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        images_dir: Path, 
        transform: transforms.Compose,
        is_training: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = self.images_dir / row.file
        
        # Load as PIL for better augmentation compatibility
        img = Image.open(image_path).convert("L")
        
        # Apply transforms
        img = self.transform(img)
        label = int(row.label)
        
        return img, label


class ModelEMA:
    """Model Exponential Moving Average for better generalization."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        
    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def build_convnext_tiny_finetune(num_classes: int, drop_path_rate: float = 0.2) -> nn.Module:
    """Build ConvNeXt-Tiny with configurable stochastic depth for fine-tuning."""
    model = timm.create_model(
        "convnext_tiny",
        pretrained=True,
        in_chans=1,  # Direct grayscale support
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def load_pretrained_weights(model: nn.Module, weights_path: Path, device: torch.device) -> bool:
    """Load pretrained weights if available."""
    if not weights_path.exists():
        print(f"[finetune] No pretrained weights found at {weights_path}, using ImageNet init")
        return False
    
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    
    # Handle both direct state_dict and checkpoint format
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    
    # Try to load, handling potential mismatches from different drop_path_rate
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"[finetune] Loaded pretrained weights from {weights_path}")
        return True
    except Exception as e:
        print(f"[finetune] Warning: Could not load weights: {e}")
        return False


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(config: FinetuneConfig, is_training: bool) -> transforms.Compose:
    """Get transforms for training or validation."""
    mean = [0.4669]
    std = [0.2796]
    
    if is_training:
        transform_list = [
            transforms.Resize((config.input_size, config.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        
        if config.use_randaug:
            # RandAugment-like transforms for grayscale medical images
            transform_list.extend([
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            EnsureNumChannels(config.input_channels),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),  # Cutout-like
        ])
    else:
        transform_list = [
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            EnsureNumChannels(config.input_channels),
            transforms.Normalize(mean=mean, std=std),
        ]
    
    return transforms.Compose(transform_list)


def get_tta_transforms(config: FinetuneConfig) -> List[transforms.Compose]:
    """Get transforms for Test-Time Augmentation."""
    mean = [0.4669]
    std = [0.2796]
    
    tta_transforms = []
    
    # Original
    tta_transforms.append(transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        EnsureNumChannels(config.input_channels),
        transforms.Normalize(mean=mean, std=std),
    ]))
    
    # Horizontal flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        EnsureNumChannels(config.input_channels),
        transforms.Normalize(mean=mean, std=std),
    ]))
    
    # Slight rotations
    for angle in [-5, 5, -10, 10]:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
            EnsureNumChannels(config.input_channels),
            transforms.Normalize(mean=mean, std=std),
        ]))
    
    # Scale variations
    for scale in [0.95, 1.05]:
        new_size = int(config.input_size * scale)
        tta_transforms.append(transforms.Compose([
            transforms.Resize((new_size, new_size)),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            EnsureNumChannels(config.input_channels),
            transforms.Normalize(mean=mean, std=std),
        ]))
    
    return tta_transforms


def get_layer_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """Get parameter groups for layer-wise learning rates.
    
    ConvNeXt structure: stem, stages.0, stages.1, stages.2, stages.3, head
    """
    groups = {
        "stem": [],
        "stages_0_1": [],
        "stages_2": [],
        "stages_3": [],
        "head": [],
    }
    
    for name, param in model.named_parameters():
        if "stem" in name:
            groups["stem"].append(param)
        elif "stages.0" in name or "stages.1" in name:
            groups["stages_0_1"].append(param)
        elif "stages.2" in name:
            groups["stages_2"].append(param)
        elif "stages.3" in name:
            groups["stages_3"].append(param)
        else:
            # head, norm_pre, and other layers
            groups["head"].append(param)
    
    return [
        groups["stem"],
        groups["stages_0_1"], 
        groups["stages_2"], 
        groups["stages_3"], 
        groups["head"]
    ]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    mixup_fn: Optional[Mixup],
    config: FinetuneConfig,
    ema: Optional[ModelEMA] = None,
) -> float:
    """Train for one epoch with mixup/cutmix."""
    model.train()
    running_loss = 0.0
    sample_count = 0
    
    for inputs, targets in tqdm(loader, desc="Train", unit="batch", leave=True, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Apply Mixup/CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        running_loss += loss.item() * inputs.size(0)
        sample_count += inputs.size(0)
    
    return running_loss / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Validate", unit="batch", leave=True, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    
    return running_loss / total, correct / total


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    dataset: OrganDatasetFinetune,
    device: torch.device,
    tta_transforms: List[transforms.Compose],
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with Test-Time Augmentation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for idx in tqdm(range(len(dataset)), desc="TTA Predict", unit="sample"):
        # Get original image path
        row = dataset.df.iloc[idx]
        image_path = dataset.images_dir / row.file
        img = Image.open(image_path).convert("L")
        label = int(row.label)
        
        # Aggregate predictions across TTA transforms
        logits_sum = None
        for transform in tta_transforms:
            input_tensor = transform(img).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            
            if logits_sum is None:
                logits_sum = outputs
            else:
                logits_sum += outputs
        
        # Average predictions
        avg_logits = logits_sum / len(tta_transforms)
        pred = avg_logits.argmax(dim=1).cpu().item()
        
        all_preds.append(pred)
        all_labels.append(label)
    
    return np.array(all_preds), np.array(all_labels)


def run_finetuning(config: FinetuneConfig) -> None:
    """Main fine-tuning loop."""
    ensure_output_directories()
    set_seed(config.seed)
    
    # Device selection - ConvNeXt works better on CPU for MPS compatibility issues
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    # ConvNeXt has known issues with MPS backend
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        # Try MPS but warn about potential issues
        print("[finetune] Warning: ConvNeXt on MPS may have issues. If training fails, consider CPU.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[finetune] Using device: {device}")
    
    # Load data
    train_df = load_labels(DATASET_CONFIG.train_labels)
    val_df = load_labels(DATASET_CONFIG.val_labels)
    num_classes = len(train_df["label"].unique())
    
    print(f"[finetune] Train samples: {len(train_df)}, Val samples: {len(val_df)}, Classes: {num_classes}")
    
    # Create datasets
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    train_dataset = OrganDatasetFinetune(train_df, DATASET_CONFIG.train_images, train_transform, is_training=True)
    val_dataset = OrganDatasetFinetune(val_df, DATASET_CONFIG.val_images, val_transform, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for mixup
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Build model
    model = build_convnext_tiny_finetune(num_classes, config.drop_path_rate)
    
    # Try to load pretrained weights from previous training
    weights_path = OUTPUT_CONFIG.models_root / f"{config.base_model_name}_weights.pth"
    if weights_path.exists():
        load_pretrained_weights(model, weights_path, device)
    else:
        # Check training_logs for best checkpoint
        logs_root = Path(__file__).resolve().parents[2] / "training_logs" / config.base_model_name
        try:
            candidates = sorted(logs_root.rglob("best_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                load_pretrained_weights(model, candidates[0], device)
        except Exception:
            pass
        print("[finetune] Using ImageNet pretrained weights as base")
    
    model = model.to(device)
    
    # Initialize EMA
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None
    
    # Setup Mixup/CutMix
    mixup_fn = None
    if config.mixup_alpha > 0 or config.cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_prob,
            switch_prob=config.mixup_switch_prob,
            mode='batch',
            label_smoothing=config.label_smoothing,
            num_classes=num_classes,
        )
    
    # Loss function
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    
    # Optimizer with layer-wise LR (discriminative fine-tuning)
    layer_groups = get_layer_groups(model)
    param_groups = [
        {"params": layer_groups[0], "lr": config.lr * config.lr_mult_stages_0_1},  # stem: lowest LR
        {"params": layer_groups[1], "lr": config.lr * config.lr_mult_stages_0_1},  # stages.0-1
        {"params": layer_groups[2], "lr": config.lr * config.lr_mult_stages_2},     # stages.2
        {"params": layer_groups[3], "lr": config.lr * config.lr_mult_stages_3},     # stages.3
        {"params": layer_groups[4], "lr": config.lr},                               # head: full LR
    ]
    optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    
    # Cosine scheduler with warmup
    n_iter_per_epoch = len(train_loader)
    num_steps = int(config.epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)
    
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.min_lr,
        warmup_lr_init=config.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "ema_val_accuracy": [],
    }
    
    best_val_acc = 0.0
    best_ema_acc = 0.0
    
    # Create output directory for this run
    run_dir = Path(__file__).resolve().parents[2] / "training_logs" / config.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[finetune] Starting fine-tuning for {config.epochs} epochs...")
    print(f"[finetune] LR schedule: stem/stages.0-1={config.lr * config.lr_mult_stages_0_1:.2e}, "
          f"stages.2={config.lr * config.lr_mult_stages_2:.2e}, "
          f"stages.3={config.lr * config.lr_mult_stages_3:.2e}, "
          f"head={config.lr:.2e}")
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_fn, config, ema)
        
        # Update scheduler (per-iteration)
        for i in range(n_iter_per_epoch):
            scheduler.step_update(epoch * n_iter_per_epoch + i)
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        # Evaluate EMA model
        ema_val_accuracy = 0.0
        if ema is not None:
            ema.model.to(device)
            _, ema_val_accuracy = evaluate(ema.model, val_loader, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["ema_val_accuracy"].append(ema_val_accuracy)
        
        print(
            f"[{config.model_name}] Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} ema_acc={ema_val_accuracy:.4f}"
        )
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "config": {
                    "model_name": config.model_name,
                    "drop_path_rate": config.drop_path_rate,
                    "input_size": config.input_size,
                },
            }
            torch.save(checkpoint, run_dir / f"best_{config.model_name}.pth")
            print(f"[finetune] New best model saved: val_acc={val_accuracy:.4f}")
        
        # Save best EMA model
        if ema is not None and ema_val_accuracy > best_ema_acc:
            best_ema_acc = ema_val_accuracy
            torch.save(ema.state_dict(), run_dir / f"best_ema_{config.model_name}.pth")
            print(f"[finetune] New best EMA model saved: ema_acc={ema_val_accuracy:.4f}")
    
    # Save final models
    torch.save(model.state_dict(), run_dir / f"last_{config.model_name}.pth")
    if ema is not None:
        torch.save(ema.state_dict(), run_dir / f"last_ema_{config.model_name}.pth")
    
    # Also save to analysis_outputs/models for prediction compatibility
    OUTPUT_CONFIG.models_root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_CONFIG.models_root / f"{config.model_name}_weights.pth")
    
    # Evaluation with TTA on best model
    print("\n[finetune] Running final evaluation with TTA on best model...")
    best_ckpt = torch.load(run_dir / f"best_{config.model_name}.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    
    tta_transforms = get_tta_transforms(config)
    print(f"[finetune] Using {len(tta_transforms)} TTA transforms...")
    preds_tta, labels_tta = predict_with_tta(model, val_dataset, device, tta_transforms)
    tta_accuracy = (preds_tta == labels_tta).mean()
    print(f"[finetune] TTA Validation Accuracy: {tta_accuracy:.4f}")
    
    # Also evaluate EMA with TTA if available
    ema_tta_accuracy = 0.0
    if ema is not None and (run_dir / f"best_ema_{config.model_name}.pth").exists():
        print("[finetune] Running TTA evaluation on best EMA model...")
        ema_state = torch.load(run_dir / f"best_ema_{config.model_name}.pth", map_location=device)
        ema_model = build_convnext_tiny_finetune(num_classes, config.drop_path_rate).to(device)
        ema_model.load_state_dict(ema_state)
        preds_ema_tta, _ = predict_with_tta(ema_model, val_dataset, device, tta_transforms)
        ema_tta_accuracy = (preds_ema_tta == labels_tta).mean()
        print(f"[finetune] EMA TTA Validation Accuracy: {ema_tta_accuracy:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(labels_tta, preds_tta)
    np.save(OUTPUT_CONFIG.models_root / f"confusion_matrix_{config.model_name}.npy", cm)
    
    # Save per-class accuracy
    class_labels = sorted(train_df["label"].unique())
    per_class_acc = {}
    for label in class_labels:
        mask = labels_tta == label
        per_class_acc[label] = float((preds_tta[mask] == label).mean()) if mask.any() else None
    per_class_df = pd.DataFrame({"label": list(per_class_acc.keys()), "accuracy": list(per_class_acc.values())})
    per_class_df.to_json(OUTPUT_CONFIG.models_root / f"per_class_accuracy_{config.model_name}.json", orient="records", indent=2)
    
    # Save training curves
    plt.figure(figsize=(14, 4))
    epochs_arr = np.arange(1, config.epochs + 1)
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_arr, history["train_loss"], label="Train Loss")
    plt.plot(epochs_arr, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_arr, history["val_accuracy"], label="Val Accuracy")
    plt.plot(epochs_arr, history["ema_val_accuracy"], label="EMA Val Accuracy", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    labels = ["Best Val", "Best EMA", "TTA", "EMA+TTA"]
    values = [best_val_acc, best_ema_acc, tta_accuracy, ema_tta_accuracy]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    bars = plt.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.ylabel("Accuracy")
    plt.title("Final Accuracies")
    plt.ylim(min(values) - 0.02, max(values) + 0.02)
    
    plt.tight_layout()
    plt.savefig(run_dir / f"training_curves_{config.model_name}.png", dpi=150)
    plt.savefig(OUTPUT_CONFIG.models_root / f"training_curves_{config.model_name}.png", dpi=150)
    plt.close()
    
    # Save summary
    summary = {
        "model_name": config.model_name,
        "base_model": config.base_model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "mixup_alpha": config.mixup_alpha,
        "cutmix_alpha": config.cutmix_alpha,
        "use_ema": config.use_ema,
        "ema_decay": config.ema_decay,
        "drop_path_rate": config.drop_path_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_accuracy": best_val_acc,
        "best_ema_accuracy": best_ema_acc,
        "tta_accuracy": float(tta_accuracy),
        "ema_tta_accuracy": float(ema_tta_accuracy),
        "input_size": config.input_size,
        "input_channels": config.input_channels,
    }
    
    with open(run_dir / f"{config.model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUTPUT_CONFIG.models_root / f"{config.model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"[finetune] Fine-tuning complete!")
    print(f"{'='*60}")
    print(f"[finetune] Best Val Accuracy:     {best_val_acc:.4f}")
    print(f"[finetune] Best EMA Accuracy:     {best_ema_acc:.4f}")
    print(f"[finetune] TTA Accuracy:          {tta_accuracy:.4f}")
    print(f"[finetune] EMA + TTA Accuracy:    {ema_tta_accuracy:.4f}")
    print(f"{'='*60}")
    print(f"[finetune] Models saved to: {run_dir}")
    print(f"\n[finetune] To generate test predictions, run:")
    print(f"    python -m src.models.predict_convnext_finetune")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ConvNeXt-Tiny for improved competition score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training params
    parser.add_argument("--epochs", type=int, default=30, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    
    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--drop-path", type=float, default=0.2, help="Stochastic depth rate")
    
    # Augmentation
    parser.add_argument("--mixup", type=float, default=0.4, help="Mixup alpha")
    parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha")
    parser.add_argument("--no-randaug", action="store_true", help="Disable RandAugment-like transforms")
    
    # EMA
    parser.add_argument("--no-ema", action="store_true", help="Disable Model EMA")
    parser.add_argument("--ema-decay", type=float, default=0.9998, help="EMA decay rate")
    
    # Other
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        drop_path_rate=args.drop_path,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        use_randaug=not args.no_randaug,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        gradient_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    run_finetuning(config)


if __name__ == "__main__":
    main()
    