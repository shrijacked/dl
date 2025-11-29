"""
Generate test predictions using the fine-tuned Swin-Tiny model with TTA.

Usage:
    python -m src.models.predict_finetune
    
    # With specific options:
    python -m src.models.predict_finetune --no-tta --use-ema
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import timm

from ..analysis.config import DATASET_CONFIG, OUTPUT_CONFIG
from ..analysis.utils import ensure_output_directories
from .train_utils import EnsureNumChannels


class TestDataset(Dataset):
    """Dataset for test predictions."""
    
    def __init__(self, manifest: pd.DataFrame, images_dir: Path, transform: transforms.Compose) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img = Image.open(self.images_dir / row.file).convert("L")
        img = self.transform(img)
        return int(row["index"]), row.file, img


def build_model(num_classes: int = 11) -> nn.Module:
    """Build Swin-Tiny model for inference."""
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
    )
    return model


def get_base_transform(input_size: int = 224) -> transforms.Compose:
    """Get base transform for inference."""
    mean = [0.4669]
    std = [0.2796]
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        EnsureNumChannels(1),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_tta_transforms(input_size: int = 224) -> List[transforms.Compose]:
    """Get TTA transforms for inference."""
    mean = [0.4669]
    std = [0.2796]
    
    tta_list = []
    
    # Original
    tta_list.append(transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        EnsureNumChannels(1),
        transforms.Normalize(mean=mean, std=std),
    ]))
    
    # Horizontal flip
    tta_list.append(transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        EnsureNumChannels(1),
        transforms.Normalize(mean=mean, std=std),
    ]))
    
    # Small rotations
    for angle in [-5, 5]:
        tta_list.append(transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
            EnsureNumChannels(1),
            transforms.Normalize(mean=mean, std=std),
        ]))
    
    # Scale variations
    for scale in [0.95, 1.05]:
        new_size = int(input_size * scale)
        tta_list.append(transforms.Compose([
            transforms.Resize((new_size, new_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            EnsureNumChannels(1),
            transforms.Normalize(mean=mean, std=std),
        ]))
    
    return tta_list


def find_best_checkpoint(model_name: str = "swin_tiny_finetuned") -> Optional[Path]:
    """Find the best checkpoint for the fine-tuned model."""
    # Priority 1: Check analysis_outputs/models
    weights_path = OUTPUT_CONFIG.models_root / f"{model_name}_weights.pth"
    if weights_path.exists():
        return weights_path
    
    # Priority 2: Check training_logs for best checkpoint
    logs_root = Path(__file__).resolve().parents[2] / "training_logs" / model_name
    
    # Try best checkpoint first
    best_ckpt = logs_root / f"best_{model_name}.pth"
    if best_ckpt.exists():
        return best_ckpt
    
    # Try EMA checkpoint
    best_ema = logs_root / f"best_ema_{model_name}.pth"
    if best_ema.exists():
        return best_ema
    
    # Fallback: find any best_*.pth
    try:
        candidates = sorted(logs_root.rglob("best_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    except Exception:
        pass
    
    return None


def load_model_weights(model: nn.Module, ckpt_path: Path, device: torch.device, use_ema: bool = False) -> None:
    """Load model weights from checkpoint."""
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "model_state" in state:
            state = state["model_state"]
        elif "state_dict" in state:
            state = state["state_dict"]
    
    model.load_state_dict(state)
    print(f"[predict] Loaded weights from {ckpt_path}")


@torch.no_grad()
def predict_batch(model: nn.Module, loader: DataLoader, device: torch.device):
    """Make predictions for a batch without TTA."""
    model.eval()
    
    indices = []
    files = []
    preds = []
    probs = []
    
    for batch in tqdm(loader, desc="Predicting", unit="batch", dynamic_ncols=True):
        b_indices, b_files, inputs = batch
        inputs = inputs.to(device, non_blocking=True)
        
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = outputs.argmax(dim=1).cpu().numpy().astype(int)
        
        indices.extend([int(i) for i in b_indices])
        files.extend(list(b_files))
        preds.extend([int(p) for p in predictions])
        probs.extend(list(probabilities))
    
    return indices, files, preds, probs


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    manifest: pd.DataFrame,
    images_dir: Path,
    device: torch.device,
    tta_transforms: List[transforms.Compose],
) -> tuple:
    """Make predictions with Test-Time Augmentation."""
    model.eval()
    
    indices = []
    files = []
    preds = []
    probs = []
    
    for idx in tqdm(range(len(manifest)), desc="TTA Predicting", unit="sample", dynamic_ncols=True):
        row = manifest.iloc[idx]
        image_path = images_dir / row.file
        img = Image.open(image_path).convert("L")
        
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
        probabilities = torch.softmax(avg_logits, dim=1).cpu().numpy()[0]
        pred = avg_logits.argmax(dim=1).cpu().item()
        
        indices.append(int(row["index"]))
        files.append(row.file)
        preds.append(int(pred))
        probs.append(probabilities)
    
    return indices, files, preds, probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions with fine-tuned Swin-Tiny")
    parser.add_argument("--model-name", type=str, default="swin_tiny_finetuned", help="Model name")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA model weights")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (only without TTA)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint")
    args = parser.parse_args()
    
    ensure_output_directories()
    
    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[predict] Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        if args.use_ema:
            # Try EMA checkpoint first
            logs_root = Path(__file__).resolve().parents[2] / "training_logs" / args.model_name
            ckpt_path = logs_root / f"best_ema_{args.model_name}.pth"
            if not ckpt_path.exists():
                ckpt_path = find_best_checkpoint(args.model_name)
        else:
            ckpt_path = find_best_checkpoint(args.model_name)
    
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {args.model_name}. "
            f"Run fine-tuning first: python -m src.models.finetune_swin_tiny"
        )
    
    # Load model
    model = build_model(num_classes=11)
    load_model_weights(model, ckpt_path, device, use_ema=args.use_ema)
    model = model.to(device)
    model.eval()
    
    # Load test manifest
    manifest = pd.read_csv(DATASET_CONFIG.test_manifest)
    if not {"index", "file"}.issubset(manifest.columns):
        raise ValueError(f"Test manifest missing required columns")
    manifest = manifest.sort_values("index").reset_index(drop=True)
    
    print(f"[predict] Test samples: {len(manifest)}")
    
    # Make predictions
    if args.no_tta:
        # Standard batch prediction
        transform = get_base_transform()
        dataset = TestDataset(manifest, DATASET_CONFIG.test_images, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        indices, files, preds, probs = predict_batch(model, loader, device)
    else:
        # TTA prediction
        tta_transforms = get_tta_transforms()
        print(f"[predict] Using {len(tta_transforms)} TTA transforms")
        indices, files, preds, probs = predict_with_tta(
            model, manifest, DATASET_CONFIG.test_images, device, tta_transforms
        )
    
    # Create output directory
    out_dir = OUTPUT_CONFIG.models_root / "predictions" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save submission
    sub_df = pd.DataFrame({"index": indices, "id": preds}).sort_values("index")
    sub_df.to_csv(out_dir / "submission.csv", index=False)
    print(f"[predict] Submission saved to {out_dir / 'submission.csv'}")
    
    # Save detailed predictions
    prob_cols = {f"p{i}": [float(p[i]) for p in probs] for i in range(11)}
    detail_df = pd.DataFrame({
        "index": indices,
        "file": files,
        "pred": preds,
        **prob_cols
    }).sort_values("index")
    detail_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"[predict] Detailed predictions saved to {out_dir / 'test_predictions.csv'}")
    
    print(f"\n[predict] Done! Submission ready at: {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

