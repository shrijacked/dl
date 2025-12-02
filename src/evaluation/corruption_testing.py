#!/usr/bin/env python3
"""
Actual Corruption Robustness Testing

Runs all 11 models on ImageNet-C style corrupted validation data.
Generates corruptions on-the-fly to avoid storage overhead.

Estimated time on M2 Pro: 3-5 hours for all models

Usage:
    python -m src.evaluation.corruption_testing                    # All models
    python -m src.evaluation.corruption_testing --model resnet50   # Single model
    python -m src.evaluation.corruption_testing --resume           # Resume from checkpoint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Corruption functions
from skimage import filters, util
from scipy.ndimage import gaussian_filter, map_coordinates
import warnings
warnings.filterwarnings('ignore')

# Project imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.evaluation.config import EVAL_CONFIG, MODEL_NAMES, CLASS_NAMES


# ============================================================================
# CORRUPTION FUNCTIONS (ImageNet-C style, adapted for grayscale)
# ============================================================================

def gaussian_noise(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Add Gaussian noise."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    img = np.array(img) / 255.
    noisy = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1)
    return (noisy * 255).astype(np.uint8)


def shot_noise(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Add shot (Poisson) noise."""
    c = [60, 25, 12, 5, 3][severity - 1]
    img = np.array(img) / 255.
    noisy = np.clip(np.random.poisson(img * c) / float(c), 0, 1)
    return (noisy * 255).astype(np.uint8)


def impulse_noise(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Add impulse (salt and pepper) noise."""
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    img = np.array(img).copy()
    flat = img.flatten()
    n_salt = int(c * len(flat) / 2)
    n_pepper = int(c * len(flat) / 2)
    salt_idx = np.random.choice(len(flat), n_salt, replace=False)
    pepper_idx = np.random.choice(len(flat), n_pepper, replace=False)
    flat[salt_idx] = 255
    flat[pepper_idx] = 0
    return flat.reshape(img.shape)


def defocus_blur(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply defocus blur."""
    c = [3, 4, 6, 8, 10][severity - 1]
    img = np.array(img, dtype=np.float32)
    blurred = gaussian_filter(img, sigma=c)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def glass_blur(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply glass blur effect."""
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    sigma, max_delta, iterations = c
    
    img = np.array(img, dtype=np.float32)
    for _ in range(iterations):
        for h in range(img.shape[0] - max_delta):
            for w in range(img.shape[1] - max_delta):
                dx, dy = np.random.randint(-max_delta, max_delta + 1, 2)
                h_prime, w_prime = h + dy, w + dx
                h_prime = np.clip(h_prime, 0, img.shape[0] - 1)
                w_prime = np.clip(w_prime, 0, img.shape[1] - 1)
                img[h, w], img[h_prime, w_prime] = img[h_prime, w_prime], img[h, w]
    
    blurred = gaussian_filter(img, sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply motion blur."""
    c = [10, 15, 15, 15, 20][severity - 1]
    img = np.array(img, dtype=np.float32)
    
    # Simple horizontal motion blur kernel
    kernel = np.zeros((c, c))
    kernel[c // 2, :] = 1.0 / c
    
    from scipy.ndimage import convolve
    blurred = convolve(img, kernel, mode='reflect')
    return np.clip(blurred, 0, 255).astype(np.uint8)


def zoom_blur(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply zoom blur."""
    c = [1.11, 1.16, 1.21, 1.26, 1.31][severity - 1]
    
    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]
    
    out = np.zeros_like(img, dtype=np.float32)
    for zoom in np.linspace(1, c, 10):
        zh, zw = int(h / zoom), int(w / zoom)
        top, left = (h - zh) // 2, (w - zw) // 2
        
        cropped = img[top:top+zh, left:left+zw]
        from scipy.ndimage import zoom as scipy_zoom
        zoomed = scipy_zoom(cropped, zoom, order=1)
        
        # Crop to original size
        zh2, zw2 = zoomed.shape[:2]
        top2, left2 = (zh2 - h) // 2, (zw2 - w) // 2
        out += zoomed[top2:top2+h, left2:left2+w]
    
    return np.clip(out / 10, 0, 255).astype(np.uint8)


def snow(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Add snow effect."""
    c = [(0.1, 0.2, 1, 0.6, 8, 3, 0.8),
         (0.1, 0.2, 1, 0.5, 10, 4, 0.8),
         (0.15, 0.3, 1.75, 0.55, 10, 4, 0.7),
         (0.25, 0.3, 2.25, 0.6, 12, 6, 0.65),
         (0.3, 0.3, 1.25, 0.65, 14, 12, 0.6)][severity - 1]
    
    img = np.array(img, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=img.shape, loc=c[0], scale=c[1])
    snow_layer = np.clip(snow_layer, 0, 1)
    snow_layer = gaussian_filter(snow_layer, sigma=c[4])
    
    img = c[6] * img + (1 - c[6]) * np.maximum(img, snow_layer * c[2])
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def frost(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Add frost effect."""
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    
    img = np.array(img, dtype=np.float32) / 255.
    
    # Generate frost pattern
    frost_pattern = np.random.uniform(0.8, 1.0, size=img.shape)
    frost_pattern = gaussian_filter(frost_pattern, sigma=2)
    
    img = c[0] * img + c[1] * frost_pattern * img
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def fog(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Add fog effect."""
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    
    img = np.array(img, dtype=np.float32) / 255.
    
    # Generate fog
    h, w = img.shape[:2]
    fog_layer = np.ones_like(img) * 0.8
    
    img = img + c[1] * fog_layer
    img = np.clip(img / (1 + c[1]), 0, 1)
    
    return (img * 255).astype(np.uint8)


def brightness(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Adjust brightness."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    
    img = np.array(img, dtype=np.float32) / 255.
    img = img + c
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Adjust contrast."""
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    
    img = np.array(img, dtype=np.float32) / 255.
    mean = np.mean(img)
    img = (img - mean) * c + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def elastic_transform(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply elastic transformation."""
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
    
    img = np.array(img, dtype=np.float32)
    shape = img.shape
    
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), c[1]) * c[0]
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), c[1]) * c[0]
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.clip(y + dy, 0, shape[0] - 1).astype(int),
               np.clip(x + dx, 0, shape[1] - 1).astype(int))
    
    return img[indices].astype(np.uint8)


def pixelate(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Pixelate the image."""
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    
    h, w = img.shape[:2]
    img = Image.fromarray(img)
    img = img.resize((int(w * c), int(h * c)), Image.BOX)
    img = img.resize((w, h), Image.NEAREST)
    return np.array(img)


def jpeg_compression(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    c = [25, 18, 15, 10, 7][severity - 1]
    
    from io import BytesIO
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=c)
    buffer.seek(0)
    img = Image.open(buffer)
    return np.array(img)


# Corruption registry
CORRUPTIONS = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'glass_blur': glass_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'snow': snow,
    'frost': frost,
    'fog': fog,
    'brightness': brightness,
    'contrast': contrast,
    'elastic_transform': elastic_transform,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression,
}


# ============================================================================
# DATASET AND MODEL LOADING
# ============================================================================

class CorruptedDataset(Dataset):
    """Dataset that applies corruption on-the-fly."""
    
    def __init__(
        self, 
        images_dir: Path,
        labels_csv: Path,
        corruption: str,
        severity: int = 3,
        transform=None
    ):
        self.images_dir = Path(images_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.corrupt_fn = CORRUPTIONS.get(corruption)
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = self.images_dir / row['file']
        label = int(row['label'])
        
        # Load grayscale image
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        
        # Apply corruption
        if self.corrupt_fn:
            img_array = self.corrupt_fn(img_array, self.severity)
        
        # Convert back to PIL and apply transforms
        img = Image.fromarray(img_array, mode='L')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_device() -> torch.device:
    """Get best available device (MPS for M2 Pro)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Load a trained model."""
    from src.model_architectures import get_model
    
    # Load model architecture
    model = get_model(model_name, num_classes=11, pretrained=False)
    
    # Find weights file
    weights_path = EVAL_CONFIG.results_root / f"{model_name}_weights.pth"
    
    if not weights_path.exists():
        # Try alternative locations
        alt_paths = [
            EVAL_CONFIG.results_root / f"best_{model_name}.pth",
            EVAL_CONFIG.training_logs / model_name / f"best_{model_name}.pth",
        ]
        for alt in alt_paths:
            if alt.exists():
                weights_path = alt
                break
    
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        # Handle different state dict formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from {weights_path.name}")
    else:
        print(f"  Warning: No weights found for {model_name}")
    
    model = model.to(device)
    model.eval()
    return model


def evaluate_model_on_corruption(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on corrupted data. Returns (accuracy, inference_time_ms)."""
    correct = 0
    total = 0
    total_time = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            start_time = time.time()
            outputs = model(images)
            if device.type == 'mps':
                torch.mps.synchronize()
            total_time += time.time() - start_time
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    avg_time_ms = (total_time / total) * 1000
    
    return accuracy, avg_time_ms


# ============================================================================
# MAIN TESTING LOGIC
# ============================================================================

def run_corruption_testing(
    models: Optional[List[str]] = None,
    severity: int = 3,
    batch_size: int = 32,
    num_workers: int = 4,
    resume: bool = False
) -> None:
    """Run actual corruption testing on all specified models."""
    
    print("=" * 70)
    print("     CORRUPTION ROBUSTNESS TESTING (Actual Inference)")
    print("=" * 70)
    
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Severity: {severity}")
    print(f"Batch size: {batch_size}")
    
    # Setup paths
    val_images = project_root / "dataset" / "val" / "images_val"
    val_labels = project_root / "dataset" / "val" / "labels_val.csv"
    
    if not val_images.exists():
        print(f"Error: Validation images not found at {val_images}")
        return
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale to 3-channel
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Models to evaluate
    if models is None:
        models = MODEL_NAMES
    
    # Results storage
    results_path = EVAL_CONFIG.tables_dir / "corruption_robustness_actual.csv"
    checkpoint_path = EVAL_CONFIG.reports_dir / "corruption_checkpoint.json"
    
    # Load checkpoint if resuming
    completed = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            completed = json.load(f)
        print(f"\nResuming from checkpoint ({len(completed)} model-corruption pairs completed)")
    
    all_results = []
    corruptions = list(CORRUPTIONS.keys())
    
    total_combinations = len(models) * len(corruptions)
    completed_count = len(completed)
    
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining: {total_combinations - completed_count}")
    
    start_time = time.time()
    
    for model_idx, model_name in enumerate(models):
        print(f"\n{'='*60}")
        print(f"Model {model_idx + 1}/{len(models)}: {model_name}")
        print('='*60)
        
        try:
            model = load_model(model_name, device)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        model_results = {'model': model_name}
        
        for corr_idx, corruption in enumerate(corruptions):
            key = f"{model_name}_{corruption}"
            
            if key in completed:
                model_results[corruption] = completed[key]
                print(f"  [{corr_idx+1}/{len(corruptions)}] {corruption}: {completed[key]*100:.2f}% (cached)")
                continue
            
            # Create dataset and dataloader
            dataset = CorruptedDataset(
                val_images, val_labels, corruption, severity, transform
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=True
            )
            
            # Evaluate
            try:
                accuracy, avg_time = evaluate_model_on_corruption(model, dataloader, device)
                model_results[corruption] = accuracy
                completed[key] = accuracy
                
                print(f"  [{corr_idx+1}/{len(corruptions)}] {corruption}: {accuracy*100:.2f}% ({avg_time:.2f} ms/img)")
                
                # Save checkpoint
                with open(checkpoint_path, 'w') as f:
                    json.dump(completed, f, indent=2)
                    
            except Exception as e:
                print(f"  [{corr_idx+1}/{len(corruptions)}] {corruption}: ERROR - {e}")
                model_results[corruption] = None
        
        # Compute summary stats
        valid_accs = [v for k, v in model_results.items() if k != 'model' and v is not None]
        if valid_accs:
            model_results['mean_accuracy'] = np.mean(valid_accs)
            model_results['std_accuracy'] = np.std(valid_accs)
        
        all_results.append(model_results)
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        pairs_done = len(completed) - completed_count
        if pairs_done > 0:
            avg_per_pair = elapsed / pairs_done
            remaining_pairs = total_combinations - len(completed)
            eta_seconds = remaining_pairs * avg_per_pair
            eta_minutes = eta_seconds / 60
            print(f"\n  ETA: {eta_minutes:.1f} minutes remaining")
        
        # Free memory
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Save final results
    EVAL_CONFIG.ensure_directories()
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mean_accuracy', ascending=False)
    results_df.to_csv(results_path, index=False)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("CORRUPTION TESTING COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n=== Robustness Ranking (Actual) ===")
    for _, row in results_df.head(11).iterrows():
        if 'mean_accuracy' in row and pd.notna(row['mean_accuracy']):
            print(f"  {row['model']:30s} | Mean: {row['mean_accuracy']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Corruption Robustness Testing")
    parser.add_argument('--model', type=str, default=None, help='Single model to test')
    parser.add_argument('--severity', type=int, default=3, choices=[1,2,3,4,5])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    models = [args.model] if args.model else None
    
    run_corruption_testing(
        models=models,
        severity=args.severity,
        batch_size=args.batch_size,
        num_workers=args.workers,
        resume=args.resume
    )


if __name__ == "__main__":
    main()

