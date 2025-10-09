from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_json


@dataclass
class FrequencyMetrics:
    split: str
    samples: int
    mean_low_freq: float
    mean_high_freq: float
    high_to_low_ratio: float


@dataclass
class AttackResult:
    attack: str
    epsilon: float
    step_size: Optional[float]
    steps: Optional[int]
    accuracy: float


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: Optional[List[int]] = None, image_size: int = 64) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        path = self.image_paths[idx]
        image = read_image(path).resize((self.image_size, self.image_size))
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.labels is None:
            return tensor, None
        return tensor, int(self.labels[idx])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_image_paths(split: str, max_samples: Optional[int], random_state: int) -> Tuple[List[Path], Optional[List[int]]]:
    rng = np.random.default_rng(random_state)
    if split == "train":
        labels_df = load_labels(DATASET_CONFIG.train_labels)
        if max_samples is not None and len(labels_df) > max_samples:
            labels_df = labels_df.sample(max_samples, random_state=random_state)
        paths = [DATASET_CONFIG.train_images / name for name in labels_df["file"]]
        labels = labels_df["label"].astype(int).tolist()
        return paths, labels
    if split == "val":
        labels_df = load_labels(DATASET_CONFIG.val_labels)
        if max_samples is not None and len(labels_df) > max_samples:
            labels_df = labels_df.sample(max_samples, random_state=random_state)
        paths = [DATASET_CONFIG.val_images / name for name in labels_df["file"]]
        labels = labels_df["label"].astype(int).tolist()
        return paths, labels
    if split == "test":
        if DATASET_CONFIG.test_manifest.exists():
            manifest = pd.read_csv(DATASET_CONFIG.test_manifest)
            files = manifest["file"].tolist()
        else:
            files = [p.name for p in sorted(DATASET_CONFIG.test_images.glob("*.png"))]
        if max_samples is not None and len(files) > max_samples:
            indices = rng.choice(len(files), size=max_samples, replace=False)
            files = [files[i] for i in indices]
        paths = [DATASET_CONFIG.test_images / name for name in files]
        return paths, None
    raise ValueError(f"Unsupported split {split}")


def _radial_energy(arr: np.ndarray) -> Tuple[float, float]:
    fft = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.abs(fft)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = np.sqrt(cx ** 2 + cy ** 2)
    low_threshold = max_radius * 0.2
    high_threshold = max_radius * 0.6

    low_mask = radius <= low_threshold
    high_mask = radius >= high_threshold

    low_energy = float(magnitude[low_mask].mean()) if low_mask.any() else float("nan")
    high_energy = float(magnitude[high_mask].mean()) if high_mask.any() else float("nan")
    return low_energy, high_energy


def _run_frequency_analysis(sample_per_split: int, random_state: int) -> List[FrequencyMetrics]:
    metrics: List[FrequencyMetrics] = []
    for split in ["train", "val", "test"]:
        paths, _ = _sample_image_paths(split, sample_per_split, random_state)
        if not paths:
            continue
        lows: List[float] = []
        highs: List[float] = []
        for path in tqdm(paths, desc=f"FFT analysis ({split})", unit="img"):
            arr = np.array(read_image(path), dtype=np.float32) / 255.0
            low, high = _radial_energy(arr)
            lows.append(low)
            highs.append(high)
        lows_arr = np.array(lows)
        highs_arr = np.array(highs)
        ratio = float(np.nanmean(highs_arr / np.maximum(lows_arr, 1e-8)))
        metrics.append(FrequencyMetrics(
            split=split,
            samples=len(paths),
            mean_low_freq=float(np.nanmean(lows_arr)),
            mean_high_freq=float(np.nanmean(highs_arr)),
            high_to_low_ratio=ratio,
        ))

    out_path = OUTPUT_CONFIG.reports / "robustness_frequency_metrics.json"
    save_json({m.split: m.__dict__ for m in metrics}, out_path)
    return metrics


def _plot_average_spectrum(sample_paths: List[Path], split: str, out_path: Path) -> None:
    if not sample_paths:
        return
    spectra = []
    for path in tqdm(sample_paths, desc=f"Average spectrum ({split})", unit="img"):
        arr = np.array(read_image(path), dtype=np.float32) / 255.0
        fft = np.fft.fftshift(np.fft.fft2(arr))
        log_mag = np.log(np.abs(fft) + 1e-6)
        spectra.append(log_mag)
    avg = np.mean(np.stack(spectra), axis=0)
    plt.figure(figsize=(4, 4))
    plt.imshow(avg, cmap="inferno")
    plt.title(f"Average log-spectrum ({split})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _prepare_dataloaders(train_paths: List[Path], train_labels: List[int], val_paths: List[Path], val_labels: List[int], batch_size: int, num_workers: int, image_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ImageDataset(train_paths, train_labels, image_size=image_size)
    val_dataset = ImageDataset(val_paths, val_labels, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def _train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.as_tensor(targets, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            pbar.set_postfix(loss=loss.item())
        history["train_loss"].append(running_loss / total)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = torch.as_tensor(targets, device=device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)
        history["val_loss"].append(val_loss / total)
        history["val_acc"].append(correct / total)
        print(f"Epoch {epoch + 1}: train_loss={history['train_loss'][-1]:.4f} val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}")

    return history


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.as_tensor(targets, device=device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
    return correct / total if total else 0.0


def _fgsm_attack(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()
    perturbed = inputs + epsilon * inputs.grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0).detach()


def _pgd_attack(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float, step_size: float, steps: int) -> torch.Tensor:
    ori = inputs.clone().detach()
    perturbed = inputs.clone().detach()
    for _ in range(steps):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        with torch.no_grad():
            perturbed = perturbed + step_size * perturbed.grad.sign()
            perturbed = torch.max(torch.min(perturbed, ori + epsilon), ori - epsilon)
            perturbed = torch.clamp(perturbed, 0.0, 1.0)
    return perturbed.detach()


def _run_attack(model: nn.Module, loader: DataLoader, device: torch.device, attack: str, epsilon: float, step_size: Optional[float] = None, steps: Optional[int] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc=f"Attack {attack} eps={epsilon:.3f}"):
        inputs = inputs.to(device)
        targets = torch.as_tensor(targets, device=device)
        if attack == "fgsm":
            adv = _fgsm_attack(model, inputs, targets, epsilon)
        elif attack == "pgd":
            if step_size is None or steps is None:
                raise ValueError("PGD attack requires step_size and steps")
            adv = _pgd_attack(model, inputs, targets, epsilon, step_size, steps)
        else:
            raise ValueError(f"Unsupported attack {attack}")
        outputs = model(adv)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)
    return correct / total if total else 0.0


def _save_attack_samples(model: nn.Module, loader: DataLoader, device: torch.device, epsilon: float, out_dir: Path) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = torch.as_tensor(targets, device=device)
        adv = _pgd_attack(model, inputs, targets, epsilon=epsilon, step_size=epsilon / 4, steps=10)
        for j in range(min(inputs.size(0), 4)):
            clean = inputs[j].detach().cpu().squeeze().numpy()
            adv_img = adv[j].detach().cpu().squeeze().numpy()
            diff = np.abs(clean - adv_img)
            fig, axes = plt.subplots(1, 3, figsize=(6, 2))
            axes[0].imshow(clean, cmap="gray")
            axes[0].set_title("clean")
            axes[1].imshow(adv_img, cmap="gray")
            axes[1].set_title("adv")
            axes[2].imshow(diff, cmap="inferno")
            axes[2].set_title("|diff|")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{idx:03d}_{j}.png")
            plt.close()
        if idx >= 2:
            break


def _occlusion_map(model: nn.Module, image: torch.Tensor, target: int, device: torch.device, patch_size: int = 8, stride: int = 4) -> np.ndarray:
    model.eval()
    _, _, h, w = image.shape
    base_prob = torch.softmax(model(image.to(device)), dim=1)[0, target].item()
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = image.clone()
            occluded[:, :, y : y + patch_size, x : x + patch_size] = 0.0
            prob = torch.softmax(model(occluded.to(device)), dim=1)[0, target].item()
            delta = base_prob - prob
            heatmap[y : y + patch_size, x : x + patch_size] += delta
            counts[y : y + patch_size, x : x + patch_size] += 1

    counts[counts == 0] = 1
    heatmap /= counts
    return heatmap


def _run_occlusion_analysis(model: nn.Module, val_loader: DataLoader, device: torch.device, output_dir: Path, num_samples: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    collected = 0
    for inputs, targets in val_loader:
        for idx in range(inputs.size(0)):
            image = inputs[idx : idx + 1]
            target = int(targets[idx])
            heatmap = _occlusion_map(model, image, target, device)
            img_np = image.squeeze().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title(f"label={target}")
            im = axes[1].imshow(heatmap, cmap="Reds")
            axes[1].set_title("occlusion delta")
            axes[1].axis("off")
            axes[0].axis("off")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(output_dir / f"occlusion_{collected:02d}.png")
            plt.close()

            collected += 1
            if collected >= num_samples:
                return


def run_robustness_deepdive(
    frequency_samples: int = 512,
    train_samples: int = 6000,
    val_samples: int = 2000,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    image_size: int = 64,
    random_state: int = 42,
) -> None:
    ensure_output_directories()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Frequency domain analysis
    print("Running frequency domain analysis...")
    freq_metrics = _run_frequency_analysis(frequency_samples, random_state)
    for metric in freq_metrics:
        sample_paths, _ = _sample_image_paths(metric.split, min(64, metric.samples), random_state)
        _plot_average_spectrum(sample_paths, metric.split, OUTPUT_CONFIG.figures / f"freq_avg_spectrum_{metric.split}.png")

    # Model training for adversarial analysis
    print("Preparing data loaders for adversarial probing...")
    train_paths, train_labels = _sample_image_paths("train", train_samples, random_state)
    val_paths, val_labels = _sample_image_paths("val", val_samples, random_state)
    num_classes = len(set(train_labels))

    train_loader, val_loader = _prepare_dataloaders(
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        image_size=image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleCNN(num_classes=num_classes).to(device)

    print("Training baseline model for adversarial probing...")
    history = _train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    base_acc = _evaluate(model, val_loader, device)
    print(f"Validation accuracy (clean): {base_acc:.4f}")
    save_json({"history": history, "clean_accuracy": base_acc}, OUTPUT_CONFIG.reports / "robustness_adversarial_training.json")

    # Adversarial attacks
    attack_results: List[AttackResult] = []
    for epsilon in [0.01, 0.03, 0.07]:
        acc = _run_attack(model, val_loader, device, attack="fgsm", epsilon=epsilon)
        attack_results.append(AttackResult("fgsm", epsilon, None, None, acc))
    for epsilon in [0.03, 0.07]:
        step = epsilon / 4
        acc = _run_attack(model, val_loader, device, attack="pgd", epsilon=epsilon, step_size=step, steps=10)
        attack_results.append(AttackResult("pgd", epsilon, step, 10, acc))

    attack_report = {
        "clean_accuracy": base_acc,
        "attacks": [a.__dict__ for a in attack_results],
    }
    save_json(attack_report, OUTPUT_CONFIG.reports / "robustness_adversarial_results.json")

    sample_loader = DataLoader(ImageDataset(val_paths[:32], val_labels[:32], image_size), batch_size=8, shuffle=False)
    _save_attack_samples(model, sample_loader, device, epsilon=0.07, out_dir=OUTPUT_CONFIG.figures / "robustness_adversarial_samples")

    # Occlusion analysis
    print("Generating occlusion sensitivity maps...")
    _run_occlusion_analysis(model, sample_loader, device, OUTPUT_CONFIG.figures / "robustness_occlusion", num_samples=8)

    print("Robustness deep dive complete.")


if __name__ == "__main__":
    run_robustness_deepdive()

