from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, metrics, util

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_table


@dataclass
class PerturbationResult:
    filename: str
    perturbation: str
    psnr: float
    ssim: float


def _add_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    noisy = util.random_noise(arr / 255.0, mode="gaussian", var=(sigma / 255.0) ** 2)
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


def _add_motion_blur(image: Image.Image, radius: int) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    blurred = filters.gaussian(arr, sigma=radius)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred, mode="L")


def _adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    mean = arr.mean()
    adjusted = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted, mode="L")


def _metrics(original: Image.Image, perturbed: Image.Image) -> Dict[str, float]:
    orig = np.array(original, dtype=np.float32)
    pert = np.array(perturbed, dtype=np.float32)
    psnr = metrics.peak_signal_noise_ratio(orig, pert, data_range=255)
    ssim = metrics.structural_similarity(orig, pert, data_range=255)
    return {"psnr": float(psnr), "ssim": float(ssim)}


def _plot_perturbations(original: Image.Image, variants: Dict[str, Image.Image], out_path: Path) -> None:
    cols = len(variants) + 1
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("original")
    axes[0].axis("off")
    for idx, (name, img) in enumerate(variants.items(), start=1):
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(name)
        axes[idx].axis("off")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_robustness_probes(sample_size: int = 16) -> None:
    ensure_output_directories()

    labels_df = load_labels(DATASET_CONFIG.train_labels)
    sample_files = labels_df.sample(min(sample_size, len(labels_df)), random_state=42)["file"].tolist()

    results: List[PerturbationResult] = []
    for file_name in sample_files:
        image_path = DATASET_CONFIG.train_images / file_name
        original = read_image(image_path)
        perturbations = {
            "gaussian_noise": _add_gaussian_noise(original, sigma=15.0),
            "motion_blur": _add_motion_blur(original, radius=2),
            "contrast_up": _adjust_contrast(original, factor=1.5),
            "contrast_down": _adjust_contrast(original, factor=0.6),
        }

        for name, perturbed in perturbations.items():
            metrics_dict = _metrics(original, perturbed)
            results.append(PerturbationResult(
                filename=file_name,
                perturbation=name,
                psnr=metrics_dict["psnr"],
                ssim=metrics_dict["ssim"],
            ))

        _plot_perturbations(original, perturbations, OUTPUT_CONFIG.figures / f"perturbations_{file_name}.png")

    results_df = pd.DataFrame(r.__dict__ for r in results)
    save_table(results_df, OUTPUT_CONFIG.tables / "robustness_metrics.csv")
