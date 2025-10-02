from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import filters

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_table


@dataclass
class GeometricStats:
    file: str
    edge_density: float
    horiz_flip_diff: float
    vert_flip_diff: float


def _edge_density(image_array: np.ndarray, threshold: float = 50.0) -> float:
    edges = filters.sobel(image_array)
    return float((edges > threshold).mean())


def _sobel_edges(image: Image.Image) -> np.ndarray:
    arr = np.array(image, dtype=np.float32)
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edge_x = np.abs(np.convolve(arr.flatten(), sobel_x.flatten(), mode="same"))
    edge_y = np.abs(np.convolve(arr.flatten(), sobel_y.flatten(), mode="same"))
    magnitude = np.sqrt(edge_x ** 2 + edge_y ** 2)
    return magnitude


def _compare_flips(image: Image.Image) -> Dict[str, float]:
    arr = np.array(image, dtype=np.float32)
    horiz = np.fliplr(arr)
    vert = np.flipud(arr)
    diff_h = np.mean(np.abs(arr - horiz))
    diff_v = np.mean(np.abs(arr - vert))
    return {"horiz_flip_diff": float(diff_h), "vert_flip_diff": float(diff_v)}


def _plot_flip_histograms(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["horiz_flip_diff"], bins=30, color="slateblue")
    axes[0].set_title("Horizontal flip difference")
    axes[1].hist(df["vert_flip_diff"], bins=30, color="seagreen")
    axes[1].set_title("Vertical flip difference")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_geometric_analysis(sample_size: int = 1024) -> None:
    ensure_output_directories()

    labels_df = load_labels(DATASET_CONFIG.train_labels)
    sampled = labels_df.sample(min(sample_size, len(labels_df)), random_state=42)

    stats = []
    for row in sampled.itertuples():
        image = read_image(DATASET_CONFIG.train_images / row.file)
        arr = np.array(image, dtype=np.float32)
        edge_density = _edge_density(arr)
        flips = _compare_flips(image)
        stats.append(GeometricStats(
            file=row.file,
            edge_density=edge_density,
            horiz_flip_diff=flips["horiz_flip_diff"],
            vert_flip_diff=flips["vert_flip_diff"],
        ))

    df = pd.DataFrame(s.__dict__ for s in stats)
    save_table(df, OUTPUT_CONFIG.tables / "geometric_stats.csv")
    _plot_flip_histograms(df, OUTPUT_CONFIG.figures / "flip_differences.png")
