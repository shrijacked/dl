from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import chunked, ensure_output_directories, load_labels, read_image, save_json, save_table


@dataclass
class ImageStats:
    file: str
    mean: float
    std: float
    min: int
    max: int


@dataclass
class AggregateStats:
    dataset: str
    mean_of_means: float
    std_of_means: float
    overall_std: float
    min_pixel: int
    max_pixel: int


def _compute_image_stats(image: Image.Image, filename: str) -> ImageStats:
    arr = np.array(image, dtype=np.float32)
    return ImageStats(
        file=filename,
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=int(arr.min()),
        max=int(arr.max()),
    )


def _aggregate_stats(stats: Iterable[ImageStats], dataset: str) -> AggregateStats:
    stats_list = list(stats)
    means = np.array([s.mean for s in stats_list])
    stds = np.array([s.std for s in stats_list])
    mins = np.array([s.min for s in stats_list])
    maxs = np.array([s.max for s in stats_list])

    return AggregateStats(
        dataset=dataset,
        mean_of_means=float(means.mean()),
        std_of_means=float(means.std()),
        overall_std=float(stds.mean()),
        min_pixel=int(mins.min()),
        max_pixel=int(maxs.max()),
    )


def _plot_histogram(images: Iterable[Image.Image], title: str, out_path: Path) -> None:
    pixel_values = np.concatenate([
        np.array(img, dtype=np.float32).flatten()
        for img in images
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pixel_values, bins=50, color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_image_stats(sample_size: int = 512) -> None:
    ensure_output_directories()

    for split_name, images_dir, labels_path in [
        ("train", DATASET_CONFIG.train_images, DATASET_CONFIG.train_labels),
        ("val", DATASET_CONFIG.val_images, DATASET_CONFIG.val_labels),
    ]:
        labels_df = load_labels(labels_path)
        stats_records = []
        sampled_images = []
        for chunk in chunked(labels_df.itertuples(), size=512):
            for row in chunk:
                image_path = images_dir / row.file
                image = read_image(image_path)
                stats = _compute_image_stats(image, row.file)
                stats_records.append(stats)
                if len(sampled_images) < sample_size:
                    sampled_images.append(image)

        per_image_df = pd.DataFrame([s.__dict__ for s in stats_records])
        agg_stats = _aggregate_stats(stats_records, split_name)

        save_table(per_image_df, OUTPUT_CONFIG.tables / f"{split_name}_image_stats.csv")
        save_json(agg_stats.__dict__, OUTPUT_CONFIG.reports / f"{split_name}_image_summary.json")
        _plot_histogram(sampled_images, f"Pixel distribution ({split_name})", OUTPUT_CONFIG.figures / f"{split_name}_pixel_histogram.png")

    if DATASET_CONFIG.test_images.exists():
        sample_paths = sorted(DATASET_CONFIG.test_images.glob("*.png"))[:sample_size]
        images = [read_image(path) for path in sample_paths]
        _plot_histogram(images, "Pixel distribution (test sample)", OUTPUT_CONFIG.figures / "test_pixel_histogram.png")
