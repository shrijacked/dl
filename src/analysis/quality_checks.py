from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_json


def _sample_images(labels_df: pd.DataFrame, images_dir: Path, per_class: int = 6) -> Dict[int, List[Path]]:
    samples: Dict[int, List[Path]] = {}
    grouped = labels_df.groupby("label")
    for label, group in grouped:
        file_names = group["file"].tolist()
        random.shuffle(file_names)
        selected = file_names[:per_class]
        samples[label] = [images_dir / name for name in selected]
    return samples


def _plot_grid(samples: Dict[int, List[Path]], out_path: Path, title: str) -> None:
    rows = len(samples)
    cols = max(len(paths) for paths in samples.values())
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = [axes]
    for row_idx, (label, paths) in enumerate(sorted(samples.items())):
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx < len(paths):
                img = read_image(paths[col_idx])
                ax.imshow(img, cmap="gray")
                ax.set_title(f"label {label}")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _check_missing_files(labels_df: pd.DataFrame, images_dir: Path) -> Tuple[int, List[str]]:
    missing = []
    for file_name in labels_df["file"]:
        if not (images_dir / file_name).exists():
            missing.append(file_name)
    return len(missing), missing


def run_quality_checks(per_class: int = 6) -> None:
    ensure_output_directories()

    for split_name, images_dir, labels_path in [
        ("train", DATASET_CONFIG.train_images, DATASET_CONFIG.train_labels),
        ("val", DATASET_CONFIG.val_images, DATASET_CONFIG.val_labels),
    ]:
        labels_df = load_labels(labels_path)
        samples = _sample_images(labels_df, images_dir, per_class=per_class)
        _plot_grid(samples, OUTPUT_CONFIG.figures / f"{split_name}_grid.png", f"Sample grid ({split_name})")

        missing_count, missing_files = _check_missing_files(labels_df, images_dir)
        report = {
            "split": split_name,
            "missing_count": missing_count,
            "missing_files": missing_files,
        }
        save_json(report, OUTPUT_CONFIG.reports / f"{split_name}_missing_files.json")

    if DATASET_CONFIG.test_manifest.exists():
        manifest_df = pd.read_csv(DATASET_CONFIG.test_manifest)
        required_columns = {"index", "file"}
        if not required_columns.issubset(manifest_df.columns):
            raise ValueError(f"Manifest missing columns: {required_columns}")
        sample_paths = manifest_df["file"].sample(min(len(manifest_df), per_class * 4), random_state=42)
        paths = {0: [DATASET_CONFIG.test_images / name for name in sample_paths]}
        _plot_grid(paths, OUTPUT_CONFIG.figures / "test_grid.png", "Sample grid (test)")
