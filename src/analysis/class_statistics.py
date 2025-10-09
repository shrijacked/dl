from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import load_labels, save_json


def _compose_stats(df: pd.DataFrame) -> Dict:
    counts = df["label"].value_counts().sort_index()
    stats = {
        "total": int(counts.sum()),
        "per_class": counts.to_dict(),
        "proportions": (counts / counts.sum()).to_dict(),
    }
    return stats


def _pixel_stats(csv_path: Path, labels_df: pd.DataFrame) -> Dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing stats CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    merged = df.merge(labels_df[["file", "label"]], on="file", how="left")
    grouped = merged.groupby("label")
    summary = {}
    for label, group in grouped:
        summary[int(label)] = {
            "mean_mean": float(group["mean"].mean()),
            "mean_std": float(group["std"].mean()),
            "mean_min": float(group["min"].mean()),
            "mean_max": float(group["max"].mean()),
        }
    return summary


def build_class_statistics() -> None:
    train_df = load_labels(DATASET_CONFIG.train_labels)
    val_df = load_labels(DATASET_CONFIG.val_labels)

    stats = {
        "train": _compose_stats(train_df),
        "val": _compose_stats(val_df),
        "train_pixel_stats": _pixel_stats(OUTPUT_CONFIG.tables / "train_image_stats.csv", train_df),
        "val_pixel_stats": _pixel_stats(OUTPUT_CONFIG.tables / "val_image_stats.csv", val_df),
    }

    save_json(stats, OUTPUT_CONFIG.reports / "class_statistics.json")


if __name__ == "__main__":
    build_class_statistics()
