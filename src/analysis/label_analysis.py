from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, save_json, save_table


def _compute_stats(df: pd.DataFrame) -> Dict:
    counts = df["label"].value_counts().sort_index()
    stats = {
        "total": int(counts.sum()),
        "per_class": counts.to_dict(),
        "proportions": (counts / counts.sum()).to_dict(),
    }
    return stats


def _plot_distribution(train_counts: pd.Series, val_counts: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = pd.DataFrame({
        "train": train_counts,
        "val": val_counts.reindex(train_counts.index, fill_value=0),
    })
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Label distribution: train vs val")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_label_analysis() -> None:
    ensure_output_directories()

    train_df = load_labels(DATASET_CONFIG.train_labels)
    val_df = load_labels(DATASET_CONFIG.val_labels)

    train_counts = train_df["label"].value_counts().sort_index()
    val_counts = val_df["label"].value_counts().sort_index()

    stats = {
        "train": _compute_stats(train_df),
        "val": _compute_stats(val_df),
    }

    summary_table = pd.DataFrame({
        "train_count": train_counts,
        "train_pct": train_counts / train_counts.sum(),
        "val_count": val_counts.reindex(train_counts.index, fill_value=0),
        "val_pct": val_counts.reindex(train_counts.index, fill_value=0) / val_counts.sum(),
    }).reset_index().rename(columns={"index": "label"})

    save_table(summary_table, OUTPUT_CONFIG.tables / "label_distribution.csv")
    save_json(stats, OUTPUT_CONFIG.reports / "label_distribution.json")
    _plot_distribution(train_counts, val_counts, OUTPUT_CONFIG.figures / "label_distribution.png")
