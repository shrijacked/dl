from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.auto import tqdm

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_json, save_table


@dataclass
class ClassPerformance:
    label: int
    train_count: int
    val_count: int
    val_accuracy: float
    correct_predictions: int


def _split_paths(split: str) -> tuple[Path, Path]:
    if split == "train":
        return DATASET_CONFIG.train_images, DATASET_CONFIG.train_labels
    if split == "val":
        return DATASET_CONFIG.val_images, DATASET_CONFIG.val_labels
    raise ValueError(f"Unsupported split: {split}")


def _sample_by_class(df: pd.DataFrame, max_per_class: Optional[int], random_state: int) -> pd.DataFrame:
    if max_per_class is None:
        return df

    sampled_frames = []
    for _, group in df.groupby("label"):
        sampled_frames.append(group.sample(min(len(group), max_per_class), random_state=random_state))
    return pd.concat(sampled_frames, ignore_index=True)


def _load_split(split: str, max_per_class: Optional[int], random_state: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    images_dir, labels_path = _split_paths(split)
    labels_df = load_labels(labels_path)
    sampled_df = _sample_by_class(labels_df, max_per_class, random_state)

    features: List[np.ndarray] = []
    targets: List[int] = []
    for row in tqdm(sampled_df.itertuples(), total=len(sampled_df), desc=f"Loading {split} images"):
        image = read_image(images_dir / row.file)
        features.append(np.array(image, dtype=np.float32).flatten() / 255.0)
        targets.append(int(row.label))

    return np.stack(features), np.array(targets, dtype=np.int64), sampled_df


def _train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    epochs: int = 5,
    batch_size: int = 1024,
    random_state: int = 42,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="adaptive",
        eta0=0.01,
        random_state=random_state,
        fit_intercept=True,
    )

    n_samples = X_train.shape[0]
    rng = np.random.RandomState(random_state)

    for epoch in range(epochs):
        order = rng.permutation(n_samples)
        X_shuffled = X_train[order]
        y_shuffled = y_train[order]
        pbar = tqdm(
            range(0, n_samples, batch_size),
            desc=f"Training epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        for start in pbar:
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            if epoch == 0 and start == 0:
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)
        pbar.close()

    return model


def _per_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_counts: pd.Series,
    val_counts: pd.Series,
) -> List[ClassPerformance]:
    labels = sorted(set(train_counts.index).union(set(val_counts.index)))
    performances: List[ClassPerformance] = []

    for label in labels:
        label = int(label)
        val_mask = y_true == label
        val_count = int(val_counts.get(label, 0))
        correct = int((y_pred[val_mask] == label).sum()) if val_mask.any() else 0
        accuracy = float(correct / val_count) if val_count > 0 else float("nan")
        performances.append(ClassPerformance(
            label=label,
            train_count=int(train_counts.get(label, 0)),
            val_count=val_count,
            val_accuracy=accuracy,
            correct_predictions=correct,
        ))

    return performances


def _save_confusion_matrix(cm: np.ndarray, labels: Iterable[int]) -> None:
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = OUTPUT_CONFIG.tables / "class_imbalance_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index_label="actual")

    plt.figure(figsize=(10, 8))
    with np.errstate(invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm.astype(np.float64), row_sums, where=row_sums != 0)
    sns.heatmap(normalized, xticklabels=labels, yticklabels=labels, cmap="mako", annot=False, fmt=".2f")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Validation confusion matrix (row-normalised)")
    plt.tight_layout()
    fig_path = OUTPUT_CONFIG.figures / "class_imbalance_confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()


def run_class_imbalance_analysis(max_train_per_class: Optional[int] = 1500, random_state: int = 42) -> None:
    ensure_output_directories()

    X_train, y_train, train_df = _load_split("train", max_train_per_class, random_state)
    X_val, y_val, val_df = _load_split("val", None, random_state)

    print("Training SGD logistic baseline...")
    model = _train_baseline(
        X_train,
        y_train,
        classes=np.unique(y_train),
        epochs=5,
        batch_size=1024,
        random_state=random_state,
    )

    print("Scoring validation set...")
    y_pred = model.predict(X_val)
    overall_accuracy = accuracy_score(y_val, y_pred)

    train_counts = train_df["label"].value_counts().sort_index()
    val_counts = val_df["label"].value_counts().sort_index()
    per_class_stats = _per_class_performance(y_val, y_pred, train_counts, val_counts)

    metrics_df = pd.DataFrame([s.__dict__ for s in per_class_stats])
    metrics_path = OUTPUT_CONFIG.tables / "class_imbalance_per_class_accuracy.csv"
    save_table(metrics_df, metrics_path)

    summary = {
        "overall_accuracy": overall_accuracy,
        "train_sampled_total": int(len(train_df)),
        "val_total": int(len(val_df)),
        "max_train_per_class": max_train_per_class,
    }
    summary_path = OUTPUT_CONFIG.reports / "class_imbalance_summary.json"
    save_json(summary, summary_path)

    labels_sorted = sorted(metrics_df["label"].tolist())
    cm = confusion_matrix(y_val, y_pred, labels=labels_sorted)
    _save_confusion_matrix(cm, labels_sorted)


if __name__ == "__main__":
    run_class_imbalance_analysis()

