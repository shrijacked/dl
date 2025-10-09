from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from scipy.fft import dct

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_json, save_table


@dataclass
class SplitHashes:
    split: str
    total_images: int
    duplicate_pairs: int
    unique_hashes: int


@dataclass
class SuspectExample:
    file: str
    label: int
    predicted: int
    confidence: float


def _sample_paths(split: str, max_samples: Optional[int], random_state: int) -> Tuple[List[Path], Optional[List[int]]]:
    if split == "train":
        labels_df = load_labels(DATASET_CONFIG.train_labels)
        if max_samples is not None and len(labels_df) > max_samples:
            labels_df = labels_df.sample(max_samples, random_state=random_state)
        return [DATASET_CONFIG.train_images / name for name in labels_df["file"]], labels_df["label"].astype(int).tolist()
    if split == "val":
        labels_df = load_labels(DATASET_CONFIG.val_labels)
        if max_samples is not None and len(labels_df) > max_samples:
            labels_df = labels_df.sample(max_samples, random_state=random_state)
        return [DATASET_CONFIG.val_images / name for name in labels_df["file"]], labels_df["label"].astype(int).tolist()
    raise ValueError(f"Unsupported split: {split}")


def _compute_phash(image: Image.Image) -> str:
    resized = image.resize((32, 32), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    dct_rows = dct(arr, type=2, norm="ortho", axis=0)
    dct_cols = dct(dct_rows, type=2, norm="ortho", axis=1)
    dct_low = dct_cols[:8, :8]
    median = np.median(dct_low)
    bits = dct_low > median
    hash_bytes = np.packbits(bits.astype(np.uint8), axis=None)
    return hash_bytes.tobytes().hex()


def _collect_hashes(paths: List[Path]) -> Dict[str, List[Path]]:
    hash_map: Dict[str, List[Path]] = {}
    for path in tqdm(paths, desc="Hashing images", unit="img"):
        try:
            image = read_image(path)
            h = _compute_phash(image)
        except FileNotFoundError:
            continue
        hash_map.setdefault(h, []).append(path)
    return hash_map


def _save_duplicate_report(split: str, hash_map: Dict[str, List[Path]], limit: int = 20) -> None:
    duplicates = {h: [str(p.relative_to(DATASET_CONFIG.root)) for p in paths] for h, paths in hash_map.items() if len(paths) > 1}
    summary = {
        "split": split,
        "duplicate_groups": len(duplicates),
        "duplicate_examples": duplicates,
    }
    save_json(summary, OUTPUT_CONFIG.reports / f"data_quality_duplicates_{split}.json")

    preview_dir = OUTPUT_CONFIG.figures / f"data_quality_duplicates_{split}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for paths in duplicates.values():
        if count >= limit:
            break
        fig, axes = plt.subplots(1, len(paths), figsize=(2 * len(paths), 2))
        if len(paths) == 1:
            axes = [axes]
        for ax, path_str in zip(axes, paths):
            path = DATASET_CONFIG.root / path_str
            ax.imshow(read_image(path), cmap="gray")
            ax.set_title(path.name)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(preview_dir / f"group_{count:03d}.png")
        plt.close(fig)
        count += 1


def _prepare_dataframe(paths: List[Path], labels: List[int]) -> pd.DataFrame:
    records = []
    for path, label in zip(paths, labels):
        records.append({
            "path": path,
            "label": label,
        })
    return pd.DataFrame(records)


def _extract_features(paths: List[Path]) -> np.ndarray:
    features = []
    for path in tqdm(paths, desc="Extracting features", unit="img"):
        image = read_image(path).resize((32, 32), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0
        features.append(arr.flatten())
    return np.stack(features)


def _train_confident_model(train_paths: List[Path], train_labels: List[int], val_paths: List[Path], val_labels: List[int], random_state: int) -> RandomForestClassifier:
    X_train = _extract_features(train_paths)
    X_val = _extract_features(val_paths)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"RandomForest validation accuracy: {val_acc:.4f}")
    return model


def _flag_suspects(model: RandomForestClassifier, paths: List[Path], labels: List[int], threshold: float = 0.2) -> List[SuspectExample]:
    X = _extract_features(paths)
    y = np.array(labels)
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    max_conf = proba.max(axis=1)

    suspects = []
    for path, label, pred, conf in zip(paths, y, preds, max_conf):
        if conf < threshold or pred != label:
            suspects.append(SuspectExample(
                file=path.name,
                label=int(label),
                predicted=int(pred),
                confidence=float(conf),
            ))
    return suspects


def _summarise_duplicates(hash_map: Dict[str, List[Path]], split: str) -> SplitHashes:
    total = sum(len(paths) for paths in hash_map.values())
    duplicates = sum(1 for paths in hash_map.values() if len(paths) > 1)
    return SplitHashes(
        split=split,
        total_images=total,
        duplicate_pairs=duplicates,
        unique_hashes=len(hash_map),
    )


def run_data_quality_assessment(
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    random_state: int = 42,
) -> None:
    ensure_output_directories()

    print("Checking for near-duplicate images via perceptual hashing...")
    duplicate_summaries: List[SplitHashes] = []
    for split, max_samples in [("train", max_train_samples), ("val", max_val_samples)]:
        paths, _ = _sample_paths(split, max_samples, random_state)
        hash_map = _collect_hashes(paths)
        duplicate_summaries.append(_summarise_duplicates(hash_map, split))
        _save_duplicate_report(split, hash_map)

    duplicates_df = pd.DataFrame([s.__dict__ for s in duplicate_summaries])
    save_table(duplicates_df, OUTPUT_CONFIG.tables / "data_quality_duplicate_summary.csv")

    print("Training baseline classifier for label audit...")
    train_paths, train_labels = _sample_paths("train", max_train_samples, random_state)
    val_paths, val_labels = _sample_paths("val", max_val_samples, random_state)

    model = _train_confident_model(train_paths, train_labels, val_paths, val_labels, random_state)
    suspects = _flag_suspects(model, train_paths, train_labels)
    suspects_df = pd.DataFrame([s.__dict__ for s in suspects]) if suspects else pd.DataFrame(columns=["file", "label", "predicted", "confidence"])
    if not suspects_df.empty:
        suspects_df.sort_values(by="confidence", inplace=True)
    save_table(suspects_df, OUTPUT_CONFIG.tables / "data_quality_suspect_labels.csv")

    if not suspects_df.empty:
        top_suspects = suspects_df.head(20)
        review_dir = OUTPUT_CONFIG.figures / "data_quality_suspects"
        review_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in top_suspects.iterrows():
            fig, axes = plt.subplots(1, 1, figsize=(3, 3))
            path = DATASET_CONFIG.train_images / row["file"]
            axes.imshow(read_image(path), cmap="gray")
            axes.set_title(f"label={row['label']} pred={row['predicted']} conf={row['confidence']:.2f}")
            axes.axis("off")
            plt.tight_layout()
            plt.savefig(review_dir / f"suspect_{idx:03d}.png")
            plt.close(fig)

    summary = {
        "duplicates": [s.__dict__ for s in duplicate_summaries],
        "suspect_count": len(suspects),
        "suspect_threshold": 0.2,
    }
    save_json(summary, OUTPUT_CONFIG.reports / "data_quality_summary.json")

    print("Data quality assessment complete.")


if __name__ == "__main__":
    run_data_quality_assessment()

