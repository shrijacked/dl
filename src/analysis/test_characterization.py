from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from skimage import feature, filters
from tqdm.auto import tqdm

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import chunked, ensure_output_directories, load_labels, read_image, save_json, save_table


BINS_PIXEL = 256
LBP_P = 24
LBP_R = 3
LBP_BINS = LBP_P + 2
IMAGE_DOWNSAMPLE = (64, 64)


@dataclass
class SplitSummary:
    split: str
    num_images: int
    mean_intensity: float
    std_intensity: float
    edge_density_mean: float
    edge_density_std: float
    lbp_entropy: float
    anomaly_rate: Optional[float]
    anomaly_score_mean: Optional[float]
    anomaly_score_std: Optional[float]


def _collect_paths(split: str, max_images: Optional[int], random_state: int) -> List[Path]:
    if split in {"train", "val"}:
        images_dir, labels_path = (
            (DATASET_CONFIG.train_images, DATASET_CONFIG.train_labels)
            if split == "train"
            else (DATASET_CONFIG.val_images, DATASET_CONFIG.val_labels)
        )
        labels_df = load_labels(labels_path)
        if max_images is not None:
            labels_df = labels_df.sample(min(max_images, len(labels_df)), random_state=random_state)
        return [images_dir / name for name in labels_df["file"]]

    if split == "test":
        if DATASET_CONFIG.test_manifest.exists():
            manifest = pd.read_csv(DATASET_CONFIG.test_manifest)
            files = manifest["file"].tolist()
            if max_images is not None:
                files = files[:max_images]
            return [DATASET_CONFIG.test_images / name for name in files]
        image_paths = sorted(DATASET_CONFIG.test_images.glob("*.png"))
        if max_images is not None:
            image_paths = image_paths[:max_images]
        return image_paths

    raise ValueError(f"Unsupported split: {split}")


def _edge_density(arr: np.ndarray, threshold: float = 0.1) -> float:
    edges = filters.sobel(arr)
    return float((edges > threshold).mean())


def _compute_lbp(arr: np.ndarray) -> np.ndarray:
    lbp = feature.local_binary_pattern(arr, P=LBP_P, R=LBP_R, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=False)
    return hist.astype(np.float64)


def _normalise_hist(hist: np.ndarray) -> np.ndarray:
    total = hist.sum()
    if total == 0:
        return np.ones_like(hist) / len(hist)
    return hist / total


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _wasserstein_distance(p: np.ndarray, q: np.ndarray, range_min: float, range_max: float) -> float:
    bin_width = (range_max - range_min) / len(p)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)) * bin_width)


def _entropy(prob: np.ndarray) -> float:
    eps = 1e-12
    prob = np.clip(prob, eps, 1.0)
    return float(-np.sum(prob * np.log(prob)))


def _extract_split_features(paths: List[Path]) -> Dict[str, np.ndarray]:
    pixel_hist = np.zeros(BINS_PIXEL, dtype=np.float64)
    edge_values: List[float] = []
    lbp_hist = np.zeros(LBP_BINS, dtype=np.float64)
    downsampled_vectors: List[np.ndarray] = []

    for path in tqdm(paths, desc="Processing images", unit="img"):
        image = read_image(path)
        arr = np.array(image, dtype=np.float32) / 255.0

        hist, _ = np.histogram(arr.flatten(), bins=BINS_PIXEL, range=(0.0, 1.0), density=False)
        pixel_hist += hist

        edge_values.append(_edge_density(arr))

        lbp_hist += _compute_lbp((arr * 255.0).astype(np.uint8))

        downsampled = image.resize(IMAGE_DOWNSAMPLE, Image.BILINEAR)
        downsampled_vectors.append(np.array(downsampled, dtype=np.float32).flatten() / 255.0)

    features = {
        "pixel_hist": pixel_hist,
        "edge_values": np.array(edge_values, dtype=np.float32),
        "lbp_hist": lbp_hist,
        "vectors": np.stack(downsampled_vectors) if downsampled_vectors else np.empty((0, IMAGE_DOWNSAMPLE[0] * IMAGE_DOWNSAMPLE[1])),
    }
    return features


def _fit_transform_features(train_vectors: np.ndarray, other_vectors: Dict[str, np.ndarray], n_components: int = 50, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray], PCA, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vectors)

    components = min(n_components, train_scaled.shape[1]) if train_scaled.size else n_components
    pca = PCA(n_components=components, random_state=random_state)
    train_pca = pca.fit_transform(train_scaled)

    transformed = {}
    for split, vecs in other_vectors.items():
        if vecs.size == 0:
            transformed[split] = np.empty((0, components))
            continue
        scaled = scaler.transform(vecs)
        transformed[split] = pca.transform(scaled)

    return train_pca, transformed, pca, scaler


def _run_anomaly_detection(train_pca: np.ndarray, transformed: Dict[str, np.ndarray], random_state: int = 42) -> Dict[str, Dict[str, float]]:
    if train_pca.size == 0:
        return {split: {"anomaly_rate": None, "score_mean": None, "score_std": None} for split in transformed}

    model = IsolationForest(random_state=random_state, contamination="auto")
    model.fit(train_pca)

    results = {}
    for split, data in transformed.items():
        if data.size == 0:
            results[split] = {"anomaly_rate": None, "score_mean": None, "score_std": None}
            continue
        scores = model.decision_function(data)
        predictions = model.predict(data)
        anomaly_rate = float(np.mean(predictions == -1))
        results[split] = {
            "anomaly_rate": anomaly_rate,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
    return results


def _plot_histograms(hist_by_split: Dict[str, np.ndarray], title: str, out_path: Path, bins: int, range_min: float, range_max: float) -> None:
    plt.figure(figsize=(8, 5))
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    for split, hist in hist_by_split.items():
        prob = _normalise_hist(hist)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(centers, prob, label=split)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_edge_hist(edge_by_split: Dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for split, values in edge_by_split.items():
        plt.hist(values, bins=50, range=(0, 1), alpha=0.5, label=split, density=True)
    plt.title("Edge density distribution")
    plt.xlabel("Edge density")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_lbp_hist(lbp_by_split: Dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    x = np.arange(LBP_BINS)
    width = 0.8 / len(lbp_by_split)
    for idx, (split, hist) in enumerate(lbp_by_split.items()):
        prob = _normalise_hist(hist)
        plt.bar(x + idx * width, prob, width=width, label=split)
    plt.title("LBP histogram comparison")
    plt.xlabel("LBP code")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def run_test_characterization(max_images: Optional[int] = 5000, random_state: int = 42) -> None:
    ensure_output_directories()

    split_paths = {
        split: _collect_paths(split, max_images, random_state)
        for split in ["train", "val", "test"]
    }

    split_features: Dict[str, Dict[str, np.ndarray]] = {}
    for split, paths in split_paths.items():
        if not paths:
            split_features[split] = {"pixel_hist": np.zeros(BINS_PIXEL), "edge_values": np.zeros(0), "lbp_hist": np.zeros(LBP_BINS), "vectors": np.empty((0, IMAGE_DOWNSAMPLE[0] * IMAGE_DOWNSAMPLE[1]))}
            continue
        print(f"Processing {split} split ({len(paths)} images)...")
        split_features[split] = _extract_split_features(paths)

    pixel_hist_norm = {split: _normalise_hist(features["pixel_hist"]) for split, features in split_features.items()}
    lbp_hist_norm = {split: _normalise_hist(features["lbp_hist"]) for split, features in split_features.items()}
    edge_values = {split: features["edge_values"] for split, features in split_features.items()}
    vectors = {split: features["vectors"] for split, features in split_features.items()}

    print("Running PCA feature extraction and anomaly detection...")
    train_pca, transformed, _, _ = _fit_transform_features(
        vectors["train"],
        {split: vec for split, vec in vectors.items() if split != "train"},
    )

    all_pca = {"train": train_pca, **transformed}
    anomaly_results = _run_anomaly_detection(train_pca, {split: feats for split, feats in all_pca.items() if split != "train"})
    anomaly_results["train"] = {"anomaly_rate": 0.0, "score_mean": 0.0, "score_std": 0.0}

    summaries: List[SplitSummary] = []
    for split in ["train", "val", "test"]:
        hist = pixel_hist_norm[split]
        lbp_prob = lbp_hist_norm[split]
        edge_vals = edge_values[split]
        anomaly = anomaly_results.get(split, {"anomaly_rate": None, "score_mean": None, "score_std": None})

        summaries.append(
            SplitSummary(
                split=split,
                num_images=len(split_paths[split]),
                mean_intensity=float(np.sum(hist * np.linspace(0, 1, BINS_PIXEL))),
                std_intensity=float(np.sqrt(np.sum(hist * (np.linspace(0, 1, BINS_PIXEL) ** 2)) - np.sum(hist * np.linspace(0, 1, BINS_PIXEL)) ** 2)),
                edge_density_mean=float(edge_vals.mean()) if edge_vals.size else float("nan"),
                edge_density_std=float(edge_vals.std()) if edge_vals.size else float("nan"),
                lbp_entropy=_entropy(lbp_prob),
                anomaly_rate=anomaly["anomaly_rate"],
                anomaly_score_mean=anomaly["score_mean"],
                anomaly_score_std=anomaly["score_std"],
            )
        )

    comparison_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    shift_metrics = {
        f"{a}_vs_{b}": {
            "pixel_kl": _kl_divergence(pixel_hist_norm[a], pixel_hist_norm[b]),
            "pixel_wasserstein": _wasserstein_distance(pixel_hist_norm[a], pixel_hist_norm[b], 0.0, 1.0),
            "edge_mean_delta": float(abs(edge_values[a].mean() - edge_values[b].mean())) if edge_values[a].size and edge_values[b].size else None,
            "lbp_kl": _kl_divergence(lbp_hist_norm[a], lbp_hist_norm[b]),
        }
        for a, b in comparison_pairs
    }

    summary_df = pd.DataFrame([s.__dict__ for s in summaries])
    save_table(summary_df, OUTPUT_CONFIG.tables / "test_characterization_summary.csv")
    save_json(shift_metrics, OUTPUT_CONFIG.reports / "test_characterization_shift_metrics.json")

    _plot_histograms(pixel_hist_norm, "Pixel intensity distribution", OUTPUT_CONFIG.figures / "test_characterization_pixel_hist.png", BINS_PIXEL, 0.0, 1.0)
    _plot_edge_hist(edge_values, OUTPUT_CONFIG.figures / "test_characterization_edge_hist.png")
    _plot_lbp_hist(lbp_hist_norm, OUTPUT_CONFIG.figures / "test_characterization_lbp_hist.png")

    print("Test characterization analysis complete. Outputs saved to analysis_outputs.")


if __name__ == "__main__":
    run_test_characterization()

