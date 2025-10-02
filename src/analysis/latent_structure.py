from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_json


@dataclass
class EmbeddingResult:
    method: str
    explained_variance: Tuple[float, ...]


def _flatten_image(path: Path) -> np.ndarray:
    img = read_image(path)
    return np.array(img, dtype=np.float32).flatten()


def run_latent_structure(sample_size: int = 2048) -> None:
    ensure_output_directories()

    labels_df = load_labels(DATASET_CONFIG.train_labels)
    sampled = labels_df.sample(min(sample_size, len(labels_df)), random_state=42)

    features = np.stack([
        _flatten_image(DATASET_CONFIG.train_images / row.file)
        for row in sampled.itertuples()
    ])
    labels = sampled["label"].to_numpy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=50, random_state=42)
    pca_features = pca.fit_transform(features_scaled)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    tsne_embeddings = tsne.fit_transform(pca_features)

    embedding_df = pd.DataFrame({
        "dim1": tsne_embeddings[:, 0],
        "dim2": tsne_embeddings[:, 1],
        "label": labels,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding_df["dim1"], embedding_df["dim2"], c=embedding_df["label"], cmap="tab20", s=10, alpha=0.7)
    ax.set_title("t-SNE embedding of sampled training images")
    plt.tight_layout()
    fig.savefig(OUTPUT_CONFIG.figures / "latent_tsne.png")
    plt.close(fig)

    result = EmbeddingResult(
        method="PCA->tSNE",
        explained_variance=tuple(pca.explained_variance_ratio_[:10]),
    )
    save_json(result.__dict__, OUTPUT_CONFIG.reports / "latent_structure.json")
