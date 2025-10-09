from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage import filters
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import DATASET_CONFIG, OUTPUT_CONFIG
from .utils import ensure_output_directories, load_labels, read_image, save_json, save_table


SCALES = [32, 64, 128]


@dataclass
class ScaleMetrics:
    split: str
    scale: int
    mean_intensity: float
    std_intensity: float
    edge_density: float


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: Optional[List[int]] = None, image_size: int = 64) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        path = self.image_paths[idx]
        image = read_image(path).resize((self.image_size, self.image_size))
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.labels is None:
            return tensor, None
        return tensor, self.labels[idx]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net[:-1]:
            x = layer(x)
        return x


def _sample_split(split: str, max_samples: Optional[int], random_state: int) -> Tuple[List[Path], Optional[List[int]]]:
    rng = np.random.default_rng(random_state)
    if split in {"train", "val"}:
        labels_path = DATASET_CONFIG.train_labels if split == "train" else DATASET_CONFIG.val_labels
        images_dir = DATASET_CONFIG.train_images if split == "train" else DATASET_CONFIG.val_images
        labels_df = load_labels(labels_path)
        if max_samples is not None and len(labels_df) > max_samples:
            labels_df = labels_df.sample(max_samples, random_state=random_state)
        paths = [images_dir / name for name in labels_df["file"]]
        labels = labels_df["label"].astype(int).tolist()
        return paths, labels
    if split == "test":
        if DATASET_CONFIG.test_manifest.exists():
            manifest = pd.read_csv(DATASET_CONFIG.test_manifest)
            files = manifest["file"].tolist()
        else:
            files = [p.name for p in sorted(DATASET_CONFIG.test_images.glob("*.png"))]
        if max_samples is not None and len(files) > max_samples:
            indices = rng.choice(len(files), size=max_samples, replace=False)
            files = [files[i] for i in indices]
        paths = [DATASET_CONFIG.test_images / name for name in files]
        return paths, None
    raise ValueError(f"Unsupported split: {split}")


def _edge_density(arr: np.ndarray, threshold: float = 30.0) -> float:
    edges = filters.sobel(arr)
    mask = edges > threshold
    return float(mask.mean())


def _multiscale_metrics(sample_paths: List[Path], split: str) -> List[ScaleMetrics]:
    metrics: List[ScaleMetrics] = []
    for scale in SCALES:
        means = []
        stds = []
        edges = []
        for path in tqdm(sample_paths, desc=f"Multi-scale {split}@{scale}", unit="img"):
            image = read_image(path)
            resized = image.resize((scale, scale), Image.BILINEAR)
            arr = np.array(resized, dtype=np.float32)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            edges.append(_edge_density(arr))
        metrics.append(
            ScaleMetrics(
                split=split,
                scale=scale,
                mean_intensity=float(np.mean(means)),
                std_intensity=float(np.mean(stds)),
                edge_density=float(np.mean(edges)),
            )
        )
    return metrics


def _prepare_loaders(
    train_paths: List[Path],
    train_labels: List[int],
    val_paths: List[Path],
    val_labels: List[int],
    batch_size: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ImageDataset(train_paths, train_labels, image_size)
    val_dataset = ImageDataset(val_paths, val_labels, image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _train_model(
    model: SimpleCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        with tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{epochs}") as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = torch.as_tensor(targets, device=device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
                pbar.set_postfix(loss=loss.item())
        history["train_loss"].append(running_loss / total)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs = inputs.to(device)
                targets = torch.as_tensor(targets, device=device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)
        history["val_loss"].append(val_loss / total)
        history["val_acc"].append(correct / total)
        print(
            f"Epoch {epoch + 1}: train_loss={history['train_loss'][-1]:.4f} "
            f"val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}"
        )

    return history


def _extract_embeddings(model: SimpleCNN, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting embeddings"):
            inputs = inputs.to(device)
            feats = model.penultimate(inputs)
            features.append(feats.cpu().numpy())
            labels.extend(targets)
    return np.concatenate(features, axis=0), np.array(labels, dtype=np.int64)


def _compute_similarity_matrix(features: np.ndarray, labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    centroids = {}
    for label in sorted(np.unique(labels)):
        centroids[label] = features[labels == label].mean(axis=0)
    centroid_df = pd.DataFrame.from_dict(centroids, orient="index")
    centroid_df.index.name = "label"

    labels_sorted = centroid_df.index.to_list()
    num = len(labels_sorted)
    sim_matrix = np.zeros((num, num), dtype=np.float32)
    for i, label_i in enumerate(labels_sorted):
        for j, label_j in enumerate(labels_sorted):
            a = centroid_df.loc[label_i]
            b = centroid_df.loc[label_j]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            sim_matrix[i, j] = sim
    similarity_df = pd.DataFrame(sim_matrix, index=labels_sorted, columns=labels_sorted)
    return centroid_df, similarity_df


def _grad_cam(model: SimpleCNN, image: torch.Tensor, target: int, device: torch.device) -> np.ndarray:
    target_layer = model.net[10]  # ReLU after third conv
    activations = None
    gradients = None

    def forward_hook(_, __, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
        return grad_input

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    image = image.to(device).requires_grad_(True)
    output = model(image)
    loss = output[0, target]
    loss.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze()
    cam = cam.cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def _save_gradcam_samples(
    model: SimpleCNN,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    num_samples: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for inputs, targets in loader:
        for idx in range(inputs.size(0)):
            image = inputs[idx : idx + 1]
            label = int(targets[idx])
            cam = _grad_cam(model, image, label, device)
            img_np = image.detach().cpu().squeeze().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(f"label={label}")
            axes[1].imshow(img_np, cmap="gray")
            axes[1].imshow(cam, cmap="jet", alpha=0.5)
            axes[1].axis("off")
            axes[1].set_title("Grad-CAM")
            plt.tight_layout()
            plt.savefig(out_dir / f"gradcam_{saved:02d}.png")
            plt.close()

            saved += 1
            if saved >= num_samples:
                return


def run_feature_exploration(
    multiscale_samples: int = 1000,
    train_samples: int = 6000,
    val_samples: int = 2000,
    image_size: int = 64,
    batch_size: int = 128,
    epochs: int = 3,
    lr: float = 1e-3,
    random_state: int = 42,
) -> None:
    ensure_output_directories()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    print("Running multi-scale analysis...")
    scale_results: List[ScaleMetrics] = []
    for split in ["train", "val", "test"]:
        paths, _ = _sample_split(split, multiscale_samples, random_state)
        if not paths:
            continue
        scale_results.extend(_multiscale_metrics(paths, split))

    scale_df = pd.DataFrame([s.__dict__ for s in scale_results])
    save_table(scale_df, OUTPUT_CONFIG.tables / "feature_multiscale_stats.csv")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=scale_df, x="scale", y="edge_density", hue="split", marker="o")
    plt.title("Edge density across scales")
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFIG.figures / "feature_multiscale_edge_density.png")
    plt.close()

    print("Training feature extractor for attention and similarity analyses...")
    train_paths, train_labels = _sample_split("train", train_samples, random_state)
    val_paths, val_labels = _sample_split("val", val_samples, random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(set(train_labels))).to(device)

    train_loader, val_loader = _prepare_loaders(train_paths, train_labels, val_paths, val_labels, batch_size, image_size)
    history = _train_model(model, train_loader, val_loader, device, epochs, lr)
    save_json({"history": history}, OUTPUT_CONFIG.reports / "feature_training_history.json")

    print("Generating Grad-CAM visualisations...")
    sample_loader = DataLoader(ImageDataset(val_paths[:64], val_labels[:64], image_size), batch_size=8, shuffle=False)
    _save_gradcam_samples(model, sample_loader, device, OUTPUT_CONFIG.figures / "feature_gradcam", num_samples=12)

    print("Computing inter-class similarity metrics...")
    feature_loader = DataLoader(ImageDataset(val_paths, val_labels, image_size), batch_size=batch_size, shuffle=False)
    embeddings, emb_labels = _extract_embeddings(model, feature_loader, device)
    centroid_df, similarity_df = _compute_similarity_matrix(embeddings, emb_labels)

    save_table(centroid_df.reset_index(), OUTPUT_CONFIG.tables / "feature_class_centroids.csv")
    save_table(similarity_df.reset_index(), OUTPUT_CONFIG.tables / "feature_interclass_similarity.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Inter-class cosine similarity")
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFIG.figures / "feature_interclass_similarity.png")
    plt.close()

    summary = {
        "multiscale_samples": multiscale_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "epochs": epochs,
        "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else None,
    }
    save_json(summary, OUTPUT_CONFIG.reports / "feature_exploration_summary.json")

    print("Feature exploration analysis complete.")


if __name__ == "__main__":
    run_feature_exploration()

