from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    root: Path = Path(__file__).resolve().parents[2] / "IS 2025 OrganAMNIST"
    train_images: Path = root / "train" / "images_train"
    train_labels: Path = root / "train" / "labels_train.csv"
    val_images: Path = root / "val" / "images_val"
    val_labels: Path = root / "val" / "labels_val.csv"
    test_images: Path = root / "test" / "images"
    test_manifest: Path = root / "test" / "manifest_public.csv"


@dataclass
class OutputConfig:
    root: Path = Path(__file__).resolve().parents[2] / "analysis_outputs"
    figures: Path = root / "figures"
    tables: Path = root / "tables"
    reports: Path = root / "reports"


DATASET_CONFIG = DatasetConfig()
OUTPUT_CONFIG = OutputConfig()
