"""
Evaluation Configuration

Defines paths and settings for comprehensive model evaluation.
Keeps evaluation outputs separate from analysis_outputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation outputs."""
    
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    
    # Input paths (existing results)
    results_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "analysis_outputs" / "models")
    training_logs: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "training_logs")
    
    # Output paths (new evaluation outputs)
    evaluation_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "evaluation_outputs")
    
    @property
    def confusion_matrices_dir(self) -> Path:
        return self.evaluation_root / "confusion_matrices"
    
    @property
    def figures_dir(self) -> Path:
        return self.evaluation_root / "figures"
    
    @property
    def tables_dir(self) -> Path:
        return self.evaluation_root / "tables"
    
    @property
    def reports_dir(self) -> Path:
        return self.evaluation_root / "reports"
    
    def ensure_directories(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.confusion_matrices_dir, self.figures_dir, 
                  self.tables_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Class names for OrganAMNIST dataset
CLASS_NAMES = [
    "Bladder",        # 0
    "Femur (L)",      # 1
    "Femur (R)",      # 2
    "Heart",          # 3
    "Kidney (L)",     # 4
    "Kidney (R)",     # 5
    "Liver",          # 6
    "Lung (L)",       # 7
    "Lung (R)",       # 8
    "Spleen",         # 9
    "Pancreas"        # 10
]

# Models to evaluate (11 models)
MODEL_NAMES = [
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "densenet121",
    "efficientnet_b3",
    "vit_s16",
    "swin_tiny",
    "swin_tiny_finetuned",
    "convnext_tiny",
    "convnext_tiny_finetuned"
]

# Model parameter counts (approximate, for inference time estimation)
MODEL_PARAMS = {
    "resnet50": 23.5,
    "resnet101": 42.5,
    "resnext50_32x4d": 23.0,
    "resnext101_32x8d": 86.7,
    "densenet121": 7.0,
    "efficientnet_b3": 10.7,
    "vit_s16": 21.7,
    "swin_tiny": 27.5,
    "swin_tiny_finetuned": 27.5,
    "convnext_tiny": 27.8,
    "convnext_tiny_finetuned": 27.8
}

# Model FLOPs (approximate, in GFLOPs at 224x224)
MODEL_FLOPS = {
    "resnet50": 4.1,
    "resnet101": 7.8,
    "resnext50_32x4d": 4.2,
    "resnext101_32x8d": 16.4,
    "densenet121": 2.9,
    "efficientnet_b3": 1.8,
    "vit_s16": 4.6,
    "swin_tiny": 4.5,
    "swin_tiny_finetuned": 4.5,
    "convnext_tiny": 4.5,
    "convnext_tiny_finetuned": 4.5
}

EVAL_CONFIG = EvaluationConfig()

