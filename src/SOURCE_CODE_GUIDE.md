# 💻 Source Code Guide

This comprehensive guide explains the entire codebase architecture, every module's purpose, and how to use each component for your deep learning medical image classification project.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Model Architectures](#2-model-architectures)
3. [Data Pipeline](#3-data-pipeline)
4. [Training](#4-training)
5. [Models Module](#5-models-module)
6. [Evaluation](#6-evaluation)
7. [Analysis](#7-analysis)
8. [Utilities](#8-utilities)
9. [How to Use](#9-how-to-use)

---

## 1. Project Architecture Overview

```
src/
├── __init__.py                 # Package marker
├── model_architectures.py      # Central model builder registry
├── data_pipeline/              # Data loading & transforms
│   ├── dataloaders.py          # DataLoader builders
│   └── organamnist_dataset.py  # Dataset class
├── training/                   # Training engines & runners
│   ├── engine.py               # Core training loop
│   ├── run_experiments.py      # Experiment sweeps
│   ├── run_baselines.py        # Baseline model training
│   └── train_*.py              # Architecture-specific trainers
├── models/                     # Individual model implementations
│   ├── resnet50.py             # ResNet-50
│   ├── densenet121.py          # DenseNet-121
│   ├── densenet121_adaptive.py # DenseNet with attention
│   ├── swin_multiscale.py      # Swin with multi-scale fusion
│   ├── convtransgfusion.py     # CNN-Transformer hybrid
│   └── train_utils.py          # Shared training utilities
├── evaluation/                 # Model evaluation
│   ├── run_evaluation.py       # Main evaluation runner
│   ├── clean_performance.py    # Accuracy evaluation
│   └── corruption_robustness.py # Robustness testing
├── analysis/                   # Data & result analysis
│   ├── run_pipeline.py         # Analysis pipeline runner
│   ├── label_analysis.py       # Class distribution
│   ├── data_quality.py         # Duplicate detection
│   └── robustness.py           # Perturbation analysis
└── utils/                      # Shared utilities
    ├── metrics.py              # Accuracy & loss metrics
    └── checkpointing.py        # Model save/load
```

---

## 2. Model Architectures

**File:** `model_architectures.py`

### What is it?
The central registry for all model architectures. Provides standardized builder functions that:
1. Load pretrained weights (ImageNet)
2. Adapt first conv layer for grayscale input (1 channel)
3. Replace classifier head for 11 classes
4. Return model with a `ModelRecipe` containing training hyperparameters

### Key Components:

#### ModelRecipe Dataclass
```python
@dataclass(frozen=True)
class ModelRecipe:
    name: str                    # Model identifier
    input_size: Tuple[int, int]  # Expected input (H, W)
    default_lr: float            # Recommended learning rate
    default_weight_decay: float  # Recommended weight decay
    default_batch_size: int      # Recommended batch size
    classifier_dropout: float    # Dropout rate (if any)
```

### Available Architectures:

| Model | Builder Function | Input Size | Default LR | Batch Size | Notes |
|-------|------------------|------------|------------|------------|-------|
| ResNet-50 | `build_resnet50()` | 224×224 | 0.05 | 64 | Classic CNN baseline |
| ResNet-101 | `build_resnet101()` | 224×224 | 0.05 | 48 | Deeper ResNet |
| ResNeXt-50 | `build_resnext50_32x4d()` | 224×224 | 0.05 | 64 | Grouped convolutions |
| ResNeXt-101 | `build_resnext101_32x8d()` | 224×224 | 0.05 | 48 | Largest ResNet variant |
| EfficientNet-B3 | `build_efficientnet_b3()` | 300×300 | 0.02 | 32 | Compound scaling |
| DenseNet-121 | `build_densenet121()` | 224×224 | 0.05 | 64 | Dense connections |
| DenseNet-121 Adaptive | `build_densenet121_adaptive()` | 224×224 | 0.05 | 32 | SE attention + gating |
| ConvNeXt-Tiny | `build_convnext_tiny()` | 224×224 | 0.01 | 64 | Modern CNN via timm |
| ViT-S/16 | `build_vit_s16()` | 224×224 | 5e-4 | 64 | Vision Transformer Small |
| ViT-B/16 | `build_vit_b16()` | 224×224 | 3e-4 | 32 | Vision Transformer Base |
| Swin-Tiny | `build_swin_tiny()` | 224×224 | 5e-4 | 64 | Shifted window attention |
| Swin MultiScale | `build_swin_multiscale()` | 224×224 | 5e-4 | 48 | Multi-scale fusion |
| ConvTransGFusion | `build_convtransgfusion()` | 224×224 | 5e-4 | 32 | CNN-Transformer hybrid |

### Grayscale Adaptation
The key function `_replace_first_conv()` adapts pretrained RGB models for grayscale:
```python
# Averages RGB kernel weights to create single-channel kernel
grayscale_weight = weight.mean(dim=1, keepdim=True)  # [out, 3, k, k] → [out, 1, k, k]
```

### Usage Example:
```python
from src.model_architectures import build_resnet50, build_all_models

# Single model
model, recipe = build_resnet50(num_classes=11, pretrained=True)
print(f"Model: {recipe.name}, LR: {recipe.default_lr}")

# All models
all_models = build_all_models(num_classes=11, pretrained=True)
for name, (model, recipe) in all_models.items():
    print(f"{name}: {recipe.input_size}")
```

---

## 3. Data Pipeline

### 3.1 OrganAMNIST Dataset

**File:** `data_pipeline/organamnist_dataset.py`

#### What is it?
PyTorch Dataset class for loading grayscale medical images with two modes:
1. **Folder mode**: Images organized in class subfolders
2. **Manifest mode**: CSV file with `path,label` or `file,label` columns

#### Class Structure:
```python
class OrganAMNISTDataset(Dataset):
    def __init__(
        self,
        root_dir: str,           # Base directory
        split: str,              # "train", "val", or "test"
        transform: Callable,     # Torchvision transforms
        manifest_csv: str,       # Optional CSV manifest
        class_to_index: dict,    # Label mapping
    )
    
    def __getitem__(self, index) -> Tuple[Tensor, int, str]:
        # Returns: (image_tensor, label, file_path)
```

#### Features:
- Automatic grayscale conversion (`.convert("L")`)
- Supports multiple image formats (PNG, JPG, TIFF, BMP)
- Returns file paths for debugging/analysis

---

### 3.2 DataLoaders

**File:** `data_pipeline/dataloaders.py`

#### What is it?
Factory functions for building DataLoaders with augmentation and class balancing.

#### Transform Levels:

| Level | Augmentations |
|-------|---------------|
| **weak** | Resize, HFlip, Rotation ±10° |
| **medium** | + ResizedCrop (0.9-1.0), Rotation ±15° |
| **strong** | + ResizedCrop (0.8-1.0), Affine, Rotation ±20° |

#### Class Balancing:
Uses `WeightedRandomSampler` to oversample minority classes:
```python
class_weights = {c: total / (num_classes * count) for c, count in counts.items()}
sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
```

#### Usage:
```python
from src.data_pipeline.dataloaders import build_dataloaders

loaders = build_dataloaders(
    data_root="dataset",
    input_size=(224, 224),
    batch_size=32,
    num_workers=4,
    aug_strength="strong",
    use_weighted_sampler=True,
)

train_loader = loaders["train"]
val_loader = loaders["val"]
```

---

## 4. Training

### 4.1 Training Engine

**File:** `training/engine.py`

#### What is it?
The core training loop with support for:
- Mixed precision (Mixup/CutMix)
- Learning rate scheduling (Cosine, Step)
- Warmup epochs
- Gradient clipping
- Checkpointing

#### Key Functions:

| Function | Purpose |
|----------|---------|
| `train_one_epoch()` | Single epoch training with optional Mixup/CutMix |
| `evaluate()` | Validation loop with metrics |
| `train_model()` | Complete training pipeline |
| `resolve_device()` | Auto-detect MPS/CUDA/CPU |

#### Training Pipeline:
```python
from src.training.engine import train_model

result = train_model(
    model=model,
    dataloaders={"train": train_loader, "val": val_loader},
    num_classes=11,
    out_dir="checkpoints/",
    epochs=50,
    optimizer_name="adamw",    # or "sgd"
    lr=1e-4,
    weight_decay=0.05,
    label_smoothing=0.1,
    run_tag="experiment_v1",
    scheduler="cosine",        # or "step"
    warmup_epochs=5,
    mixup_alpha=0.8,          # Set > 0 to enable
    cutmix_alpha=1.0,         # Set > 0 to enable
    grad_clip_norm=1.0,
)
```

#### Outputs Generated:
- `best_{tag}.pth` - Best model checkpoint
- `last_{tag}.pth` - Latest checkpoint
- `metrics_{tag}.json` - Training history
- `curves_{tag}.png` - Loss/accuracy plots

---

### 4.2 Experiment Runner

**File:** `training/run_experiments.py`

#### What is it?
Automated hyperparameter sweep across architectures and settings.

#### Default Sweep Grid:
```python
architectures = ["resnet50", "resnet101", "efficientnet_b3", ...]
optimizers = ["sgd", "adamw"]
label_smoothings = [0.0, 0.1]
aug_strengths = ["medium"]
learning_rates = [0.001, 0.01]
```

#### Usage:
```bash
# Run all experiments
python -m src.training.run_experiments

# Run specific architectures
python -m src.training.run_experiments --architectures resnet50 densenet121
```

---

### 4.3 Architecture-Specific Trainers

| File | Description |
|------|-------------|
| `train_adaptive_densenet.py` | Train DenseNet-121 with SE attention |
| `train_convtransgfusion.py` | Train CNN-Transformer hybrid |

Each includes optimized hyperparameters and training recipes.

---

## 5. Models Module

### 5.1 Standard Architectures

Simple wrappers calling `model_architectures.py`:

| File | Model |
|------|-------|
| `resnet50.py` | ResNet-50 |
| `resnet101.py` | ResNet-101 |
| `resnext50_32x4d.py` | ResNeXt-50 |
| `resnext101_32x8d.py` | ResNeXt-101 |
| `densenet121.py` | DenseNet-121 |
| `efficientnet_b3.py` | EfficientNet-B3 |
| `vit_s16.py` | ViT-Small |
| `vit_b16.py` | ViT-Base |
| `swin_tiny.py` | Swin-Tiny |
| `convnext_tiny.py` | ConvNeXt-Tiny |

---

### 5.2 Enhanced Architectures

#### DenseNet-121 Adaptive
**File:** `densenet121_adaptive.py`

**Innovations:**
1. **SE Blocks** - Squeeze-and-Excitation channel attention
2. **Per-layer Gating** - Learnable gates for feature selection
3. **Adaptive Fusion** - Weighted combination of dense block features

```python
class SEBlock(nn.Module):
    """Channel attention via squeeze-excitation"""
    # Squeeze: Global average pool → FC → ReLU → FC → Sigmoid
    # Excitation: Multiply input by attention weights

class EnhancedDenseBlock(nn.ModuleDict):
    """Dense block with per-layer gating"""
    # Each layer output scaled by learned gate ∈ (0, 1)
```

---

#### Swin MultiScale
**File:** `swin_multiscale.py`

**Innovations:**
1. **Multi-scale Feature Extraction** - Features from all 4 Swin stages
2. **Attention-weighted Fusion** - Learns to combine scales
3. **Deep Supervision** - Auxiliary heads at each scale

```python
# Stage outputs for Swin-Tiny:
# Stage 1: 56×56, 96 channels
# Stage 2: 28×28, 192 channels  
# Stage 3: 14×14, 384 channels
# Stage 4: 7×7, 768 channels

class MultiScaleFusion(nn.Module):
    """Project and fuse features from all stages"""
    # 1. Project each stage to common dimension
    # 2. Generate attention weights from context
    # 3. Weighted sum of projections
    # 4. SE attention on fused output
```

**Training with Deep Supervision:**
```python
from src.models.swin_multiscale import SwinMultiScale, MultiScaleLoss

model = SwinMultiScale(num_classes=11, use_aux_heads=True)
criterion = MultiScaleLoss(aux_weight=0.4)

# During training:
main_logits, aux_logits = model(images)
loss = criterion((main_logits, aux_logits), targets)
```

---

#### ConvTransGFusion (CNN-Transformer Hybrid)
**File:** `convtransgfusion.py`

**Architecture:**
```
Input (224×224×1)
    ├── ConvNeXt Branch (local features)
    │   ├── Stage 1: 56×56, 96ch
    │   ├── Stage 2: 28×28, 192ch
    │   ├── Stage 3: 14×14, 384ch
    │   └── Stage 4: 7×7, 768ch
    │
    └── Swin Transformer Branch (global attention)
        ├── Stage 1: 56×56, 96ch
        ├── Stage 2: 28×28, 192ch
        ├── Stage 3: 14×14, 384ch
        └── Stage 4: 7×7, 768ch

    ↓ Feature Alignment (bilinear interpolation)
    ↓ AGFF (Attention-Guided Feature Fusion)
        ├── Channel Attention
        ├── Spatial Attention
        └── Weighted Fusion
    ↓ Classification Head
    ↓ Output (11 classes)
```

---

### 5.3 Training Utilities

**File:** `models/train_utils.py`

**Key Components:**

| Class/Function | Purpose |
|----------------|---------|
| `TrainingConfig` | Dataclass with all training hyperparameters |
| `OrganDataset` | Simple Dataset for training |
| `run_training()` | Complete training pipeline |
| `prepare_datasets()` | Load and transform datasets |
| `_load_class_weights()` | Load precomputed class weights |

---

### 5.4 Prediction Scripts

| File | Purpose |
|------|---------|
| `predict.py` | Run inference with trained model |
| `predict_finetune.py` | Inference with finetuned models |
| `predict_swin_multiscale.py` | Inference with Swin MultiScale |
| `predict_convnext_finetune.py` | Inference with finetuned ConvNeXt |

---

## 6. Evaluation

**Directory:** `evaluation/`

### 6.1 Evaluation Runner

**File:** `run_evaluation.py`

**Usage:**
```bash
# Run all evaluations
python -m src.evaluation.run_evaluation

# Clean performance only
python -m src.evaluation.run_evaluation --clean

# Robustness only
python -m src.evaluation.run_evaluation --robust
```

---

### 6.2 Clean Performance

**File:** `clean_performance.py`

**Metrics Computed:**
- Overall accuracy
- Per-class accuracy
- Macro F1 score
- Confusion matrices
- Inference time

---

### 6.3 Corruption Robustness

**File:** `corruption_robustness.py`

**15 Corruption Types Tested:**

| Category | Corruptions |
|----------|-------------|
| Noise | Gaussian, Shot, Impulse |
| Blur | Defocus, Glass, Motion, Zoom |
| Weather | Snow, Frost, Fog, Brightness |
| Digital | Contrast, Elastic, Pixelate, JPEG |

**Outputs:**
- Per-model corruption accuracy
- Category-wise rankings
- Relative robustness scores
- Corruption heatmaps

---

## 7. Analysis

**Directory:** `analysis/`

### 7.1 Analysis Pipeline

**File:** `run_pipeline.py`

**Execution Order:**
1. Label analysis → Class distribution
2. Image statistics → Pixel histograms
3. Quality checks → Duplicate detection
4. Robustness probes → Perturbation testing
5. Latent structure → PCA + t-SNE
6. Geometric analysis → Edge density, symmetry

**Usage:**
```bash
python -m src.analysis.run_pipeline
```

---

### 7.2 Analysis Modules

| Module | Purpose | Key Outputs |
|--------|---------|-------------|
| `label_analysis.py` | Class distribution | `label_distribution.json`, bar charts |
| `class_statistics.py` | Per-class pixel stats | `class_statistics.json` |
| `class_imbalance.py` | Imbalance impact | Confusion matrix, per-class accuracy |
| `image_stats.py` | Image-level statistics | `*_image_stats.csv`, histograms |
| `data_quality.py` | Duplicates, suspect labels | `data_quality_*.json` |
| `quality_checks.py` | Missing files, grid samples | Sample grids, QA reports |
| `latent_structure.py` | Dimensionality reduction | t-SNE plots, PCA variance |
| `geometric.py` | Edge density, flip differences | `geometric_stats.csv` |
| `robustness.py` | Perturbation effects | PSNR/SSIM metrics |
| `robustness_deepdive.py` | Adversarial attacks | FGSM/PGD results |
| `feature_exploration.py` | Multi-scale features, Grad-CAM | Feature visualizations |
| `test_characterization.py` | Train/val/test comparison | Distribution shift metrics |

---

### 7.3 Configuration

**File:** `analysis/config.py`

**Dataset Paths:**
```python
DATASET_CONFIG = DatasetConfig(
    root=Path("dataset"),
    train_images=Path("dataset/train/images_train"),
    train_labels=Path("dataset/train/labels_train.csv"),
    val_images=Path("dataset/val/images_val"),
    val_labels=Path("dataset/val/labels_val.csv"),
    test_images=Path("dataset/test/images"),
)

OUTPUT_CONFIG = OutputConfig(
    root=Path("analysis_outputs"),
    models_root=Path("analysis_outputs/models"),
    figures=Path("analysis_outputs/figures"),
    tables=Path("analysis_outputs/tables"),
    reports=Path("analysis_outputs/reports"),
)
```

**Environment Variables:**
- `DATASET_ROOT` - Override dataset location
- `OUTPUT_ROOT` - Override output location

---

## 8. Utilities

### 8.1 Metrics

**File:** `utils/metrics.py`

```python
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    per_class_accuracy: Dict[int, float]

# Functions:
compute_accuracy(logits, targets) -> float
compute_per_class_accuracy(logits, targets, num_classes) -> Dict
aggregate_epoch_metrics(losses, logits, targets, num_classes) -> EpochMetrics
```

---

### 8.2 Checkpointing

**File:** `utils/checkpointing.py`

```python
def save_checkpoint(state: Dict, filename: str) -> None:
    """Save model checkpoint with optimizer state"""

def load_checkpoint(filename: str, map_location=None) -> Dict:
    """Load checkpoint for resuming training"""

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
```

---

## 9. How to Use

### 9.1 Quick Start: Train a Model

```python
from src.model_architectures import build_resnet50
from src.data_pipeline.dataloaders import build_dataloaders
from src.training.engine import train_model

# Build model
model, recipe = build_resnet50(num_classes=11, pretrained=True)

# Build data loaders
loaders = build_dataloaders(
    data_root="dataset",
    input_size=recipe.input_size,
    batch_size=recipe.default_batch_size,
    num_workers=4,
    aug_strength="strong",
)

# Train
result = train_model(
    model=model,
    dataloaders=loaders,
    num_classes=11,
    out_dir="checkpoints/",
    epochs=50,
    optimizer_name="sgd",
    lr=recipe.default_lr,
    weight_decay=recipe.default_weight_decay,
    label_smoothing=0.1,
    run_tag="resnet50_v1",
)

print(f"Best accuracy: {result['best']['accuracy']:.4f}")
```

---

### 9.2 Run Full Experiment Sweep

```bash
# Train all architectures with hyperparameter sweep
python -m src.training.run_experiments \
    --data-root dataset \
    --out-root training_logs

# Train specific architectures
python -m src.training.run_experiments \
    --architectures resnet50 densenet121 swin_tiny
```

---

### 9.3 Run Complete Analysis

```bash
# Run all analysis modules
python -m src.analysis.run_pipeline
```

---

### 9.4 Evaluate Trained Models

```bash
# Evaluate all trained models
python -m src.evaluation.run_evaluation

# Clean performance only
python -m src.evaluation.run_evaluation --clean

# Robustness testing only
python -m src.evaluation.run_evaluation --robust
```

---

### 9.5 Train Specific Advanced Models

```bash
# Train adaptive DenseNet with SE attention
python -m src.models.densenet121_adaptive

# Train Swin with multi-scale fusion
python -m src.models.swin_multiscale --epochs 50 --aux-weight 0.4

# Train CNN-Transformer hybrid
python -m src.training.train_convtransgfusion
```

---

### 9.6 Make Predictions

```python
import torch
from src.model_architectures import build_swin_tiny
from torchvision import transforms
from PIL import Image

# Load model
model, recipe = build_swin_tiny(num_classes=11)
model.load_state_dict(torch.load("checkpoints/swin_tiny_best.pth"))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(recipe.input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

img = Image.open("sample.png")
x = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()
    prob = torch.softmax(logits, dim=1)[0, pred].item()

print(f"Predicted class: {pred}, Confidence: {prob:.4f}")
```

---

## 📁 File Reference Summary

### Root Files
| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `model_architectures.py` | Central model registry (13 architectures) |

### data_pipeline/
| File | Purpose |
|------|---------|
| `organamnist_dataset.py` | PyTorch Dataset class |
| `dataloaders.py` | DataLoader factory with augmentation |

### training/
| File | Purpose |
|------|---------|
| `engine.py` | Core training loop |
| `run_experiments.py` | Hyperparameter sweep |
| `run_baselines.py` | Baseline training |
| `train_*.py` | Architecture-specific trainers |
| `evaluate_best_on_test.py` | Test set evaluation |

### models/
| File | Purpose |
|------|---------|
| `*.py` (10 files) | Standard architecture wrappers |
| `densenet121_adaptive.py` | DenseNet with SE + gating |
| `swin_multiscale.py` | Multi-scale Swin fusion |
| `convtransgfusion.py` | CNN-Transformer hybrid |
| `train_utils.py` | Shared training utilities |
| `predict*.py` | Inference scripts |
| `finetune_*.py` | Finetuning scripts |

### evaluation/
| File | Purpose |
|------|---------|
| `run_evaluation.py` | Main evaluation runner |
| `clean_performance.py` | Accuracy evaluation |
| `corruption_robustness.py` | Robustness testing |
| `corruption_testing.py` | Corruption generators |
| `config.py` | Evaluation configuration |

### analysis/
| File | Purpose |
|------|---------|
| `run_pipeline.py` | Analysis orchestrator |
| `config.py` | Path configuration |
| `utils.py` | Shared helpers |
| `label_analysis.py` | Class distribution |
| `image_stats.py` | Pixel statistics |
| `class_*.py` | Class-level analysis |
| `data_quality.py` | Duplicate detection |
| `quality_checks.py` | QA reports |
| `robustness*.py` | Perturbation testing |
| `latent_structure.py` | PCA + t-SNE |
| `geometric.py` | Edge/symmetry analysis |
| `feature_exploration.py` | Grad-CAM, embeddings |
| `test_characterization.py` | Distribution shift |

### utils/
| File | Purpose |
|------|---------|
| `metrics.py` | Accuracy/loss computation |
| `checkpointing.py` | Model save/load |

---

## 🎯 Key Takeaways

1. **Model Zoo**: 13 architectures from CNNs to Vision Transformers
2. **Automatic Grayscale**: All models adapted for single-channel input
3. **Class Balancing**: Built-in weighted sampling for imbalanced data
4. **Flexible Training**: Mixup, CutMix, label smoothing, cosine scheduling
5. **Advanced Architectures**: Custom attention, multi-scale fusion, hybrid models
6. **Comprehensive Evaluation**: Clean accuracy + 15 corruption types
7. **Full Analysis Suite**: From data quality to adversarial robustness


# ⚙️ Complete Source Code Guide

This is the “teach me every module” reference for the entire `src/` tree. Use it when you need to understand how training, evaluation, analysis, and inference pieces connect. Pair it with the figures/tables/reports guides for the full story.

---

## Contents

1. [Orientation Map](#1-orientation-map)  
2. [Execution Flows](#2-execution-flows)  
3. [Model Registry (`model_architectures.py`)](#3-model-registry-model_architecturespy)  
4. [Data Pipeline (`data_pipeline/`)](#4-data-pipeline-data_pipeline)  
5. [Training Stack (`training/`)](#5-training-stack-training)  
6. [Model Implementations (`models/`)](#6-model-implementations-models)  
7. [Evaluation Suite (`evaluation/`)](#7-evaluation-suite-evaluation)  
8. [Analysis Suite (`analysis/`)](#8-analysis-suite-analysis)  
9. [Utilities (`utils/`)](#9-utilities-utils)  
10. [End-to-End Workflows](#10-end-to-end-workflows)  
11. [Quick File Reference](#11-quick-file-reference)

---

## 1. Orientation Map

```
src/
├── __init__.py
├── model_architectures.py        # Model builder registry
├── data_pipeline/                # Dataset + DataLoader factories
├── training/                     # Engines, experiments, scripts
├── models/                       # Architecture-specific trainers/predictors
├── evaluation/                   # Clean + corruption evaluation
├── analysis/                     # Exploratory & diagnostic pipeline
└── utils/                        # Shared helpers
```

Always start with the registry (`model_architectures.py`) and flow outward: dataloaders feed the training engine, which produces checkpoints that evaluation & analysis consume.

---

## 2. Execution Flows

| Flow | Command | Touches | Outputs |
|------|---------|---------|---------|
| **Baseline training** | `python -m src.training.run_baselines` | Registry → dataloaders → `training/engine.py` | `training_logs/`, checkpoints |
| **Experiment sweep** | `python -m src.training.run_experiments` | Adds sweep grid + logging | Multiple runs with varied hparams |
| **Custom trainer** | `python -m src.models.swin_multiscale --epochs 50` | Uses architecture-specific script | Checkpoints tuned for that model |
| **Evaluation** | `python -m src.evaluation.run_evaluation --clean --robust` | Loads checkpoints → evaluation suite | `evaluation_outputs/` tables & figs |
| **Analysis** | `python -m src.analysis.run_pipeline` | Sequenced analysis modules | `analysis_outputs/` across reports/tables/figs |
| **Inference** | `python -m src.models.predict --model swin_tiny_finetuned` | Loads model, runs dataloaders/inference | Submission CSV or console predictions |

---

## 3. Model Registry (`model_architectures.py`)

### 3.1 ModelRecipe
```python
@dataclass(frozen=True)
class ModelRecipe:
    name: str
    input_size: Tuple[int, int]
    default_lr: float
    default_weight_decay: float
    default_batch_size: int
    classifier_dropout: float | None = None
```

### 3.2 Builders (selection)

| Builder | Backbone | Highlights |
|---------|----------|------------|
| `build_resnet50()` | ResNet-50 | 224² input, SGD LR 0.05 |
| `build_densenet121()` | DenseNet-121 | 7M params, 0.94 ms inference |
| `build_densenet121_adaptive()` | DenseNet + SE + gating | Adds adaptive per-layer scaling |
| `build_efficientnet_b3()` | EfficientNet-B3 | 300² input, fastest inference |
| `build_convnext_tiny()` | ConvNeXt-Tiny | Modern CNN with LayerNorm |
| `build_vit_s16()` / `build_vit_b16()` | Vision Transformers | Patch size 16, dropout-ready |
| `build_swin_tiny()` | Swin Transformer | Shifted-window attention |
| `build_swin_multiscale()` | Custom Swin | Stage fusion + auxiliary heads |
| `build_convtransgfusion()` | ConvNeXt + Swin hybrid | Attention-guided fusion |

Every builder:
1. Loads pretrained ImageNet weights.  
2. Calls `_replace_first_conv()` to average RGB kernels → grayscale.  
3. Installs an 11-class classifier.  
4. Returns `(model, recipe)` so training scripts know the defaults.

### 3.3 Quick usage
```python
from src.model_architectures import build_resnet50
model, recipe = build_resnet50(num_classes=11, pretrained=True)
```

---

## 4. Data Pipeline (`data_pipeline/`)

### 4.1 Dataset (`organamnist_dataset.py`)

| Feature | Description |
|---------|-------------|
| Multi-mode | Works with folder structures or CSV manifests. |
| Returns | `(tensor, label, relative_path)` for debugging duplicate detection or mislabels. |
| Formats | Accepts PNG/JPG/TIFF/BMP and forces grayscale conversion. |

Constructor signature:
```python
OrganAMNISTDataset(
    root_dir, split, transform, manifest_csv=None, class_to_index=None
)
```

### 4.2 Dataloaders (`dataloaders.py`)

| Option | Effect |
|--------|--------|
| `aug_strength` | `weak` (resize/flip), `medium` (crop + ±15°), `strong` (affine, ±20°). |
| `use_weighted_sampler` | Enables `WeightedRandomSampler` based on class counts. |
| `input_size` | Pulled from `ModelRecipe`; ensures consistent resizing. |

```python
from src.data_pipeline.dataloaders import build_dataloaders
dls = build_dataloaders(
    data_root="dataset",
    input_size=recipe.input_size,
    batch_size=recipe.default_batch_size,
    aug_strength="strong",
    use_weighted_sampler=True,
)
```

---

## 5. Training Stack (`training/`)

### 5.1 Engine (`training/engine.py`)

| Function | Job |
|----------|-----|
| `train_model()` | High-level orchestrator (warmup, scheduler, logging, checkpointing). |
| `train_one_epoch()` | Forward/backward pass, optional MixUp/CutMix, AMP, grad clipping. |
| `evaluate()` | Validation loop returning `EpochMetrics`. |
| `resolve_device()` | Picks CUDA / MPS / CPU. |

**Features supported:** label smoothing, cosine/step schedulers, warmup epochs, automatic mixed precision, MixUp, CutMix, EMA (via scripts), gradient clipping.

### 5.2 Scripts

| Script | Description |
|--------|-------------|
| `run_baselines.py` | Trains a curated set of models with default recipes. |
| `run_experiments.py` | Sweeps architectures × optimizers × learning rates × augment strength. |
| `run_adaptive_*.py` | Architecture-specific fine-tuning scripts. |
| `train_convtransgfusion.py` | Hybrid CNN/Transformer training entry. |
| `evaluate_best_on_test.py` | Loads best checkpoint, runs inference on test set. |

Each script imports the registry + dataloaders + engine—meaning you can drop in new architectures with minimal glue.

---

## 6. Model Implementations (`models/`)

### 6.1 Standard wrappers
Files such as `resnet50.py`, `efficientnet_b3.py`, `vit_s16.py`, and `convnext_tiny.py`:
1. Parse CLI args (epochs, LR, output dir).  
2. Build model/recipe via registry.  
3. Build dataloaders with recipe defaults.  
4. Call `train_model()` and log metrics.

### 6.2 Enhanced/custom scripts

| File | Why open it |
|------|-------------|
| `densenet121_adaptive.py` | Contains SE blocks, per-layer gating, adaptive fusion logic. |
| `swin_multiscale.py` | Multi-stage feature collection, attention-weighted fusion, auxiliary head training. |
| `convtransgfusion.py` | Dual-branch (ConvNeXt + Swin) with attention-guided feature fusion. |
| `finetune_swin_tiny.py` | Adds layer-wise LR decay, MixUp/CutMix, EMA for high-accuracy Swin fine-tuning. |
| `predict*.py` | Inference helpers for specific models (plain Swin, finetuned ConvNeXt, etc.). |

### 6.3 Shared helpers (`models/train_utils.py`)
Defines `TrainingConfig`, dataset prep functions, and a `run_training()` wrapper for scripts that want a simpler API.

---

## 7. Evaluation Suite (`evaluation/`)

| Component | Description |
|-----------|-------------|
| `run_evaluation.py` | CLI entry: `--clean`, `--robust`, `--models resnet50 swin_tiny`. Loads checkpoints automatically. |
| `clean_performance.py` | Computes accuracy, macro F1, per-class accuracy, confusion matrices, inference time. |
| `corruption_robustness.py` | Applies 15 corruptions (noise/blur/weather/digital) and records accuracy + relative robustness. |
| `corruption_testing.py` | Individual corruption generators (Gaussian noise, motion blur, fog, pixelate, etc.). |
| `config.py` | Points to checkpoint directory and output paths. |

Outputs populate `evaluation_outputs/` (tables, reports, heatmaps) which are explained in the evaluation guide outside `src/`.

---

## 8. Analysis Suite (`analysis/`)

| Module | Output |
|--------|--------|
| `run_pipeline.py` | Orchestrates the full pipeline: label stats → quality → robustness → latent structure → geometric features. |
| `label_analysis.py` | `label_distribution.json`, bar charts. |
| `class_statistics.py` | Per-class pixel stats. |
| `class_imbalance.py` | Balanced accuracy probes, confusion matrices. |
| `image_stats.py` | `train_image_stats.csv`, `val_image_stats.csv`. |
| `data_quality.py` | Duplicate detection, suspect label reports. |
| `quality_checks.py` | Sample grids, missing file audits. |
| `robustness.py` | Perturbation PSNR/SSIM metrics + example grids. |
| `robustness_deepdive.py` | FGSM/PGD adversarial evaluation. |
| `latent_structure.py` | PCA variance, t-SNE embeddings. |
| `geometric.py` | Edge density + flip difference tables. |
| `feature_exploration.py` | Grad-CAMs, multi-scale stats. |
| `test_characterization.py` | Train/val/test shift metrics (pixel, edge, LBP). |

Configuration lives in `analysis/config.py`, which defines dataset paths and output directories (overridable via `DATASET_ROOT`, `OUTPUT_ROOT` env vars).

---

## 9. Utilities (`utils/`)

| File | Purpose |
|------|---------|
| `metrics.py` | Defines `EpochMetrics`, `compute_accuracy`, `compute_per_class_accuracy`, and aggregators used by training/eval loops. |
| `checkpointing.py` | `save_checkpoint`, `load_checkpoint`, and `ensure_dir` used globally. |

Keep utility code minimal here; anything model-specific belongs in `models/` or `training/`.

---

## 10. End-to-End Workflows

### 10.1 Train a baseline
```python
from src.model_architectures import build_resnet50
from src.data_pipeline.dataloaders import build_dataloaders
from src.training.engine import train_model

model, recipe = build_resnet50(num_classes=11, pretrained=True)
dls = build_dataloaders(
    data_root="dataset",
    input_size=recipe.input_size,
    batch_size=recipe.default_batch_size,
    aug_strength="strong",
    use_weighted_sampler=True,
)
train_model(
    model=model,
    dataloaders=dls,
    num_classes=11,
    out_dir="checkpoints/",
    epochs=50,
    optimizer_name="sgd",
    lr=recipe.default_lr,
    weight_decay=recipe.default_weight_decay,
    label_smoothing=0.1,
    run_tag="resnet50_v1",
)
```

### 10.2 Run experiment sweep
```bash
python -m src.training.run_experiments \
  --architectures resnet50 densenet121 swin_tiny \
  --learning-rates 0.01 0.001 \
  --aug-strength medium
```

### 10.3 Full analysis + evaluation
```bash
python -m src.analysis.run_pipeline
python -m src.evaluation.run_evaluation --clean --robust
```

### 10.4 Advanced scripts
```bash
python -m src.models.densenet121_adaptive --epochs 50
python -m src.models.swin_multiscale --epochs 50 --aux-weight 0.4
python -m src.training.train_convtransgfusion
```

### 10.5 Inference snippet
```python
from src.model_architectures import build_swin_tiny
from torchvision import transforms
from PIL import Image
import torch

model, recipe = build_swin_tiny(num_classes=11)
model.load_state_dict(torch.load("checkpoints/swin_tiny_best.pth"))
model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(recipe.input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
x = tfm(Image.open("sample.png")).unsqueeze(0)
with torch.no_grad():
    pred = torch.softmax(model(x), dim=1).argmax(dim=1).item()
```

---

## 11. Quick File Reference

| Directory | Files to know | Why |
|-----------|---------------|-----|
| Root | `model_architectures.py` | Add/modify model recipes, grayscale adaptation logic. |
| `data_pipeline/` | `organamnist_dataset.py`, `dataloaders.py` | Adjust augmentation, sampler, normalization. |
| `training/` | `engine.py`, `run_experiments.py`, `run_baselines.py` | Modify training loop, add schedules, launch sweeps. |
| `models/` | `densenet121_adaptive.py`, `swin_multiscale.py`, `convtransgfusion.py`, `predict*.py` | Explore custom architectures and inference helpers. |
| `evaluation/` | `run_evaluation.py`, `clean_performance.py`, `corruption_robustness.py` | Generate clean + corruption metrics. |
| `analysis/` | `run_pipeline.py`, `label_analysis.py`, `robustness.py`, etc. | Produce figures/tables/reports for data/model diagnostics. |
| `utils/` | `metrics.py`, `checkpointing.py` | Shared helpers used by training/eval/analysis. |

With this guide you can jump from any high-level question (“where do MixUp settings live?”) to the exact file and section in seconds.