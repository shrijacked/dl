# OrganAMNIST Deep Learning Classification Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

> **A comprehensive deep learning solution for medical image classification featuring 13+ architectures, extensive analysis pipelines, and state-of-the-art results on the OrganAMNIST dataset.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Achievements](#-key-achievements)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Analysis Pipeline](#-analysis-pipeline)
- [Technical Details](#-technical-details)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project addresses the **OrganAMNIST Challenge** — a medical image classification task requiring the identification of **11 different abdominal organs** from grayscale CT scan images. The project implements a complete machine learning pipeline from data analysis through model training to comprehensive evaluation.

### What Makes This Project Unique

1. **Comprehensive Model Zoo**: 13+ architectures from classical CNNs to cutting-edge Vision Transformers
2. **Custom Architectures**: 3 novel architectures designed specifically for medical imaging
3. **Extensive Analysis**: Complete data quality, distribution, and robustness analysis pipeline
4. **Production-Ready**: HPC cluster support, reproducible experiments, and comprehensive documentation
5. **State-of-the-Art Results**: 99.69% validation accuracy with robust performance under corruptions

---

## 🏆 Key Achievements

| Metric | Value | Model |
|--------|-------|-------|
| **Best Validation Accuracy** | **99.69%** | Swin-Tiny Finetuned |
| **Best Robustness (Corrupted)** | **76.88%** | ConvNeXt-Tiny Finetuned |
| **Most Efficient** | 7M params, 0.94ms | DenseNet-121 |
| **Fastest Inference** | 0.77ms | EfficientNet-B3 |
| **Classes Covered** | 11 organs | Abdominal CT scans |
| **Total Training Samples** | 34,561 images | OrganAMNIST dataset |

### Performance Highlights

- ✅ **99.69% validation accuracy** — Top-tier performance across all organs
- ✅ **100% accuracy** on 9 out of 11 organ classes by best models
- ✅ **76.88% mean accuracy** under 15 types of image corruptions
- ✅ **Robust to distribution shift** — Detailed train/val/test analysis
- ✅ **Minimal overfitting** — Strong generalization through regularization

---

## 📊 Dataset

### OrganAMNIST Dataset Specifications

The OrganAMNIST dataset is derived from abdominal CT scans, containing grayscale images of 11 different organs.

| Split | Samples | Purpose |
|-------|---------|---------|
| **Training** | 34,561 | Model training |
| **Validation** | 6,491 | Hyperparameter tuning |
| **Test** | Hidden | Final evaluation (unlabeled) |

### The 11 Organ Classes

| Class ID | Organ | Training Samples | Proportion | Difficulty |
|----------|-------|------------------|------------|------------|
| 0 | **Bladder** | 1,956 | 5.66% | Easy |
| 1 | **Femur (Left)** | 1,390 | 4.02% | Easy |
| 2 | **Femur (Right)** | 1,357 | 3.93% | Easy |
| 3 | **Heart** | 1,474 | 4.26% | **Hard** ⚠️ |
| 4 | **Kidney (Left)** | 3,963 | 11.47% | Medium |
| 5 | **Kidney (Right)** | 3,817 | 11.04% | Medium |
| 6 | **Liver** | 6,164 | 17.84% | Easy |
| 7 | **Lung (Left)** | 3,919 | 11.34% | Medium |
| 8 | **Lung (Right)** | 3,929 | 11.37% | **Hard** ⚠️ |
| 9 | **Spleen** | 3,031 | 8.77% | Medium |
| 10 | **Pancreas** | 3,561 | 10.30% | Medium |

### Image Specifications

- **Format**: Grayscale PNG images
- **Resolution**: 28×28 pixels (original) → 224×224 (resized for training)
- **Bit Depth**: 8-bit (0-255 pixel values)
- **Preprocessing**: Normalized with mean=0.5, std=0.5
- **Augmentation**: Rotation, flips, crops, affine transforms

### Class Imbalance Characteristics

- **Imbalance Ratio**: ~4.5:1 (Liver vs Femur classes)
- **Mitigation**: Weighted random sampling, class-weighted loss
- **Challenge**: Heart (4.26%) and Femurs (3.93-4.02%) are underrepresented

---

## 🏗️ Model Architectures

### Architecture Taxonomy

```
Classical CNNs                    Hybrid Architectures           Vision Transformers
├── ResNet Family                 ├── DenseViT (Custom)          ├── ViT Family
│   ├── ResNet-50                 └── ConvTransGFusion (Custom)  │   ├── ViT-S/16
│   ├── ResNet-101                                               │   └── ViT-B/16
│   └── ResNeXt                                                  │
│       ├── ResNeXt-50 (32×4d)                                   └── Swin Transformer
│       └── ResNeXt-101 (32×8d)                                      ├── Swin-Tiny
│                                                                    └── Swin-MultiScale (Custom)
├── Dense Connections
│   ├── DenseNet-121
│   └── DenseNet-121 Adaptive (Custom)
│
└── Efficient Architectures
    ├── EfficientNet-B3
    └── ConvNeXt-Tiny
```

### Model Comparison Table

| Model | Type | Params (M) | FLOPs (G) | Val Acc (%) | Corruption Acc (%) | Inference (ms) |
|-------|------|-----------|-----------|-------------|-------------------|----------------|
| **Swin-Tiny Finetuned** 🥇 | Transformer | 27.5 | 4.5 | **99.69** | 75.36 | 1.17 |
| **ConvNeXt-Tiny Finetuned** 🥈 | CNN | 27.8 | 4.5 | 99.60 | **76.88** | 1.17 |
| **DenseNet-121** 🥉 | CNN | 7.0 | 2.9 | 99.61 | 73.96 | **0.94** |
| **EfficientNet-B3** | CNN | 10.7 | 1.8 | 99.32 | 74.57 | **0.77** |
| ResNet-50 | CNN | 23.5 | 4.1 | 99.14 | 72.44 | 1.11 |
| ResNet-101 | CNN | 42.5 | 7.8 | 99.15 | 73.42 | 1.67 |
| ResNeXt-50 (32×4d) | CNN | 23.0 | 4.2 | 98.94 | 73.22 | 1.13 |
| ResNeXt-101 (32×8d) | CNN | 86.7 | 16.4 | 98.94 | 74.20 | 2.96 |
| Swin-Tiny | Transformer | 27.5 | 4.5 | 99.63 | 74.55 | 1.17 |
| ConvNeXt-Tiny | CNN | 27.8 | 4.5 | 97.32 | 74.10 | 1.17 |
| ViT-S/16 | Transformer | 21.7 | 4.6 | 98.43 | 70.28 | 1.19 |

**Legend**: 🥇 Best Accuracy | 🥈 Most Robust | 🥉 Most Efficient

---

## 🌟 Custom Model Architectures

### 1. DenseNet-121 Adaptive

**Innovation**: Enhanced DenseNet with Squeeze-and-Excitation attention and adaptive per-layer gating.

```
Key Components:
├── SE Attention Blocks → Channel-wise recalibration in every dense layer
├── Adaptive Gating → Input-dependent scaling of layer contributions
│   └── GlobalAvgPool → MLP → Sigmoid → Per-Layer Gates (0-1)
├── Enhanced Transition Layers → SE attention between dense blocks
└── 4 Dense Blocks → (6, 12, 24, 16 layers) with growth rate 32
```

**Why it works**:
- **SE blocks**: Learn which feature channels are important for each input
- **Adaptive gating**: Dynamically weight layer contributions based on input content
- **Compact**: Only 9M parameters vs 7M for vanilla DenseNet

**Performance**: Competitive accuracy with improved feature selection

---

### 2. Swin-MultiScale

**Innovation**: Hierarchical Swin Transformer with multi-scale feature fusion and deep supervision.

```
Architecture Flow:
├── Swin-Tiny Backbone (4 stages)
│   ├── Stage 1: 56×56, 96 channels ─────┐
│   ├── Stage 2: 28×28, 192 channels ────┤
│   ├── Stage 3: 14×14, 384 channels ────┤ → Multi-Scale Fusion
│   └── Stage 4: 7×7, 768 channels  ─────┘
│
├── Scale Projection → Project each stage to 512-dim
├── Attention-Weighted Fusion
│   ├── Concat all scales → [B, 4×512]
│   ├── Weight Network → Linear → ReLU → Linear → Softmax
│   └── Weighted Sum → [B, 512] → SE Attention
│
├── Auxiliary Heads (for training)
│   └── Stage outputs → Linear → Dropout → 11 classes
│
└── Main Classification Head
    └── Fused features → Dropout(0.1) → Linear(512 → 11)
```

**Training Loss**: `main_loss + 0.4 × average(aux_losses)`

**Why it works**:
- **Multi-scale fusion**: Captures both fine-grained details and global context
- **Deep supervision**: Auxiliary heads regularize intermediate layers
- **Attention weights**: Learn optimal scale combination per input

**Performance**: Enhanced feature representation through multi-scale aggregation

---

### 3. ConvTransGFusion (Hybrid CNN-Transformer)

**Innovation**: Dual-branch architecture combining ConvNeXt (local features) and Swin Transformer (global attention) with attention-guided feature fusion.

```
Dual-Branch Architecture:

ConvNeXt Branch (CNN)              Swin Branch (Transformer)
├── Stage 1: 56×56, 96ch           ├── Stage 1: 56×56, 96ch
├── Stage 2: 28×28, 192ch          ├── Stage 2: 28×28, 192ch
├── Stage 3: 14×14, 384ch          ├── Stage 3: 14×14, 384ch
└── Stage 4: 7×7, 768ch            └── Stage 4: 7×7, 768ch
        ↓                                    ↓
        └──────────────┬─────────────────────┘
                       ↓
            Feature Alignment (Bilinear Interpolation)
                       ↓
        ╔══════════════════════════════════════╗
        ║  AGFF (Attention-Guided Feature Fusion)  ║
        ╠══════════════════════════════════════╣
        ║ 1. Feature Calibration                   ║
        ║    ├── Separate LayerNorm for each       ║
        ║    ├── Project to 384-dim                ║
        ║    ├── Learnable branch weights          ║
        ║    └── Concat [384|384] = 768            ║
        ║                                           ║
        ║ 2. Channel Attention                     ║
        ║    └── GAP → FC(768→48) → FC(48→768)    ║
        ║        → Sigmoid → Scale features        ║
        ║                                           ║
        ║ 3. Spatial Attention                     ║
        ║    └── Conv1×1(768→1) → Sigmoid         ║
        ║        → Scale features                  ║
        ║                                           ║
        ║ 4. Fusion                                ║
        ║    └── Concat [channel|spatial]          ║
        ║        → Conv1×1 → BatchNorm → GELU      ║
        ╚══════════════════════════════════════╝
                       ↓
            Classification Head
            └── GAP → LayerNorm → Dropout → Linear(768→11)
```

**Why it works**:
- **Complementary features**: CNN captures local textures, Transformer captures global structure
- **Adaptive fusion**: Input-dependent weighting of CNN vs Transformer
- **Multi-level attention**: Both channel (what) and spatial (where) attention
- **Pretrained initialization**: Swin branch starts with ImageNet knowledge

**Performance**: Balanced local-global feature representation

---

## 📁 Project Structure

```
dl-project/
├── 📂 dataset/                          # OrganAMNIST dataset
│   ├── train/
│   │   ├── images_train/                # 34,561 training images
│   │   └── labels_train.csv             # Training labels
│   ├── val/
│   │   ├── images_val/                  # 6,491 validation images
│   │   └── labels_val.csv               # Validation labels
│   └── test/
│       ├── images/                      # Test images (unlabeled)
│       └── manifest_public.csv          # Test manifest
│
├── 📂 src/                              # Source code
│   ├── 📄 model_architectures.py        # Central model registry (13 architectures)
│   ├── 📂 data_pipeline/
│   │   ├── organamnist_dataset.py       # Custom PyTorch Dataset
│   │   └── dataloaders.py               # DataLoader factory with augmentation
│   ├── 📂 models/                       # Model implementations
│   │   ├── resnet50.py                  # ResNet-50
│   │   ├── resnet101.py                 # ResNet-101
│   │   ├── resnext50_32x4d.py           # ResNeXt-50
│   │   ├── resnext101_32x8d.py          # ResNeXt-101
│   │   ├── densenet121.py               # DenseNet-121
│   │   ├── densenet121_adaptive.py      # 🌟 Custom Adaptive DenseNet
│   │   ├── efficientnet_b3.py           # EfficientNet-B3
│   │   ├── convnext_tiny.py             # ConvNeXt-Tiny
│   │   ├── vit_s16.py                   # Vision Transformer Small
│   │   ├── vit_b16.py                   # Vision Transformer Base
│   │   ├── swin_tiny.py                 # Swin Transformer Tiny
│   │   ├── swin_multiscale.py           # 🌟 Custom Multi-Scale Swin
│   │   ├── convtransgfusion.py          # 🌟 Custom Hybrid CNN-Transformer
│   │   ├── dense_vit.py                 # 🌟 Custom DenseViT
│   │   ├── finetune_*.py                # Finetuning scripts
│   │   ├── predict*.py                  # Inference scripts
│   │   └── train_utils.py               # Shared training utilities
│   ├── 📂 training/
│   │   ├── engine.py                    # Core training loop
│   │   ├── run_baselines.py             # Baseline experiments
│   │   ├── run_experiments.py           # Hyperparameter sweeps
│   │   └── evaluate_best_on_test.py     # Test set evaluation
│   ├── 📂 analysis/                     # Analysis pipeline (19 modules)
│   │   ├── run_pipeline.py              # Main orchestrator
│   │   ├── label_analysis.py            # Class distribution
│   │   ├── data_quality.py              # Duplicate detection
│   │   ├── feature_exploration.py       # Grad-CAM, multi-scale
│   │   ├── robustness.py                # Perturbation analysis
│   │   ├── robustness_deepdive.py       # Adversarial attacks (FGSM, PGD)
│   │   ├── latent_structure.py          # t-SNE, PCA visualization
│   │   ├── test_characterization.py     # Distribution shift detection
│   │   └── ...                          # 12 more analysis modules
│   ├── 📂 evaluation/
│   │   ├── run_evaluation.py            # Main evaluation runner
│   │   ├── clean_performance.py         # Accuracy, F1, confusion matrices
│   │   └── corruption_robustness.py     # 15 corruption types
│   └── 📂 utils/
│       ├── metrics.py                   # Custom metrics
│       └── checkpointing.py             # Model save/load
│
├── 📂 analysis_outputs/                 # Analysis results
│   ├── 📂 figures/                      # 60+ visualization plots
│   │   ├── FIGURE_GUIDE.md              # Complete figure documentation
│   │   ├── label_distribution.png
│   │   ├── latent_tsne.png
│   │   ├── feature_gradcam/             # 12 Grad-CAM visualizations
│   │   ├── robustness_adversarial_samples/
│   │   └── ...                          # Distribution, frequency, edge plots
│   ├── 📂 reports/                      # JSON metric reports
│   │   ├── REPORTS_GUIDE.md             # Complete report documentation
│   │   ├── label_distribution.json
│   │   ├── data_quality_summary.json
│   │   ├── distribution_shifts.json
│   │   ├── robustness_adversarial_results.json
│   │   └── ...                          # 17 analysis reports
│   ├── 📂 tables/                       # CSV data exports
│   │   ├── TABLES_GUIDE.md              # Complete table documentation
│   │   ├── label_distribution.csv
│   │   ├── feature_interclass_similarity.csv
│   │   └── ...                          # 14 analysis tables
│   └── 📂 models/
│       ├── weights/                     # Trained model checkpoints (.pth)
│       ├── confusion_matrix/            # Confusion matrices (.npy)
│       └── predictions/                 # Model predictions (.csv)
│
├── 📂 evaluation_outputs/               # Model evaluation results
│   ├── 📂 confusion_matrices/           # 11 per-model confusion matrices (PNG)
│   ├── 📂 figures/                      # Comparison visualizations
│   │   ├── corruption_heatmap.png
│   │   ├── robustness_ranking.png
│   │   ├── per_class_model_performance.png
│   │   ├── inference_time_comparison.png
│   │   └── model_diversity_heatmap.png
│   ├── 📂 reports/                      # JSON evaluation reports
│   │   ├── robustness_ranking.json      # 15 corruption types tested
│   │   └── class_specialists.json       # Best model per organ
│   ├── 📂 tables/                       # CSV performance tables
│   │   ├── model_comparison_table.csv
│   │   ├── per_class_performance.csv
│   │   ├── corruption_robustness_all_models.csv
│   │   └── model_diversity_correlation.csv
│   └── EVALUATION_GUIDE.md              # Complete evaluation documentation
│
├── 📂 training_logs/                    # Training artifacts per model
│   ├── adaptive_densenet/
│   ├── densenet121_adaptive/
│   └── ...                              # Checkpoints, metrics, curves
│
├── 📂 hpc/                              # HPC cluster configuration
│   ├── README_hpc.md                    # HPC setup guide
│   ├── hpc_config_example.sh
│   ├── setup_hpc_ssh.sh
│   └── login_hpc.sh
│
├── 📄 requirements.txt                  # Python dependencies
├── 📄 LICENSE                           # MIT License
├── 📄 README.md                         # This file
├── 📄 PROJECT_DOCUMENTATION.md          # Comprehensive project overview
├── 📄 HOW.md                            # Detailed usage guide
├── 📄 MODEL_ARCHITECTURE_DIAGRAMS_PROMPT.md  # Architecture specifications
└── 📄 model_comparison_dashboard.png    # Visual comparison dashboard
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 11.8+ (for GPU training)
- **Git LFS**: For large model weights (optional)
- **Disk Space**: ~50 GB (dataset + outputs + models)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/organamnist-dl-project.git
cd organamnist-dl-project

# If using Git LFS for model weights
git lfs install
git lfs pull
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch==2.5.1` — PyTorch deep learning framework
- `torchvision==0.20.1` — Computer vision utilities
- `timm==1.0.9` — Vision model architectures (Swin, ConvNeXt, ViT)
- `scikit-learn==1.7.2` — Machine learning utilities
- `matplotlib==3.9.4` — Plotting
- `seaborn==0.13.2` — Statistical visualization
- `pandas==2.3.3` — Data manipulation
- `numpy==2.1.3` — Numerical computing
- `Pillow==11.3.0` — Image processing

### Step 4: Download Dataset

The OrganAMNIST dataset should be placed in the `dataset/` directory:

```
dataset/
├── train/
│   ├── images_train/
│   └── labels_train.csv
├── val/
│   ├── images_val/
│   └── labels_val.csv
└── test/
    ├── images/
    └── manifest_public.csv
```

---

## 💻 Usage

### Quick Start: Train a Model

```bash
# Train ResNet-50 (baseline)
python -m src.models.resnet50

# Train DenseNet-121 (efficient)
python -m src.models.densenet121

# Train Swin-Tiny (high accuracy)
python -m src.models.swin_tiny

# Train custom Adaptive DenseNet
python -m src.models.densenet121_adaptive --epochs 50 --lr 0.01
```

### Advanced Training

```bash
# Train with custom hyperparameters
python -m src.models.swin_tiny \
    --epochs 50 \
    --lr 5e-4 \
    --batch-size 64 \
    --weight-decay 0.05 \
    --out-root training_logs/swin_experiment

# Finetune a pretrained model
python -m src.models.finetune_swin_tiny \
    --checkpoint training_logs/swin_tiny/best_model.pth \
    --epochs 25 \
    --lr 5e-5
```

### Run Baseline Experiments

Train multiple models with different configurations:

```bash
# Train 8 baseline configurations
python -m src.training.run_baselines \
    --data-root dataset \
    --out-root training_logs
```

### Run Hyperparameter Sweep

```bash
# Full hyperparameter sweep
python -m src.training.run_experiments \
    --architectures resnet50 densenet121 swin_tiny \
    --learning-rates 0.01 0.001 \
    --aug-strength medium strong \
    --out-root training_logs/experiments
```

### Run Analysis Pipeline

```bash
# Complete data analysis
python -m src.analysis.run_pipeline

# Individual analyses
python -m src.analysis.label_analysis          # Class distribution
python -m src.analysis.latent_structure        # t-SNE visualization
python -m src.analysis.feature_exploration     # Grad-CAM, multi-scale
python -m src.analysis.robustness_deepdive     # Adversarial attacks
python -m src.analysis.test_characterization   # Distribution shift
```

### Run Model Evaluation

```bash
# Evaluate all trained models
python -m src.evaluation.run_evaluation

# Clean performance only
python -m src.evaluation.run_evaluation --clean

# Robustness testing only
python -m src.evaluation.run_evaluation --robust

# Specific models
python -m src.evaluation.run_evaluation \
    --models resnet50 densenet121 swin_tiny_finetuned
```

### Generate Test Predictions

```bash
# Best model predictions
python -m src.training.evaluate_best_on_test

# Specific model predictions
python -m src.models.predict --model swin_tiny_finetuned

# With test-time augmentation
python -m src.models.predict_finetune --tta
```

---

## 📈 Results

### Overall Performance Leaderboard

| Rank | Model | Val Acc (%) | Macro F1 | Params (M) | FLOPs (G) | Inference (ms) |
|------|-------|-------------|----------|------------|-----------|----------------|
| 🥇 1 | **Swin-Tiny Finetuned** | **99.69** | **0.998** | 27.5 | 4.5 | 1.17 |
| 🥈 2 | Swin-Tiny | 99.63 | 0.996 | 27.5 | 4.5 | 1.17 |
| 🥉 3 | DenseNet-121 | 99.61 | 0.996 | 7.0 | 2.9 | 0.94 |
| 4 | ConvNeXt-Tiny Finetuned | 99.60 | 0.996 | 27.8 | 4.5 | 1.17 |
| 5 | EfficientNet-B3 | 99.32 | 0.994 | 10.7 | 1.8 | 0.77 |
| 6 | ResNet-101 | 99.15 | 0.993 | 42.5 | 7.8 | 1.67 |
| 7 | ResNet-50 | 99.14 | 0.991 | 23.5 | 4.1 | 1.11 |
| 8 | ResNeXt-50 (32×4d) | 98.94 | 0.991 | 23.0 | 4.2 | 1.13 |
| 9 | ResNeXt-101 (32×8d) | 98.94 | 0.991 | 86.7 | 16.4 | 2.96 |
| 10 | ViT-S/16 | 98.43 | 0.986 | 21.7 | 4.6 | 1.19 |
| 11 | ConvNeXt-Tiny (base) | 97.32 | 0.974 | 27.8 | 4.5 | 1.17 |

### Per-Class Performance

**Best Accuracy by Organ:**

| Organ | Best Accuracy | Best Model | Avg Across Models |
|-------|---------------|------------|-------------------|
| Bladder | **100%** | Swin-Tiny Finetuned | 99.7% |
| Femur (Left) | **100%** | Multiple models | 100.0% |
| Femur (Right) | **100%** | Multiple models | 100.0% |
| Heart | **100%** | EfficientNet-B3 | 97.4% ⚠️ |
| Kidney (Left) | **100%** | ResNeXt-101 | 99.2% |
| Kidney (Right) | **100%** | EfficientNet-B3 | 99.0% |
| Liver | **100%** | Multiple models | 99.9% |
| Lung (Left) | **100%** | EfficientNet-B3 | 99.1% |
| Lung (Right) | **99.6%** | Swin-Tiny Finetuned | 97.0% ⚠️ |
| Spleen | **100%** | ResNet-101 | 99.3% |
| Pancreas | **100%** | Multiple models | 99.5% |

**Most Challenging Classes:**
- 🔴 **Heart** (Avg: 97.4%) — High contrast, limited samples
- 🔴 **Lung (Right)** (Avg: 97.0%) — Bilateral confusion, texture shift

### Class Specialists

Different architectures excel at different organs:

| Organ Category | Best Models | Why |
|----------------|-------------|-----|
| **Bones** (Femurs) | ResNet-50, ResNet-101 | CNNs excel at high-contrast edges |
| **Soft Organs** (Heart, Kidneys) | EfficientNet-B3, ResNeXt-101 | Compound scaling + large receptive fields |
| **Lungs** | EfficientNet-B3, Swin-Tiny FT | Global attention captures cavity structure |
| **Bladder** | Swin-Tiny Finetuned | Transformer attention on oval shape |
| **Liver** | ResNet family | Standard CNN features sufficient |
| **Spleen/Pancreas** | ResNet-101, ResNet-50 | Deep residual features capture texture |

### Corruption Robustness Rankings

| Rank | Model | Clean Acc | Corrupted Acc | Relative Robustness | Best Category |
|------|-------|-----------|---------------|---------------------|---------------|
| 🥇 1 | **ConvNeXt-Tiny FT** | 99.60% | **76.88%** | 77.19% | Digital (78.7%) |
| 🥈 2 | Swin-Tiny FT | 99.69% | 75.36% | 75.59% | Digital (76.8%) |
| 🥉 3 | EfficientNet-B3 | 99.32% | 74.57% | 75.08% | Noise (74.2%) |
| 4 | Swin-Tiny | 99.63% | 74.55% | 74.83% | Weather (75.6%) |
| 5 | ResNeXt-101 | 98.94% | 74.20% | 74.99% | Digital (75.8%) |
| 6 | ConvNeXt-Tiny | 97.32% | 74.10% | 76.15% | Digital (76.0%) |
| 7 | DenseNet-121 | 99.61% | 73.96% | 74.25% | Digital (75.9%) |
| 8 | ResNet-101 | 99.15% | 73.42% | 74.05% | Digital (75.0%) |
| 9 | ResNeXt-50 | 98.94% | 73.22% | 74.01% | Digital (74.8%) |
| 10 | ResNet-50 | 99.14% | 72.44% | 73.07% | Digital (74.0%) |
| 11 | ViT-S/16 | 98.43% | 70.28% | 71.40% | Digital (71.8%) |

**Corruption Categories Tested:**

1. **Noise** (3 types): Gaussian, Shot, Impulse
2. **Blur** (4 types): Defocus, Glass, Motion, Zoom
3. **Weather** (4 types): Snow, Frost, Fog, Brightness
4. **Digital** (4 types): Contrast, Elastic, Pixelate, JPEG

**Key Findings:**
- 🏆 **ConvNeXt-Tiny Finetuned** dominates 13/15 corruption types
- 💪 **Modern CNNs** (ConvNeXt, EfficientNet) > Pure Transformers for robustness
- 📉 **ViT-S/16** struggles with local corruptions (lacks CNN inductive bias)
- ✅ **Finetuning** significantly improves robustness (+2-3% across corruptions)

---

## 🔬 Analysis Pipeline

### Comprehensive Dataset Analysis

The project includes an extensive analysis pipeline producing 60+ visualizations, 17 JSON reports, and 14 CSV tables.

#### 1. Data Quality Analysis

- **Duplicate Detection**: Perceptual hash-based identification
  - Train: 921 duplicate pairs (~3% duplicate rate)
  - Val: 134 duplicate pairs (~2.3% duplicate rate)
- **Label Quality**: Random Forest baseline flagging suspicious labels (0 suspects found)
- **Missing Files**: All labeled images exist and are readable

#### 2. Dataset Characterization

**Class Distribution:**
- Liver dominates at 17.84% (6,164 samples)
- Femurs/Heart underrepresented at 3.93-4.26%
- Validation split mirrors training distribution

**Pixel Statistics:**
- Mean intensity: ~119.4 (train), ~119.9 (val)
- Full dynamic range used: 0-255
- Spike at 0 (background air), spike at 255 (bone/contrast)

**Distribution Shifts (Train → Test):**
- Pixel KL divergence: 0.143 (20× higher than train→val)
- LBP texture KL: 0.349 (texture shift detected)
- Edge density: Similar but test images slightly smoother

#### 3. Latent Structure Analysis

**t-SNE Visualization:**
- Tight clusters: Femurs (yellow/green) — visually distinctive
- Broad blobs: Kidneys/Pancreas/Spleen — high intra-class variability
- Central overlap: Challenging class boundaries

**Inter-Class Cosine Similarity (Most Confusable Pairs):**
- Kidney L ↔ Kidney R: 0.88
- Kidney L ↔ Pancreas: 0.86
- Kidney R ↔ Pancreas: 0.82
- Spleen ↔ Kidney R: 0.81

**PCA Analysis:**
- PC1 explains 32.82% variance (dominant anatomical axis)
- Top 10 PCs capture 65.75% variance

#### 4. Feature Exploration

**Grad-CAM Visualizations (12 samples):**
- Lungs: Attention on cavity boundaries
- Liver: Focus on parenchyma mass
- Kidneys: Bean-shaped region highlighting

**Multi-Scale Edge Density:**
- All splits: Edge density decreases with scale (expected)
- Test: Consistently lower edge density → smoother images
- Largest difference at finest scale (32×32)

**Occlusion Sensitivity:**
- Lungs: Sensitive regions inside cavity + pleural boundary
- Liver: Hotspots align with parenchyma
- Confirms anatomically-relevant decision regions

#### 5. Robustness Probes

**Perturbation Analysis:**
- Gaussian noise: PSNR 25.1 dB, SSIM 0.44
- Motion blur: PSNR 36.2 dB, SSIM 0.97
- Contrast adjustments: PSNR 20-21 dB

**Adversarial Attacks:**
- Clean accuracy: 95.95%
- FGSM ε=0.03: 72.00% (-23.95%)
- PGD ε=0.07: 2.35% (-93.6%) ⚠️

**Flip Asymmetry:**
- Mean absolute difference: 50-60 (on 0-255 scale)
- Validates flip augmentation creates novel samples

#### 6. Frequency Domain Analysis

**Average Fourier Spectrum:**
- Bright center: Low-frequency dominance (global shapes)
- Cross-shaped streaks: CT acquisition geometry
- Smooth decay: 1/f spectrum typical of natural images

**Test Split Differences:**
- Test: 4× higher high-frequency energy
- Indicates sharper edges or acquisition noise

---

## 🛠️ Technical Details

### Grayscale Adaptation Strategy

All pretrained models (originally trained on RGB ImageNet) are adapted for single-channel grayscale input:

```python
# Strategy: Average pretrained RGB kernels across channel dimension
with torch.no_grad():
    weight = conv.weight  # [out_channels, 3, kernel_h, kernel_w]
    grayscale_weight = weight.mean(dim=1, keepdim=True)  # [out, 1, k, k]
    new_conv.weight.copy_(grayscale_weight)
```

**Why this works:**
- Preserves learned edge/texture detectors
- Smooth initialization (no random weights)
- Empirically validated: pretrained grayscale > random init

### Data Augmentation Pipeline

Three strength levels implemented:

| Strength | Transformations |
|----------|-----------------|
| **Weak** | Random horizontal flip, rotation ±10° |
| **Medium** | + Random resized crop (0.9-1.0 scale), rotation ±15° |
| **Strong** | + Crop (0.8-1.0), rotation ±20°, affine (shear ±5°), color jitter |

**Additional techniques for top models:**
- **MixUp** (α=0.4): Interpolate between pairs of training examples
- **CutMix** (α=1.0): Patch regions from different images
- **Test-Time Augmentation (TTA)**: Average predictions over multiple augmented views

### Training Configuration

**Optimizers:**
- **SGD**: `momentum=0.9, nesterov=True` (ResNet, DenseNet)
- **AdamW**: `betas=(0.9, 0.999), eps=1e-8` (Transformers, ConvNeXt)

**Learning Rate Schedules:**
- **Cosine Annealing**: Smooth decay from `lr_max` → `lr_min=1e-6`
- **Warmup**: Linear ramp over first 5 epochs
- **StepLR**: Decay by 0.1 every 20 epochs (fallback)

**Regularization:**
- **Weight Decay**: 1e-5 to 0.1 (architecture-dependent)
- **Label Smoothing**: 0.0-0.1 (prevents overconfidence)
- **Dropout**: 0.1-0.3 in classification heads
- **Drop Path**: 0.1 (stochastic depth for Transformers)
- **Gradient Clipping**: `max_norm=1.0` (prevents exploding gradients)

**Handling Class Imbalance:**

1. **Weighted Random Sampler**:
   ```python
   weight_per_class = total_samples / (num_classes * samples_per_class)
   sample_weights = [weight_per_class[label] for label in dataset.labels]
   sampler = WeightedRandomSampler(sample_weights, len(dataset))
   ```

2. **Class-Weighted Loss**:
   ```python
   class_weights = torch.tensor([total/count for count in class_counts])
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   ```

### ModelRecipe System

Each model has an associated "recipe" containing optimal hyperparameters:

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

**Example recipes:**
- ResNet-50: `lr=0.05, wd=1e-4, bs=64`
- Swin-Tiny: `lr=5e-4, wd=0.05, bs=64`
- EfficientNet-B3: `lr=0.02, wd=1e-5, bs=32`

### Advanced Training Techniques

**Exponential Moving Average (EMA):**
```python
# Smooth model weights over training
ema_model = copy.deepcopy(model)
for epoch in training:
    # Update EMA after each step
    for param_ema, param in zip(ema_model.parameters(), model.parameters()):
        param_ema.data = decay * param_ema.data + (1 - decay) * param.data
```

**Mixed Precision Training:**
- Uses `torch.cuda.amp` for automatic mixed precision
- Speeds up training by 2-3× with negligible accuracy loss
- Reduces memory usage (enables larger batch sizes)

### Evaluation Metrics

**Per-Epoch Metrics:**
```python
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    per_class_accuracy: Dict[int, float]
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    macro_f1: float
```

**Confusion Matrix Analysis:**
- Row-normalized: Shows prediction distribution per true class
- Identifies systematic errors (e.g., Kidney L → Kidney R confusion)
- Used for ensemble diversity analysis

### HPC Deployment

**IITD Cluster Support:**
- SSH configuration scripts for bastion host
- SLURM job submission templates
- Automatic environment setup
- See `hpc/README_hpc.md` for details

**Resource Requirements:**
- GPU: NVIDIA V100/A100 (32 GB VRAM recommended)
- CPU: 16+ cores for data loading
- RAM: 64 GB minimum
- Storage: 50 GB for dataset + outputs

---

## 📚 Documentation

### Comprehensive Guides

1. **[SOURCE_CODE_GUIDE.md](src/SOURCE_CODE_GUIDE.md)** (1,120 lines)
   - Complete codebase walkthrough
   - Every module explained with usage examples
   - Architecture deep dives with code snippets

2. **[FIGURE_GUIDE.md](analysis_outputs/figures/FIGURE_GUIDE.md)** (302 lines)
   - All 60+ analysis figures explained
   - Key insights and interpretation
   - Action items derived from visualizations

3. **[REPORTS_GUIDE.md](analysis_outputs/reports/REPORTS_GUIDE.md)** (357 lines)
   - 17 JSON reports documented
   - Metric definitions and thresholds
   - How to use reports for decision-making

4. **[TABLES_GUIDE.md](analysis_outputs/tables/TABLES_GUIDE.md)** (273 lines)
   - 14 CSV tables explained
   - Column definitions and usage
   - Quick action checklists

5. **[EVALUATION_GUIDE.md](evaluation_outputs/EVALUATION_GUIDE.md)** (260 lines)
   - Model comparison methodology
   - Robustness evaluation details
   - Ensemble design recommendations

6. **[HOW.md](HOW.md)** (2,026 lines)
   - Step-by-step tutorials
   - Advanced usage patterns
   - Architecture deep dives with mathematical details

7. **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** (567 lines)
   - High-level project overview
   - Key achievements and results
   - Technical implementation summary

8. **[MODEL_ARCHITECTURE_DIAGRAMS_PROMPT.md](MODEL_ARCHITECTURE_DIAGRAMS_PROMPT.md)** (633 lines)
   - Detailed architecture specifications
   - Component-level breakdowns
   - Diagram generation prompts

### Quick Navigation

**Want to understand...**
- **The codebase?** → Start with `src/SOURCE_CODE_GUIDE.md`
- **Analysis results?** → See `analysis_outputs/*/GUIDE.md` files
- **Model performance?** → Read `evaluation_outputs/EVALUATION_GUIDE.md`
- **How to train?** → Follow `HOW.md` tutorials
- **Architecture details?** → Study `MODEL_ARCHITECTURE_DIAGRAMS_PROMPT.md`

---

## 🎓 Key Learnings & Best Practices

### What We Learned

1. **Hierarchical Transformers > Pure ViT** for medical images at this resolution
   - Swin-Tiny (99.69%) >> ViT-S/16 (98.43%)
   - Windowed attention provides better inductive bias

2. **Finetuning with Modern Regularization** provides significant gains
   - ConvNeXt-Tiny base (97.32%) → finetuned (99.60%)
   - MixUp + CutMix + EMA + lower LR = magic formula

3. **Dense Connections are Valuable**
   - DenseNet achieves 99.61% with only 7M params
   - Feature reuse reduces redundancy

4. **Attention Mechanisms Help**
   - SE blocks in Adaptive DenseNet improve feature selection
   - Multi-scale fusion in Swin-MultiScale captures context

5. **Robustness ≠ Accuracy**
   - Most accurate: Swin-Tiny FT (99.69%)
   - Most robust: ConvNeXt-Tiny FT (76.88% corrupted)
   - Need both metrics for deployment decisions

6. **CNN Inductive Bias Matters**
   - Pure ViT struggles with local corruptions (70.28%)
   - ConvNeXt (modernized CNN) excels (76.88%)

### Best Practices Demonstrated

✅ **Data Analysis Before Training**
- Comprehensive quality checks (duplicates, missing files)
- Distribution shift detection (train/val/test)
- Class imbalance quantification

✅ **Systematic Hyperparameter Search**
- Grid search over optimizers, learning rates, augmentation
- Architecture-specific recipes documented
- Reproducible experiment tracking

✅ **Multiple Evaluation Metrics**
- Accuracy, F1, per-class accuracy, confusion matrices
- Robustness testing (15 corruption types)
- Inference time and model size

✅ **Proper Train/Val/Test Splits**
- No data leakage between splits
- Stratified sampling maintains class distribution
- Duplicate detection prevents cross-split contamination

✅ **Visualization of Predictions**
- Grad-CAM attention maps
- Confusion matrices per model
- t-SNE latent space visualization

✅ **Version Control of Large Artifacts**
- Git LFS for model weights
- Structured output directories
- Comprehensive documentation

---

## 🔮 Potential Extensions

### 1. Ensemble Methods
- **Stacking**: Train meta-learner on diverse base models
- **Weighted Averaging**: Use validation performance as weights
- **Boosting**: Sequential training focusing on hard examples

**Recommended ensemble:**
- ConvNeXt-Tiny FT (robust) + Swin-Tiny FT (accurate) + DenseNet-121 (efficient)
- Diversity correlation: 0.08 (low = good for ensemble)

### 2. Self-Supervised Pretraining
- **Contrastive Learning**: SimCLR, MoCo on medical images
- **Masked Autoencoders**: MAE pretraining on unlabeled CT scans
- **Rotation Prediction**: Auxiliary task for better features

### 3. Knowledge Distillation
- **Teacher**: Swin-Tiny Finetuned (99.69%)
- **Student**: MobileNetV3 or EfficientNet-B0
- **Target**: <5M params, >99% accuracy for edge deployment

### 4. Uncertainty Estimation
- **Bayesian Deep Learning**: MC Dropout for epistemic uncertainty
- **Evidential Deep Learning**: Output uncertainty explicitly
- **Ensemble Disagreement**: Measure prediction variance

### 5. Enhanced Explainability
- **Integrated Gradients**: Better attribution than Grad-CAM
- **SHAP**: Game-theoretic feature importance
- **Concept Activation Vectors**: Learn interpretable concepts

### 6. Few-Shot Learning
- **Prototypical Networks**: Learn class prototypes
- **MAML**: Meta-learning for fast adaptation
- **Transfer to New Organs**: Adapt to unseen anatomical structures

### 7. Active Learning
- **Uncertainty Sampling**: Label most uncertain predictions
- **Query by Committee**: Use ensemble disagreement
- **Core-Set Selection**: Maximize training set diversity

### 8. Domain Adaptation
- **Adversarial Training**: DANN for train/test distribution matching
- **Self-Training**: Pseudo-label test set iteratively
- **Test-Time Adaptation**: Update batch norm stats on test data

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Models

1. Implement model in `src/models/your_model.py`
2. Add builder function to `src/model_architectures.py`
3. Add recipe with optimal hyperparameters
4. Train and document results
5. Submit PR with evaluation metrics

### Improving Analysis

1. Add new analysis module in `src/analysis/`
2. Update `run_pipeline.py` to include it
3. Document outputs in appropriate GUIDE.md
4. Include sample visualizations

### Bug Reports

- Use GitHub Issues
- Include Python version, PyTorch version, CUDA version
- Provide minimal reproducible example
- Attach relevant logs

### Feature Requests

- Open GitHub Discussion
- Describe use case and motivation
- Suggest implementation approach if possible

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Shrijak Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

### Datasets & Challenges
- **OrganAMNIST**: Part of the MedMNIST v2 benchmark
- **ImageNet**: Pretrained weights for transfer learning

### Frameworks & Libraries
- **PyTorch**: Deep learning framework
- **timm**: State-of-the-art vision model implementations
- **torchvision**: Computer vision utilities
- **scikit-learn**: Machine learning tools

### Key Papers Referenced

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
2. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
3. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
4. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer" (ICCV 2021)
5. **ConvNeXt**: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)
6. **SE-Net**: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
7. **EfficientNet**: Tan et al., "EfficientNet: Rethinking Model Scaling" (ICML 2019)

### Infrastructure
- **IITD HPC**: High-performance computing cluster for training
- **Git LFS**: Large file storage for model weights

---

## 📞 Contact

**Author**: Shrijak Kumar

**Project Repository**: [GitHub](https://github.com/yourusername/organamnist-dl-project)

**Questions?** Open a GitHub Issue or Discussion

---

## 📊 Project Statistics

- **Total Lines of Code**: ~15,000+ (Python)
- **Documentation**: ~8,000+ lines (Markdown)
- **Analysis Outputs**: 60+ figures, 17 JSON reports, 14 CSV tables
- **Trained Models**: 11 architectures × multiple configurations = 25+ checkpoints
- **Training Time**: ~200 GPU-hours total
- **Dataset Size**: 41,052 labeled images (34,561 train + 6,491 val)

---

## 🗺️ Roadmap

### Phase 1: Foundation (✅ Completed)
- [x] Dataset analysis pipeline
- [x] 13+ model architectures
- [x] Comprehensive evaluation framework
- [x] HPC deployment setup
- [x] Complete documentation

### Phase 2: Enhancement (🚧 In Progress)
- [ ] Ensemble methods implementation
- [ ] Knowledge distillation pipeline
- [ ] Uncertainty quantification
- [ ] Interactive visualization dashboard

### Phase 3: Advanced Features (📋 Planned)
- [ ] Self-supervised pretraining
- [ ] Few-shot learning support
- [ ] Domain adaptation techniques
- [ ] Real-time inference API

### Phase 4: Deployment (🔮 Future)
- [ ] Web application interface
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Cloud inference service
- [ ] Clinical validation study

---

## 🎯 Quick Reference Card

### Training Quick Commands

```bash
# Fastest to start
python -m src.models.resnet50

# Best accuracy
python -m src.models.swin_tiny
python -m src.models.finetune_swin_tiny

# Most efficient
python -m src.models.densenet121

# Most robust
python -m src.models.convnext_tiny
python -m src.models.finetune_convnext_tiny

# Custom models
python -m src.models.densenet121_adaptive
python -m src.models.swin_multiscale
python -m src.training.train_convtransgfusion
```

### Analysis Quick Commands

```bash
# Complete pipeline
python -m src.analysis.run_pipeline

# Specific analyses
python -m src.analysis.label_analysis
python -m src.analysis.latent_structure
python -m src.analysis.feature_exploration
python -m src.analysis.robustness_deepdive
```

### Evaluation Quick Commands

```bash
# All models
python -m src.evaluation.run_evaluation

# Specific evaluation
python -m src.evaluation.run_evaluation --clean
python -m src.evaluation.run_evaluation --robust
```

### Key Directories

```bash
src/                    # Source code
analysis_outputs/       # Analysis results
evaluation_outputs/     # Model evaluation
training_logs/          # Training artifacts
dataset/                # OrganAMNIST data
hpc/                    # HPC configuration
```

---

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{organamnist_dl_2025,
  author = {Shrijak Kumar},
  title = {OrganAMNIST Deep Learning Classification Project},
  year = {2025},
  url = {https://github.com/yourusername/organamnist-dl-project},
  note = {Comprehensive deep learning solution for medical image classification}
}
```

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ for advancing medical AI

</div>

