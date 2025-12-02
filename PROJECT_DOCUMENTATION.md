# Deep Learning Project: Medical Image Classification on OrganAMNIST

## 🎯 Project Overview

This project tackles the **OrganAMNIST Challenge** — a medical image classification task that involves classifying **grayscale abdominal CT scan images** into **11 different organ classes**. The project implements, trains, and extensively evaluates multiple state-of-the-art deep learning architectures, ranging from classical CNNs to modern Vision Transformers, along with custom hybrid architectures designed specifically for this task.

### Key Highlights

- **14 Neural Network Architectures** implemented and evaluated
- **3 Novel Custom Architectures** designed specifically for medical image classification
- **Comprehensive Analysis Pipeline** for data quality, robustness, and model performance
- **Best Validation Accuracy: 99.69%** (Swin-Tiny Finetuned)
- **Extensive Robustness Testing** under various image corruptions

---

## 📊 Dataset: OrganAMNIST

### Dataset Characteristics

| Split | Samples | Description |
|-------|---------|-------------|
| **Training** | 34,561 | Labeled images for training |
| **Validation** | 6,491 | Labeled images for hyperparameter tuning |
| **Test** | Hidden | Unlabeled images for final evaluation |

### The 11 Organ Classes

| Class ID | Organ | Training Samples | Proportion |
|----------|-------|------------------|------------|
| 0 | Bladder | 1,956 | 5.66% |
| 1 | Femur (Left) | 1,390 | 4.02% |
| 2 | Femur (Right) | 1,357 | 3.93% |
| 3 | Heart | 1,474 | 4.26% |
| 4 | Kidney (Left) | 3,963 | 11.47% |
| 5 | Kidney (Right) | 3,817 | 11.04% |
| 6 | Liver | 6,164 | 17.84% |
| 7 | Lung (Left) | 3,919 | 11.34% |
| 8 | Lung (Right) | 3,929 | 11.37% |
| 9 | Spleen | 3,031 | 8.77% |
| 10 | Pancreas | 3,561 | 10.30% |

### Image Specifications
- **Format**: Grayscale PNG images
- **Input Size**: 224×224 pixels (or 300×300 for EfficientNet)
- **Preprocessing**: Normalized with mean=0.5, std=0.5

---

## 🏗️ Model Architectures

### Architecture Categories

```
Classical CNNs                    Hybrid Architectures           Vision Transformers
├── ResNet Family                 └── DenseViT (Custom)          ├── ViT Family
│   ├── ResNet-18 (baseline)                                     │   ├── ViT-S/16
│   ├── ResNet-50                                                │   └── ViT-B/16
│   ├── ResNet-101                                               │
│   └── ResNeXt (grouped convs)                                  └── Swin Transformer
│       ├── ResNeXt-50 (32×4d)                                       ├── Swin-Tiny
│       └── ResNeXt-101 (32×8d)                                      └── Swin-MultiScale (Custom)
│
├── Dense Connections
│   ├── DenseNet-121
│   └── DenseNet-121 Adaptive (Custom)
│
├── Efficient Architectures
│   ├── EfficientNet-B3
│   └── ConvNeXt-Tiny
```

### 1️⃣ ResNet Family (Residual Networks)

**Core Innovation**: Skip connections that enable training of very deep networks by allowing gradients to flow directly through the network.

| Model | Depth | Parameters | Key Feature |
|-------|-------|------------|-------------|
| ResNet-50 | 50 layers | 23.5M | Bottleneck blocks (1×1→3×3→1×1) |
| ResNet-101 | 101 layers | 42.5M | 23 blocks in Stage 3 |
| ResNeXt-50 (32×4d) | 50 layers | 23.0M | 32 parallel grouped convolutions |
| ResNeXt-101 (32×8d) | 101 layers | 86.7M | Deeper + wider grouped convolutions |

### 2️⃣ DenseNet-121 & Adaptive DenseNet (Custom)

**Core Innovation**: Dense connections where each layer receives feature maps from ALL preceding layers.

**Standard DenseNet-121 Architecture**:
- 4 Dense Blocks with 6, 12, 24, 16 layers respectively
- Growth rate of 32 channels per layer
- Transition layers for downsampling

**🌟 Adaptive DenseNet-121 (Custom Architecture)**:

Our custom enhancement adds:
1. **Squeeze-and-Excitation (SE) Attention** in every dense layer for channel-wise recalibration
2. **Adaptive Per-Layer Gating** that dynamically weights feature contributions based on input content
3. **SE-Enhanced Transition Layers** for better information flow between blocks

```
Dense Layer with SE Attention:
Input → BN → ReLU → Conv1×1(128) → BN → ReLU → Conv3×3(32) → SE Block → Output

Adaptive Gating Mechanism:
[All Layer Outputs] → GlobalAvgPool → MLP → Sigmoid → Per-Layer Gates (0-1)
Each layer output is scaled by its corresponding gate value
```

### 3️⃣ EfficientNet-B3

**Core Innovation**: Compound scaling of depth, width, and resolution using Mobile Inverted Bottleneck (MBConv) blocks with Squeeze-and-Excitation.

- **Input Size**: 300×300 (larger than other models)
- **Parameters**: 10.7M (efficient!)
- **Key Feature**: Depthwise separable convolutions + SE attention

### 4️⃣ ConvNeXt-Tiny

**Core Innovation**: Modernized ConvNet design inspired by Vision Transformers — large 7×7 depthwise convolutions, LayerNorm, GELU activations, and Layer Scale.

- Uses inverted bottleneck with expansion ratio of 4
- Patchify stem (4×4 conv with stride 4)
- GELU activation instead of ReLU

### 5️⃣ Vision Transformers (ViT)

**Core Innovation**: Pure attention-based architecture that treats images as sequences of patches.

| Model | Embed Dim | Heads | Depth | MLP Ratio | Parameters |
|-------|-----------|-------|-------|-----------|------------|
| ViT-S/16 | 384 | 6 | 12 | 4 | 21.7M |
| ViT-B/16 | 768 | 12 | 12 | 4 | 86M |

**Patch Processing**:
```
224×224 image → 16×16 patches → 196 patch tokens + [CLS] token → Transformer Encoder → Classification
```

### 6️⃣ Swin Transformer & Multi-Scale Swin (Custom)

**Core Innovation**: Hierarchical vision transformer with shifted windows for efficient local-global attention.

**Standard Swin-Tiny**:
- 4 stages with 2, 2, 6, 2 transformer blocks
- Window size: 7×7
- Alternating regular and shifted window attention

**🌟 Swin-MultiScale (Custom Architecture)**:

Our custom enhancement adds:
1. **Multi-Scale Feature Extraction** from all 4 Swin stages (not just final output)
2. **Attention-Weighted Fusion** with learnable weights for each scale
3. **Auxiliary Heads for Deep Supervision** during training
4. **SE-Style Channel Attention** on fused features

```
Stage Outputs:
├── Stage 1: 56×56, 96 channels  ─────┐
├── Stage 2: 28×28, 192 channels ────┤ → Project to 512-dim → Attention-Weighted Fusion → Classification
├── Stage 3: 14×14, 384 channels ────┤
└── Stage 4: 7×7, 768 channels  ─────┘

Training Loss = main_loss + 0.4 × average(aux_losses)
```

### 7️⃣ DenseViT (Custom Hybrid Architecture)

**Core Innovation**: Combines Vision Transformer with DenseNet-style connections and parallel CNN branches.

**Key Features**:
1. **Dense Connections Between Transformer Blocks**: Each block's output is concatenated (not just added) with previous features
2. **Parallel CNN Branches**: Depthwise separable convolutions run alongside each attention block
3. **Adaptive Pathway Fusion**: Learnable weights combine global (attention) and local (conv) features
4. **Multi-Scale Feature Aggregation**: Growth features from all 12 blocks contribute to final output
5. **Differential Learning Rates**: Dense/fusion modules get higher learning rates (5×) for faster adaptation

```
Each DenseViT Block:
                    ┌── MHSA (12 heads) ──┐
Input → LayerNorm ─┤                      ├─ AdaptiveFusion → (+Input) → LayerNorm → MLP → Output
                    └── ParallelConv     ─┘
                                              ↓
                                        Bottleneck → Growth Features (64 channels)
```

---

## 🔧 Training Infrastructure

### Data Augmentation Pipeline

Three augmentation strength levels are implemented:

| Strength | Transformations |
|----------|-----------------|
| **Weak** | Random horizontal flip, rotation ±10° |
| **Medium** | + Random resized crop (0.9-1.0), rotation ±15° |
| **Strong** | + Affine transforms, crop (0.8-1.0), rotation ±20°, shear 5° |

### Training Configuration

| Hyperparameter | Value Range |
|----------------|-------------|
| **Optimizers** | SGD (momentum=0.9), AdamW |
| **Learning Rate** | 1e-5 to 0.05 (architecture-dependent) |
| **Weight Decay** | 1e-5 to 0.1 |
| **Batch Size** | 32-64 |
| **Epochs** | 25-50 |
| **Schedulers** | Cosine Annealing, StepLR |
| **Label Smoothing** | 0.0 - 0.1 |

### Advanced Training Techniques

For best-performing models (Swin-Tiny Finetuned, ConvNeXt Finetuned):
- **MixUp** (α=0.4): Interpolates between training examples
- **CutMix** (α=1.0): Patches regions from different images
- **Exponential Moving Average (EMA)**: decay=0.9998
- **Drop Path Rate**: 0.1 (stochastic depth)
- **Test-Time Augmentation (TTA)**: Multiple augmented views at inference

### Handling Class Imbalance

- **Weighted Random Sampler**: Oversamples minority classes during training
- **Class-Weighted Loss**: Inverse frequency weights for CrossEntropyLoss

---

## 📈 Model Performance Results

### Validation Accuracy Comparison

| Rank | Model | Val Accuracy | Macro F1 | Parameters | Inference Time |
|------|-------|--------------|----------|------------|----------------|
| 🥇 1 | **Swin-Tiny Finetuned** | **99.69%** | 0.998 | 27.5M | 1.17ms |
| 🥈 2 | Swin-Tiny | 99.63% | 0.996 | 27.5M | 1.17ms |
| 🥉 3 | DenseNet-121 | 99.61% | 0.996 | 7.0M | 0.94ms |
| 4 | ConvNeXt-Tiny Finetuned | 99.60% | 0.996 | 27.8M | 1.17ms |
| 5 | EfficientNet-B3 | 99.32% | 0.994 | 10.7M | 0.77ms |
| 6 | ResNet-101 | 99.15% | 0.993 | 42.5M | 1.67ms |
| 7 | ResNet-50 | 99.14% | 0.991 | 23.5M | 1.11ms |
| 8 | ResNeXt-50 (32×4d) | 98.94% | 0.991 | 23.0M | 1.13ms |
| 9 | ResNeXt-101 (32×8d) | 98.94% | 0.991 | 86.7M | 2.96ms |
| 10 | ViT-S/16 | 98.43% | 0.986 | 21.7M | 1.19ms |
| 11 | ConvNeXt-Tiny | 97.32% | 0.974 | 27.8M | 1.17ms |

### Per-Class Performance

**Most Challenging Classes**:
- **Heart** (Class 3): Lowest accuracy across many models (87.9% - 99.2%)
- **Lung (Right)** (Class 8): Second most challenging (94.8% - 96.5%)

**Easiest Classes**:
- **Femur (Left/Right)**: 100% accuracy across all models
- **Liver**: 100% accuracy for top models

### Class Specialist Models

Different models excel at different organs:

| Organ | Best Model(s) |
|-------|---------------|
| Bladder | Swin-Tiny Finetuned, ResNeXt-50 |
| Heart | EfficientNet-B3, ConvNeXt-Tiny Finetuned |
| Kidney (Left) | ResNeXt-101, Swin-Tiny Finetuned |
| Liver | ResNet-50, ResNet-101 |
| Lung (Left) | EfficientNet-B3, ResNet-101 |
| Spleen | ResNet-101, ResNeXt-50 |
| Pancreas | ResNet-50, EfficientNet-B3 |

---

## 🛡️ Robustness Analysis

### Corruption Robustness Testing

Models were tested under 15 types of image corruptions across 4 categories:

| Category | Corruptions |
|----------|-------------|
| **Noise** | Gaussian noise, Shot noise, Impulse noise |
| **Blur** | Defocus blur, Glass blur, Motion blur, Zoom blur |
| **Weather** | Snow, Frost, Fog, Brightness |
| **Digital** | Contrast, Elastic transform, Pixelate, JPEG compression |

### Robustness Rankings

| Rank | Model | Clean Acc | Corrupted Acc | Relative Robustness |
|------|-------|-----------|---------------|---------------------|
| 🥇 1 | **ConvNeXt-Tiny Finetuned** | 99.6% | **76.88%** | 77.2% |
| 🥈 2 | Swin-Tiny Finetuned | 99.69% | 75.36% | 75.6% |
| 🥉 3 | EfficientNet-B3 | 99.32% | 74.57% | 75.1% |
| 4 | Swin-Tiny | 99.63% | 74.55% | 74.8% |
| 5 | ResNeXt-101 (32×8d) | 98.94% | 74.20% | 75.0% |
| ... | ... | ... | ... | ... |
| 11 | ViT-S/16 | 98.43% | 70.28% | 71.4% |

### Key Robustness Findings

1. **ConvNeXt-Tiny Finetuned** is the most robust model overall, winning 13/15 corruption categories
2. **Hierarchical architectures** (Swin, ConvNeXt) are more robust than pure ViT
3. **Finetuning with MixUp/CutMix** significantly improves robustness
4. **Vision Transformers without pretraining** (ViT-S/16) are the least robust

---

## 📊 Comprehensive Analysis Pipeline

The project includes an extensive analysis pipeline (`src/analysis/`) that produces:

### Data Quality Analysis
- **Duplicate Detection**: Perceptual hash-based identification of duplicate images
- **Label Quality Check**: Random Forest baseline to flag suspicious labels
- **Missing File Detection**: Ensures all labeled images exist

### Dataset Characterization
- **Label Distribution Analysis**: Class imbalance visualization
- **Pixel Statistics**: Per-image and aggregate intensity histograms
- **Edge Density Analysis**: Multi-scale edge detection statistics
- **Texture Analysis**: Local Binary Pattern (LBP) histograms

### Latent Structure Analysis
- **t-SNE Visualization**: 2D embedding of image features colored by class
- **PCA Analysis**: Explained variance ratios
- **Inter-Class Similarity**: Cosine similarity between class centroids

### Feature Exploration
- **Grad-CAM Visualizations**: Attention heatmaps showing what models focus on
- **Multi-Scale Statistics**: Intensity and edge metrics at different resolutions

### Robustness Probes
- **Perturbation Analysis**: PSNR/SSIM metrics for noise, blur, contrast changes
- **Adversarial Attacks**: FGSM and PGD attack resistance
- **Occlusion Sensitivity**: Maps showing which regions affect predictions
- **Frequency Domain Analysis**: Low/high frequency energy comparisons

---

## 📁 Project Structure

```
dl-project/
├── dataset/
│   ├── train/
│   │   ├── images_train/         # 34,561 training images
│   │   └── labels_train.csv      # Training labels
│   ├── val/
│   │   ├── images_val/           # 6,491 validation images
│   │   └── labels_val.csv        # Validation labels
│   └── test/
│       ├── images/               # Test images (unlabeled)
│       └── manifest_public.csv   # Test manifest
│
├── src/
│   ├── data_pipeline/
│   │   ├── organamnist_dataset.py    # Custom Dataset class
│   │   └── dataloaders.py            # DataLoader builders with augmentation
│   │
│   ├── models/
│   │   ├── resnet50.py               # ResNet-50 implementation
│   │   ├── resnet101.py              # ResNet-101 implementation
│   │   ├── resnext50_32x4d.py        # ResNeXt-50 implementation
│   │   ├── resnext101_32x8d.py       # ResNeXt-101 implementation
│   │   ├── densenet121.py            # DenseNet-121 implementation
│   │   ├── densenet121_adaptive.py   # 🌟 Custom Adaptive DenseNet
│   │   ├── efficientnet_b3.py        # EfficientNet-B3 implementation
│   │   ├── convnext_tiny.py          # ConvNeXt-Tiny implementation
│   │   ├── vit_s16.py                # Vision Transformer Small
│   │   ├── vit_b16.py                # Vision Transformer Base
│   │   ├── swin_tiny.py              # Swin Transformer Tiny
│   │   ├── swin_multiscale.py        # 🌟 Custom Multi-Scale Swin
│   │   ├── dense_vit.py              # 🌟 Custom DenseViT Hybrid
│   │   └── train_utils.py            # Shared training utilities
│   │
│   ├── training/
│   │   ├── engine.py                 # Training loop implementation
│   │   ├── run_baselines.py          # Baseline model training script
│   │   ├── run_experiments.py        # Experiment runner
│   │   └── evaluate_best_on_test.py  # Test set evaluation
│   │
│   ├── analysis/
│   │   ├── run_pipeline.py           # Main analysis orchestrator
│   │   ├── class_imbalance.py        # Class balance analysis
│   │   ├── data_quality.py           # Duplicate/quality detection
│   │   ├── feature_exploration.py    # Grad-CAM and feature analysis
│   │   ├── robustness.py             # Perturbation robustness
│   │   ├── robustness_deepdive.py    # Adversarial attacks
│   │   ├── latent_structure.py       # t-SNE visualization
│   │   └── test_characterization.py  # Distribution shift analysis
│   │
│   ├── utils/
│   │   ├── metrics.py                # Accuracy, per-class metrics
│   │   └── checkpointing.py          # Model save/load utilities
│   │
│   └── model_architectures.py        # Central model builder registry
│
├── analysis_outputs/
│   ├── figures/                      # Visualization outputs
│   ├── reports/                      # JSON metric reports
│   ├── tables/                       # CSV exports
│   └── models/                       # Trained model weights
│
├── evaluation_outputs/
│   ├── confusion_matrices/           # Per-model confusion matrices
│   ├── figures/                      # Comparison visualizations
│   ├── reports/                      # Robustness rankings
│   └── tables/                       # Model comparison tables
│
├── training_logs/                    # Training run artifacts
├── hpc/                              # HPC cluster configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview
```

---

## 🔬 Technical Implementation Details

### Grayscale Adaptation

All pretrained models (originally for RGB ImageNet) are adapted for single-channel grayscale input:

```python
# Strategy: Average pretrained RGB kernels across channel dimension
with torch.no_grad():
    weight = conv.weight  # [out, 3, k, k]
    grayscale_weight = weight.mean(dim=1, keepdim=True)  # [out, 1, k, k]
    new_conv.weight.copy_(grayscale_weight)
```

### Model Recipe System

Each model has an associated "recipe" containing optimal hyperparameters:

```python
@dataclass(frozen=True)
class ModelRecipe:
    name: str
    input_size: Tuple[int, int]  # H, W
    default_lr: float
    default_weight_decay: float
    default_batch_size: int
    classifier_dropout: float | None = None
```

### Metrics Computation

Per-class and aggregate metrics are computed at each epoch:

```python
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    per_class_accuracy: Dict[int, float]
```

---

## 🚀 How to Run

### 1. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd dl-project

# Install Git LFS (for model weights)
git lfs install
git lfs pull

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Training a Model

```bash
# Train a specific model
python -m src.models.resnet50

# Train with custom hyperparameters
python -m src.models.densenet121_adaptive --epochs 50 --lr 0.01 --batch-size 32

# Run baseline experiments (8 configurations)
python -m src.training.run_baselines
```

### 3. Running Analysis Pipeline

```bash
# Run complete analysis
python -m src.analysis.run_pipeline

# Individual analyses
python -m src.analysis.label_analysis
python -m src.analysis.robustness
python -m src.analysis.latent_structure
```

### 4. Generating Test Predictions

```bash
# Evaluate best model on test set
python -m src.training.evaluate_best_on_test

# Generate submission file
python -m src.models.predict --model swin_tiny_finetuned
```

---

## 📚 Key Takeaways

### What We Learned

1. **Hierarchical transformers (Swin) outperform pure ViT** for medical images at this resolution
2. **Finetuning with modern regularization** (MixUp, CutMix, EMA) provides significant gains
3. **Dense connections are valuable** — both in CNNs (DenseNet) and Transformers (our DenseViT)
4. **Attention mechanisms help** — SE blocks and multi-scale fusion improve performance
5. **Robustness and accuracy don't always correlate** — the most accurate model isn't always the most robust

### Best Practices Demonstrated

- ✅ Systematic hyperparameter search across architectures
- ✅ Comprehensive data quality analysis before training
- ✅ Multiple evaluation metrics (accuracy, F1, per-class, robustness)
- ✅ Proper train/val/test splits with no data leakage
- ✅ Visualization of model predictions (Grad-CAM, confusion matrices)
- ✅ Version control of large artifacts (Git LFS)

### Potential Extensions

1. **Ensemble methods**: Combine predictions from top-performing models
2. **Self-supervised pretraining**: Contrastive learning on medical images
3. **Knowledge distillation**: Train smaller models from large ones
4. **Uncertainty estimation**: Bayesian or Monte Carlo dropout approaches
5. **Explainability**: More detailed attention analysis and feature attribution

---

## 👥 Authors

**Shrijak Kumar** (MIT License, 2025)

---

## 📖 References

1. He, K., et al. "Deep Residual Learning for Image Recognition" (ResNet)
2. Huang, G., et al. "Densely Connected Convolutional Networks" (DenseNet)
3. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words" (ViT)
4. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer" (Swin)
5. Liu, Z., et al. "A ConvNet for the 2020s" (ConvNeXt)
6. Hu, J., et al. "Squeeze-and-Excitation Networks" (SE-Net)
7. Tan, M., et al. "EfficientNet: Rethinking Model Scaling" (EfficientNet)

---

*This documentation was generated based on comprehensive analysis of the project codebase, including all model implementations, training scripts, analysis pipelines, and experimental results.*

