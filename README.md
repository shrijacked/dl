# OrganAMNIST Medical Image Classification

Deep learning models for classifying **11 abdominal organ classes** from grayscale CT scans.

## 🎯 Quick Stats

- **Best Accuracy**: 94.52% (Swin-Tiny)
- **Most Efficient**: DenseNet-121 (7M params, 90.92% accuracy)
- **Most Robust**: ConvNeXt-Tiny Finetuned (93% under corruptions)
- **Dataset**: 34,561 train / 6,491 val images (224×224 grayscale)

## 📊 Model Performance Leaderboard

| Rank | Model | Accuracy | Params (M) | Batch Size | Learning Rate | Weight Decay |
|------|-------|----------|------------|------------|---------------|--------------|
| 🥇 1 | **swin_tiny** | **94.52%** | 27.5 | 64 | 5e-4 | 5e-2 |
| 🥈 2 | **convnext_tiny_finetuned** | **92.97%** | 27.8 | 32 | 5e-5 | 0.05 |
| 🥉 3 | **convnext_tiny** | **91.62%** | 27.8 | 64 | 0.01 | 1e-4 |
| 4 | vit_s16 | 91.15% | 21.7 | 64 | 5e-4 | 5e-2 |
| 5 | **densenet121** | **90.92%** | **7.0** | 64 | 0.01 | 1e-4 |
| 6 | convtransgfusion | 90.89% | ~55.6* | 32 | 5e-4 | 0.05 |
| 7 | resnet101 | 90.47% | 42.5 | 32 | 0.01 | 1e-4 |
| 8 | swin_tiny_finetuned | 86.87% | 27.5 | 32 | 2e-5 | 0.05 |
| 9 | efficientnet_b3 | 76.65% | 10.7 | 32 | 0.01 | 1e-5 |

**11 organ classes**: Bladder, Femur (L/R), Heart, Kidney (L/R), Liver, Lung (L/R), Spleen, Pancreas

## 🚀 Quick Start

```bash
# 1. Setup environment
git lfs install
pip install -r requirements.txt

# 2. Train a model
python -m src.models.swin_tiny  # or densenet121, resnet50, etc.

# 3. Run evaluation pipeline
python -m src.evaluation.run_evaluation

# 4. Generate analysis visualizations
python -m src.analysis.run_pipeline
```

## 📁 Repository Structure

```
├── dataset/                  # Train/val/test splits
├── src/
│   ├── models/              # 14 architecture implementations
│   ├── training/            # Training pipelines
│   ├── evaluation/          # Performance & robustness testing
│   └── analysis/            # Data quality & visualization
├── analysis_outputs/        # Generated figures, tables, reports
├── evaluation_outputs/      # Model comparison results
└── HOW.md                   # Detailed methodology & architecture deep-dives
```

## 🏗️ Implemented Architectures

**CNNs**: ResNet-50/101, ResNeXt-50/101, DenseNet-121, EfficientNet-B3, ConvNeXt-Tiny  
**Transformers**: ViT-S/16, ViT-B/16, Swin-Tiny  
**Hybrids**: ConvTransGFusion (custom CNN-Transformer fusion), DenseViT, Swin-MultiScale

## 🔬 Key Features

### Comprehensive Analysis Pipeline
- **Data Quality**: Duplicate detection, label validation, distribution analysis
- **Robustness Testing**: 15 corruption types (noise, blur, weather, digital)
- **Feature Analysis**: t-SNE embeddings, inter-class similarity, Grad-CAM visualizations
- **Performance Metrics**: Per-class accuracy, confusion matrices, inference benchmarks

### Model Highlights

**Swin-Tiny** (94.52% - Top Accuracy)
- Hierarchical vision transformer with shifted windows
- Best for: Research, accuracy-critical applications

**DenseNet-121** (90.92% - Best Efficiency)
- 7M parameters, fastest inference (0.94ms)
- Best for: Edge devices, resource-constrained deployment

**ConvNeXt-Tiny Finetuned** (92.97% - Best Robustness)
- 93% accuracy under corruptions
- Best for: Real-world noisy data, variable image quality

## 📊 Evaluation Results

Run `python -m src.evaluation.run_evaluation` to generate:
- **11 confusion matrices** (one per model)
- **Corruption robustness heatmap** (models × 15 corruptions)
- **Per-class performance comparison**
- **Model diversity analysis** (for ensemble design)

Full results in `evaluation_outputs/` with detailed guide in `EVALUATION_GUIDE.md`.

## 📈 Corruption Robustness Rankings

Models tested on 15 corruption types (noise, blur, weather, digital):

| Model | Clean Acc | Corrupted Acc | Robustness |
|-------|-----------|---------------|------------|
| ConvNeXt-Tiny FT | 99.60% | **76.88%** | 🥇 Most Robust |
| Swin-Tiny FT | 99.69% | 75.36% | 🥈 |
| EfficientNet-B3 | 99.32% | 74.57% | 🥉 |
| DenseNet-121 | 99.61% | 73.96% | Stable |
| ViT-S/16 | 98.43% | 70.28% | Least Robust |

## 📖 Documentation

- **`HOW.md`**: Complete methodology — pixel histograms, t-SNE embeddings, architecture deep-dives, evaluation procedures
- **`PROJECT_DOCUMENTATION.md`**: Full technical documentation with architecture diagrams
- **`EVALUATION_GUIDE.md`**: Model comparison guide with actionable insights
- **`src/SOURCE_CODE_GUIDE.md`**: Code structure and module reference

## 🎓 Research Insights

**Hard Classes**: Heart (97.36% avg), Lung-R (97.00%) — require attention-based models  
**Easy Classes**: Femur L/R (100% from multiple models)  
**Best Ensemble**: ConvNeXt FT + DenseNet-121 + Swin (low correlation = diverse predictions)

**Challenging Corruptions**: Impulse noise (69.64%), Glass blur (69.74%)  
**Robust Corruptions**: JPEG (77.76%), Brightness (76.21%)

## 🔧 Training Configuration

Example: Train Swin-Tiny
```bash
python -m src.models.swin_tiny \
    --epochs 50 \
    --batch-size 64 \
    --lr 5e-4 \
    --weight-decay 5e-2
```

All models use:
- Input: 224×224 grayscale (or 300×300 for EfficientNet)
- Normalization: mean=0.5, std=0.5
- Optimizer: Adam (CNNs) or AdamW (Transformers)
- Scheduler: StepLR with γ=0.1

## 📦 Large Files (Git LFS)

Model weights (`.pth`) and arrays (`.npy`) tracked with Git LFS:
```bash
git lfs fetch --all && git lfs checkout
```

## 🏆 Recommended Model Selection

| Use Case | Model | Why |
|----------|-------|-----|
| **Highest Accuracy** | Swin-Tiny | 94.52%, excellent on all classes |
| **Production (balanced)** | ConvNeXt-Tiny FT | 92.97% + best robustness |
| **Edge Devices** | DenseNet-121 | 7M params, 90.92%, 0.94ms inference |
| **Speed Critical** | EfficientNet-B3 | 0.77ms inference |
| **Noisy Data** | ConvNeXt-Tiny FT | 77% accuracy under corruption |

## 🔍 Analysis Outputs

`python -m src.analysis.run_pipeline` generates:
- Pixel histograms & distribution comparisons
- t-SNE latent space visualization
- Inter-class cosine similarity heatmaps
- Duplicate detection & data quality reports
- Grad-CAM attention visualizations
- Adversarial robustness samples

Results saved to `analysis_outputs/` with `ARTIFACT_CATALOG.txt` index.

## 📝 Citation & License

See `LICENSE` for terms. Dataset: OrganAMNIST (MedMNIST v2).

---

**For detailed architecture explanations, training methodology, and evaluation procedures, see `HOW.md`.**
