# 🏆 Model Evaluation Guide

This guide explains all evaluation outputs from your model comparison experiments. It covers model rankings, robustness testing, per-class performance, and more.

---

## Table of Contents

### Reports (JSON)
1. [Robustness Ranking](#1-robustness-ranking)
2. [Class Specialists](#2-class-specialists)

### Tables (CSV)
3. [Model Comparison Table](#3-model-comparison-table)
4. [Per-Class Performance](#4-per-class-performance)
5. [Corruption Robustness (All Models)](#5-corruption-robustness-all-models)
6. [Model Diversity Correlation](#6-model-diversity-correlation)

### Figures (PNG)
7. [Visualization Guide](#7-visualization-guide)

### Confusion Matrices
8. [Per-Model Confusion Matrices](#8-per-model-confusion-matrices)

---

## 1. Robustness Ranking

**File:** `reports/robustness_ranking.json`

### What is it?
Comprehensive robustness evaluation of all 11 models tested against 15 types of image corruptions.

---

### Overall Model Rankings

| Rank | Model | Clean Acc | Corruption Acc | Relative Robustness |
|------|-------|-----------|----------------|---------------------|
| 🥇 1 | **convnext_tiny_finetuned** | 99.60% | **76.88%** | 77.19% |
| 🥈 2 | swin_tiny_finetuned | 99.69% | 75.36% | 75.59% |
| 🥉 3 | efficientnet_b3 | 99.32% | 74.57% | 75.08% |
| 4 | swin_tiny | 99.63% | 74.55% | 74.83% |
| 5 | resnext101_32x8d | 98.94% | 74.20% | 74.99% |
| 6 | convnext_tiny | 97.32% | 74.10% | 76.15% |
| 7 | densenet121 | 99.61% | 73.96% | 74.25% |
| 8 | resnet101 | 99.15% | 73.42% | 74.05% |
| 9 | resnext50_32x4d | 98.94% | 73.22% | 74.01% |
| 10 | resnet50 | 99.14% | 72.44% | 73.07% |
| 11 | vit_s16 | 98.43% | **70.28%** | 71.40% |

### Key Metrics Explained:
- **Clean Accuracy:** Performance on unperturbed validation images
- **Mean Corruption Accuracy:** Average accuracy across all 15 corruption types
- **Relative Robustness:** Corruption accuracy / Clean accuracy (how well it maintains performance)

---

### Category Rankings

#### 🔊 Noise Corruptions
*Gaussian noise, shot noise, impulse noise*

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | convnext_tiny_finetuned | 74.92% |
| 2 | efficientnet_b3 | 74.17% |
| 3 | densenet121 | 73.20% |
| 4 | swin_tiny_finetuned | 72.63% |
| 5 | swin_tiny | 72.26% |

#### 🌫️ Blur Corruptions
*Defocus blur, glass blur, motion blur, zoom blur*

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | convnext_tiny_finetuned | 76.00% |
| 2 | swin_tiny_finetuned | 74.67% |
| 3 | resnext101_32x8d | 74.49% |
| 4 | resnet101 | 74.12% |
| 5 | convnext_tiny | 74.02% |

#### 🌧️ Weather Corruptions
*Snow, frost, fog, brightness*

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | convnext_tiny_finetuned | 77.46% |
| 2 | swin_tiny_finetuned | 76.63% |
| 3 | swin_tiny | 75.60% |
| 4 | resnext101_32x8d | 74.29% |
| 5 | efficientnet_b3 | 74.01% |

#### 💻 Digital Corruptions
*Contrast, elastic transform, pixelate, JPEG compression*

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | convnext_tiny_finetuned | 78.66% |
| 2 | swin_tiny_finetuned | 76.83% |
| 3 | efficientnet_b3 | 76.72% |
| 4 | swin_tiny | 76.12% |
| 5 | convnext_tiny | 75.96% |

---

### Corruption Specialists

Which model is best for each specific corruption?

| Corruption | Best Model | Accuracy |
|------------|------------|----------|
| gaussian_noise | convnext_tiny_finetuned | 77.52% |
| shot_noise | efficientnet_b3 | 74.26% |
| impulse_noise | convnext_tiny_finetuned | 73.20% |
| defocus_blur | convnext_tiny_finetuned | 78.41% |
| glass_blur | swin_tiny_finetuned | 71.82% |
| motion_blur | convnext_tiny_finetuned | 77.36% |
| zoom_blur | convnext_tiny_finetuned | 76.64% |
| snow | convnext_tiny_finetuned | 75.60% |
| frost | convnext_tiny_finetuned | 75.92% |
| fog | convnext_tiny_finetuned | 77.82% |
| **brightness** | convnext_tiny_finetuned | **80.52%** |
| contrast | convnext_tiny_finetuned | 79.29% |
| elastic_transform | convnext_tiny_finetuned | 75.24% |
| pixelate | convnext_tiny_finetuned | 79.08% |
| **jpeg_compression** | convnext_tiny_finetuned | **81.03%** |

### Key Takeaways:
- 🏆 **ConvNeXt-Tiny (finetuned) dominates** - Best on 13 of 15 corruptions
- EfficientNet-B3 excels at shot noise
- Swin-Tiny (finetuned) excels at glass blur
- **Brightness & JPEG compression** are easiest corruptions (>80% accuracy)
- **Glass blur & impulse noise** are hardest (<73% accuracy)

---

### Summary Statistics

| Metric | Value |
|--------|-------|
| Most Robust Model | convnext_tiny_finetuned |
| Most Robust Accuracy | 76.88% |
| Least Robust Model | vit_s16 |
| Least Robust Accuracy | 70.28% |
| Mean Across Models | 73.91% |
| Std Across Models | 1.59% |

---

## 2. Class Specialists

**File:** `reports/class_specialists.json`

### What is it?
Identifies which models perform best for each anatomical class (organ).

### Best Models by Class (Top 3):

| Class | #1 Model | #2 Model | #3 Model |
|-------|----------|----------|----------|
| **Bladder** | swin_tiny_finetuned | resnext50_32x4d | resnet50 |
| **Femur (L)** | resnet50 | resnet101 | resnext50_32x4d |
| **Femur (R)** | resnet50 | resnet101 | resnext50_32x4d |
| **Heart** | efficientnet_b3 | convnext_tiny_finetuned | resnet101 |
| **Kidney (L)** | resnext101_32x8d | swin_tiny_finetuned | convnext_tiny_finetuned |
| **Kidney (R)** | efficientnet_b3 | swin_tiny | resnet101 |
| **Liver** | resnet50 | resnet101 | resnext50_32x4d |
| **Lung (L)** | efficientnet_b3 | resnet101 | densenet121 |
| **Lung (R)** | swin_tiny_finetuned | densenet121 | swin_tiny |
| **Spleen** | resnet101 | resnext50_32x4d | densenet121 |
| **Pancreas** | resnet50 | efficientnet_b3 | swin_tiny_finetuned |

### Model Appearances as Specialist:

| Model | Times in Top 3 | Primary Strength |
|-------|----------------|------------------|
| resnet50 | 6 | Femurs, Liver, Pancreas |
| resnet101 | 6 | Femurs, Spleen, Lungs |
| resnext50_32x4d | 5 | Femurs, Liver, Bladder |
| efficientnet_b3 | 4 | Heart, Kidneys, Lungs |
| swin_tiny_finetuned | 4 | Bladder, Kidneys, Lungs |
| densenet121 | 3 | Lungs, Spleen |
| swin_tiny | 2 | Kidney (R), Lung (R) |
| convnext_tiny_finetuned | 2 | Heart, Kidney (L) |
| resnext101_32x8d | 1 | Kidney (L) |

### Key Takeaways:
- **ResNet family excels at bones** (Femurs) and dense organs (Liver)
- **EfficientNet-B3 excels at soft organs** (Heart, Kidneys, Lungs)
- **Swin-Tiny (finetuned) excels at Bladder and Lung (R)**
- Consider **ensemble strategies** combining specialists

---

## 3. Model Comparison Table

**File:** `tables/model_comparison_table.csv`

### What is it?
Comprehensive comparison of all 11 models on key metrics.

### Full Comparison:

| Model | Val Acc | Macro F1 | Params (M) | FLOPs (G) | Inference (ms) | Worst Class | Worst Acc | Best Class | Best Acc |
|-------|---------|----------|------------|-----------|----------------|-------------|-----------|------------|----------|
| swin_tiny_finetuned | **99.69%** | 0.998 | 27.5 | 4.5 | 1.17 | Heart | 99.23% | Bladder | 100% |
| swin_tiny | 99.63% | 0.996 | 27.5 | 4.5 | 1.17 | Heart | 98.98% | Femur (L) | 100% |
| densenet121 | 99.61% | 0.996 | **7.0** | **2.9** | **0.94** | Heart | 98.72% | Femur (L) | 100% |
| convnext_tiny_finetuned | 99.60% | 0.996 | 27.8 | 4.5 | 1.17 | Spleen | 98.30% | Femur (L) | 100% |
| efficientnet_b3 | 99.32% | 0.994 | 10.7 | 1.8 | **0.77** | Lung (R) | 96.53% | Femur (L) | 100% |
| resnet101 | 99.15% | 0.993 | 42.5 | 7.8 | 1.67 | Lung (R) | 96.13% | Femur (L) | 100% |
| resnet50 | 99.14% | 0.991 | 23.5 | 4.1 | 1.11 | Heart | 97.19% | Femur (L) | 100% |
| resnext50_32x4d | 98.94% | 0.991 | 23.0 | 4.2 | 1.13 | Lung (R) | 96.23% | Femur (L) | 100% |
| resnext101_32x8d | 98.94% | 0.991 | **86.7** | **16.4** | **2.96** | Lung (R) | 95.04% | Femur (L) | 100% |
| vit_s16 | 98.43% | 0.986 | 21.7 | 4.6 | 1.19 | Lung (R) | 94.85% | Liver | 100% |
| convnext_tiny | 97.32% | 0.974 | 27.8 | 4.5 | 1.17 | Heart | 89.80% | Femur (L) | 100% |

### Efficiency Analysis:

#### Best Accuracy/Parameter Ratio:
1. **DenseNet121** - 99.61% with only 7M params
2. **EfficientNet-B3** - 99.32% with 10.7M params
3. **Swin-Tiny-Finetuned** - 99.69% with 27.5M params

#### Best Accuracy/Inference Ratio:
1. **EfficientNet-B3** - 99.32% @ 0.77ms
2. **DenseNet121** - 99.61% @ 0.94ms
3. **ResNet50** - 99.14% @ 1.11ms

#### Largest Model:
- **ResNeXt101-32x8d** - 86.7M params, 16.4 GFLOPs, 2.96ms

### Key Takeaways:
- 🏆 **Swin-Tiny (finetuned) has highest accuracy** (99.69%)
- ⚡ **EfficientNet-B3 is fastest** (0.77ms inference)
- 💾 **DenseNet121 is most compact** (7M params with 99.61% acc)
- ❌ **Heart and Lung (R) are hardest classes** for most models

---

## 4. Per-Class Performance

**File:** `tables/per_class_performance.csv`

### What is it?
Detailed accuracy and F1 score for each class across all 11 models.

### Class-wise Best Models:

| Class ID | Class Name | Best Model | Best Accuracy |
|----------|------------|------------|---------------|
| 0 | Bladder | swin_tiny_finetuned | **100%** |
| 1 | Femur (L) | resnet50 (+ 7 others) | **100%** |
| 2 | Femur (R) | resnet50 (+ 6 others) | **100%** |
| 3 | Heart | efficientnet_b3 | **100%** |
| 4 | Kidney (L) | resnext101_32x8d | **100%** |
| 5 | Kidney (R) | efficientnet_b3 | **100%** |
| 6 | Liver | resnet50 (+ 8 others) | **100%** |
| 7 | Lung (L) | efficientnet_b3 | **100%** |
| 8 | Lung (R) | swin_tiny_finetuned | **99.60%** |
| 9 | Spleen | resnet101 | **100%** |
| 10 | Pancreas | resnet50 (+ 2 others) | **100%** |

### Class Difficulty Analysis:

| Class | Avg Accuracy | Hardest Model | Hardest Acc |
|-------|--------------|---------------|-------------|
| Femur (L) | 99.96% | vit_s16 | 99.57% |
| Femur (R) | 99.80% | vit_s16 | 98.67% |
| Liver | 100% | All | 100% |
| Pancreas | 99.45% | convnext_tiny | 97.06% |
| Bladder | 99.33% | vit_s16 | 98.44% |
| Kidney (L) | 99.12% | convnext_tiny | 98.06% |
| Kidney (R) | 99.14% | convnext_tiny | 97.80% |
| Spleen | 99.35% | convnext_tiny_finetuned | 98.30% |
| Lung (L) | 99.49% | convnext_tiny | 97.77% |
| **Lung (R)** | **97.00%** | convnext_tiny | 93.66% |
| **Heart** | **97.36%** | convnext_tiny | 89.80% |

### Key Takeaways:
- ✅ **Liver achieves 100% accuracy** across ALL models
- ✅ **Femurs are easy** - All models score >98.5%
- ⚠️ **Heart is hardest** - Average 97.36%, ConvNeXt-Tiny only 89.80%
- ⚠️ **Lung (R) is second hardest** - Average 97.00%
- ConvNeXt-Tiny (non-finetuned) struggles most across classes

---

## 5. Corruption Robustness (All Models)

**File:** `tables/corruption_robustness_all_models.csv`

### What is it?
Complete breakdown of each model's accuracy on all 15 corruption types.

### Columns Explained:
| Column | Description |
|--------|-------------|
| `model` | Model name |
| `clean_accuracy` | Accuracy on original images |
| `mean_corruption_accuracy` | Average across all corruptions |
| `relative_robustness` | mean_corruption / clean |
| `noise_accuracy` | Average of noise corruptions |
| `blur_accuracy` | Average of blur corruptions |
| `weather_accuracy` | Average of weather corruptions |
| `digital_accuracy` | Average of digital corruptions |
| *Individual corruptions* | Accuracy on each specific corruption |

### Corruption Difficulty Ranking (Averaged Across Models):

| Rank | Corruption | Avg Accuracy | Category |
|------|------------|--------------|----------|
| 1 (Easiest) | jpeg_compression | 77.76% | Digital |
| 2 | brightness | 76.21% | Weather |
| 3 | contrast | 76.14% | Digital |
| 4 | defocus_blur | 75.82% | Blur |
| 5 | pixelate | 75.53% | Digital |
| 6 | fog | 74.76% | Weather |
| 7 | motion_blur | 74.69% | Blur |
| 8 | elastic_transform | 72.63% | Digital |
| 9 | zoom_blur | 74.04% | Blur |
| 10 | frost | 73.21% | Weather |
| 11 | snow | 72.05% | Weather |
| 12 | gaussian_noise | 73.00% | Noise |
| 13 | shot_noise | 71.75% | Noise |
| 14 | glass_blur | 69.74% | Blur |
| 15 (Hardest) | impulse_noise | 69.64% | Noise |

### Model Stability (Std Dev Across Corruptions):

| Model | Mean Acc | Std Dev | Interpretation |
|-------|----------|---------|----------------|
| convnext_tiny_finetuned | 76.88% | 2.15% | Most stable |
| swin_tiny_finetuned | 75.36% | 2.31% | Very stable |
| efficientnet_b3 | 74.57% | 2.18% | Stable |
| vit_s16 | 70.28% | 2.29% | Least accurate |

### Key Takeaways:
- **JPEG compression is easiest** - Models handle it best
- **Impulse noise & glass blur are hardest** - <70% average accuracy
- Weather corruptions are generally easier than noise
- Finetuned models are more stable across corruptions

---

## 6. Model Diversity Correlation

**File:** `tables/model_diversity_correlation.csv`

### What is it?
Correlation matrix showing how similar/different model predictions are. Useful for **ensemble selection**.

### How to Read:
- **1.0** = Identical predictions (diagonal)
- **High positive (>0.7)** = Models agree often
- **Low/Negative (<0.3)** = Models disagree (good for ensemble diversity)

### Full Correlation Matrix:

|  | resnet50 | resnet101 | resnext50 | resnext101 | dense121 | eff_b3 | vit_s16 | swin | swin_ft | convnext | convnext_ft |
|--|----------|-----------|-----------|------------|----------|--------|---------|------|---------|----------|-------------|
| resnet50 | 1.00 | 0.61 | 0.69 | 0.48 | 0.80 | 0.45 | 0.69 | 0.70 | 0.70 | 0.84 | **0.08** |
| resnet101 | 0.61 | 1.00 | 0.91 | 0.89 | 0.31 | 0.94 | 0.78 | 0.45 | 0.23 | 0.47 | 0.24 |
| resnext50 | 0.69 | 0.91 | 1.00 | 0.92 | 0.32 | 0.85 | 0.80 | 0.38 | 0.53 | 0.64 | 0.28 |
| resnext101 | 0.48 | 0.89 | 0.92 | 1.00 | 0.19 | 0.94 | 0.83 | 0.33 | 0.42 | 0.49 | 0.42 |
| densenet121 | 0.80 | 0.31 | 0.32 | 0.19 | 1.00 | 0.20 | 0.50 | 0.65 | 0.50 | 0.67 | 0.16 |
| efficientnet_b3 | 0.45 | 0.94 | 0.85 | 0.94 | 0.20 | 1.00 | 0.75 | 0.39 | 0.22 | 0.32 | 0.48 |
| vit_s16 | 0.69 | 0.78 | 0.80 | 0.83 | 0.50 | 0.75 | 1.00 | 0.66 | 0.62 | 0.79 | 0.22 |
| swin_tiny | 0.70 | 0.45 | 0.38 | 0.33 | 0.65 | 0.39 | 0.66 | 1.00 | 0.39 | 0.54 | **0.08** |
| swin_tiny_ft | 0.70 | 0.23 | 0.53 | 0.42 | 0.50 | 0.22 | 0.62 | 0.39 | 1.00 | 0.83 | 0.27 |
| convnext_tiny | 0.84 | 0.47 | 0.64 | 0.49 | 0.67 | 0.32 | 0.79 | 0.54 | 0.83 | 1.00 | **-0.07** |
| **convnext_ft** | **0.08** | 0.24 | 0.28 | 0.42 | 0.16 | 0.48 | 0.22 | **0.08** | 0.27 | **-0.07** | 1.00 |

### Most Similar Model Pairs:
| Pair | Correlation | Implication |
|------|-------------|-------------|
| efficientnet_b3 ↔ resnext101 | **0.94** | Very similar predictions |
| efficientnet_b3 ↔ resnet101 | **0.94** | Very similar predictions |
| resnext50 ↔ resnext101 | **0.92** | Same architecture family |
| resnet101 ↔ resnext50 | **0.91** | ResNet variants cluster |

### Most Diverse Model Pairs (Best for Ensemble):
| Pair | Correlation | Implication |
|------|-------------|-------------|
| convnext_tiny ↔ convnext_ft | **-0.07** | Opposite predictions! |
| convnext_ft ↔ resnet50 | **0.08** | Very different |
| convnext_ft ↔ swin_tiny | **0.08** | Very different |
| convnext_ft ↔ densenet121 | **0.16** | Very different |

### Recommended Ensemble Combinations:

1. **Diverse Trio (Best):**
   - convnext_tiny_finetuned + densenet121 + swin_tiny
   - Low correlation (0.16, 0.08, 0.65)

2. **Accuracy-Focused:**
   - swin_tiny_finetuned + efficientnet_b3 + convnext_tiny_finetuned
   - Top performers with moderate diversity

3. **Speed-Focused:**
   - efficientnet_b3 + densenet121 + resnet50
   - Fast inference with diversity

### Key Takeaways:
- 🎯 **ConvNeXt-Tiny (finetuned) is most unique** - Negative/low correlation with all others
- ResNet family members correlate highly (0.61-0.94)
- Finetuning significantly changes prediction patterns
- Use diverse models for ensembles, not similar ones

---

## 7. Visualization Guide

**Directory:** `figures/`

### Available Figures:

| Figure | Description |
|--------|-------------|
| `corruption_heatmap.png` | Heatmap showing each model's accuracy on each corruption type |
| `inference_time_comparison.png` | Bar chart comparing model inference times |
| `model_diversity_heatmap.png` | Visual representation of the correlation matrix |
| `per_class_model_performance.png` | How each model performs on each anatomical class |
| `robustness_ranking.png` | Visual ranking of models by robustness |

### What to Look For:

#### corruption_heatmap.png
- **Bright colors** = High accuracy
- **Dark colors** = Low accuracy
- Look for **vertical patterns** (corruption difficulty)
- Look for **horizontal patterns** (model strengths)

#### inference_time_comparison.png
- Compare model speed vs accuracy trade-offs
- Identify fastest models for deployment

#### model_diversity_heatmap.png
- **Blue** = High correlation (similar)
- **Red** = Low/negative correlation (diverse)
- Identify ensemble candidates

#### per_class_model_performance.png
- Spot which models excel at specific organs
- Identify problematic classes needing attention

---

## 8. Per-Model Confusion Matrices

**Directory:** `confusion_matrices/`

### Available Files:
```
confusion_matrix_convnext_tiny_finetuned.png
confusion_matrix_convnext_tiny.png
confusion_matrix_densenet121.png
confusion_matrix_efficientnet_b3.png
confusion_matrix_resnet101.png
confusion_matrix_resnet50.png
confusion_matrix_resnext101_32x8d.png
confusion_matrix_resnext50_32x4d.png
confusion_matrix_swin_tiny_finetuned.png
confusion_matrix_swin_tiny.png
confusion_matrix_vit_s16.png
```

### How to Read Confusion Matrices:
- **Rows** = Actual/True labels
- **Columns** = Predicted labels
- **Diagonal** = Correct predictions (should be dark/high)
- **Off-diagonal** = Misclassifications (should be light/low)

### What to Look For:
1. **Strong diagonal** = Good model performance
2. **Off-diagonal clusters** = Systematic confusion between classes
3. **Empty rows** = Class never predicted correctly
4. **Empty columns** = Class never predicted

### Common Patterns Found:
- **Heart ↔ Other classes** - Most common confusion
- **Lung (L) ↔ Lung (R)** - Bilateral organ confusion
- **Kidney (L) ↔ Kidney (R)** - Bilateral organ confusion

---

## 📈 Summary: Key Findings

### 🏆 Model Rankings

| Category | Best Model | Metric |
|----------|------------|--------|
| **Overall Accuracy** | swin_tiny_finetuned | 99.69% |
| **Robustness** | convnext_tiny_finetuned | 76.88% corruption acc |
| **Efficiency (Params)** | densenet121 | 7M params |
| **Efficiency (Speed)** | efficientnet_b3 | 0.77ms |
| **Ensemble Diversity** | convnext_tiny_finetuned | -0.07 to 0.48 correlations |

### ⚠️ Problem Areas

| Issue | Details | Recommendation |
|-------|---------|----------------|
| Heart classification | 89-100% range, most variable | More Heart samples, data augmentation |
| Lung (R) classification | 93-99.6% range | Balance with Lung (L) samples |
| Impulse noise | 65.6-73.2% accuracy | Train with noise augmentation |
| Glass blur | 66-72% accuracy | Include blur augmentation |

### 🔧 Recommendations

1. **For Production (Accuracy):** Use `swin_tiny_finetuned`
2. **For Production (Speed):** Use `efficientnet_b3`
3. **For Robustness:** Use `convnext_tiny_finetuned`
4. **For Ensemble:** Combine `convnext_tiny_finetuned` + `densenet121` + `swin_tiny`
5. **For Edge Devices:** Use `densenet121` (smallest, fast, accurate)

---

## 📁 File Reference

### Reports (JSON)
| File | Purpose |
|------|---------|
| `robustness_ranking.json` | Full robustness analysis with rankings |
| `class_specialists.json` | Best models per class |

### Tables (CSV)
| File | Rows | Purpose |
|------|------|---------|
| `model_comparison_table.csv` | 13 | Full model comparison |
| `per_class_performance.csv` | 13 | Class-wise accuracy/F1 |
| `corruption_robustness_all_models.csv` | 13 | 15 corruption accuracies |
| `model_diversity_correlation.csv` | 13 | 11×11 correlation matrix |

### Figures (PNG)
| File | Purpose |
|------|---------|
| `corruption_heatmap.png` | Corruption accuracy heatmap |
| `inference_time_comparison.png` | Speed comparison |
| `model_diversity_heatmap.png` | Correlation visualization |
| `per_class_model_performance.png` | Per-class performance |
| `robustness_ranking.png` | Robustness ranking chart |

### Confusion Matrices (PNG)
11 individual confusion matrix visualizations, one per model.

