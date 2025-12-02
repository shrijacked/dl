# 🏆 Complete Model Evaluation Guide

All evaluation artifacts live under `evaluation_outputs/`. This guide mirrors the “teach me everything” style used for figures/tables/reports: for each JSON/CSV/PNG you’ll get **what it measures, the key numbers, and how to act on them**.

---

## Contents Overview

1. [Robustness Ranking Report](#1-robustness-ranking-report)  
2. [Class Specialists Report](#2-class-specialists-report)  
3. [Model Comparison Table](#3-model-comparison-table)  
4. [Per-Class Performance Table](#4-per-class-performance-table)  
5. [Corruption Robustness Table](#5-corruption-robustness-table)  
6. [Model Diversity Correlation](#6-model-diversity-correlation)  
7. [Evaluation Figures](#7-evaluation-figures)  
8. [Per-Model Confusion Matrices](#8-per-model-confusion-matrices)  
9. [Summary & Recommended Actions](#9-summary--recommended-actions)  
10. [File Reference](#10-file-reference)

---

## 1. Robustness Ranking Report

**File:** `reports/robustness_ranking.json`

**What it contains:** Clean accuracy, mean corruption accuracy (15 corruptions), and relative robustness (corruption/clean) for all 11 models.

### Overall leaderboard

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

### Category winners

| Category | Best Model(s) | Notes |
|----------|---------------|-------|
| Noise (Gaussian/Shot/Impulse) | convnext_tiny_finetuned (74.9%), efficientnet_b3 (74.2%) | Noise is hardest overall. |
| Blur | convnext_tiny_finetuned (76.0%) | Swin finetuned is close (74.7%). |
| Weather | convnext_tiny_finetuned (77.5%) | Handles brightness/fog well. |
| Digital | convnext_tiny_finetuned (78.7%) | Also best on JPEG (81%). |

**Specialists per corruption:** ConvNeXt-Tiny finetuned dominates 13/15 corruptions; EfficientNet-B3 wins shot noise; Swin-Tiny finetuned wins glass blur.

**Why it matters:** Choose models based on deployment constraints—ConvNeXt FT for robustness-first, Swin FT for accuracy, EfficientNet-B3 when speed + robustness balance is needed.

---

## 2. Class Specialists Report

**File:** `reports/class_specialists.json`

**Purpose:** Lists the top 3 models for each organ class (based on accuracy/F1).

| Class | Top Model | Runners-Up | Interpretation |
|-------|-----------|------------|----------------|
| Bladder | swin_tiny_finetuned | resnext50_32x4d, resnet50 | Swin FT locks 100% accuracy. |
| Femur (L/R) | resnet50 / resnet101 | resnext50 | ResNet family excels at bones. |
| Heart | efficientnet_b3 | convnext_tiny_finetuned, resnet101 | EfficientNet handles high contrast regions. |
| Kidney (L) | resnext101_32x8d | swin_tiny_finetuned, convnext_tiny_finetuned | Larger receptive fields help. |
| Kidney (R) | efficientnet_b3 | swin_tiny, resnet101 | EfficientNet best on asymmetric organs. |
| Lung (L/R) | efficientnet_b3 / swin_tiny_finetuned | densenet121 | Attention-driven models win on lungs. |
| Pancreas | resnet50 | efficientnet_b3, swin_tiny_finetuned | Residual models catch subtle textures. |

**Use cases:**  
- Build ensembles that mix specialists (e.g., EfficientNet-B3 for soft organs + Swin FT for bladder + ResNet50 for bones).  
- Prioritize data augmentation for organs without a dominant specialist.

---

## 3. Model Comparison Table

**File:** `tables/model_comparison_table.csv`

| Model | Val Acc | Macro F1 | Params (M) | FLOPs (G) | Inference (ms) | Worst Class | Best Class |
|-------|---------|----------|------------|-----------|----------------|-------------|------------|
| **swin_tiny_finetuned** | **99.69%** | 0.998 | 27.5 | 4.5 | 1.17 | Heart (99.23%) | Bladder (100%) |
| swin_tiny | 99.63% | 0.996 | 27.5 | 4.5 | 1.17 | Heart (98.98%) | Femur L (100%) |
| **densenet121** | 99.61% | 0.996 | **7.0** | **2.9** | **0.94** | Heart (98.72%) | Femur L (100%) |
| convnext_tiny_finetuned | 99.60% | 0.996 | 27.8 | 4.5 | 1.17 | Spleen (98.3%) | Femur L (100%) |
| **efficientnet_b3** | 99.32% | 0.994 | 10.7 | 1.8 | **0.77** | Lung R (96.53%) | Femur L (100%) |
| resnet101 | 99.15% | 0.993 | 42.5 | 7.8 | 1.67 | Lung R (96.13%) | Femur L (100%) |
| resnet50 | 99.14% | 0.991 | 23.5 | 4.1 | 1.11 | Heart (97.19%) | Femur L (100%) |
| resnext50_32x4d | 98.94% | 0.991 | 23.0 | 4.2 | 1.13 | Lung R (96.23%) | Femur L (100%) |
| resnext101_32x8d | 98.94% | 0.991 | **86.7** | **16.4** | **2.96** | Lung R (95.04%) | Femur L (100%) |
| vit_s16 | 98.43% | 0.986 | 21.7 | 4.6 | 1.19 | Lung R (94.85%) | Liver (100%) |
| convnext_tiny | 97.32% | 0.974 | 27.8 | 4.5 | 1.17 | Heart (89.80%) | Femur L (100%) |

**Reading tips**
- Choose DenseNet121 when memory/speed-constrained.  
- EfficientNet-B3 is the inference-speed champion (0.77 ms).  
- ConvNeXt Tiny (non-finetuned) lags both in accuracy and robustness—only use after finetuning.

---

## 4. Per-Class Performance Table

**File:** `tables/per_class_performance.csv`

**Best model per class**

| Class | Best Accuracy | Best Model |
|-------|---------------|------------|
| 0 (Bladder) | 100% | swin_tiny_finetuned |
| 1 (Femur L) | 100% | resnet50 (+ others) |
| 2 (Femur R) | 100% | resnet50 (+ others) |
| 3 (Heart) | 100% | efficientnet_b3 |
| 4 (Kidney L) | 100% | resnext101_32x8d |
| 5 (Kidney R) | 100% | efficientnet_b3 |
| 6 (Liver) | 100% | many |
| 7 (Lung L) | 100% | efficientnet_b3 |
| 8 (Lung R) | 99.6% | swin_tiny_finetuned |
| 9 (Spleen) | 100% | resnet101 |
| 10 (Pancreas) | 100% | resnet50 (+ others) |

**Hard classes (average across models):**

| Class | Avg Acc | Reason |
|-------|---------|--------|
| Heart | 97.36% | High contrast, limited samples. |
| Lung (R) | 97.00% | Bilateral confusion + texture shift. |
| Pancreas | 99.45% | Thin structure, benefits from residual nets. |

**Takeaway:** Deploy class-aware monitoring (especially Heart/Lung R) even with top models.

---

## 5. Corruption Robustness Table

**File:** `tables/corruption_robustness_all_models.csv`

**Columns:** clean accuracy, mean corruption accuracy, relative robustness, per-category averages (noise/blur/weather/digital), plus every individual corruption.

### Corruption difficulty (averaged over all models)

| Rank | Corruption | Avg Accuracy |
|------|------------|--------------|
| 1 (easiest) | JPEG compression | 77.76% |
| 2 | Brightness | 76.21% |
| 3 | Contrast | 76.14% |
| … | … | … |
| 14 | Glass blur | 69.74% |
| 15 (hardest) | Impulse noise | 69.64% |

### Model stability (std dev across corruptions)

| Model | Mean Corruption | Std Dev |
|-------|-----------------|---------|
| convnext_tiny_finetuned | 76.88% | 2.15% |
| swin_tiny_finetuned | 75.36% | 2.31% |
| efficientnet_b3 | 74.57% | 2.18% |
| vit_s16 | 70.28% | 2.29% |

**Usage:**  
- When benchmarking new augmentations, compare them against this table.  
- Use per-corruption columns to target synthetic data generation (e.g., add impulse noise, glass blur).

---

## 6. Model Diversity Correlation

**File:** `tables/model_diversity_correlation.csv`

**Interpretation:** Pearson correlation of prediction vectors; low/negative values = diverse behavior → better ensembles.

| Notable Pairs | Correlation | Insight |
|---------------|-------------|---------|
| efficientnet_b3 ↔ resnet101 | **0.94** | Very similar outputs; redundant in ensemble. |
| convnext_tiny_finetuned ↔ resnet50 | **0.08** | Highly diverse; good ensemble combo. |
| convnext_tiny_finetuned ↔ convnext_tiny | **-0.07** | Finetuning dramatically alters decisions. |
| swin_tiny ↔ convnext_tiny_finetuned | 0.08 | Another diverse pair. |

**Recommended ensembles**
1. **Diverse/robust:** convnext_tiny_finetuned + densenet121 + swin_tiny.  
2. **Speed-focused:** efficientnet_b3 + densenet121 + resnet50.  
3. **Accuracy-focused:** swin_tiny_finetuned + convnext_tiny_finetuned + efficientnet_b3.

---

## 7. Evaluation Figures

**Directory:** `evaluation_outputs/figures/`

| Figure | What to inspect |
|--------|-----------------|
| `corruption_heatmap.png` | Visual row/column patterns → hardest corruptions (vertical) & weak models (horizontal). |
| `robustness_ranking.png` | Bar chart of mean corruption accuracy, easier to show in slides. |
| `per_class_model_performance.png` | Which model dominates each organ. |
| `inference_time_comparison.png` | Speed vs architecture; pair with deployment constraints. |
| `model_diversity_heatmap.png` | Visual version of correlation matrix; look for cool colors to pick ensemble members. |

Use these PNGs for presentations or to quickly sanity-check numeric tables.

---

## 8. Per-Model Confusion Matrices

**Directory:** `evaluation_outputs/confusion_matrices/`

Available PNGs:
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

**What to look for**
1. Diagonal strength → overall accuracy.  
2. Row-specific leaks → problematic true classes.  
3. Compare matrices between base vs finetuned variants (e.g., convnext tiny) to see where improvements occurred.  
4. Validate class specialist claims visually (e.g., EfficientNet’s heart row).

---

## 9. Summary & Recommended Actions

| Topic | Insight | Next Step |
|-------|---------|-----------|
| Accuracy champion | Swin-Tiny finetuned hits 99.69% | Use as baseline for accuracy-critical deployments. |
| Robustness champion | ConvNeXt-Tiny finetuned: 76.9% mean corruption | Prefer when noise/weather robustness is key. |
| Efficiency champion | DenseNet121 (7M params, 0.94 ms) / EfficientNet-B3 (0.77 ms) | Ideal for edge devices. |
| Weak classes | Heart & Lung (R) across many models | Collect more samples, apply class-specific augmentation, monitor post-deployment. |
| Hard corruptions | Impulse noise, glass blur | Augment training data with synthetic impulse noise and blur kernels. |
| Ensemble planning | ConvNeXt FT has lowest correlation with others | Combine with DenseNet121 or Swin for diverse committees. |

---

## 10. File Reference

| Type | File | Purpose |
|------|------|---------|
| Report | `reports/robustness_ranking.json` | All robustness metrics + rankings. |
| Report | `reports/class_specialists.json` | Per-organ model specialists. |
| Table | `tables/model_comparison_table.csv` | Accuracy/efficiency summary. |
| Table | `tables/per_class_performance.csv` | Per-class accuracy/F1 by model. |
| Table | `tables/corruption_robustness_all_models.csv` | Accuracy per corruption type. |
| Table | `tables/model_diversity_correlation.csv` | Prediction correlation matrix. |
| Figures | `figures/*.png` | Visual summaries (heatmaps, rankings, speed). |
| Figures | `confusion_matrices/*.png` | Model-specific confusion matrices. |

Keep this guide handy when comparing new models or preparing reports—you can jump directly to the relevant section for clean metrics, robustness behavior, or ensemble design.
