# 📊 Complete Analysis Tables Guide

The CSV files under `analysis_outputs/tables` capture every quantitative slice of the OrganAMNIST analysis pipeline. This guide mirrors the “teach me every file” format: each section tells you **what the table is**, **how to read it**, **numbers to remember**, and **actions to take**.

---

## Table of Contents

1. [Label Distribution Table](#1-label-distribution-table)  
2. [Duplicate Summary Table](#2-duplicate-summary-table)  
3. [Suspect Label Table](#3-suspect-label-table)  
4. [Test Anomaly Scores](#4-test-anomaly-scores)  
5. [Test Characterization Summary](#5-test-characterization-summary)  
6. [Confusion Matrix Table](#6-confusion-matrix-table)  
7. [Per-Class Accuracy Table](#7-per-class-accuracy-table)  
8. [Interclass Similarity Matrix](#8-interclass-similarity-matrix)  
9. [Feature Class Centroids](#9-feature-class-centroids)  
10. [Feature Multiscale Stats](#10-feature-multiscale-stats)  
11. [Robustness Metrics Table](#11-robustness-metrics-table)  
12. [Geometric Stats Table](#12-geometric-stats-table)  
13. [Per-Image Pixel Stats](#13-per-image-pixel-stats)  
14. [Summary & Quick Actions](#14-summary--quick-actions)  
15. [File Reference](#15-file-reference)

---

## 1. Label Distribution Table

**File:** `label_distribution.csv`

| Column | Meaning |
|--------|---------|
| `label` | Organ ID (0–10). |
| `train_count`, `train_pct` | Raw count & percent in training split. |
| `val_count`, `val_pct` | Same for validation split. |

**Highlights**
- Class 6 (Liver) = 17.84% of train (6,164 images).  
- Classes 1–3 (Femurs + Heart) each ≤ 4.26%.  
- Validation mirrors train but has slightly more lungs (classes 7–8).

**Why it matters:** Use the table to derive class weights, to ensure stratified sampling, and to verify that your train/val splits are consistent before training.

---

## 2. Duplicate Summary Table

**File:** `data_quality_duplicate_summary.csv`

```
split,total_images,duplicate_pairs,unique_hashes
train,34561,921,33525
val,6491,134,6343
```

**Interpretation**
- Duplicate rate ≈ 3.0% in train (1,036 redundant images).  
- Duplicate rate ≈ 2.3% in val (148 redundant images).  

**Action:** Remove or flag duplicates before final training to avoid inflated metrics or train/val leakage.

---

## 3. Suspect Label Table

**File:** `data_quality_suspect_labels.csv`

The table structure (`file,label,predicted,confidence`) is empty—no mislabeled images were detected. Still handy when you rerun the pipeline after major dataset edits.

---

## 4. Test Anomaly Scores

**File:** `test_anomaly_scores.csv`

| Split | Anomaly Rate | Mean Score | Std |
|-------|--------------|-----------|-----|
| Train | 0.0% | 0.000 | 0.000 |
| Val | 0.46% | 0.083 | 0.0269 |
| **Test** | **0.58%** | **0.0859** | **0.0307** |

**Usage:** Confirms that test images are slightly more “unusual” vs train/val, reinforcing the domain shift highlighted in the reports & figures.

---

## 5. Test Characterization Summary

**File:** `test_characterization_summary.csv`

| Metric | Train | Val | Test | Reading |
|--------|-------|-----|------|---------|
| Mean Intensity | 0.467 | 0.470 | 0.460 | Test images are a bit darker. |
| Std Intensity | 0.280 | 0.281 | **0.263** | Lower contrast on test. |
| Edge Density Mean | 0.0254 | 0.0285 | 0.0281 | Slightly higher edges in val/test. |
| LBP Entropy | 2.728 | 2.721 | **2.638** | Test textures are simpler. |
| Anomaly Rate | 0.0% | 0.46% | **0.58%** | Matches Section 4. |

**Why it matters:** This table aggregates every shift metric into one place, ideal for monitoring after each augmentation experiment.

---

## 6. Confusion Matrix Table

**File:** `class_imbalance_confusion_matrix.csv`

**Format:** 11×11 table (rows = actual, columns = predicted). Row totals match validation sample counts.

**Key patterns**
- Class 0 (Bladder) → heavily misclassified as class 2 (235 samples).  
- Class 2 → perfectly classified (225/225).  
- Class 7 (Left Lung) mostly stays on-diagonal (920/1,033).  

**How to use:**  
- Compute per-class recall/precision without re-running evaluation.  
- Identify systematic confusions (0→2, 4→2, 5→2, 6→2, 8→2).  
- Feed rows into cost-sensitive training strategies (e.g., penalize predictions of class 2 when true label is 4/5/6/8).

---

## 7. Per-Class Accuracy Table

**File:** `class_imbalance_per_class_accuracy.csv`

| Label | Val Samples | Accuracy | Status |
|-------|-------------|----------|--------|
| 0 | 321 | **7.79%** | ❌ Very poor |
| 1 | 233 | 85.41% | ✅ Good |
| 2 | 225 | **100%** | 🏆 Perfect |
| 4 | 568 | 17.43% | ❌ Poor |
| 5 | 637 | 19.00% | ❌ Poor |
| 7 | 1,033 | 89.06% | ✅ Excellent |

**Insights**
- Tiered view (Excellent / Good / Moderate / Poor) guides targeted improvements.  
- Combine with the confusion matrix to see whether low accuracy comes from class imbalance or texture similarity.

---

## 8. Interclass Similarity Matrix

**File:** `feature_interclass_similarity.csv`

**Definition:** Cosine similarity of class centroids (0–1). High values = features look alike → expect confusion.

| Example Pairs | Similarity | Meaning |
|---------------|------------|---------|
| 4 ↔ 5 | 0.88 | Left vs Right Kidney (hard). |
| 4 ↔ 10 | 0.86 | Kidney vs Pancreas similarity. |
| 7 ↔ 8 | 0.79 | Left vs Right lung. |
| 1 ↔ 6 | 0.37 | Femur vs Liver (easy). |

**How to use:** Drive curriculum strategies (contrastive learning on high-similarity pairs) and diagnose why certain confusion matrix entries stay large.

---

## 9. Feature Class Centroids

**File:** `feature_class_centroids.csv`

**Structure:** 11 rows × 256 columns. Each row is the average embedding vector for that class.

**Practical uses**
- Prototype-based inference (nearest centroid).  
- Visualizing which feature dimensions activate strongly per organ.  
- Initializing smaller student models with these centroids.

**Example:** Feature dimension 9 spikes for classes 7 & 8, aligning with lung-specific patterns from Grad-CAM.

---

## 10. Feature Multiscale Stats

**File:** `feature_multiscale_stats.csv`

| Split | Scale | Mean | Std | Edge Density |
|-------|-------|------|-----|--------------|
| Train | 32 | 118.36 | 54.84 | 0.344 |
| Train | 64 | 118.35 | 56.56 | 0.207 |
| Train | 128 | 118.36 | 57.12 | 0.073 |
| Val | 32 | 119.33 | 56.00 | 0.355 |
| Test | 32 | 119.21 | 51.91 | **0.315** |
| Test | 128 | 119.20 | 54.11 | **0.063** |

**Takeaways**
- Edge density drops with scale (expected).  
- Test edge density is consistently lower → smoother images.  
- Helps justify multi-scale fusion in custom models (DenseViT, Swin-MultiScale).

---

## 11. Robustness Metrics Table

**File:** `robustness_metrics.csv`

| Perturbation | Avg PSNR | Avg SSIM | Comment |
|--------------|----------|----------|---------|
| gaussian_noise | 25.1 dB | 0.44 | Highly destructive. |
| motion_blur | 36.2 dB | 0.97 | Hardly alters structure. |
| contrast_up | 21.2 dB | 0.88 | Moderate visual impact. |
| contrast_down | 20.5 dB | 0.71 | Noticeable but not catastrophic. |

**Why it matters:** PSNR/SSIM statistics explain the qualitative perturbation grids from the figures guide. Use them to set noise levels for robustness training.

---

## 12. Geometric Stats Table

**File:** `geometric_stats.csv`

| Column | Details |
|--------|---------|
| `file` | Image filename. |
| `edge_density` | Fraction of pixels classified as edges (Canny). |
| `horiz_flip_diff`, `vert_flip_diff` | Mean absolute difference between original and flipped versions (0–255 scale). |

**Observations**
- Edge densities are near zero for most CT slices (organs are smooth).  
- Flip differences range 34–111 → images are far from symmetric.  
- Validate whether flip augmentation produces meaningful variety (it does—Section 6.2 of figures guide).

---

## 13. Per-Image Pixel Stats

**Files:** `train_image_stats.csv` (34,561 rows) and `val_image_stats.csv` (6,493 rows)

| Column | Meaning |
|--------|---------|
| `mean`, `std`, `min`, `max` | Pixel statistics per image (0–255). |

**Use cases**
- Spot broken files (min=max).  
- Detect extreme brightness/contrast outliers for manual inspection.  
- Feed aggregate statistics into auto-normalization routines.  
- Pair with label information to analyze class-dependent brightness.

Example rows show extremes (mean 26.85 vs 205.92, std up to 88.26) proving that augmentation strategies should handle wide intensity ranges.

---

## 14. Summary & Quick Actions

| Finding | Table(s) | Action |
|---------|----------|--------|
| Class 6 dominance, classes 0–3 scarcity | Label distribution, per-class accuracy | Apply class-weighted loss, augment minorities. |
| Duplicate presence | Duplicate summary, duplicate listings (JSON) | Remove redundant slices pre-training. |
| Confusion cluster (4/5/9/10) | Confusion matrix, similarity matrix | Add class-specific augmentations, metric learning. |
| Test shift (texture/contrast) | Test characterization summary, anomaly scores, multiscale stats | Use brightness/contrast augmentation and domain adaptation. |
| Robustness weakest under noise | Robustness metrics table | Inject Gaussian noise during training; evaluate after adversarial training. |
| Flip augmentation validity | Geometric stats | Keep horizontal flips; vertical flips also meaningful but double-check anatomy. |

---

## 15. File Reference

| File | Rows | Purpose |
|------|------|---------|
| `label_distribution.csv` | 13 | Class counts/percentages. |
| `data_quality_duplicate_summary.csv` | 4 | Duplicate statistics. |
| `data_quality_suspect_labels.csv` | 0–few | Potential label issues (none currently). |
| `test_anomaly_scores.csv` | 5 | Out-of-distribution detection per split. |
| `test_characterization_summary.csv` | 5 | Combined split comparison metrics. |
| `class_imbalance_confusion_matrix.csv` | 13 | Confusions at a glance. |
| `class_imbalance_per_class_accuracy.csv` | 13 | Validation accuracy per class. |
| `feature_interclass_similarity.csv` | 13 | Cosine similarity matrix (11×11). |
| `feature_class_centroids.csv` | 13 | 256-dim feature prototypes. |
| `feature_multiscale_stats.csv` | 11 | Mean/std/edges at different resolutions. |
| `robustness_metrics.csv` | 66 | PSNR/SSIM under perturbations. |
| `geometric_stats.csv` | 1,026 | Edge density + flip asymmetry. |
| `train_image_stats.csv` | 34,561 | Per-image stats (train). |
| `val_image_stats.csv` | 6,493 | Per-image stats (val). |

Keep this guide open alongside the **reports** and **figures** walkthroughs to get a complete, context-rich understanding of every artifact generated by the analysis pipeline.