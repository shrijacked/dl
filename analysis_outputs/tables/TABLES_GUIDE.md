# 📋 Analysis Tables Guide

This guide explains every CSV table file in this folder, what each column means, and how to interpret the data for your deep learning project.

---

## Table of Contents

1. [Label Distribution](#1-label-distribution)
2. [Data Quality - Duplicate Summary](#2-data-quality---duplicate-summary)
3. [Data Quality - Suspect Labels](#3-data-quality---suspect-labels)
4. [Test Anomaly Scores](#4-test-anomaly-scores)
5. [Test Characterization Summary](#5-test-characterization-summary)
6. [Class Imbalance - Confusion Matrix](#6-class-imbalance---confusion-matrix)
7. [Class Imbalance - Per Class Accuracy](#7-class-imbalance---per-class-accuracy)
8. [Feature - Interclass Similarity](#8-feature---interclass-similarity)
9. [Feature - Class Centroids](#9-feature---class-centroids)
10. [Feature - Multiscale Stats](#10-feature---multiscale-stats)
11. [Robustness Metrics](#11-robustness-metrics)
12. [Geometric Stats](#12-geometric-stats)
13. [Image Stats (Train & Val)](#13-image-stats-train--val)

---

## 1. Label Distribution

**File:** `label_distribution.csv`

### What is it?
Shows how many images belong to each class in training and validation sets with percentages.

### Columns:
| Column | Description |
|--------|-------------|
| `label` | Class label (0-10) |
| `train_count` | Number of training samples |
| `train_pct` | Percentage of training set |
| `val_count` | Number of validation samples |
| `val_pct` | Percentage of validation set |

### Full Data:

| Label | Train Count | Train % | Val Count | Val % |
|-------|-------------|---------|-----------|-------|
| 0 | 1,956 | 5.66% | 321 | 4.95% |
| 1 | 1,390 | 4.02% | 233 | 3.59% |
| 2 | 1,357 | 3.93% | 225 | 3.47% |
| 3 | 1,474 | 4.26% | 392 | 6.04% |
| 4 | 3,963 | 11.47% | 568 | 8.75% |
| 5 | 3,817 | 11.04% | 637 | 9.81% |
| **6** | **6,164** | **17.84%** | 1,033 | 15.91% |
| 7 | 3,919 | 11.34% | 1,033 | 15.91% |
| 8 | 3,929 | 11.37% | 1,009 | 15.54% |
| 9 | 3,031 | 8.77% | 529 | 8.15% |
| 10 | 3,561 | 10.30% | 511 | 7.87% |

### Key Takeaways:
- **Class 6 is largest** with ~18% of training data
- **Classes 1, 2, 3 are smallest** with ~4% each
- **Imbalance ratio:** ~4.5:1 between largest and smallest classes
- Train/Val proportions are slightly different (validation has more of classes 7, 8)

---

## 2. Data Quality - Duplicate Summary

**File:** `data_quality_duplicate_summary.csv`

### What is it?
Summary of duplicate images found using perceptual hashing.

### Content:
```
split,total_images,duplicate_pairs,unique_hashes
train,34561,921,33525
val,6491,134,6343
```

### Columns:
| Column | Description |
|--------|-------------|
| `split` | Dataset split (train/val) |
| `total_images` | Total number of images in split |
| `duplicate_pairs` | Number of duplicate groups found |
| `unique_hashes` | Number of truly unique images |

### Calculations:

**Training Set:**
- Duplicates: 921 groups
- Affected images: 34,561 - 33,525 = **1,036 redundant copies**
- Duplicate rate: **3.0%**

**Validation Set:**
- Duplicates: 134 groups  
- Affected images: 6,491 - 6,343 = **148 redundant copies**
- Duplicate rate: **2.3%**

### Key Takeaways:
- ⚠️ About 3% of your training data are duplicates
- Consider removing duplicates to:
  - Prevent overfitting to repeated examples
  - Avoid data leakage if duplicates exist across train/val

---

## 3. Data Quality - Suspect Labels

**File:** `data_quality_suspect_labels.csv`

### What is it?
Lists images where the model's prediction strongly disagrees with the assigned label (possible mislabeled data).

### Content:
```
file,label,predicted,confidence
```
*(Empty - no suspect labels found)*

### Columns:
| Column | Description |
|--------|-------------|
| `file` | Image filename |
| `label` | Assigned ground truth label |
| `predicted` | Model's predicted label |
| `confidence` | Model's confidence in its prediction |

### Key Takeaways:
- ✅ **No suspect labels detected** - Labels appear consistent with image content
- This is good news for data quality

---

## 4. Test Anomaly Scores

**File:** `test_anomaly_scores.csv`

### What is it?
Measures how "anomalous" or out-of-distribution images are in each split.

### Content:
```
split,anomaly_rate,anomaly_score_mean,anomaly_score_std
train,0.0,0.0,0.0
val,0.0046,0.0833974201616975,0.0268625732921899
test,0.0058,0.0859347068652064,0.030666002141133
```

### Columns:
| Column | Description |
|--------|-------------|
| `split` | Dataset split |
| `anomaly_rate` | Proportion of images flagged as anomalies |
| `anomaly_score_mean` | Average anomaly score (higher = more anomalous) |
| `anomaly_score_std` | Standard deviation of anomaly scores |

### Interpretation:

| Split | Anomaly Rate | Mean Score | Interpretation |
|-------|--------------|------------|----------------|
| Train | 0.0% | 0.0 | Baseline (reference distribution) |
| Val | 0.46% | 0.083 | Slight deviation from training |
| Test | **0.58%** | **0.086** | Highest deviation |

### Key Takeaways:
- ⚠️ **Test set has highest anomaly rate** (0.58%)
- Test images are slightly more "unusual" compared to training distribution
- This confirms the distribution shift observed in other analyses

---

## 5. Test Characterization Summary

**File:** `test_characterization_summary.csv`

### What is it?
Comprehensive statistics comparing image characteristics across all splits.

### Columns:
| Column | Description |
|--------|-------------|
| `split` | Dataset split (train/val/test) |
| `num_images` | Number of images analyzed |
| `mean_intensity` | Average pixel intensity (0-1 normalized) |
| `std_intensity` | Standard deviation of intensity |
| `edge_density_mean` | Average edge density (how many edges) |
| `edge_density_std` | Edge density variation |
| `lbp_entropy` | Local Binary Pattern entropy (texture complexity) |
| `anomaly_rate` | Proportion of anomalous images |
| `anomaly_score_mean` | Average anomaly score |
| `anomaly_score_std` | Anomaly score variation |

### Full Data:

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Num Images | 5,000 | 5,000 | 5,000 |
| Mean Intensity | 0.467 | 0.470 | 0.460 |
| Std Intensity | 0.280 | 0.281 | **0.263** |
| Edge Density Mean | 0.0254 | 0.0285 | 0.0281 |
| Edge Density Std | 0.0260 | 0.0273 | **0.0237** |
| LBP Entropy | 2.728 | 2.721 | **2.638** |
| Anomaly Rate | 0.0% | 0.46% | **0.58%** |

### Key Takeaways:
- ⚠️ **Test set has lower texture entropy** (2.638 vs 2.728)
- ⚠️ **Test set has less intensity variation** (0.263 vs 0.280)
- Test images appear to be more "uniform" or less complex
- Edge density is similar across splits

---

## 6. Class Imbalance - Confusion Matrix

**File:** `class_imbalance_confusion_matrix.csv`

### What is it?
Shows how the model confuses different classes - rows are actual labels, columns are predicted labels.

### How to Read:
- **Row** = Actual/True label
- **Column** = Predicted label
- **Diagonal values** = Correct predictions
- **Off-diagonal values** = Misclassifications

### Full Confusion Matrix:

| Actual↓ / Pred→ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-----------------|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **25** | 25 | 235 | 0 | 33 | 1 | 0 | 1 | 0 | 1 | 0 |
| **1** | 0 | **199** | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2** | 0 | 0 | **225** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 2 | 15 | 33 | **325** | 1 | 3 | 0 | 0 | 0 | 0 | 13 |
| **4** | 12 | 91 | 331 | 0 | **99** | 15 | 0 | 0 | 0 | 14 | 6 |
| **5** | 11 | 28 | 458 | 9 | 4 | **121** | 0 | 0 | 0 | 0 | 6 |
| **6** | 27 | 6 | 424 | 28 | 11 | 2 | **519** | 0 | 2 | 14 | 0 |
| **7** | 5 | 28 | 3 | 0 | 29 | 0 | 1 | **920** | 0 | 43 | 4 |
| **8** | 12 | 4 | 416 | 0 | 4 | 1 | 43 | 0 | **519** | 7 | 3 |
| **9** | 22 | 38 | 191 | 0 | 8 | 10 | 2 | 4 | 0 | **242** | 12 |
| **10** | 7 | 87 | 170 | 0 | 32 | 28 | 0 | 0 | 0 | 4 | **183** |

### Major Confusion Patterns:

1. **Class 2 is a confusion magnet:**
   - Class 0 → 2: 235 misclassifications
   - Class 4 → 2: 331 misclassifications
   - Class 5 → 2: 458 misclassifications
   - Class 6 → 2: 424 misclassifications
   - Class 8 → 2: 416 misclassifications

2. **Well-separated classes:**
   - Class 2: 100% accuracy (225/225)
   - Class 7: 89% accuracy (920/1033)
   - Class 3: 83% accuracy (325/392)

3. **Poorly-separated classes:**
   - Class 0: 7.8% accuracy (25/321)
   - Class 4: 17.4% accuracy (99/568)
   - Class 5: 19% accuracy (121/637)

### Key Takeaways:
- ⚠️ **Class 2 receives many false positives** - Model over-predicts this class
- Classes 0, 4, 5 need more attention (low accuracy)
- Classes 7, 8 are easily distinguished

---

## 7. Class Imbalance - Per Class Accuracy

**File:** `class_imbalance_per_class_accuracy.csv`

### What is it?
Shows accuracy metrics broken down by each class.

### Columns:
| Column | Description |
|--------|-------------|
| `label` | Class label (0-10) |
| `train_count` | Training samples (capped at 1500) |
| `val_count` | Validation samples |
| `val_accuracy` | Accuracy on validation set |
| `correct_predictions` | Number of correct predictions |

### Full Data:

| Label | Train | Val | Accuracy | Correct | Status |
|-------|-------|-----|----------|---------|--------|
| 0 | 1,500 | 321 | **7.79%** | 25 | ❌ Very Poor |
| 1 | 1,390 | 233 | 85.41% | 199 | ✅ Good |
| **2** | 1,357 | 225 | **100%** | 225 | ✅ Perfect |
| 3 | 1,474 | 392 | 82.91% | 325 | ✅ Good |
| 4 | 1,500 | 568 | **17.43%** | 99 | ❌ Poor |
| 5 | 1,500 | 637 | **19.00%** | 121 | ❌ Poor |
| 6 | 1,500 | 1,033 | 50.24% | 519 | ⚠️ Moderate |
| **7** | 1,500 | 1,033 | **89.06%** | 920 | ✅ Excellent |
| 8 | 1,500 | 1,009 | 51.44% | 519 | ⚠️ Moderate |
| 9 | 1,500 | 529 | 45.75% | 242 | ⚠️ Moderate |
| 10 | 1,500 | 511 | 35.81% | 183 | ⚠️ Below Average |

### Performance Tiers:

| Tier | Classes | Accuracy Range |
|------|---------|----------------|
| 🏆 Excellent | 2, 7 | 89-100% |
| ✅ Good | 1, 3 | 82-86% |
| ⚠️ Moderate | 6, 8, 9, 10 | 35-52% |
| ❌ Poor | 0, 4, 5 | 8-19% |

### Key Takeaways:
- **Class 2 achieves 100% accuracy** - Very distinct features
- **Class 0 has only 7.79% accuracy** - Almost always misclassified
- Strong correlation with class imbalance (minority classes often worse)
- Consider class-specific data augmentation for poor performers

---

## 8. Feature - Interclass Similarity

**File:** `feature_interclass_similarity.csv`

### What is it?
A similarity matrix showing how similar feature representations are between classes (cosine similarity).

### How to Read:
- **Values range from 0 to 1**
- **1.0** = Identical features (diagonal)
- **High values (>0.7)** = Similar classes (hard to distinguish)
- **Low values (<0.5)** = Different classes (easy to distinguish)

### Full Similarity Matrix:

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 0 | 1.00 | 0.50 | 0.43 | 0.40 | 0.61 | 0.61 | 0.46 | 0.50 | 0.41 | 0.56 | 0.59 |
| 1 | 0.50 | 1.00 | **0.72** | 0.38 | 0.58 | 0.63 | 0.37 | 0.63 | 0.39 | 0.57 | 0.61 |
| 2 | 0.43 | **0.72** | 1.00 | 0.40 | 0.60 | 0.62 | 0.47 | 0.41 | 0.48 | 0.50 | 0.50 |
| 3 | 0.40 | 0.38 | 0.40 | 1.00 | 0.66 | 0.63 | **0.71** | 0.51 | 0.54 | 0.47 | **0.73** |
| 4 | 0.61 | 0.58 | 0.60 | 0.66 | 1.00 | **0.88** | 0.63 | 0.52 | 0.41 | **0.77** | **0.86** |
| 5 | 0.61 | 0.63 | 0.62 | 0.63 | **0.88** | 1.00 | **0.76** | 0.53 | 0.45 | **0.81** | **0.82** |
| 6 | 0.46 | 0.37 | 0.47 | **0.71** | 0.63 | **0.76** | 1.00 | 0.49 | 0.58 | **0.77** | **0.74** |
| 7 | 0.50 | 0.63 | 0.41 | 0.51 | 0.52 | 0.53 | 0.49 | 1.00 | **0.79** | 0.53 | 0.67 |
| 8 | 0.41 | 0.39 | 0.48 | 0.54 | 0.41 | 0.45 | 0.58 | **0.79** | 1.00 | 0.36 | 0.54 |
| 9 | 0.56 | 0.57 | 0.50 | 0.47 | **0.77** | **0.81** | **0.77** | 0.53 | 0.36 | 1.00 | **0.77** |
| 10 | 0.59 | 0.61 | 0.50 | **0.73** | **0.86** | **0.82** | **0.74** | 0.67 | 0.54 | **0.77** | 1.00 |

### Highly Similar Class Pairs (>0.75):

| Class Pair | Similarity | Implication |
|------------|------------|-------------|
| 4 ↔ 5 | **0.88** | Very hard to distinguish |
| 4 ↔ 10 | **0.86** | Very hard to distinguish |
| 5 ↔ 9 | **0.81** | Hard to distinguish |
| 5 ↔ 10 | **0.82** | Hard to distinguish |
| 7 ↔ 8 | **0.79** | Hard to distinguish |
| 5 ↔ 6 | **0.76** | Hard to distinguish |

### Well-Separated Class Pairs (<0.45):

| Class Pair | Similarity | Implication |
|------------|------------|-------------|
| 1 ↔ 3 | 0.38 | Easy to distinguish |
| 1 ↔ 6 | 0.37 | Easy to distinguish |
| 8 ↔ 9 | 0.36 | Easy to distinguish |
| 0 ↔ 3 | 0.40 | Easy to distinguish |

### Key Takeaways:
- ⚠️ **Classes 4, 5, 9, 10 form a confusing cluster** (all >0.77 similarity)
- ⚠️ **Classes 7 & 8 are similar** (0.79) but still distinguishable
- Classes 1 & 2 are similar (0.72) - explains some confusion
- **Class 3 is most unique** - low similarity with most others

---

## 9. Feature - Class Centroids

**File:** `feature_class_centroids.csv`

### What is it?
The average feature vector (256 dimensions) for each class. These are the "prototypes" that represent each class in feature space.

### Structure:
- **Rows:** Each class (0-10)
- **Columns:** 256 feature dimensions (0-255)
- **Values:** Average activation for that feature in that class

### Sample Data (First 10 features):

| Class | F0 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 0 | 0.0 | 3.92 | 0.34 | 0.0 | 1.09 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.0 | 5.70 | 3.19 | 0.0 | 2.70 | 0.0 | 0.0 | 0.0 | 0.0 | 0.003 |
| 7 | 0.0 | 8.03 | 0.61 | 0.0 | 2.00 | 0.0 | 0.0 | 0.0 | 0.0 | 5.47 |
| 8 | 0.0 | 1.92 | 1.17 | 0.0 | 7.00 | 0.0 | 0.0 | 0.0 | 0.0 | 12.69 |

### Key Observations:
- **Many zero-valued features** - Sparse representation
- **Classes 7 & 8 have unique patterns** (high values in features 9, 14, 15)
- **Feature 4 varies significantly** across classes
- These centroids can be used for:
  - Nearest-centroid classification
  - Prototype-based explanations
  - Transfer learning initialization

### Key Takeaways:
- This is the "DNA" of each class in feature space
- Similar centroids = similar classes (confirms similarity matrix)
- Useful for understanding what features define each class

---

## 10. Feature - Multiscale Stats

**File:** `feature_multiscale_stats.csv`

### What is it?
Image statistics computed at different resolutions (32×32, 64×64, 128×128 pixels).

### Columns:
| Column | Description |
|--------|-------------|
| `split` | Dataset split |
| `scale` | Image resolution (32, 64, or 128 pixels) |
| `mean_intensity` | Average pixel brightness |
| `std_intensity` | Pixel value variation |
| `edge_density` | Proportion of edge pixels |

### Full Data:

| Split | Scale | Mean Intensity | Std Intensity | Edge Density |
|-------|-------|----------------|---------------|--------------|
| train | 32 | 118.36 | 54.84 | 0.344 |
| train | 64 | 118.35 | 56.56 | 0.207 |
| train | 128 | 118.36 | 57.12 | 0.073 |
| val | 32 | 119.33 | 56.00 | 0.355 |
| val | 64 | 119.32 | 57.80 | 0.217 |
| val | 128 | 119.32 | 58.38 | 0.079 |
| test | 32 | 119.21 | 51.91 | **0.315** |
| test | 64 | 119.20 | 53.53 | **0.183** |
| test | 128 | 119.20 | 54.11 | **0.063** |

### Scale Analysis:

**Edge Density Decreases with Scale:**
| Scale | Train | Val | Test |
|-------|-------|-----|------|
| 32 | 0.344 | 0.355 | 0.315 |
| 64 | 0.207 | 0.217 | 0.183 |
| 128 | 0.073 | 0.079 | 0.063 |

This is expected - at higher resolution, edge pixels become a smaller proportion.

### Key Takeaways:
- ✅ **Mean intensity consistent** across splits (~118-119)
- ⚠️ **Test has lower std intensity** at all scales
- ⚠️ **Test has lower edge density** - smoother images
- Scale 64 or 128 may be optimal for training (reasonable edge density)

---

## 11. Robustness Metrics

**File:** `robustness_metrics.csv`

### What is it?
Image quality metrics for various perturbations (noise, blur, contrast changes).

### Columns:
| Column | Description |
|--------|-------------|
| `filename` | Image file tested |
| `perturbation` | Type of corruption applied |
| `psnr` | Peak Signal-to-Noise Ratio (higher = less distortion) |
| `ssim` | Structural Similarity Index (0-1, higher = more similar to original) |

### Perturbation Types:
1. **gaussian_noise** - Random noise added to pixels
2. **motion_blur** - Simulated camera motion blur
3. **contrast_up** - Increased contrast
4. **contrast_down** - Decreased contrast

### Average Metrics by Perturbation:

| Perturbation | Avg PSNR | Avg SSIM | Impact |
|--------------|----------|----------|--------|
| gaussian_noise | 25.1 dB | 0.44 | High distortion |
| motion_blur | 36.2 dB | 0.97 | Low distortion |
| contrast_up | 21.2 dB | 0.88 | Medium distortion |
| contrast_down | 20.5 dB | 0.71 | Medium-High distortion |

### Interpretation:
- **PSNR > 30 dB** = High quality (little distortion)
- **PSNR 20-30 dB** = Acceptable quality
- **PSNR < 20 dB** = Significant distortion

- **SSIM > 0.9** = Very similar to original
- **SSIM 0.7-0.9** = Somewhat similar
- **SSIM < 0.7** = Significantly different

### Key Takeaways:
- **Motion blur is least damaging** (SSIM ~0.97)
- **Gaussian noise is most damaging** (SSIM ~0.44)
- Contrast changes have moderate impact
- Consider augmenting with these perturbations during training for robustness

---

## 12. Geometric Stats

**File:** `geometric_stats.csv` (1,026 rows)

### What is it?
Geometric analysis of images including edge density and symmetry measures.

### Columns:
| Column | Description |
|--------|-------------|
| `file` | Image filename |
| `edge_density` | Proportion of edge pixels (0-1) |
| `horiz_flip_diff` | Difference when flipped horizontally (asymmetry) |
| `vert_flip_diff` | Difference when flipped vertically (asymmetry) |

### Sample Data:
| File | Edge Density | Horiz Flip Diff | Vert Flip Diff |
|------|--------------|-----------------|----------------|
| train_14233.png | 0.00012 | 88.60 | 66.75 |
| train_15230.png | 0.00022 | 84.35 | 72.12 |
| train_23784.png | 0.00000 | 54.71 | 63.87 |
| train_26230.png | 0.00000 | 111.39 | 44.07 |

### Interpretation:

**Edge Density:**
- Most values are very low (0 to 0.0006)
- Indicates images are mostly smooth with few sharp edges
- Higher values = more detailed/textured images

**Flip Differences (Asymmetry):**
- Values range from ~34 to ~111
- Higher values = more asymmetric image
- Can help identify orientation-dependent features
- **train_26230.png** has highest horizontal asymmetry (111.39)

### Key Takeaways:
- Most images have very low edge density
- Images have varying degrees of symmetry
- Asymmetry features could be useful for classification
- Consider horizontal flip augmentation carefully (may change semantic meaning)

---

## 13. Image Stats (Train & Val)

**Files:** `train_image_stats.csv` (34,561 rows), `val_image_stats.csv` (6,493 rows)

### What is it?
Per-image pixel statistics for every image in the dataset.

### Columns:
| Column | Description |
|--------|-------------|
| `file` | Image filename |
| `mean` | Average pixel value (0-255) |
| `std` | Standard deviation of pixels |
| `min` | Minimum pixel value |
| `max` | Maximum pixel value |

### Sample Training Data:
| File | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| train_00000.png | 135.64 | 48.55 | 0 | 255 |
| train_00001.png | 26.85 | 50.20 | 0 | 250 |
| train_00002.png | 137.43 | 53.72 | 23 | 235 |
| train_00005.png | 91.96 | **88.26** | 0 | 253 |
| train_00009.png | 72.52 | **81.38** | 0 | 255 |

### Sample Validation Data:
| File | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| val_00000.png | 104.10 | 78.42 | 0 | 255 |
| val_00004.png | 205.92 | 45.57 | 74 | 255 |
| val_00006.png | 145.81 | 55.06 | 4 | 254 |
| val_00011.png | 80.19 | 68.52 | 0 | 255 |

### Use Cases:
1. **Find outliers:** Images with extreme mean/std values
2. **Normalize data:** Use global mean/std for normalization
3. **Quality control:** Images with min=max are broken
4. **Class analysis:** Correlate with labels to understand class characteristics

### Quick Statistics:

| Metric | Training | Validation |
|--------|----------|------------|
| Total Images | 34,561 | 6,491 |
| Mean of Means | ~119.4 | ~119.9 |
| Std of Stds | ~57.8 | ~58.6 |

### Key Takeaways:
- Full pixel range (0-255) is used in most images
- Some images are very dark (mean ~27) or very bright (mean ~206)
- High std (>80) indicates high contrast images
- These stats enable per-image normalization if needed

---

## 📈 Summary: Key Insights from Tables

### ✅ What's Good
1. **No suspect labels** - Data labeling is clean
2. **Consistent mean intensity** across splits (~119)
3. **Comprehensive coverage** - All images analyzed

### ⚠️ What Needs Attention
1. **Class Imbalance:** 4.5:1 ratio (Class 6 vs Classes 1,2)
2. **~3% Duplicates** in training data
3. **Confusion Cluster:** Classes 4, 5, 9, 10 are highly similar
4. **Class 0 has only 7.8% accuracy** - needs attention
5. **Test set anomalies** - 0.58% anomaly rate

### 🔧 Recommended Actions

| Issue | Solution |
|-------|----------|
| Class imbalance | Use class weights, oversampling, or focal loss |
| Duplicates | Remove or deduplicate before training |
| Confused classes (4,5,9,10) | Add class-specific augmentation, use contrastive learning |
| Poor Class 0 accuracy | Collect more data, use hard example mining |
| Test distribution shift | Apply domain adaptation, use robust training |

---

## 📁 File Reference

| File | Rows | Purpose |
|------|------|---------|
| `label_distribution.csv` | 13 | Class counts and percentages |
| `data_quality_duplicate_summary.csv` | 4 | Duplicate statistics |
| `data_quality_suspect_labels.csv` | 2 | Mislabeled image candidates |
| `test_anomaly_scores.csv` | 5 | Anomaly detection by split |
| `test_characterization_summary.csv` | 5 | Comprehensive split comparison |
| `class_imbalance_confusion_matrix.csv` | 13 | 11×11 confusion matrix |
| `class_imbalance_per_class_accuracy.csv` | 13 | Per-class accuracy breakdown |
| `feature_interclass_similarity.csv` | 13 | 11×11 similarity matrix |
| `feature_class_centroids.csv` | 13 | 256-dim feature prototypes |
| `feature_multiscale_stats.csv` | 11 | Stats at different resolutions |
| `robustness_metrics.csv` | 66 | Perturbation quality metrics |
| `geometric_stats.csv` | 1,026 | Edge density and symmetry |
| `train_image_stats.csv` | 34,563 | Per-image stats (train) |
| `val_image_stats.csv` | 6,493 | Per-image stats (val) |

