# 📊 Analysis Reports Guide

This guide explains every JSON report file in this folder, what each field means, and how to interpret the results for your deep learning project.

---

## Table of Contents

1. [Pipeline Log](#1-pipeline-log)
2. [Label Distribution](#2-label-distribution)
3. [Class Statistics](#3-class-statistics)
4. [Class Imbalance Summary](#4-class-imbalance-summary)
5. [Image Summaries](#5-image-summaries)
6. [Missing Files Reports](#6-missing-files-reports)
7. [Data Quality Summary](#7-data-quality-summary)
8. [Duplicate Detection](#8-duplicate-detection)
9. [Distribution Shifts](#9-distribution-shifts)
10. [Test Characterization Shift Metrics](#10-test-characterization-shift-metrics)
11. [Frequency Analysis Metrics](#11-frequency-analysis-metrics)
12. [Latent Structure](#12-latent-structure)
13. [Feature Exploration Summary](#13-feature-exploration-summary)
14. [Feature Training History](#14-feature-training-history)
15. [Robustness - Adversarial Results](#15-robustness---adversarial-results)
16. [Robustness - Adversarial Training](#16-robustness---adversarial-training)
17. [Eval Summary Baseline](#17-eval-summary-baseline)

---

## 1. Pipeline Log

**File:** `pipeline.log`

### What is it?
A text log file that records when each analysis step was run during the pipeline execution.

### Content:
```
2025-10-02 22:17:07,803 [INFO] Starting analysis pipeline
2025-10-02 22:17:07,804 [INFO] Running label analysis
2025-10-02 22:17:08,026 [INFO] Running image statistics
2025-10-02 22:17:46,084 [INFO] Running quality checks
2025-10-02 22:17:48,992 [INFO] Running robustness probes
2025-10-02 22:17:51,265 [INFO] Running latent structure analysis
2025-10-02 22:17:58,059 [INFO] Running geometric analysis
2025-10-02 22:17:59,965 [INFO] Pipeline finished successfully
```

### Key Takeaways:
- **Pipeline Duration:** ~53 seconds (from 22:17:07 to 22:17:59)
- **Steps Executed:** Label analysis → Image stats → Quality checks → Robustness → Latent structure → Geometric analysis
- **Status:** ✅ Successfully completed

---

## 2. Label Distribution

**File:** `label_distribution.json`

### What is it?
Shows how many images belong to each class (0-10) in your training and validation sets.

### Content Breakdown:

#### Training Set
| Class | Count | Proportion |
|-------|-------|------------|
| 0 | 1,956 | 5.66% |
| 1 | 1,390 | 4.02% |
| 2 | 1,357 | 3.93% |
| 3 | 1,474 | 4.26% |
| 4 | 3,963 | 11.47% |
| 5 | 3,817 | 11.04% |
| **6** | **6,164** | **17.84%** (largest) |
| 7 | 3,919 | 11.34% |
| 8 | 3,929 | 11.37% |
| 9 | 3,031 | 8.77% |
| 10 | 3,561 | 10.30% |
| **Total** | **34,561** | 100% |

#### Validation Set
| Class | Count | Proportion |
|-------|-------|------------|
| 0 | 321 | 4.95% |
| 1 | 233 | 3.59% |
| 2 | 225 | 3.47% |
| 3 | 392 | 6.04% |
| 4 | 568 | 8.75% |
| 5 | 637 | 9.81% |
| 6 | 1,033 | 15.91% |
| 7 | 1,033 | 15.91% |
| 8 | 1,009 | 15.54% |
| 9 | 529 | 8.15% |
| 10 | 511 | 7.87% |
| **Total** | **6,491** | 100% |

### Key Takeaways:
- **Imbalanced Dataset:** Class 6 has the most samples (~18% in train), while classes 1, 2, 3 have the fewest (~4%)
- **Imbalance Ratio:** Roughly **4.5:1** between largest and smallest classes
- **Why it matters:** Your model may perform better on majority classes. Consider:
  - Using class weights during training
  - Oversampling minority classes
  - Using focal loss

---

## 3. Class Statistics

**File:** `class_statistics.json`

### What is it?
Detailed pixel-level statistics for each class, showing the visual characteristics of images in each category.

### Key Fields Explained:

For each class (0-10), you get:
- **`mean_mean`**: Average pixel brightness across all images in that class
- **`mean_std`**: Standard deviation of pixel values (how much contrast/variation)
- **`mean_min`**: Minimum average pixel value found
- **`mean_max`**: Maximum average pixel value found

### Visual Characteristics by Class:

| Class | Avg Brightness (Train) | Std Dev | Interpretation |
|-------|------------------------|---------|----------------|
| 0 | 109.96 | 35.14 | Medium brightness, moderate contrast |
| 1 | 182.43 | 45.82 | **Bright images**, high variation |
| 2 | 180.14 | 45.51 | **Bright images**, high variation |
| 3 | 122.17 | **81.91** | Medium brightness, **very high contrast** |
| 4 | 123.88 | 53.37 | Medium brightness |
| 5 | 134.27 | 52.61 | Medium-high brightness |
| 6 | 131.44 | 54.88 | Medium-high brightness |
| 7 | 75.80 | **76.47** | **Dark images**, high contrast |
| 8 | 67.93 | **75.02** | **Darkest images**, high contrast |
| 9 | 134.97 | 49.80 | Medium-high brightness |
| 10 | 125.41 | 52.72 | Medium brightness |

### Key Takeaways:
- **Classes 1 & 2:** Brightest images (~180-210 mean pixel value)
- **Classes 7 & 8:** Darkest images (~68-77 mean pixel value)
- **Classes 3, 7, 8:** Highest contrast (std ~75-82)
- **Why it matters:** Different classes have distinct visual signatures that the model can learn

---

## 4. Class Imbalance Summary

**File:** `class_imbalance_summary.json`

### Content:
```json
{
  "overall_accuracy": 0.5202588199044831,
  "train_sampled_total": 16221,
  "val_total": 6491,
  "max_train_per_class": 1500
}
```

### Fields Explained:
- **`overall_accuracy`**: 52.03% - Baseline accuracy (likely from a simple model or random baseline)
- **`train_sampled_total`**: 16,221 - Number of training samples used in this analysis
- **`val_total`**: 6,491 - Total validation samples
- **`max_train_per_class`**: 1,500 - Maximum samples taken per class for balanced evaluation

### Key Takeaways:
- **52% baseline accuracy** for 11 classes is slightly better than random guessing (~9%)
- The analysis capped each class at 1,500 samples to evaluate fairly

---

## 5. Image Summaries

**Files:** `train_image_summary.json`, `val_image_summary.json`

### Training Set Summary:
```json
{
  "dataset": "train",
  "mean_of_means": 119.39705833970214,
  "std_of_means": 39.07733255186564,
  "overall_std": 57.82924329512946,
  "min_pixel": 0,
  "max_pixel": 255
}
```

### Validation Set Summary:
```json
{
  "dataset": "val",
  "mean_of_means": 119.89671667131616,
  "std_of_means": 38.83622185983735,
  "overall_std": 58.55688908691124,
  "min_pixel": 0,
  "max_pixel": 255
}
```

### Fields Explained:
- **`mean_of_means`**: Average brightness across all images (~119, which is mid-gray)
- **`std_of_means`**: How much image brightness varies (~39)
- **`overall_std`**: Average pixel variation within images (~58)
- **`min_pixel`/`max_pixel`**: Full 0-255 range is used

### Key Takeaways:
- ✅ **Train and Val are similar** - Mean values are nearly identical (119.4 vs 119.9)
- ✅ **No normalization issues** - Full pixel range is used
- ✅ **Good for training** - Similar distributions mean validation will be representative

---

## 6. Missing Files Reports

**Files:** `train_missing_files.json`, `val_missing_files.json`

### Content:
```json
{
  "split": "train",
  "missing_count": 0,
  "missing_files": []
}
```

```json
{
  "split": "val",
  "missing_count": 0,
  "missing_files": []
}
```

### Key Takeaways:
- ✅ **No missing files** - All images referenced in labels exist
- ✅ **Data integrity verified** - No broken file references

---

## 7. Data Quality Summary

**File:** `data_quality_summary.json`

### Content:
```json
{
  "duplicates": [
    {
      "split": "train",
      "total_images": 34561,
      "duplicate_pairs": 921,
      "unique_hashes": 33525
    },
    {
      "split": "val",
      "total_images": 6491,
      "duplicate_pairs": 134,
      "unique_hashes": 6343
    }
  ],
  "suspect_count": 0,
  "suspect_threshold": 0.2
}
```

### Fields Explained:
- **`duplicate_pairs`**: Groups of identical images found
- **`unique_hashes`**: Number of truly unique images
- **`suspect_count`**: Near-duplicate or suspicious images (threshold 0.2)

### Calculations:

**Training Set:**
- Total: 34,561 images
- Duplicate groups: 921
- Unique images: 33,525
- **Duplicate rate:** ~3% of images have duplicates

**Validation Set:**
- Total: 6,491 images
- Duplicate groups: 134
- Unique images: 6,343
- **Duplicate rate:** ~2% of images have duplicates

### Key Takeaways:
- ⚠️ **~3% duplicates in training** - Some images appear multiple times
- Consider removing duplicates to prevent data leakage
- Duplicates might inflate accuracy if the same image appears in train AND val

---

## 8. Duplicate Detection

**Files:** `data_quality_duplicates_train.json`, `data_quality_duplicates_val.json`

### What is it?
Lists all duplicate image groups with their file paths.

### Example (Training):
```json
{
  "split": "train",
  "duplicate_groups": 921,
  "duplicate_examples": {
    "997a60a12e2cb73e": [
      "train/images_train/train_00014.png",
      "train/images_train/train_26030.png"
    ],
    "d5d5d330c81ae639": [
      "train/images_train/train_00036.png",
      "train/images_train/train_15827.png",
      "train/images_train/train_31786.png"
    ]
  }
}
```

### Fields Explained:
- **Hash key** (e.g., `997a60a12e2cb73e`): Perceptual hash of the image
- **Array of paths**: All images that share this hash (are duplicates)

### Key Takeaways:
- Some images appear **2-3 times** in the dataset
- Use these lists to deduplicate your training data
- Consider keeping only one copy of each unique image

---

## 9. Distribution Shifts

**File:** `distribution_shifts.json`

### Content:
```json
{
  "distribution_shift_metrics": {
    "train_vs_val": {
      "pixel_kl": 0.007268887328846223,
      "pixel_wasserstein": 0.012955969082579311,
      "edge_mean_delta": 0.003097517415881157,
      "lbp_kl": 0.0006510564021210889
    },
    "train_vs_test": {
      "pixel_kl": 0.14266392855389706,
      "pixel_wasserstein": 0.015934481577970644,
      "edge_mean_delta": 0.0027028732001781464,
      "lbp_kl": 0.349278428322088
    },
    "val_vs_test": {
      "pixel_kl": 0.16228791203377035,
      "pixel_wasserstein": 0.023635722678048303,
      "edge_mean_delta": 0.00039464421570301056,
      "lbp_kl": 0.35390446099309164
    }
  },
  "class_statistics_overview": {
    "train_total": 34561,
    "val_total": 6491
  },
  "class_weights_present": true
}
```

### Metrics Explained:

| Metric | What it Measures | Lower = Better |
|--------|------------------|----------------|
| **pixel_kl** | KL divergence of pixel histograms | ✅ |
| **pixel_wasserstein** | Wasserstein distance (Earth Mover's Distance) | ✅ |
| **edge_mean_delta** | Difference in edge detection responses | ✅ |
| **lbp_kl** | KL divergence of Local Binary Patterns (texture) | ✅ |

### Comparison Table:

| Comparison | Pixel KL | Wasserstein | Edge Delta | LBP KL |
|------------|----------|-------------|------------|--------|
| Train ↔ Val | **0.007** | 0.013 | 0.003 | **0.0007** |
| Train ↔ Test | 0.143 | 0.016 | 0.003 | 0.349 |
| Val ↔ Test | 0.162 | 0.024 | 0.0004 | 0.354 |

### Key Takeaways:
- ✅ **Train & Val are very similar** - KL divergence ~0.007 (nearly identical)
- ⚠️ **Test set is different** - KL divergence ~0.14-0.16 (20x larger shift)
- ⚠️ **Texture shift (LBP)** - Test has significantly different textures (0.35 vs 0.0007)
- **Why it matters:** Model may perform worse on test due to domain shift

---

## 10. Test Characterization Shift Metrics

**File:** `test_characterization_shift_metrics.json`

### Content:
```json
{
  "train_vs_val": {
    "pixel_kl": 0.007268887328846223,
    "pixel_wasserstein": 0.012955969082579311,
    "edge_mean_delta": 0.003097517415881157,
    "lbp_kl": 0.0006510564021210889
  },
  "train_vs_test": {
    "pixel_kl": 0.14266392855389706,
    "pixel_wasserstein": 0.015934481577970644,
    "edge_mean_delta": 0.0027028732001781464,
    "lbp_kl": 0.349278428322088
  },
  "val_vs_test": {
    "pixel_kl": 0.16228791203377035,
    "pixel_wasserstein": 0.023635722678048303,
    "edge_mean_delta": 0.00039464421570301056,
    "lbp_kl": 0.35390446099309164
  }
}
```

### Key Takeaways:
Same as Distribution Shifts - this file focuses specifically on test set characterization:
- ⚠️ **Test distribution differs significantly from training**
- The texture patterns (LBP) show the largest shift
- Consider domain adaptation techniques for better test performance

---

## 11. Frequency Analysis Metrics

**File:** `robustness_frequency_metrics.json`

### Content:
```json
{
  "train": {
    "split": "train",
    "samples": 512,
    "mean_low_freq": 73.0646858625114,
    "mean_high_freq": 0.646524703304749,
    "high_to_low_ratio": 0.008946219597920817
  },
  "val": {
    "split": "val",
    "samples": 512,
    "mean_low_freq": 74.88701258599758,
    "mean_high_freq": 0.6534784882096574,
    "high_to_low_ratio": 0.008833942682739751
  },
  "test": {
    "split": "test",
    "samples": 512,
    "mean_low_freq": 71.19235471636057,
    "mean_high_freq": 2.402956433943473,
    "high_to_low_ratio": 0.03491979830004045
  }
}
```

### Fields Explained:
- **`mean_low_freq`**: Average low-frequency content (smooth areas, overall structure)
- **`mean_high_freq`**: Average high-frequency content (edges, fine details, noise)
- **`high_to_low_ratio`**: Ratio of detail to structure

### Comparison:

| Split | Low Freq | High Freq | Ratio |
|-------|----------|-----------|-------|
| Train | 73.06 | 0.65 | 0.009 |
| Val | 74.89 | 0.65 | 0.009 |
| **Test** | 71.19 | **2.40** | **0.035** |

### Key Takeaways:
- ⚠️ **Test images have 4x more high-frequency content**
- Test images are **sharper/noisier** than train/val
- This could be due to:
  - Different camera/capture settings
  - Different compression
  - More detailed images
- **Why it matters:** Models trained on smoother images may struggle with sharper test images

---

## 12. Latent Structure

**File:** `latent_structure.json`

### Content:
```json
{
  "method": "PCA->tSNE",
  "explained_variance": [
    0.3281986117362976,
    0.07902063429355621,
    0.06733036041259766,
    0.06143045425415039,
    0.0276905857026577,
    0.025507405400276184,
    0.020137546584010124,
    0.01795702986419201,
    0.01617165096104145,
    0.014062561094760895
  ]
}
```

### Fields Explained:
- **`method`**: PCA followed by t-SNE for dimensionality reduction
- **`explained_variance`**: How much each PCA component captures

### Cumulative Variance Explained:

| Component | Individual | Cumulative |
|-----------|------------|------------|
| PC1 | 32.82% | 32.82% |
| PC2 | 7.90% | 40.72% |
| PC3 | 6.73% | 47.45% |
| PC4 | 6.14% | 53.59% |
| PC5 | 2.77% | 56.36% |
| PC6 | 2.55% | 58.91% |
| PC7 | 2.01% | 60.92% |
| PC8 | 1.80% | 62.72% |
| PC9 | 1.62% | 64.34% |
| PC10 | 1.41% | 65.75% |

### Key Takeaways:
- **First component explains ~33%** - There's one dominant pattern in your data
- **Top 4 components explain ~54%** - Reasonably good dimensionality reduction
- **Top 10 components explain ~66%** - Still significant unexplained variance
- Check `latent_tsne.png` in figures to see how classes cluster

---

## 13. Feature Exploration Summary

**File:** `feature_exploration_summary.json`

### Content:
```json
{
  "multiscale_samples": 1000,
  "train_samples": 6000,
  "val_samples": 2000,
  "epochs": 3,
  "final_val_accuracy": 0.927
}
```

### Fields Explained:
- **`multiscale_samples`**: 1,000 images used for multi-scale feature analysis
- **`train_samples`**: 6,000 images used for feature training
- **`val_samples`**: 2,000 images for feature validation
- **`epochs`**: 3 training epochs
- **`final_val_accuracy`**: **92.7% accuracy** achieved

### Key Takeaways:
- ✅ **92.7% validation accuracy** with feature-based model
- Achieved in just 3 epochs
- This suggests features are highly discriminative for your classes

---

## 14. Feature Training History

**File:** `feature_training_history.json`

### Content:
```json
{
  "history": {
    "train_loss": [
      1.148596003373464,
      0.3897874044577281,
      0.2525862370332082
    ],
    "val_loss": [
      1.5619534740447998,
      0.18675406217575075,
      0.23895027351379394
    ],
    "val_acc": [
      0.5095,
      0.9555,
      0.927
    ]
  }
}
```

### Training Progression:

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1 | 1.149 | 1.562 | 50.95% |
| 2 | 0.390 | 0.187 | **95.55%** |
| 3 | 0.253 | 0.239 | 92.70% |

### Key Takeaways:
- **Rapid convergence** - Loss drops dramatically after epoch 1
- **Peak accuracy at epoch 2** - 95.55% (higher than epoch 3!)
- ⚠️ **Slight overfitting** - Val loss increased in epoch 3
- Consider early stopping at epoch 2

---

## 15. Robustness - Adversarial Results

**File:** `robustness_adversarial_results.json`

### Content:
```json
{
  "clean_accuracy": 0.9595,
  "attacks": [
    {"attack": "fgsm", "epsilon": 0.01, "accuracy": 0.8455},
    {"attack": "fgsm", "epsilon": 0.03, "accuracy": 0.72},
    {"attack": "fgsm", "epsilon": 0.07, "accuracy": 0.477},
    {"attack": "pgd", "epsilon": 0.03, "step_size": 0.0075, "steps": 10, "accuracy": 0.1715},
    {"attack": "pgd", "epsilon": 0.07, "step_size": 0.0175, "steps": 10, "accuracy": 0.0235}
  ]
}
```

### Attack Types Explained:

**FGSM (Fast Gradient Sign Method):**
- Single-step attack
- Adds perturbation in the direction of the gradient
- `epsilon` controls perturbation magnitude (0.01 = 1%, 0.07 = 7% of pixel range)

**PGD (Projected Gradient Descent):**
- Multi-step iterative attack (10 steps here)
- More powerful than FGSM
- `step_size` is the perturbation per step

### Accuracy Under Attack:

| Attack | Epsilon | Accuracy | Drop from Clean |
|--------|---------|----------|-----------------|
| Clean | - | 95.95% | - |
| FGSM | 0.01 | 84.55% | -11.4% |
| FGSM | 0.03 | 72.00% | -23.95% |
| FGSM | 0.07 | 47.70% | -48.25% |
| PGD | 0.03 | 17.15% | -78.8% |
| PGD | 0.07 | **2.35%** | -93.6% |

### Key Takeaways:
- ✅ **95.95% clean accuracy** - Model performs well on unperturbed images
- ⚠️ **Highly vulnerable to adversarial attacks**
- PGD with ε=0.07 reduces accuracy to just 2.35%
- FGSM attacks are less effective than PGD
- Consider adversarial training to improve robustness

---

## 16. Robustness - Adversarial Training

**File:** `robustness_adversarial_training.json`

### Content:
```json
{
  "history": {
    "train_loss": [
      1.148596003373464,
      0.3897874044577281,
      0.2525862370332082,
      0.17969076001644135,
      0.1285493994951248
    ],
    "val_loss": [
      1.5619534740447998,
      0.18675406217575075,
      0.23895027351379394,
      0.09573566579818725,
      0.13478737592697143
    ],
    "val_acc": [
      0.5095,
      0.9555,
      0.927,
      0.97,
      0.9595
    ]
  },
  "clean_accuracy": 0.9595
}
```

### Extended Training Progression:

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1 | 1.149 | 1.562 | 50.95% |
| 2 | 0.390 | 0.187 | 95.55% |
| 3 | 0.253 | 0.239 | 92.70% |
| 4 | 0.180 | **0.096** | **97.00%** |
| 5 | 0.129 | 0.135 | 95.95% |

### Key Takeaways:
- **Best performance at epoch 4** - 97% accuracy, lowest val loss (0.096)
- Epoch 5 shows slight overfitting (val loss increased)
- Final clean accuracy: 95.95%

---

## 17. Eval Summary Baseline

**File:** `eval_summary_baseline.json`

### Content:
```json
{
  "timestamp": "2025-10-09T23:53:51.980646Z",
  "trained_this_run": false,
  "epochs": 0,
  "val_final_loss": null,
  "val_final_accuracy": null
}
```

### Fields Explained:
- **`timestamp`**: When the evaluation was run
- **`trained_this_run`**: Whether training occurred (false = used pre-trained)
- **`epochs`**: Number of training epochs (0 = no training)
- **`val_final_loss`/`val_final_accuracy`**: null because no training occurred

### Key Takeaways:
- This was a baseline evaluation run without training
- Used to establish a starting point before experiments

---

## 📈 Summary: Key Findings

### ✅ What's Good
1. **No missing files** - Data integrity verified
2. **Train/Val distributions match** - Similar pixel statistics
3. **92.7-97% validation accuracy** achieved
4. **Clear class separation** - 33% variance explained by first component

### ⚠️ What Needs Attention
1. **Class Imbalance** - 4.5:1 ratio between largest/smallest classes
2. **~3% Duplicates** - 921 duplicate groups in training
3. **Test Distribution Shift** - Significant difference from train/val
4. **Adversarial Vulnerability** - PGD attack drops accuracy to 2.35%
5. **Higher frequency in test** - Test images are sharper/noisier

### 🔧 Recommended Actions
1. Remove or address duplicate images
2. Use class weights or oversampling for imbalanced classes
3. Apply domain adaptation for test set
4. Consider adversarial training for robustness
5. Add data augmentation with frequency variations

---

## 📁 File Reference

| File | Purpose |
|------|---------|
| `pipeline.log` | Execution log |
| `label_distribution.json` | Class counts and proportions |
| `class_statistics.json` | Per-class pixel statistics |
| `class_imbalance_summary.json` | Imbalance analysis |
| `train_image_summary.json` | Training set overview |
| `val_image_summary.json` | Validation set overview |
| `train_missing_files.json` | Missing file check (train) |
| `val_missing_files.json` | Missing file check (val) |
| `data_quality_summary.json` | Duplicate summary |
| `data_quality_duplicates_train.json` | Training duplicates list |
| `data_quality_duplicates_val.json` | Validation duplicates list |
| `distribution_shifts.json` | Distribution comparison |
| `test_characterization_shift_metrics.json` | Test shift metrics |
| `robustness_frequency_metrics.json` | Frequency analysis |
| `latent_structure.json` | PCA/t-SNE analysis |
| `feature_exploration_summary.json` | Feature model summary |
| `feature_training_history.json` | Feature training metrics |
| `robustness_adversarial_results.json` | Adversarial attack results |
| `robustness_adversarial_training.json` | Extended training history |
| `eval_summary_baseline.json` | Baseline evaluation info |

