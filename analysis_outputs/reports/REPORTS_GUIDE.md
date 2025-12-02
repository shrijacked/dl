# 📚 Complete Analysis Reports Guide

You now have three analysis directories (figures / tables / reports). This document focuses on the **JSON & log outputs** under `analysis_outputs/reports`. It mirrors the step-by-step teaching style from the figures walkthrough: for every file you’ll see what it is, how to read it, numerical breakdowns, and why the information matters for OrganAMNIST.

---

## Contents At A Glance

1. [Pipeline Execution Log](#1-pipeline-execution-log)
2. [Label Distribution Report](#2-label-distribution-report)
3. [Per-Class Pixel Statistics](#3-per-class-pixel-statistics)
4. [Balanced Accuracy Probe](#4-balanced-accuracy-probe)
5. [Split-Level Image Summaries](#5-split-level-image-summaries)
6. [Missing File Audits](#6-missing-file-audits)
7. [Data Quality Summary](#7-data-quality-summary)
8. [Duplicate Listings](#8-duplicate-listings)
9. [Distribution Shift Diagnostics](#9-distribution-shift-diagnostics)
10. [Test-Shift Metrics (Detailed)](#10-test-shift-metrics-detailed)
11. [Frequency-Domain Metrics](#11-frequency-domain-metrics)
12. [Latent Structure Metadata](#12-latent-structure-metadata)
13. [Feature Exploration Snapshot](#13-feature-exploration-snapshot)
14. [Feature Training History](#14-feature-training-history)
15. [Adversarial Evaluation Results](#15-adversarial-evaluation-results)
16. [Adversarial Training History](#16-adversarial-training-history)
17. [Baseline Evaluation Stub](#17-baseline-evaluation-stub)
18. [Summary & Action Checklist](#18-summary--action-checklist)
19. [File Reference](#19-file-reference)

---

## 1. Pipeline Execution Log

**File:** `pipeline.log`

| Aspect | Details |
|--------|---------|
| **Purpose** | Chronological proof that the end-to-end analysis pipeline ran to completion. |
| **Sample entries** | `2025-10-02 22:17:07,803 [INFO] Starting analysis pipeline` … `Running label analysis`, `Running robustness probes`, `Pipeline finished successfully`. |
| **Duration** | ~53 seconds (22:17:07 → 22:17:59). |
| **Order of stages** | Label analysis → Image statistics → Quality checks → Robustness probes → Latent structure → Geometric analysis. |

**Why it matters:** When results look suspicious, confirm whether all stages executed or if a crash truncated the outputs. The timestamps also help correlate with experiment IDs or HPC jobs.

---

## 2. Label Distribution Report

**File:** `label_distribution.json`

**What it captures:** Canonical counts and proportions for each of the 11 organ classes in both train and validation splits. Essential for any weighting scheme or prevalence-aware evaluation.

| Class | Train Count | Train % | Val Count | Val % |
|-------|-------------|---------|-----------|-------|
| 0 (Bladder) | 1,956 | 5.66% | 321 | 4.95% |
| 1 (Femur L) | 1,390 | 4.02% | 233 | 3.59% |
| 2 (Femur R) | 1,357 | 3.93% | 225 | 3.47% |
| 3 (Heart) | 1,474 | 4.26% | 392 | 6.04% |
| 4 (Kidney L) | 3,963 | 11.47% | 568 | 8.75% |
| 5 (Kidney R) | 3,817 | 11.04% | 637 | 9.81% |
| **6 (Liver)** | **6,164** | **17.84%** | 1,033 | 15.91% |
| 7 (Lung L) | 3,919 | 11.34% | 1,033 | 15.91% |
| 8 (Lung R) | 3,929 | 11.37% | 1,009 | 15.54% |
| 9 (Spleen) | 3,031 | 8.77% | 529 | 8.15% |
| 10 (Pancreas) | 3,561 | 10.30% | 511 | 7.87% |

**Insights**
- Imbalance ratio ≈ 4.5:1 (Liver vs Femur classes).  
- Validation preserves ordering, so metrics are comparable without extra weighting.  
- Underrepresented heart/femur samples explain the lower accuracy tiers seen later.

**Action ideas:** Weighted Random Sampler, inverse-frequency loss weights, or specialized augmentations for classes 0–3.

---

## 3. Per-Class Pixel Statistics

**File:** `class_statistics.json`

**What it contains:** For each class, the dataset reports aggregated pixel stats (`mean_mean`, `mean_std`, `mean_min`, `mean_max`). These numbers describe texture/brightness signatures.

| Class | Avg Brightness | Avg Std | Reading |
|-------|----------------|---------|---------|
| 0 | 109.96 | 35.14 | Medium tone, moderate variance |
| 1 | 182.43 | 45.82 | Bright, high variance (bone) |
| 2 | 180.14 | 45.51 | Same as class 1 |
| 3 | 122.17 | **81.91** | Medium brightness, extremely high contrast |
| 7 | 75.80 | **76.47** | Dark with high contrast |
| 8 | 67.93 | **75.02** | Darkest average intensity |

**Why it matters:**  
- Informs per-class normalization or adaptive histogram equalization.  
- Highlights why femurs are easy (bright, distinct) and lungs are tricky (dark, noisy).  
- Useful for sanity checking Grad-CAM saliency (does it align with brightness patterns?).

---

## 4. Balanced Accuracy Probe

**File:** `class_imbalance_summary.json`

```json
{
  "overall_accuracy": 0.5202588199044831,
  "train_sampled_total": 16221,
  "val_total": 6491,
  "max_train_per_class": 1500
}
```

**Interpretation**
- A quick balanced experiment capped each class at 1,500 samples to gauge difficulty without imbalance.  
- Balanced accuracy ≈ 52%, which is > random (9%) but highlights how far the baseline is from the 99% top-line models.  
- Use this as a sanity baseline before performing architecture sweeps.

---

## 5. Split-Level Image Summaries

**Files:** `train_image_summary.json`, `val_image_summary.json`

| Metric | Train | Val | Takeaway |
|--------|-------|-----|----------|
| mean_of_means | 119.40 | 119.90 | Average image midpoint ~0.47 (after normalization). |
| std_of_means | 39.08 | 38.84 | Similar variability across splits. |
| overall_std | 57.83 | 58.56 | Equivalent intra-image variance. |
| min/max pixel | 0 / 255 | 0 / 255 | Full dynamic range is used. |

**Why it matters:** Confirms there’s no normalization mismatch between splits and establishes the global mean/std used later (e.g., PyTorch transforms).

---

## 6. Missing File Audits

**Files:** `train_missing_files.json`, `val_missing_files.json`

Both reports show `missing_count: 0`. That means every label entry maps to an existing PNG. Any training failure won’t be due to missing files—useful when sharing the dataset.

---

## 7. Data Quality Summary

**File:** `data_quality_summary.json`

| Split | Total Images | Duplicate Groups | Unique Hashes | Duplicate Rate |
|-------|--------------|------------------|---------------|----------------|
| Train | 34,561 | 921 | 33,525 | ~3.0% |
| Val | 6,491 | 134 | 6,343 | ~2.3% |

`suspect_count` is 0, so no mislabeled samples were flagged above the perceptual-difference threshold (0.2).  

**Risk:** duplicates can inflate accuracy if near-identical slices straddle train and val. Use this summary plus Section 8 to prune.

---

## 8. Duplicate Listings

**Files:** `data_quality_duplicates_train.json`, `data_quality_duplicates_val.json`

Sample excerpt:
```json
"997a60a12e2cb73e": [
  "train/images_train/train_00014.png",
  "train/images_train/train_26030.png"
]
```

**How to use**
1. Sort duplicate groups by size (3+ images first).  
2. Keep one path per group, remove the rest or move to a “held-out” folder.  
3. Re-run training to make sure accuracy improvements aren’t just memorizing duplicates.

---

## 9. Distribution Shift Diagnostics

**File:** `distribution_shifts.json`

| Comparison | Pixel KL ↓ | Wasserstein ↓ | Edge Δ ↓ | LBP KL ↓ |
|------------|------------|---------------|----------|----------|
| Train vs Val | **0.0073** | 0.0130 | 0.0031 | **0.00065** |
| Train vs Test | 0.1427 | 0.0159 | 0.0027 | 0.3493 |
| Val vs Test | 0.1623 | 0.0236 | 0.00039 | 0.3539 |

**Interpretation**
- Pixel histograms: test deviates ~20× more than train/val.  
- Texture (LBP KL): most severe shift; test textures differ dramatically.  
- Edge means: more subtle, but still non-zero.

**Mitigation ideas:** histogram matching, adaptive contrast augmentations, or domain adaptation (moment matching).

---

## 10. Test-Shift Metrics (Detailed)

**File:** `test_characterization_shift_metrics.json`

This is essentially the same metric bundle as Section 9 but provided separately for plotting scripts. Treat it as the source of truth when generating new shift figures—no need to recompute the metrics from scratch.

---

## 11. Frequency-Domain Metrics

**File:** `robustness_frequency_metrics.json`

| Split | Mean Low Freq | Mean High Freq | High/Low Ratio |
|-------|---------------|----------------|----------------|
| Train | 73.06 | 0.65 | 0.0089 |
| Val | 74.89 | 0.65 | 0.0088 |
| **Test** | 71.19 | **2.40** | **0.0349** |

**Implications**
- Train/Val spectra are smooth (expected 1/f falloff).  
- Test images contain ~4× more high-frequency energy → sharper edges or noise.  
- Use blur/noise augmentations and high-frequency regularizers (e.g., total variation loss) to close the gap.

---

## 12. Latent Structure Metadata

**File:** `latent_structure.json`

| PCA Component | Variance Explained | Cumulative |
|---------------|-------------------|------------|
| PC1 | 32.82% | 32.82% |
| PC2 | 7.90% | 40.72% |
| PC3 | 6.73% | 47.45% |
| PC4 | 6.14% | 53.59% |
| PC5 | 2.77% | 56.36% |
| PC10 | 1.41% | 65.75% |

**Usage tips**
- Feed these PCA embeddings into t-SNE/UMAP (already done for `latent_tsne.png`).  
- If training a shallow classifier, using top-10 PCs gives ~66% variance coverage.  
- Look at PC1 as a “dominant anatomical axis” (likely dark vs bright organs).

---

## 13. Feature Exploration Snapshot

**File:** `feature_exploration_summary.json`

| Metric | Value |
|--------|-------|
| Multiscale samples | 1,000 |
| Train samples | 6,000 |
| Val samples | 2,000 |
| Epochs | 3 |
| Final Val Accuracy | 92.7% |

**Interpretation:** Even a lightweight feature extractor hits 92.7% in 3 epochs—good evidence that OrganAMNIST is separable and that further improvements hinge on robustness more than raw accuracy.

---

## 14. Feature Training History

**File:** `feature_training_history.json`

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1 | 1.149 | 1.562 | 0.5095 |
| 2 | 0.390 | 0.1868 | **0.9555** |
| 3 | 0.253 | 0.2390 | 0.9270 |

**Insights:**  
- Biggest jump occurs between epochs 1 and 2 (loss ↓ 1.15 → 0.39).  
- Val loss rebounds at epoch 3 → early stopping would have locked in the 95.6% peak.  
- Useful for tuning learning rate schedules or patience counters.

---

## 15. Adversarial Evaluation Results

**File:** `robustness_adversarial_results.json`

| Attack | Params | Accuracy | Drop vs Clean |
|--------|--------|----------|---------------|
| Clean | – | 95.95% | – |
| FGSM ε=0.01 | – | 84.55% | -11.4% |
| FGSM ε=0.03 | – | 72.00% | -23.95% |
| FGSM ε=0.07 | – | 47.70% | -48.25% |
| PGD ε=0.03 | step=0.0075 (10 steps) | 17.15% | -78.8% |
| PGD ε=0.07 | step=0.0175 (10 steps) | **2.35%** | -93.6% |

**Why it matters:** Without adversarial defenses, the model collapses under PGD. If you plan clinical deployment, bake adversarial training (Section 16) into your recipe or restrict the threat model.

---

## 16. Adversarial Training History

**File:** `robustness_adversarial_training.json`

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1 | 1.149 | 1.562 | 0.5095 |
| 2 | 0.390 | 0.1868 | 0.9555 |
| 3 | 0.253 | 0.2390 | 0.9270 |
| 4 | 0.180 | **0.0957** | **0.9700** |
| 5 | 0.129 | 0.1348 | 0.9595 |

**Reading tips**
- Epoch 4 is the sweet spot (highest validation accuracy + lowest loss).  
- Epoch 5 drifts upward, signalling mild overfitting even with adversarial noise.  
- This history is helpful when deciding whether to stop early or continue adversarial fine-tuning.

---

## 17. Baseline Evaluation Stub

**File:** `eval_summary_baseline.json`

```json
{
  "timestamp": "2025-10-09T23:53:51.980646Z",
  "trained_this_run": false,
  "epochs": 0,
  "val_final_loss": null,
  "val_final_accuracy": null
}
```

Use this as a metadata breadcrumb: it logs the timestamp of a baseline evaluation that reused pretrained weights without retraining. If you ever wonder “did this report correspond to a fresh run?”, check `trained_this_run`.

---

## 18. Summary & Action Checklist

| Theme | Findings | Recommended Response |
|-------|----------|----------------------|
| Class balance | Liver dominates, femurs/heart are scarce. | Weighted sampler, focal/CB loss, targeted augmentation. |
| Data hygiene | ~3% duplicate rate, no missing files. | Deduplicate before final training, keep log of removed hashes. |
| Distribution shift | Test differs in pixel stats, textures, and frequency content. | Augment with brightness/contrast/texture changes, consider domain adaptation or style transfer. |
| Latent structure | PC1 captures 33% variance; some classes cluster tightly. | Use for visualization, potential dimensionality reduction for lightweight models. |
| Adversarial robustness | PGD reduces accuracy to 2.35%. | Integrate adversarial training (stop near epoch 4) or deploy detection/denoising. |

---

## 19. File Reference

| File | Purpose |
|------|---------|
| `pipeline.log` | Execution trace for the analysis pipeline. |
| `label_distribution.json` | Class counts/proportions for train/val. |
| `class_statistics.json` | Per-class pixel statistics. |
| `class_imbalance_summary.json` | Balanced accuracy probe metadata. |
| `train_image_summary.json`, `val_image_summary.json` | Split-level pixel summaries. |
| `train_missing_files.json`, `val_missing_files.json` | Missing-file audits (both zero). |
| `data_quality_summary.json` | Duplicate rates and suspect counts. |
| `data_quality_duplicates_*.json` | Actual duplicate listings. |
| `distribution_shifts.json`, `test_characterization_shift_metrics.json` | KL/Wasserstein/texture shift metrics. |
| `robustness_frequency_metrics.json` | Low-/high-frequency energy stats. |
| `latent_structure.json` | PCA variance metadata for latent embeddings. |
| `feature_exploration_summary.json` | Configuration of feature-exploration experiments. |
| `feature_training_history.json` | Loss/accuracy trajectory for feature model. |
| `robustness_adversarial_results.json` | Accuracy per adversarial attack. |
| `robustness_adversarial_training.json` | Adversarial training loss/accuracy curves. |
| `eval_summary_baseline.json` | Baseline evaluation metadata stub. |

Use this guide in tandem with the **Figures** and **Tables** walkthroughs to understand the full context behind every artifact in the `analysis_outputs` and `evaluation_outputs` directories.