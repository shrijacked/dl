# Analysis Figures Walkthrough

This guide summarizes every visualization under `analysis_outputs/figures`. Use it as a quick reference when presenting findings or revisiting diagnostics.


---
## 1. Dataset Distribution & Visual Samples

### `label_distribution.png`
- **What to look for:** Train (blue) vs validation (orange) counts for the 11 organ classes.
- **Insights:** Liver (class 6) dominates; heart and femurs are minority classes. Imbalance handling (weighted sampler/loss) is recommended.

### `train_grid.png`, `val_grid.png`, `test_grid.png`
- **Content:** 6 example slices per class for train/val, randomly sampled rows for the unlabeled test set.
- **Insights:** Helps recognize organ morphology and notice visually similar classes (e.g., kidneys vs pancreas).

### `train_pixel_histogram.png`, `val_pixel_histogram.png`, `test_pixel_histogram.png`
- **Content:** Per-split pixel intensity histograms (0–255).
- **Insights:** Train/val match closely; test has slightly fewer saturated pixels and a different dark-pixel spike → mild distribution shift.

### `distribution_comparison_plots.png`
- **Content:** Mean pixel intensity per split.
- **Insights:** Verifies train/val alignment and quantifies the small shift in the test set.

---

## 2. Latent Structure & Class Similarity

### `latent_tsne.png`
- **Content:** t-SNE embedding of sampled training features, colored by label.
- **Insights:** Femur classes form tight clusters (easy); central classes overlap (harder). Guides which classes need extra augmentation.

### `feature_interclass_similarity.png`
- **Content:** Cosine similarity matrix between class centroids.
- **Insights:** Left/right kidneys (classes 4/5) and pancreas (10) have >0.8 similarity → expect confusion. Femurs are most distinct (~0.4).

### `class_imbalance_confusion_matrix.png`
- **Content:** Row-normalized validation confusion matrix for the reference model.
- **Insights:** Diagonal is almost perfect; remaining errors occur among kidney/pancreas/lung classes.

---

## 3. Frequency & Texture Diagnostics

### `frequency_analysis.png`, `freq_avg_spectrum_train.png`, `freq_avg_spectrum_val.png`, `freq_avg_spectrum_test.png`
- **Content:** Average 2-D log-spectra per split.
- **Insights:** Strong low-frequency energy (bright center) and cross-shaped streaks reflect smooth anatomical structures. Compare train vs test spectra for shift.

### `test_characterization_pixel_hist.png`
- **Content:** Overlaid intensity PDFs for train/val/test (values normalized to 0–1).
- **Insights:** Confirms the subtle but consistent pixel-distribution drift in the test split.

### `test_characterization_edge_hist.png`
- **Content:** Edge density histograms (Canny detector) across splits.
- **Insights:** Test images generally have slightly fewer edges, implying smoother textures relative to train/val.

### `test_characterization_lbp_hist.png`
- **Content:** Local Binary Pattern histogram comparison.
- **Insights:** Test split shows a pronounced spike at code 25, meaning different fine-grained textures; useful for designing texture-aware augmentations.

### `feature_multiscale_edge_density.png`
- **Content:** Edge density versus scale (32, 64, 128 px) per split.
- **Insights:** All splits lose edge density at coarser scales; the test split remains consistently lower, reinforcing the smoothness observation.

---

## 4. Feature Attribution & Importance Maps

### `feature_gradcam/gradcam_*.png`
- **Content:** Paired original image and Grad-CAM heatmap for various labels.
- **Insights:** Checks whether the model focuses on the organ region (desired) or background artifacts (undesired). Useful for qualitative trust.

### `robustness_occlusion/occlusion_*.png`
- **Content:** Occlusion sensitivity maps (delta in confidence when sliding a mask).
- **Insights:** Dark-red regions indicate pixels critical for the prediction; confirms which subregions drive decisions.

### `flip_differences.png`
- **Content:** Histograms of per-image L1 differences after horizontal/vertical flips.
- **Insights:** Images are far from symmetric, so flip augmentations inject meaningful variance rather than duplicating data.

---

## 5. Robustness & Perturbation Probes

### `perturbations_train_*.png.png`
- **Content:** Rows showing a training slice under clean, Gaussian noise, motion blur, contrast up, and contrast down settings.
- **Insights:** Visualizes the corruption suite used in robustness tests and highlights which degradations are most destructive.

### `robustness_adversarial_samples/sample_*_*.png`
- **Content:** Triplets (clean, adversarial, absolute difference) generated via FGSM/PGD-style attacks.
- **Insights:** Demonstrates that imperceptible perturbations can flip predictions, emphasizing the need for adversarial defenses.

---

## 6. Data Quality & Deduplication

### `data_quality_duplicates_train/group_*.png`, `data_quality_duplicates_val/group_*.png`
- **Content:** Side-by-side thumbnails of perceptually similar images flagged as duplicates.
- **Insights:** Use to prune near-identical slices, preventing leakage between splits and reducing overfitting to repeated anatomy.

---

## 7. Quick How-To

1. **Need class context?** Start with the distribution plot and sample grids.
2. **Investigating misclassifications?** Cross-reference the t-SNE, similarity heatmap, and confusion matrix.
3. **Checking for domain shift?** Inspect pixel/edge/LBP histograms plus frequency plots.
4. **Explaining predictions?** Use Grad-CAM and occlusion maps.
5. **Assessing robustness?** Review perturbation rows, flip histograms, and adversarial samples.
6. **Validating data hygiene?** Scan the duplicate groups.

Keeping this flow handy will save time whenever you revisit the OrganAMNIST analyses or prepare slides/reports.

