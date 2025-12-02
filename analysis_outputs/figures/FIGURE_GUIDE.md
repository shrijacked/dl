# 📚 Complete Guide to Your Analysis Figures

This document mirrors the step-by-step walkthrough style used in the interactive explanation. Each section focuses on a theme, then drills into every figure with **what it shows**, **key insights**, and **why it matters** for the OrganAMNIST project.

---

## 📊 SECTION 1: Dataset Understanding & Distribution

### 1.1 `label_distribution.png` — Class Imbalance Analysis
**What it shows**: Bar chart comparing training (blue) and validation (orange) counts for label IDs 0–10.

**Key insights**
- Class 6 (Liver) is the clear majority (~6.2k samples), roughly 4.5× larger than the smallest classes.
- Classes 1–3 (Femur Left/Right, Heart) sit near 1.4k samples and are the most underrepresented.
- Classes 4–8 (Kidneys and Lungs) form the middle band around 3.8–4.0k.
- Validation distribution mirrors training almost perfectly, so evaluation remains fair despite the imbalance.

**Why it matters**: Class imbalance biases the model toward majority organs unless we counteract it via weighted sampling, class-weighted loss, or targeted augmentation for minority classes such as the Heart.

---

### 1.2 `train_grid.png` — Visual Sample Gallery
**What it shows**: A grid of six samples per class (11 × 6 = 66 total) from the training split.

**Organ cheat sheet**
| Label | Organ | Visual signature |
| ----- | ----- | ---------------- |
| 0 | Bladder | Oval cavity, medium intensity |
| 1 | Femur (Left) | Bright cortical bone with thick edge |
| 2 | Femur (Right) | Mirror of label 1 |
| 3 | Heart | Rounded shape with darker center |
| 4 | Kidney (Left) | Bean-like, bright rim, dark medulla |
| 5 | Kidney (Right) | Mirror of label 4 |
| 6 | Liver | Large bright mass, coarse texture |
| 7 | Lung (Left) | Large dark void with bright boundary |
| 8 | Lung (Right) | Similar to label 7 |
| 9 | Spleen | Irregular blob, intermediate intensity |
| 10 | Pancreas | Thin, irregular shape, noisy texture |

**Why it matters**: Seeing multiple samples per class helps identify visually similar organs (Left vs Right Kidney) that drive misclassifications and informs augmentation ideas.

---

### 1.3 `train_pixel_histogram.png`, `val_pixel_histogram.png`, `test_pixel_histogram.png`
**What they show**: Pixel intensity histograms for each split (0–255 scale).

**Key observations**
- Huge spike near 0 represents background air in CT scans.
- Secondary spike near 255 comes from very bright bone/contrast regions.
- Train and val distributions overlap almost perfectly.
- Test split has a slightly higher mass near intensity 0 and slightly fewer saturated pixels, hinting at mild distribution shift.

**Why it matters**: A different intensity profile on test images can cause accuracy drop unless normalization/augmentation makes the model robust to those shifts.

---

### 1.4 `distribution_comparison_plots.png`
**What it shows**: Mean pixel intensity per split (train, val, test).

**Key insights**
- Train and val both average ~0.46 (after normalization).
- Test averages slightly lower (~0.45), confirming the histogram observation numerically.

**Why it matters**: Even a small mean shift signals that we should keep an eye on domain adaptation and potentially rely on histogram-matching augmentations.

---

## 📊 SECTION 2: Latent Space & Class Structure

### 2.1 `latent_tsne.png` — t-SNE Visualization
**What it shows**: 2D embedding of high-dimensional features extracted from the model, color-coded by class.

**How to read it**
- Tight clusters → visually distinctive/easy classes.
- Broad blobs → high intra-class variability.
- Overlapping colors → challenging class boundaries.

**Key insights**
- Yellow/green clusters on the left (femurs) are compact, meaning the model separates them easily.
- A red cluster at the bottom and purple clusters on the right are also well isolated.
- Central region shows heavy overlap among multiple colors (kidneys, pancreas, spleen, lungs), aligning with real confusion patterns.

**Why it matters**: Guides where to invest effort—overlapping clusters need extra augmentation, specialized losses, or secondary classifiers.

---

### 2.2 `feature_interclass_similarity.png` — Cosine Similarity Matrix
**What it shows**: Pairwise cosine similarity between class centroids in feature space.

**Key confusing pairs**
| Class Pair | Cosine Sim | Interpretation |
| ---------- | ---------- | -------------- |
| 4 ↔ 5 | 0.88 | Left vs Right Kidney |
| 4 ↔ 10 | 0.86 | Left Kidney vs Pancreas |
| 5 ↔ 10 | 0.82 | Right Kidney vs Pancreas |
| 9 ↔ 5 | 0.81 | Spleen vs Right Kidney |

Classes 1 and 2 (Femurs) score ~0.4 with others, so they are distinguishable.

**Why it matters**: Preemptively indicates where confusion will happen even before inspecting the confusion matrix; tells you which organs deserve class-specific augmentation or auxiliary heads.

---

### 2.3 `class_imbalance_confusion_matrix.png`
**What it shows**: Row-normalized confusion matrix on the validation set so every row sums to 1.0.

**Key observations**
- Most diagonal cells are >0.95, signaling near-perfect accuracy.
- Row for label 7 (Left Lung) spreads more mass into columns 6 and 8, meaning leaks into liver/right lung predictions.
- Femur rows (1 & 2) have tiny off-diagonal entries pointing to mutual confusion, which matches anatomical symmetry.

**Why it matters**: Pinpoints exactly which true labels suffer, so you can verify whether the similarity matrix predictions hold and craft mitigation strategies.

---

## 📊 SECTION 3: Frequency Domain Analysis

### 3.1 `frequency_analysis.png` & `freq_avg_spectrum_*.png`
**What it shows**: Average log-magnitude of the 2D Fourier transform per split.

**Key insights**
- Bright center = dominance of low-frequency content (global organ shapes).
- Cross-shaped streaks indicate strong horizontal/vertical structures from CT acquisition geometry.
- Smooth decay toward edges follows the 1/f spectrum common to natural/scientific images.
- Comparing train vs test spectra helps detect acquisition-related distribution shifts.

**Why it matters**: Knowing that low frequencies dominate tells us architectural choices (e.g., larger receptive fields) and anti-aliasing augmentations will be beneficial. It also helps diagnose spurious high-frequency adversarial noise.

---

### 3.2 `distribution_comparison_plots.png`
Discussed earlier (Section 1.4). Use it as the numeric check after seeing the frequency plots.

---

## 📊 SECTION 4: Test Set Characterization (Shift Analysis)

### 4.1 `test_characterization_pixel_hist.png`
**What it shows**: Probability density of normalized pixel intensities for train/val/test overlaid.

**Key insights**
- All three splits have a giant spike at 0 (air/background).
- Train/val spikes are higher than test, implying test slices contain fewer pure background pixels.
- Test shows small peaks around 0.02–0.04 that train/val lack.
- The bright-end spike near 1.0 is slightly smaller for test.

**Why it matters**: Confirms a measurable but manageable photometric shift; augmenting with random brightness/contrast or histogram equalization can close the gap.

---

### 4.2 `test_characterization_edge_hist.png`
**What it shows**: Distribution of edge density (fraction of edge pixels) across splits.

**Key insights**
- All splits cluster at low edge density (<0.1) because organs are smooth.
- Test distribution is slightly shifted toward higher densities, meaning certain test slices contain more detailed structures or noise.

**Why it matters**: Encourages including edge-based augmentations (e.g., sharpening/blur) so the model remains stable despite structural detail differences.

---

### 4.3 `test_characterization_lbp_hist.png`
**What it shows**: Local Binary Pattern histogram comparison (codes 0–25).

**Key insights**
- Train/val curves almost overlap with peaks at codes 11–14.
- Test curve diverges sharply at code 25 (strong spike) and differs around codes 10–13.

**Why it matters**: Texture statistics differ noticeably on test images—without texture-aware augmentations or domain adaptation, the model might underperform on submission data.

---

## 📊 SECTION 5: Feature Exploration & Model Attention

### 5.1 `feature_gradcam/gradcam_*.png`
**What it shows**: Side-by-side original slice and Grad-CAM heatmap for several labels.

**Key insights**
- Label 8 (Right Lung): Hotspots hug the lung boundary, confirming the model pays attention to the hollow cavity.
- Label 7 (Left Lung): Distributed hotspots across the lung interior; ensures the model isn’t relying on background.
- Label 6 (Liver): Concentrated attention on the right-lower quadrant where the liver mass sits.

**Why it matters**: Grad-CAM builds trust in clinical contexts by proving predictions depend on the relevant anatomy rather than noise or borders.

---

### 5.2 `feature_multiscale_edge_density.png`
**What it shows**: Edge density measured at multiple downsample scales (approx. 32, 64, 128 px).

**Key insights**
- All curves drop as the scale coarsens because fine edges vanish.
- Test split stays consistently below train/val at every scale, reinforcing earlier texture observations.
- The separation is largest at the finest scale, indicating a deficit of fine-grained texture in test images.

**Why it matters**: Suggests using multi-scale feature aggregation (already present in Swin/DenseViT) and possibly training with smoother augmentations to narrow the distribution gap.

---

## 📊 SECTION 6: Robustness Analysis

### 6.1 `perturbations_train_*.png.png`
**What it shows**: Rows of the same training slice subjected to Gaussian noise, motion blur, contrast boosts/cuts.

**Key insights**
- Noise reduces fine details but organ outlines remain.
- Motion blur smears boundaries, testing the model’s reliance on edge sharpness.
- Contrast adjustments simulate scanner setting changes.

**Why it matters**: Visual confirmation of the corruption suite used for robustness benchmarking; helps interpret downstream corruption accuracy tables.

---

### 6.2 `flip_differences.png`
**What it shows**: Histograms of mean absolute differences between original images and their horizontal/vertical flips.

**Key insights**
- Peaks around 50–60 indicate most images change noticeably when flipped—organs are not symmetric.
- Vertical and horizontal distributions are similar but not identical.

**Why it matters**: Validates that flip augmentations actually create novel training samples (especially helpful for mirroring left/right organs).

---

### 6.3 `robustness_adversarial_samples/sample_*_*.png`
**What it shows**: Clean vs adversarial images plus absolute difference heatmaps.

**Key insights**
- Human-visible difference is negligible, yet the model prediction can flip.
- Difference map reveals high-frequency speckle noise typical of FGSM/PGD.

**Why it matters**: Highlights vulnerability of medical models to adversarial perturbations; motivates adversarial training or certified defenses if deployment requires it.

---

### 6.4 `robustness_occlusion/occlusion_*.png`
**What it shows**: Occlusion sensitivity maps—redder regions mean occluding them hurts confidence the most.

**Key insights**
- For lungs, the most sensitive areas cluster inside the cavity and near the pleural boundary.
- For liver, hotspots align with the parenchyma rather than background.

**Why it matters**: Confirms decision-critical regions align with anatomy, complementing Grad-CAM and providing a second interpretability check.

---

## 📊 SECTION 7: Data Quality Analysis

### 7.1 `data_quality_duplicates_train/group_*.png` & `data_quality_duplicates_val/group_*.png`
**What it shows**: Perceptual-hash–based duplicate groups; each image pair looks almost identical.

**Key insights**
- Roughly 20 duplicate clusters exist per split.
- Some duplicates are exact copies, others are adjacent CT slices with negligible variation.

**Why it matters**: Duplicates inflate effective dataset size and risk leaking between splits if not handled. Use the groups to prune or consolidate samples before training.

---

### 7.2 `val_grid.png`
Same structure as the training grid but for validation samples—handy for a sanity check that validation data has similar visual diversity and no corruption artifacts.

---

## 📊 SECTION 8: Additional Grids

### 8.1 `test_grid.png`
**What it shows**: Strip of randomly chosen test images (labels shown as 0 because ground truth is hidden).

**Key insights**
- Visual quality matches train/val.
- Confirms the types of anatomy expected at evaluation time.

**Why it matters**: Provides intuition about what the leaderboard data looks like even without labels, ensuring augmentations remain realistic for test-time predictions.

---

## 📚 Comprehensive Summary & Next Steps

| Category | Figures | Takeaways |
| -------- | ------- | --------- |
| Data distribution | `label_distribution.png` | Address heavy liver dominance and minor classes. |
| Visual samples | `*_grid.png` | Recognize organ morphology; spot confusing pairs. |
| Pixel statistics | `*_pixel_histogram.png`, `distribution_comparison_plots.png` | Train/val aligned; test slightly darker. |
| Latent structure | `latent_tsne.png` | Some clusters overlap → prioritize those classes. |
| Class similarity | `feature_interclass_similarity.png` | Kidneys/pancreas appear similar; expect confusion. |
| Confusion behavior | `class_imbalance_confusion_matrix.png` | Errors match similarity predictions. |
| Frequency & texture | `freq_avg_spectrum_*.png`, `test_characterization_*`, `feature_multiscale_edge_density.png` | Test split smoother with different textures; prepare augmentations accordingly. |
| Interpretability | `feature_gradcam/`, `robustness_occlusion/` | Model attends to organ regions—good sign. |
| Robustness | `perturbations_*.png.png`, `flip_differences.png`, `robustness_adversarial_samples/` | Clean-vs-corrupted comparisons, asymmetry validation, adversarial vulnerability. |
| Data hygiene | `data_quality_duplicates_*` | Remove/flag duplicates to prevent leakage. |

**Key project actions**
1. Use weighted sampling or focal loss to mitigate class imbalance.
2. Apply targeted augmentation for kidney/pancreas to reduce confusion.
3. Incorporate brightness/contrast and texture augmentations to bridge the train-test shift.
4. Continue using interpretability checks (Grad-CAM + occlusion) for clinical trust.
5. Harden against adversarial/perturbation attacks via adversarial training or test-time adaptation.
6. Prune duplicate groups before final training.

This markdown is intentionally detailed so you can skim to any figure and instantly recall what it communicates and how it impacts modeling decisions.

