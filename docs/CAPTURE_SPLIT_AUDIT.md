# Capture-timestamp audit (9 November 2026)

This note supplements the existing FFE five-fold evaluation. It does not redefine the published experimental results.

## Why the audit was needed

The original split uses species–freshness stratification and SHA256 grouping. This keeps byte-identical files from crossing train, validation and test partitions, but does not ensure that separate images from a common camera burst, fish, or acquisition session are separated.

We grouped the filename pattern `IMG_YYYYMMDD_HHMMSS` by the complete species-class folder and capture timestamp to the second. Within each fold, a test image was flagged when its key also appeared in that fold's training or validation images.

| Fold | Test images | Matching capture timestamp | Unparseable timestamp | No matched timestamp |
|---|---:|---:|---:|---:|
| 1 | 878 | 324 | 1 | 553 |
| 2 | 878 | 333 | 1 | 544 |
| 3 | 879 | 274 | 1 | 604 |
| 4 | 878 | 331 | 0 | 547 |
| 5 | 877 | 298 | 0 | 579 |
| **Total** | **4,390** | **1,560 (35.54%)** | **3** | **2,827** |

Matching timestamps indicate possible shared capture moments, not proven identical fish. Different timestamps may still belong to the same fish or session.

## Post hoc sensitivity analysis on the 2,827 test images without a matched timestamp

Using **saved out-of-fold predictions**, we recomputed Macro F1 for severe synthetic darkening without correction and after validation-selected Gamma Correction. The original fitted models were not retrained, and no new grouped cross-validation was performed.

| Classifier | Uncorrected Macro F1 | Gamma Macro F1 | Gamma gain | Not Fresh classified as Highly Fresh, before / after |
|---|---:|---:|---:|---:|
| FCN | 68.94% | 70.92% | +1.98 percentage points | 56 / 41 |
| Logistic regression | 67.96% | 70.94% | +2.97 percentage points | 71 / 51 |
| LightGBM | 68.53% | 71.60% | +3.07 percentage points | 73 / 51 |

The direction of the Gamma benefit persists, but the restricted subset is selected after inspection and does **not** establish independent-subject accuracy. No new confidence intervals were computed for this subset.

## Publication limits

- Do not claim that SHA256 grouping achieves fish-identity or acquisition-session independence.
- Do not claim field robustness, clinical food safety, or physical illumination in lux based on these synthetic changes.
- For a stronger generalization claim, retrain and evaluate with a defensible group-aware partition, ideally using known fish or session IDs. Grouping by timestamp alone would be an incomplete surrogate.
- The dataset metadata discrepancy (catalogue count 4,392 versus 4,390 images in the analyzed archive) is documented elsewhere in the repository.
- Results remain exploratory and should receive author and supervisor review before journal submission.
