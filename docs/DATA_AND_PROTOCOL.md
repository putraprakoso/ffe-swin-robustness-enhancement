# Dataset and experimental protocol

## FFE source and counts

Source: [Mendeley Data — FFE v1](https://data.mendeley.com/datasets/xzyx7pbr3w/1), DOI `10.17632/xzyx7pbr3w.1`.

The provider's description reports 4,392 images, whereas the source ZIP audited for these experiments yielded **4,390** valid images. This mismatch has not been independently reconciled. The experiments and result tables consistently refer to the 4,390-image local audit.

Archived source name: `xzyx7pbr3w-1.zip`.

Archived source SHA256:

```text
9feb992a2f027a8e31e4ae196a9bf311a8258b2410e5e919178e2eb4c1e231fa
```

Observed label counts: Highly Fresh = 1,764; Fresh = 1,320; Not Fresh = 1,306 (total 4,390), across eight species. Labels are based on storage days rather than independent lab or microbiological measurements.

## Splitting and preprocessing

- Five outer folds using `StratifiedGroupKFold`, stratification by species + freshness and grouping by exact byte SHA256 hash.
- A first inner split within each outer training portion supplies training and validation subsets.
- Deep input uses 224 × 224 RGB and ImageNet normalization; training augmentation occurs within the training set.
- Swin-Tiny backbone is fine-tuned separately in each outer fold; embeddings are 768-dimensional.
- Classifiers are trained on the same fold-specific embeddings: LR, LightGBM, and FCN. FCN StandardScaler is fit on training data only.
- Baseline parity was checked by comparing predictions from the reloaded saved models with the original stored predictions for all test folds: zero mismatches.

## Paired degradation tests (08)

Seven conditions: original; dark multipliers 0.8 and 0.6; bright multipliers 1.2 and 1.4; Gaussian blur sigma 0.7 and 1.5. Each condition is evaluated on matching test IDs in each fold. All 4,390 images have one outer-fold test prediction per condition and per classifier.

## Enhancement (09)

For original and both synthetic-dark conditions, compare none, Gamma Correction and CLAHE on the LAB-L channel. The candidate gamma settings are 0.70 and 0.85; CLAHE clip limits are 1.5 and 2.5. Parameters are selected separately per fold using validation samples, averaged across three classifier Macro F1 scores under two dark conditions. The selected method is then tested on the fold's held-out samples. Normal-image harm is reported as a control, not used to choose hyperparameters.

## Caveats

1. No specimen ID or acquisition-session ID exists in the current manifest; splitting by identical bytes does not guarantee physical independence.
2. Photometric degradations are synthetic, not an independent real-world acquisition study.
3. Image-level paired bootstrap intervals may be overconfident under within-fish dependence.
4. Multiple comparisons and post-hoc selection of enhancement as a follow-up limit confirmatory claims.
5. Exact dependencies and Colab environments change over time; refer to notebook output configuration records from the original workspace for version checks.
