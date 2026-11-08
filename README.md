# FFE: Swin-Tiny Robustness and Image Enhancement

**Reproducibility package for the study of fish-eye freshness classification under synthetic image degradation and test-time enhancement.**

This repository contains a focused research pipeline using the **Freshness of the Fish Eyes (FFE)** dataset, Swin Transformer-Tiny feature extraction, and three classification heads: fully connected neural network (FCN), logistic regression (LR), and LightGBM. We evaluate darkening, brightening and Gaussian blur, then compare **gamma correction** and **CLAHE** for darkened images.

> **Research status:** Completed experiments on one publicly available dataset. Findings are exploratory; the repository does **not** establish real-world generalization or physical food safety. The linked GitHub repository is intended to remain private during manuscript preparation and blind review, pending the journal's disclosure rules.

## Highlights from saved results

| Scenario / method | Swin + FCN Macro F1 | Swin + LR Macro F1 | Swin + LightGBM Macro F1 |
|---|---:|---:|---:|
| Original, no correction | 0.7645 | 0.7638 | 0.7629 |
| Dark strong, no correction | 0.7157 | 0.7094 | 0.7171 |
| Dark strong, gamma correction | 0.7377 | 0.7405 | **0.7465** |
| Dark strong, CLAHE | 0.7072 | 0.6922 | 0.6989 |

In these tests, gamma correction partly recovered predictive performance under synthetic darkening, while CLAHE reduced Macro F1. Applying enhancement to *unmodified* images could also degrade performance; therefore universal pre-processing is **not supported** by these results. The enhancement experiment was devised after inspecting the robustness results (exploratory follow-up).

See `docs/RESULTS.md` for full caveats and `results/` for CSV tables and original figures.

## Dataset and attribution

- Original dataset: [The Freshness of the Fish Eyes Dataset](https://data.mendeley.com/datasets/xzyx7pbr3w/1), DOI [`10.17632/xzyx7pbr3w.1`](https://doi.org/10.17632/xzyx7pbr3w.1), creators Prasetyo, Adityo, Suciati and Fatichah (2022), CC BY 4.0.
- Associated paper: Prasetyo et al. (2022), *Information Processing in Agriculture*, DOI [`10.1016/j.inpa.2022.01.002`](https://doi.org/10.1016/j.inpa.2022.01.002).
- **Important count discrepancy:** the dataset catalog describes **4,392** images, while this project's archived ZIP was audited and yielded **4,390 valid images**. The exact analyzed archive fingerprint is documented in `docs/DATA_AND_PROTOCOL.md`. Do not substitute sample counts or assume the difference has been resolved.
- Class labels are storage-day categories (*Highly Fresh*, *Fresh*, *Not Fresh*), not independent biochemical or safety certificates.

The raw images are **not included** in this repository. Obtain them from the dataset provider and observe the attribution requirements. The original ZIP used by this study was named `xzyx7pbr3w-1.zip`.

## Repository layout

```text
notebooks/
  ffe_core.py
  00_dataset_audit_and_splits.ipynb
  02_swin_finetuning.ipynb
  03_deep_embeddings_and_baselines.ipynb
  04_deep_fcn_only.ipynb
  08_paired_robustness_jri.ipynb
  09_image_enhancement_jri.ipynb
results/
  08_robustness/       # pooled metrics, bootstrap summaries, figures
  09_enhancement/      # pooled metrics, bootstrap summaries, figures
scripts/
  check_results.py     # lightweight consistency check of saved CSVs
docs/
  START_HERE_ID.md
  DATA_AND_PROTOCOL.md
  RESULTS.md
```

## Reproduce in Google Colab

The original experiments used a **Google Drive workspace**. All notebook code defaults to:

```text
/content/drive/MyDrive/tugas_akhir/FFE_Publication_2026/
  01_data/raw/xzyx7pbr3w-1.zip
  02_notebooks/ffe_core.py
  03_artifacts/{splits,checkpoints,features,models}/
  04_results/{metrics,figures}/
  06_logs/
```

If necessary, set `FFE_PROJECT` in a notebook's setup cell **before importing** `ffe_core` to point at another Drive folder. Copy `notebooks/ffe_core.py` into that project's `02_notebooks/` directory. A GPU runtime (e.g., Colab T4) is recommended; full reproducibility of backbone training is compute-intensive. The original trained checkpoints are not committed to Git.

Run notebooks in this order:

1. `00_dataset_audit_and_splits.ipynb` — audit images and create grouped five-fold splits.
2. `02_swin_finetuning.ipynb` — train Swin-Tiny backbones per fold.
3. `03_deep_embeddings_and_baselines.ipynb` — extract 768-dimensional embeddings and fit LR and LightGBM.
4. `04_deep_fcn_only.ipynb` — train/reload FCN with training-fold scaling; **focused replacement** for the original fusion-heavy notebook 04.
5. `08_paired_robustness_jri.ipynb` — compare classifiers on original/dark/bright/blur inputs, with baseline parity check.
6. `09_image_enhancement_jri.ipynb` — tune Gamma/CLAHE parameters on validation data only; compare on matched held-out images.

If using the **existing** Google Drive project, the generated artifacts from 00–04 already exist and one can rerun 08 and 09 without training the backbone again. Before trusting an independent rerun, check the parity CSV and environment versions.

```bash
python scripts/check_results.py
```

This checks the **included summary tables**; it is not a substitute for rerunning the image-inference pipeline.

## Methodological boundaries

- Outer `StratifiedGroupKFold(5)` groups **identical SHA256 images**, stratified by species × freshness. It does **not** group by individual fish or acquisition session.
- Perturbations are synthetic: dark multipliers `0.8`/`0.6`, bright multipliers `1.2`/`1.4`, Gaussian blur sigma `0.7`/`1.5`.
- Gamma candidates `0.70`/`0.85`; CLAHE clip limits `1.5`/`2.5`; fold-level selection on validation examples (mean Macro F1 over three classifiers and two darkening conditions).
- Paired image-bootstrap intervals do not fully capture potential within-fish/session dependence or multiple-comparison selection.
- The notebook 09 follow-up was planned **after** inspecting notebook 08 results. It is exploratory, not an independent preregistered confirmation.
- The notebook 04 reduction is logically based on the original FCN implementation; it has not been separately executed from scratch. The published summary numbers derive from the original completed pipeline, and their parity was verified in notebook 08.

## Research software, data and publication

No original photos, pretrained weights or private Drive files are committed here. This repository contains research code and aggregate findings only. Software license, author citation metadata and public-release timing should be determined by the project owner before opening the repository to the public. Avoid revealing author identities in any anonymized peer-review supplement.
