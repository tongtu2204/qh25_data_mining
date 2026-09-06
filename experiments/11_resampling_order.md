# Resampling-order pilot — 2026-09-06

## What was established from the paper

Source: https://doi.org/10.1186/s12911-025-02894-z (Methods, Results, Tables 5/7, Conclusions).

- D1: SMOTE; D2: random majority undersampling to match the minority class.
- An 80/20 train/test split, PCA retaining 95% variance and XGBoost grid search scored with ROC-AUC are described.
- Exact resampling/split order, seed, encoding, SMOTE neighbors and the D2 sample-size discrepancy are not reproducible from the text alone.
- D1 Table 5 balanced evaluation has 1,944 rows (976/968); D2 Table 7 has 274,354 without PCA and 21,218 with PCA, both approximately balanced. PCA cannot explain a change in row count.
- Full D1 SMOTE at 1:1 produces 9,722 rows with the repo cleaning. sklearn's 20% split yields **1,945**, not exactly the paper's 1,944. The prior approximate count match was suggestive, not exact proof of processing order.

## Executed pilot

Both full input files were loaded; D2 was read directly from ZIP. Existing results 01–10 were not overwritten.

1. **Control:** stratified raw split → preprocessing fitted on train → train-only resampling → optional train-fitted PCA → XGBoost. Evaluate on original test and a 1:1 random subset of the same real test, using identical predictions without retraining.
2. **Inferred paper arm:** D1 full-data preprocessing → SMOTE 1:1 → unstratified split; D2 raw-data undersampling 1:1 → unstratified split → train-fitted preprocessing. Then optional train-fitted PCA → XGBoost.

The D1 inferred arm deliberately allows pre-split preprocessing/SMOTE information sharing and is a diagnostic reconstruction, not an unbiased generalization estimate. D2 random undersampling does not synthesize examples; the chief issue here is altered evaluation prevalence, not automatic SMOTE-type leakage.

Both arms retain the existing repo's label conversion, numeric scaling, one-hot encoding and IQR clipping to isolate resampling/evaluation changes. D2's binary threshold 0.5 is a repo assumption, not verified author code. Binary missing values are checked before conversion; the experiment fails if any exist. None were found in these runs.

Use saved step-05 tuned XGBoost parameters in both PCA conditions, seed 42, threshold 0.5. **No new grid search, CV, seed search or threshold optimization was performed.** This pilot does not reproduce every paper step. It is not a fresh independently selected final-model estimate. Library versions differ from the earlier Windows runs; metadata records installed versions, and the control was rerun in the same environment.

## Results with PCA

| Dataset | Protocol/evaluation | Test rows | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---|---|---:|---:|---:|---:|---:|
| D1 | Control / original test | 1,022 | 77.79% | 15.29% | 78.00% | 0.2557 |
| D1 | Same control model / balanced real test | 100 | 80.00% | 81.25% | 78.00% | 0.7959 |
| D1 | Inferred paper arm / resampled test | 1,945 | 83.19% | 78.63% | 91.03% | 0.8438 |
| D1 | Paper Table 5(b) | 1,944 | 95% | 93% | 96% | 0.95 |
| D2 | Control / original test | 1,108,453 | 96.79% | 57.06% | 99.47% | 0.7252 |
| D2 | Same control model / balanced real test | 94,384 | 98.01% | 96.65% | 99.47% | 0.9804 |
| D2 | Inferred paper arm / resampled test | 94,385 | 98.12% | 96.93% | 99.38% | 0.9814 |
| D2 | Paper Table 7 with PCA | 21,218 | 98% | 97% | 99% | 0.98 |

Full metrics including no-PCA, confusion counts, ROC-AUC, average precision, MCC and specificity are saved in results/<dataset>/11_resampling_order/seed_42/comparison.csv. In the D2 inferred arm, no-PCA accuracy/F1 are 95.33%/0.9546, versus PCA 98.12%/0.9814. The two variants use the same train/test rows.

D2 is close numerically to the paper under balanced evaluation, **not proof of an identical implementation or superiority**. D1 remains well below the paper with these fixed parameters. Balanced-test comparisons demonstrate why the old low precision need not imply weaker discrimination. D1 balanced real test has only 100 rows and is imprecise. No between-protocol significance claim is made.

## Authors' concluding claims

The authors conclude PCA + XGBoost improves predictive accuracy and computation, citing up to 95%/98% accuracy on D1/D2. They attribute interpretability to SHAP and computation benefits to dimensionality reduction and parallelism; they report more than threefold total runtime reduction against the referenced workflow. That is not a universal OpenMP speedup. They propose early stroke-risk decision-support tools and future expansion to other tasks, larger real-world data and other parallel/distributed methods. This pilot tests only the ML resampling/evaluation question, not SHAP or runtime claims.

## Reproduce

From the repo root (quotes required around paths with spaces):

```powershell
python -m experiments.11_resampling_order --dataset dataset1 --data "C:/path/healthcare-dataset-stroke-data.csv"
python -m experiments.11_resampling_order --dataset dataset2 --data "C:/path/healthcare_data_2GB.csv.zip"
```

Requires the repo dependencies including xgboost and imbalanced-learn. See each metadata.json for versions and exact parameters. Models and compressed predictions are generated locally; only code, this note, comparison CSVs and metadata are intended for Git.
