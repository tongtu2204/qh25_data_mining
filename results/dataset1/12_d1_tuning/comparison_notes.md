# D1 tuning comparison

Completed run recovered and verified on 2026-09-10. All 16 metric rows were checked against stored prediction arrays; all four searches contain 24 candidates.

## Design

- Two protocols: split original D1 first and SMOTE only inside train/CV; or the inferred paper protocol with full-data preprocessing and SMOTE before the split.
- Two feature variants per protocol: no PCA or PCA retaining 95% variance.
- 24 randomized parameter candidates per variant, three stratified folds, selected by mean CV ROC-AUC: 288 search fits plus refits and OOF fits.
- Thresholds: 0.5 and three training out-of-fold thresholds maximizing recall subject to specificity targets of 80%, 85%, 90%. These are validation targets, not guarantees on test.
- All preprocessing in the original-test branch is fitted inside the training fold. Thresholds are frozen before evaluating either PCA option on test.
- Original test: 1,022 real rows / 50 positive. Inferred paper test: 1,945 rows / 970 positive, including synthetic rows.

## Interpretation

On the inferred paper arm, tuning increased no-PCA accuracy from 86.48% to 96.30%, precision to 96.67%, recall to 95.88%, and F1 to 0.9627. PCA increased accuracy from 83.19% to 95.12%, recall from 91.03% to 98.87%, with precision 91.95% and F1 0.9528 at threshold 0.5. Paper Table 5(b) reports rounded values: accuracy 95%, precision(1) 93%, recall(1) 96%, F1(1) 0.95. These are numerical comparisons only: the processing order is inferred and the test data differ.

On the original test, no-PCA at 0.5 changed from 32 TP / 196 FP to 33 TP / 172 FP, improving recall 64% to 66%, accuracy 79.06% to 81.51% and F1 0.2302 to 0.2588. PCA at 0.5 changed from 39 TP / 216 FP to 40 TP / 224 FP: recall 78% to 80% but more false alarms and slightly lower F1. This does not establish a general improvement over the author on real unseen cases.

The higher-CV-AUC variant is no-PCA in the inferred arm and PCA in the original-test arm. These choices are based on CV, not the maximum test score. Low thresholds can inflate recall while increasing false positives. D1 test has only 50 positives: one additional detection changes recall by two percentage points.

## Complete test comparison

| Protocol | PCA | Threshold policy | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TP | FP | FN | TN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| split_then_resample | False | default_0.5 | 0.5000 | 81.51% | 16.10% | 66.00% | 82.30% | 0.2588 | 0.8201 | 33 | 172 | 17 | 800 |
| split_then_resample | False | oof_specificity_0.80 | 0.4354 | 76.32% | 13.64% | 72.00% | 76.54% | 0.2293 | 0.8201 | 36 | 228 | 14 | 744 |
| split_then_resample | False | oof_specificity_0.85 | 0.4870 | 80.04% | 15.00% | 66.00% | 80.76% | 0.2444 | 0.8201 | 33 | 187 | 17 | 785 |
| split_then_resample | False | oof_specificity_0.90 | 0.5448 | 85.42% | 18.06% | 56.00% | 86.93% | 0.2732 | 0.8201 | 28 | 127 | 22 | 845 |
| split_then_resample | True | default_0.5 | 0.5000 | 77.10% | 15.15% | 80.00% | 76.95% | 0.2548 | 0.8118 | 40 | 224 | 10 | 748 |
| split_then_resample | True | oof_specificity_0.80 | 0.5468 | 79.75% | 15.72% | 72.00% | 80.14% | 0.2581 | 0.8118 | 36 | 193 | 14 | 779 |
| split_then_resample | True | oof_specificity_0.85 | 0.6385 | 84.34% | 16.87% | 56.00% | 85.80% | 0.2593 | 0.8118 | 28 | 138 | 22 | 834 |
| split_then_resample | True | oof_specificity_0.90 | 0.7111 | 87.57% | 16.52% | 38.00% | 90.12% | 0.2303 | 0.8118 | 19 | 96 | 31 | 876 |
| resample_then_split_inferred | False | default_0.5 | 0.5000 | 96.30% | 96.67% | 95.88% | 96.72% | 0.9627 | 0.9931 | 930 | 32 | 40 | 943 |
| resample_then_split_inferred | False | oof_specificity_0.80 | 0.1251 | 89.97% | 83.90% | 98.87% | 81.13% | 0.9077 | 0.9931 | 959 | 184 | 11 | 791 |
| resample_then_split_inferred | False | oof_specificity_0.85 | 0.1989 | 93.47% | 89.73% | 98.14% | 88.82% | 0.9375 | 0.9931 | 952 | 109 | 18 | 866 |
| resample_then_split_inferred | False | oof_specificity_0.90 | 0.3113 | 94.96% | 93.00% | 97.22% | 92.72% | 0.9506 | 0.9931 | 943 | 71 | 27 | 904 |
| resample_then_split_inferred | True | default_0.5 | 0.5000 | 95.12% | 91.95% | 98.87% | 91.38% | 0.9528 | 0.9885 | 959 | 84 | 11 | 891 |
| resample_then_split_inferred | True | oof_specificity_0.80 | 0.2005 | 92.34% | 86.88% | 99.69% | 85.03% | 0.9285 | 0.9885 | 967 | 146 | 3 | 829 |
| resample_then_split_inferred | True | oof_specificity_0.85 | 0.3466 | 93.78% | 89.49% | 99.18% | 88.41% | 0.9408 | 0.9885 | 962 | 113 | 8 | 862 |
| resample_then_split_inferred | True | oof_specificity_0.90 | 0.5569 | 95.32% | 92.63% | 98.45% | 92.21% | 0.9545 | 0.9885 | 955 | 76 | 15 | 899 |

## Limits and reproduction

This is an exploratory continuation using previously observed test splits. OOF predictions use parameters selected on the same training CV; this is not nested CV. The inferred arm has preprocessing/SMOTE before train/test and shares information across the split. Its high scores must not be interpreted as unbiased generalization performance. No test-based seed or threshold search was used. SMOTENC, class weighting, IQR alternatives and PCA 99% have not yet been tested in this stage.

Exact parameters, versions, search scores and thresholds are in adjacent JSON/CSV files. Run from repository root:

```powershell
python -m experiments.12_d1_tuning --data "C:/path/healthcare-dataset-stroke-data.csv" --trials 24
```

Paper source: https://doi.org/10.1186/s12911-025-02894-z (Table 5). Prior-run control source: results/dataset1/11_resampling_order/seed_42/comparison.csv.
