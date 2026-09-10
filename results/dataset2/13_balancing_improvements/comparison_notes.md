# Experiment 13 — D2 balancing improvements

The experiment compares random undersampling and class-weighted XGBoost, each with and without PCA. D2 is split before fitting; all preprocessing and any resampling are learned from training data only. Tuning uses at most 150,000 rows from an inner training subset. The final models are then fitted on all 4,433,808 outer-training rows when the method supports it.

The input contains 5,769,190 raw rows and 5,542,261 rows after the repository's duplicate/cleaning procedure. The unchanged test set contains 1,108,453 rows and 47,192 positives.

## Test results

| Variant | Threshold policy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TP | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| undersampling + PCA | default 0.5 | 98.47% | 74.01% | 98.86% | 98.46% | 0.8465 | 0.9987 | 46,652 | 16,382 |
| undersampling + PCA | selected specificity 0.85 | 86.11% | 23.46% | 100.00% | 85.50% | 0.3801 | 0.9987 | 47,192 | 153,928 |
| undersampling + PCA | selected specificity 0.90 | 91.22% | 32.65% | 99.98% | 90.83% | 0.4922 | 0.9987 | 47,183 | 97,349 |
| undersampling + PCA | selected specificity 0.95 | 95.18% | 46.90% | 99.84% | 94.97% | 0.6383 | 0.9987 | 47,118 | 53,337 |

At threshold 0.5, the new undersampling + PCA model improves on the earlier tuned balancing + PCA result on the same original test: Accuracy 96.64% → 98.47%, Precision 55.90% → 74.01%, F1 0.7157 → 0.8465 and ROC-AUC 0.9976 → 0.9987. Recall changes from 99.44% to 98.86% (−0.58 percentage points), so the improvement is mainly substantially fewer false positives while retaining very high recall.

The author's D2 PCA result is reported on a balanced test set at approximately 98% accuracy, 99% recall and 0.98 F1. The new figures above are measured on the unchanged natural-prevalence test set (4.26% positive), so precision and F1 must not be compared as if the test distributions were identical.

## Limitations

- The final D2 run used six random-search candidates per variant to complete reliably on the full data; the exact run budget is recorded in `metadata.json`.
- Thresholds are selected on an inner held-out portion of the outer training set and transferred to the full-train refit.
- The test split is reused from earlier experiments and is not external validation.
- The public source ZIP is recorded by SHA-256 in `metadata.json`; raw data are not committed.

Exact search results, selected parameters and test predictions are stored in the adjacent CSV and JSON files.
