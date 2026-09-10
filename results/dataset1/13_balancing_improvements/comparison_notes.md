# Experiment 13 — D1 balancing improvements

The experiment compares SMOTE, SMOTENC and class-weighted XGBoost, each with and without PCA. Preprocessing and resampling are fitted only on the training data. Hyperparameters are selected by standardized partial ROC-AUC at maximum FPR 0.10; thresholds are frozen from training OOF predictions before the unchanged test set is evaluated.

The D1 split contains 4,088 training rows and 1,022 test rows; the test set contains only 50 positive cases. Therefore, one additional detected stroke changes recall by two percentage points.

## Test results for validation-selected operating points

| Target specificity | Selected variant | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TP | FP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | weighted + PCA | 0.6704 | 84.34% | 19.44% | 70.00% | 85.08% | 0.3043 | 0.8241 | 35 | 145 |
| 0.90 | weighted, no PCA | 0.1305 | 87.57% | 18.70% | 46.00% | 89.71% | 0.2659 | 0.8263 | 23 | 100 |
| 0.95 | weighted, no PCA | 0.1902 | 91.88% | 23.81% | 30.00% | 95.06% | 0.2655 | 0.8263 | 15 | 48 |

At the default threshold 0.5, the highest F1 among the tested variants is SMOTENC without PCA: Accuracy 92.95%, Precision 29.63%, Recall 32.00%, Specificity 96.09%, F1 0.3077 and ROC-AUC 0.8290. This is a different operating point from the recall-oriented 85% specificity result.

Compared with the earlier split-then-SMOTE + PCA control (Recall 78%, Precision 15.29%, Specificity 77.78%, F1 0.2557), the new 85%-specificity policy trades 8 percentage points of recall for about 7.3 points of specificity, 4.2 points of precision and 0.0486 F1. It is therefore a better-controlled operating point, not a uniform improvement on every metric.

The author's D1 table reports approximately 95% accuracy, 93% precision, 96% recall and 0.95 F1 on a balanced test set. Those figures are not directly comparable with this unchanged, naturally imbalanced test set (50 positives / 1,022 rows).

## Limitations

- The test split is the same previously used split, not an external validation set.
- D1 OOF predictions are generated after hyperparameter selection and are not nested CV.
- The specificity targets are selected on training OOF data; they are not guaranteed on test.
- Because there are only 50 positive test cases, uncertainty around recall is substantial.

Exact search results, selected parameters and row-level predictions are stored in the adjacent CSV, JSON and ignored `local/` files.
