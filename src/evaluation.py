import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


def evaluate_predictions(
    y_true,
    y_pred,
    y_prob,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, cm


def get_cv_scoring():
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }


def run_cross_validation(
    pipeline,
    X,
    y,
    n_splits=10,
    random_state=42,
    n_jobs=-1,
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    start = time.perf_counter()

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=get_cv_scoring(),
        n_jobs=n_jobs,
        return_train_score=False,
    )

    total_time = (
        time.perf_counter()
        - start
    )

    rows = []

    for i in range(n_splits):
        rows.append({
            "fold": i + 1,
            "accuracy": float(
                scores["test_accuracy"][i]
            ),
            "precision": float(
                scores["test_precision"][i]
            ),
            "recall": float(
                scores["test_recall"][i]
            ),
            "f1": float(
                scores["test_f1"][i]
            ),
            "roc_auc": float(
                scores["test_roc_auc"][i]
            ),
            "fit_time": float(
                scores["fit_time"][i]
            ),
            "score_time": float(
                scores["score_time"][i]
            ),
        })

    fold_results = pd.DataFrame(
        rows
    )

    summary = {
        "accuracy_mean":
            float(
                fold_results[
                    "accuracy"
                ].mean()
            ),
        "accuracy_std":
            float(
                fold_results[
                    "accuracy"
                ].std()
            ),
        "precision_mean":
            float(
                fold_results[
                    "precision"
                ].mean()
            ),
        "precision_std":
            float(
                fold_results[
                    "precision"
                ].std()
            ),
        "recall_mean":
            float(
                fold_results[
                    "recall"
                ].mean()
            ),
        "recall_std":
            float(
                fold_results[
                    "recall"
                ].std()
            ),
        "f1_mean":
            float(
                fold_results[
                    "f1"
                ].mean()
            ),
        "f1_std":
            float(
                fold_results[
                    "f1"
                ].std()
            ),
        "roc_auc_mean":
            float(
                fold_results[
                    "roc_auc"
                ].mean()
            ),
        "roc_auc_std":
            float(
                fold_results[
                    "roc_auc"
                ].std()
            ),
        "total_cv_time":
            float(
                total_time
            ),
    }

    return (
        fold_results,
        summary,
    )