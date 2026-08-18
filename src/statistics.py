import time

import numpy as np
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    ttest_rel,
)
from sklearn.metrics import (
    cohen_kappa_score,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


def get_scoring():
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "mcc": make_scorer(
            matthews_corrcoef
        ),
        "cohen_kappa": make_scorer(
            cohen_kappa_score
        ),
    }


def run_cv_scores(
    estimator,
    X,
    y,
    cv,
    n_jobs=1,
):
    start = time.perf_counter()

    scores = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=get_scoring(),
        n_jobs=n_jobs,
        return_train_score=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    rows = []

    n_folds = len(
        scores["test_accuracy"]
    )

    for i in range(n_folds):
        rows.append({
            "fold":
                i + 1,

            "accuracy":
                float(
                    scores[
                        "test_accuracy"
                    ][i]
                ),

            "precision":
                float(
                    scores[
                        "test_precision"
                    ][i]
                ),

            "recall":
                float(
                    scores[
                        "test_recall"
                    ][i]
                ),

            "f1":
                float(
                    scores[
                        "test_f1"
                    ][i]
                ),

            "roc_auc":
                float(
                    scores[
                        "test_roc_auc"
                    ][i]
                ),

            "mcc":
                float(
                    scores[
                        "test_mcc"
                    ][i]
                ),

            "cohen_kappa":
                float(
                    scores[
                        "test_cohen_kappa"
                    ][i]
                ),

            "fit_time":
                float(
                    scores[
                        "fit_time"
                    ][i]
                ),

            "score_time":
                float(
                    scores[
                        "score_time"
                    ][i]
                ),
        })

    return (
        pd.DataFrame(rows),
        elapsed,
    )


def statistical_tests(
    scores_without_pca,
    scores_with_pca,
):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
        "cohen_kappa",
    ]

    rows = []

    for metric in metrics:
        a = np.asarray(
            scores_without_pca[
                metric
            ],
            dtype=float,
        )

        b = np.asarray(
            scores_with_pca[
                metric
            ],
            dtype=float,
        )

        if len(a) != len(b):
            raise ValueError(
                "Paired samples must have "
                "the same number of folds."
            )

        t_stat, t_pvalue = (
            ttest_rel(
                b,
                a,
            )
        )

        u_stat, u_pvalue = (
            mannwhitneyu(
                b,
                a,
                alternative="two-sided",
            )
        )

        mean_without = float(
            np.mean(a)
        )

        mean_with = float(
            np.mean(b)
        )

        mean_difference = (
            mean_with
            - mean_without
        )

        rows.append({
            "metric":
                metric,

            "without_pca_mean":
                mean_without,

            "with_pca_mean":
                mean_with,

            "mean_difference":
                float(
                    mean_difference
                ),

            "paired_t_stat":
                float(
                    t_stat
                ),

            "paired_t_pvalue":
                float(
                    t_pvalue
                ),

            "mannwhitney_u_stat":
                float(
                    u_stat
                ),

            "mannwhitney_u_pvalue":
                float(
                    u_pvalue
                ),

            "t_significant_0.05":
                bool(
                    t_pvalue < 0.05
                ),

            "mw_significant_0.05":
                bool(
                    u_pvalue < 0.05
                ),
        })

    return pd.DataFrame(
        rows
    )


def build_cv(
    n_splits=10,
    random_state=42,
):
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )