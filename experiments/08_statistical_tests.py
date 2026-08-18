import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from src.pca import load_npz_dataset
from src.statistics import (
    build_cv,
    run_cv_scores,
    statistical_tests,
)
from src.tuning import get_sampler


def print_title(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_best_params(
    dataset_name,
):
    path = (
        Path("results")
        / dataset_name
        / "05_hyperparameter_search"
        / "best_params.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return (
        metadata[
            "tuning"
        ][
            "best_params"
        ]
    )


def build_xgboost(
    best_params,
    random_state=42,
):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        **best_params,
    )


def build_without_pca_pipeline(
    dataset_name,
    best_params,
    random_state=42,
):
    sampler = get_sampler(
        dataset_name=dataset_name,
        random_state=random_state,
    )

    model = build_xgboost(
        best_params=best_params,
        random_state=random_state,
    )

    return Pipeline([
        (
            "sampler",
            sampler,
        ),
        (
            "model",
            model,
        ),
    ])


def build_with_pca_pipeline(
    dataset_name,
    best_params,
    random_state=42,
):
    sampler = get_sampler(
        dataset_name=dataset_name,
        random_state=random_state,
    )

    pca = PCA(
        n_components=0.95,
        svd_solver="full",
    )

    model = build_xgboost(
        best_params=best_params,
        random_state=random_state,
    )

    return Pipeline([
        (
            "sampler",
            sampler,
        ),
        (
            "pca",
            pca,
        ),
        (
            "model",
            model,
        ),
    ])


def print_cv_summary(
    name,
    df,
):
    print()
    print(name)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
        "cohen_kappa",
    ]

    for metric in metrics:
        mean = df[
            metric
        ].mean()

        std = df[
            metric
        ].std()

        print(
            f"{metric:15s}: "
            f"{mean:.6f} "
            f"± {std:.6f}"
        )


def save_metric_comparison_plot(
    test_results,
    output_path,
):
    plot_df = test_results[
        [
            "metric",
            "without_pca_mean",
            "with_pca_mean",
        ]
    ].copy()

    plot_df = plot_df.set_index(
        "metric"
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_ylabel(
        "Mean 10-fold CV score"
    )

    ax.set_xlabel(
        "Metric"
    )

    ax.set_title(
        "Cross-validation performance: "
        "Without PCA vs With PCA"
    )

    ax.set_ylim(
        bottom=0
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def run_statistical_analysis(
    dataset_name,
    cv_splits=10,
    n_jobs=1,
):
    output_dir = (
        Path("results")
        / dataset_name
        / "08_statistical_tests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    data = load_npz_dataset(
        dataset_name
    )

    X_train = data[
        "X_train"
    ]

    y_train = data[
        "y_train"
    ]

    print_title(
        f"{dataset_name.upper()} - LOAD DATA"
    )

    print(
        f"X_train       : "
        f"{X_train.shape}"
    )

    print(
        f"Positive      : "
        f"{y_train.sum():,}"
    )

    print(
        f"Negative      : "
        f"{len(y_train) - y_train.sum():,}"
    )

    # ============================================================
    # PARAMETERS
    # ============================================================

    best_params = load_best_params(
        dataset_name
    )

    print_title(
        "FIXED XGBOOST PARAMETERS"
    )

    for key, value in best_params.items():
        print(
            f"{key:20s}: "
            f"{value}"
        )

    print()
    print(
        "Only experimental difference: "
        "PCA ON / OFF"
    )

    # ============================================================
    # BUILD PIPELINES
    # ============================================================

    without_pca = (
        build_without_pca_pipeline(
            dataset_name=
                dataset_name,
            best_params=
                best_params,
            random_state=42,
        )
    )

    with_pca = (
        build_with_pca_pipeline(
            dataset_name=
                dataset_name,
            best_params=
                best_params,
            random_state=42,
        )
    )

    # ============================================================
    # IMPORTANT:
    # Create fold indices ONCE.
    #
    # Passing the same list of splits to both models ensures
    # that fold 1 without PCA is directly paired with
    # fold 1 with PCA, etc.
    # ============================================================

    cv_generator = build_cv(
        n_splits=cv_splits,
        random_state=42,
    )

    cv_splits_list = list(
        cv_generator.split(
            X_train,
            y_train,
        )
    )

    # ============================================================
    # WITHOUT PCA
    # ============================================================

    print_title(
        f"{cv_splits}-FOLD CV - WITHOUT PCA"
    )

    (
        scores_without,
        time_without,
    ) = run_cv_scores(
        estimator=without_pca,
        X=X_train,
        y=y_train,
        cv=cv_splits_list,
        n_jobs=n_jobs,
    )

    print(
        scores_without.to_string(
            index=False
        )
    )

    print_cv_summary(
        "WITHOUT PCA SUMMARY",
        scores_without,
    )

    print(
        f"\nCV time: "
        f"{time_without:.2f} s"
    )

    # ============================================================
    # WITH PCA
    # ============================================================

    print_title(
        f"{cv_splits}-FOLD CV - WITH PCA"
    )

    (
        scores_with,
        time_with,
    ) = run_cv_scores(
        estimator=with_pca,
        X=X_train,
        y=y_train,
        cv=cv_splits_list,
        n_jobs=n_jobs,
    )

    print(
        scores_with.to_string(
            index=False
        )
    )

    print_cv_summary(
        "WITH PCA SUMMARY",
        scores_with,
    )

    print(
        f"\nCV time: "
        f"{time_with:.2f} s"
    )

    # ============================================================
    # STATISTICAL TESTS
    # ============================================================

    print_title(
        "STATISTICAL TESTS"
    )

    test_results = statistical_tests(
        scores_without_pca=
            scores_without,
        scores_with_pca=
            scores_with,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    print(
        test_results.to_string(
            index=False
        )
    )

    # ============================================================
    # PRIMARY PAPER-LIKE RESULT
    # ============================================================

    accuracy_row = (
        test_results[
            test_results[
                "metric"
            ] == "accuracy"
        ]
        .iloc[0]
    )

    print_title(
        "PRIMARY RESULT - ACCURACY"
    )

    print(
        f"Without PCA mean     : "
        f"{accuracy_row['without_pca_mean']:.6f}"
    )

    print(
        f"With PCA mean        : "
        f"{accuracy_row['with_pca_mean']:.6f}"
    )

    print(
        f"Mean difference      : "
        f"{accuracy_row['mean_difference']:+.6f}"
    )

    print()
    print(
        f"Paired t-test p-value: "
        f"{accuracy_row['paired_t_pvalue']:.12g}"
    )

    print(
        f"Mann-Whitney p-value : "
        f"{accuracy_row['mannwhitney_u_pvalue']:.12g}"
    )

    print()

    if (
        accuracy_row[
            "paired_t_pvalue"
        ] < 0.05
    ):
        print(
            "Paired t-test        : "
            "SIGNIFICANT (p < 0.05)"
        )
    else:
        print(
            "Paired t-test        : "
            "NOT SIGNIFICANT"
        )

    if (
        accuracy_row[
            "mannwhitney_u_pvalue"
        ] < 0.05
    ):
        print(
            "Mann-Whitney U       : "
            "SIGNIFICANT (p < 0.05)"
        )
    else:
        print(
            "Mann-Whitney U       : "
            "NOT SIGNIFICANT"
        )

    # ============================================================
    # SAVE FOLD SCORES
    # ============================================================

    scores_without.to_csv(
        output_dir
        / "cv_without_pca.csv",
        index=False,
    )

    scores_with.to_csv(
        output_dir
        / "cv_with_pca.csv",
        index=False,
    )

    paired = scores_without[
        [
            "fold"
        ]
    ].copy()

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
        "cohen_kappa",
    ]:
        paired[
            f"{metric}_without_pca"
        ] = scores_without[
            metric
        ]

        paired[
            f"{metric}_with_pca"
        ] = scores_with[
            metric
        ]

        paired[
            f"{metric}_difference"
        ] = (
            scores_with[
                metric
            ]
            - scores_without[
                metric
            ]
        )

    paired.to_csv(
        output_dir
        / "paired_fold_scores.csv",
        index=False,
    )

    test_results.to_csv(
        output_dir
        / "statistical_tests.csv",
        index=False,
    )

    # ============================================================
    # FIGURE
    # ============================================================

    save_metric_comparison_plot(
        test_results=
            test_results,
        output_path=(
            output_dir
            / "pca_vs_no_pca_metrics.png"
        ),
    )

    # ============================================================
    # METADATA
    # ============================================================

    metadata = {
        "dataset":
            dataset_name,

        "cv_folds":
            cv_splits,

        "random_state":
            42,

        "n_jobs":
            n_jobs,

        "comparison": {
            "without_pca":
                "Sampler -> XGBoost",

            "with_pca":
                "Sampler -> PCA -> XGBoost",
        },

        "same_cv_folds":
            True,

        "same_xgboost_params":
            True,

        "xgboost_params":
            best_params,

        "cv_time_without_pca":
            float(
                time_without
            ),

        "cv_time_with_pca":
            float(
                time_with
            ),

        "primary_metric":
            "accuracy",

        "primary_result": {
            "without_pca_mean":
                float(
                    accuracy_row[
                        "without_pca_mean"
                    ]
                ),

            "with_pca_mean":
                float(
                    accuracy_row[
                        "with_pca_mean"
                    ]
                ),

            "mean_difference":
                float(
                    accuracy_row[
                        "mean_difference"
                    ]
                ),

            "paired_t_pvalue":
                float(
                    accuracy_row[
                        "paired_t_pvalue"
                    ]
                ),

            "mannwhitney_u_pvalue":
                float(
                    accuracy_row[
                        "mannwhitney_u_pvalue"
                    ]
                ),
        },

        "note": (
            "Paired t-test is directly appropriate "
            "because PCA and non-PCA results use the "
            "same cross-validation folds. "
            "Mann-Whitney U is included to reproduce "
            "the statistical tests reported in the paper, "
            "although it assumes independent samples."
        ),
    }

    with open(
        output_dir
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print_title(
        "OUTPUT"
    )

    print(
        f"Saved to: "
        f"{output_dir.resolve()}"
    )

    print()
    print("Files:")

    for file in sorted(
        output_dir.iterdir()
    ):
        print(
            f"  - {file.name}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "08 - Statistical comparison "
            "of XGBoost with and without PCA"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "dataset1",
            "dataset2",
        ],
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of CV jobs. "
            "Use 1 or 2 for dataset2 "
            "to avoid excessive CPU/RAM."
        ),
    )

    args = parser.parse_args()

    run_statistical_analysis(
        dataset_name=
            args.dataset,
        cv_splits=
            args.cv,
        n_jobs=
            args.n_jobs,
    )


if __name__ == "__main__":
    main()