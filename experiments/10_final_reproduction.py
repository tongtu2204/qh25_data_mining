import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")


PAPER_REFERENCE = {
    "dataset1": {
        "paper_cv_accuracy": 0.9532,
    },
    "dataset2": {
        "paper_cv_accuracy": 0.99028,
        "paper_mcc": 0.9634,
        "paper_cohen_kappa": 0.9632,
    },
}


def print_title(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def load_json(path):
    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        return None

    return pd.read_csv(path)


def safe_value(
    obj,
    *keys,
    default=np.nan,
):
    current = obj

    for key in keys:
        if current is None:
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def load_dataset_results(
    dataset_name,
):
    base = (
        RESULTS_DIR
        / dataset_name
    )

    step03 = load_json(
        base
        / "03_pca_analysis"
        / "metadata.json"
    )

    step04 = load_csv(
        base
        / "04_xgboost_baseline"
        / "baseline_summary.csv"
    )

    step05_summary = load_csv(
        base
        / "05_hyperparameter_search"
        / "summary.csv"
    )

    step05_meta = load_json(
        base
        / "05_hyperparameter_search"
        / "best_params.json"
    )

    step06_cv = load_json(
        base
        / "06_model_evaluation"
        / "cross_validation_summary.json"
    )

    step06_test = load_json(
        base
        / "06_model_evaluation"
        / "final_test_metrics.json"
    )

    step07_original = load_csv(
        base
        / "07_shap_analysis"
        / "shap_projected_original_importance.csv"
    )

    step07_pc = load_csv(
        base
        / "07_shap_analysis"
        / "shap_pc_importance.csv"
    )

    step08 = load_csv(
        base
        / "08_statistical_tests"
        / "statistical_tests.csv"
    )

    step08_meta = load_json(
        base
        / "08_statistical_tests"
        / "metadata.json"
    )

    return {
        "step03":
            step03,

        "step04":
            step04,

        "step05_summary":
            step05_summary,

        "step05_meta":
            step05_meta,

        "step06_cv":
            step06_cv,

        "step06_test":
            step06_test,

        "step07_original":
            step07_original,

        "step07_pc":
            step07_pc,

        "step08":
            step08,

        "step08_meta":
            step08_meta,
    }


def build_model_progression(
    dataset_name,
    results,
):
    rows = []

    step04 = results[
        "step04"
    ]

    if step04 is not None:

        for _, row in step04.iterrows():

            rows.append({
                "dataset":
                    dataset_name,

                "stage":
                    "baseline",

                "variant":
                    row[
                        "variant"
                    ],

                "features":
                    row.get(
                        "features",
                        np.nan,
                    ),

                "accuracy":
                    row.get(
                        "accuracy",
                        np.nan,
                    ),

                "precision":
                    row.get(
                        "precision",
                        np.nan,
                    ),

                "recall":
                    row.get(
                        "recall",
                        np.nan,
                    ),

                "f1":
                    row.get(
                        "f1",
                        np.nan,
                    ),

                "roc_auc":
                    row.get(
                        "roc_auc",
                        np.nan,
                    ),

                "mcc":
                    row.get(
                        "mcc",
                        np.nan,
                    ),

                "cohen_kappa":
                    row.get(
                        "cohen_kappa",
                        np.nan,
                    ),

                "train_time":
                    row.get(
                        "train_time",
                        np.nan,
                    ),
            })

    step05 = results[
        "step05_summary"
    ]

    if step05 is not None:

        for _, row in step05.iterrows():

            rows.append({
                "dataset":
                    dataset_name,

                "stage":
                    "balancing_tuning",

                "variant":
                    row[
                        "variant"
                    ],

                "features":
                    row.get(
                        "features",
                        np.nan,
                    ),

                "accuracy":
                    row.get(
                        "accuracy",
                        np.nan,
                    ),

                "precision":
                    row.get(
                        "precision",
                        np.nan,
                    ),

                "recall":
                    row.get(
                        "recall",
                        np.nan,
                    ),

                "f1":
                    row.get(
                        "f1",
                        np.nan,
                    ),

                "roc_auc":
                    row.get(
                        "roc_auc",
                        np.nan,
                    ),

                "mcc":
                    row.get(
                        "mcc",
                        np.nan,
                    ),

                "cohen_kappa":
                    row.get(
                        "cohen_kappa",
                        np.nan,
                    ),

                "train_time":
                    row.get(
                        "train_time",
                        np.nan,
                    ),
            })

    return pd.DataFrame(
        rows
    )


def build_final_model_summary(
    dataset_name,
    results,
):
    cv = results[
        "step06_cv"
    ]

    test = results[
        "step06_test"
    ]

    meta05 = results[
        "step05_meta"
    ]

    paper = PAPER_REFERENCE[
        dataset_name
    ]

    return {
        "dataset":
            dataset_name,

        "sampler":
            safe_value(
                meta05,
                "tuned_pipeline",
                "sampler",
            ),

        "pca_components":
            safe_value(
                meta05,
                "tuned_pipeline",
                "pca_components",
            ),

        "pca_explained_variance":
            safe_value(
                meta05,
                "tuned_pipeline",
                "pca_explained_variance",
            ),

        "cv_accuracy":
            safe_value(
                cv,
                "accuracy_mean",
            ),

        "cv_accuracy_std":
            safe_value(
                cv,
                "accuracy_std",
            ),

        "cv_precision":
            safe_value(
                cv,
                "precision_mean",
            ),

        "cv_recall":
            safe_value(
                cv,
                "recall_mean",
            ),

        "cv_f1":
            safe_value(
                cv,
                "f1_mean",
            ),

        "cv_roc_auc":
            safe_value(
                cv,
                "roc_auc_mean",
            ),

        "test_accuracy":
            safe_value(
                test,
                "accuracy",
            ),

        "test_precision":
            safe_value(
                test,
                "precision",
            ),

        "test_recall":
            safe_value(
                test,
                "recall",
            ),

        "test_f1":
            safe_value(
                test,
                "f1",
            ),

        "test_roc_auc":
            safe_value(
                test,
                "roc_auc",
            ),

        "test_mcc":
            safe_value(
                test,
                "mcc",
            ),

        "test_cohen_kappa":
            safe_value(
                test,
                "cohen_kappa",
            ),

        "paper_cv_accuracy":
            paper.get(
                "paper_cv_accuracy",
                np.nan,
            ),

        "paper_mcc":
            paper.get(
                "paper_mcc",
                np.nan,
            ),

        "paper_cohen_kappa":
            paper.get(
                "paper_cohen_kappa",
                np.nan,
            ),
    }


def build_pca_statistical_summary(
    dataset_name,
    results,
):
    df = results[
        "step08"
    ]

    if df is None:
        return pd.DataFrame()

    df = df.copy()

    df.insert(
        0,
        "dataset",
        dataset_name,
    )

    return df


def build_shap_summary(
    dataset_name,
    results,
    top_n=10,
):
    df = results[
        "step07_original"
    ]

    if df is None:
        return pd.DataFrame()

    output = (
        df
        .head(top_n)
        .copy()
    )

    output.insert(
        0,
        "rank",
        np.arange(
            1,
            len(output) + 1,
        ),
    )

    output.insert(
        0,
        "dataset",
        dataset_name,
    )

    return output


def build_pc_shap_summary(
    dataset_name,
    results,
):
    df = results[
        "step07_pc"
    ]

    if df is None:
        return pd.DataFrame()

    output = df.copy()

    output.insert(
        0,
        "dataset",
        dataset_name,
    )

    return output


def load_openmp_benchmark():
    path = (
        RESULTS_DIR
        / "pca_benchmark"
        / "benchmark_results.csv"
    )

    return load_csv(
        path
    )


def save_model_metric_plot(
    progression,
    output_path,
):
    metrics = [
        "accuracy",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
    ]

    data = (
        progression
        .set_index(
            "variant"
        )[
            metrics
        ]
    )

    ax = data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Model configuration"
    )

    ax.set_title(
        "Model performance progression"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_final_dataset_comparison(
    final_summary,
    output_path,
):
    metrics = [
        "test_accuracy",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_mcc",
    ]

    data = (
        final_summary
        .set_index(
            "dataset"
        )[
            metrics
        ]
    )

    ax = data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Final model comparison"
    )

    plt.xticks(
        rotation=0,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_openmp_plot(
    benchmark,
    output_path,
):
    if benchmark is None:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        benchmark["n"],
        benchmark["speedup"],
        marker="o",
    )

    ax.axhline(
        1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Matrix size"
    )

    ax.set_ylabel(
        "Speedup"
    )

    ax.set_title(
        "OpenMP Jacobi speedup"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_dataset_summary(
    dataset_name,
    progression,
    final_row,
    shap,
):
    print_title(
        f"{dataset_name.upper()} - MODEL PROGRESSION"
    )

    show_cols = [
        "variant",
        "features",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mcc",
        "cohen_kappa",
    ]

    print(
        progression[
            show_cols
        ].to_string(
            index=False
        )
    )

    print_title(
        f"{dataset_name.upper()} - FINAL MODEL"
    )

    print(
        f"Sampler               : "
        f"{final_row['sampler']}"
    )

    print(
        f"PCA components        : "
        f"{final_row['pca_components']}"
    )

    print(
        f"PCA explained variance: "
        f"{final_row['pca_explained_variance']:.6f}"
    )

    print()
    print(
        f"CV Accuracy           : "
        f"{final_row['cv_accuracy']:.6f}"
    )

    print(
        f"CV Recall             : "
        f"{final_row['cv_recall']:.6f}"
    )

    print(
        f"CV F1                 : "
        f"{final_row['cv_f1']:.6f}"
    )

    print(
        f"CV ROC-AUC            : "
        f"{final_row['cv_roc_auc']:.6f}"
    )

    print()
    print(
        f"Test Accuracy         : "
        f"{final_row['test_accuracy']:.6f}"
    )

    print(
        f"Test Precision        : "
        f"{final_row['test_precision']:.6f}"
    )

    print(
        f"Test Recall           : "
        f"{final_row['test_recall']:.6f}"
    )

    print(
        f"Test F1               : "
        f"{final_row['test_f1']:.6f}"
    )

    print(
        f"Test ROC-AUC          : "
        f"{final_row['test_roc_auc']:.6f}"
    )

    print(
        f"Test MCC              : "
        f"{final_row['test_mcc']:.6f}"
    )

    print(
        f"Test Cohen Kappa      : "
        f"{final_row['test_cohen_kappa']:.6f}"
    )

    if not shap.empty:

        print_title(
            f"{dataset_name.upper()} - TOP PROJECTED FEATURES"
        )

        print(
            shap.to_string(
                index=False
            )
        )


def run_final_summary():
    output_dir = (
        RESULTS_DIR
        / "final_reproduction"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = [
        "dataset1",
        "dataset2",
    ]

    all_progression = []
    all_final = []
    all_stats = []
    all_shap = []
    all_pc_shap = []

    for dataset_name in datasets:

        results = (
            load_dataset_results(
                dataset_name
            )
        )

        progression = (
            build_model_progression(
                dataset_name,
                results,
            )
        )

        final_row = (
            build_final_model_summary(
                dataset_name,
                results,
            )
        )

        stats = (
            build_pca_statistical_summary(
                dataset_name,
                results,
            )
        )

        shap = (
            build_shap_summary(
                dataset_name,
                results,
                top_n=10,
            )
        )

        pc_shap = (
            build_pc_shap_summary(
                dataset_name,
                results,
            )
        )

        all_progression.append(
            progression
        )

        all_final.append(
            final_row
        )

        if not stats.empty:
            all_stats.append(
                stats
            )

        if not shap.empty:
            all_shap.append(
                shap
            )

        if not pc_shap.empty:
            all_pc_shap.append(
                pc_shap
            )

        print_dataset_summary(
            dataset_name=
                dataset_name,
            progression=
                progression,
            final_row=
                final_row,
            shap=
                shap,
        )

        save_model_metric_plot(
            progression=
                progression,
            output_path=(
                output_dir
                / (
                    f"{dataset_name}_"
                    "model_progression.png"
                )
            ),
        )

    progression_df = pd.concat(
        all_progression,
        ignore_index=True,
    )

    final_df = pd.DataFrame(
        all_final
    )

    progression_df.to_csv(
        output_dir
        / "model_progression.csv",
        index=False,
    )

    final_df.to_csv(
        output_dir
        / "final_model_summary.csv",
        index=False,
    )

    if all_stats:

        stats_df = pd.concat(
            all_stats,
            ignore_index=True,
        )

        stats_df.to_csv(
            output_dir
            / "pca_statistical_summary.csv",
            index=False,
        )

    if all_shap:

        shap_df = pd.concat(
            all_shap,
            ignore_index=True,
        )

        shap_df.to_csv(
            output_dir
            / "top_original_features.csv",
            index=False,
        )

    if all_pc_shap:

        pc_shap_df = pd.concat(
            all_pc_shap,
            ignore_index=True,
        )

        pc_shap_df.to_csv(
            output_dir
            / "pc_shap_importance.csv",
            index=False,
        )

    benchmark = (
        load_openmp_benchmark()
    )

    if benchmark is not None:

        benchmark.to_csv(
            output_dir
            / "openmp_benchmark_summary.csv",
            index=False,
        )

        save_openmp_plot(
            benchmark=
                benchmark,
            output_path=(
                output_dir
                / "openmp_speedup.png"
            ),
        )

    save_final_dataset_comparison(
        final_summary=
            final_df,
        output_path=(
            output_dir
            / "final_dataset_comparison.png"
        ),
    )

    # ============================================================
    # PAPER COMPARISON
    # ============================================================

    comparison_rows = []

    for _, row in final_df.iterrows():

        comparison_rows.append({
            "dataset":
                row["dataset"],

            "metric":
                "CV Accuracy",

            "paper":
                row[
                    "paper_cv_accuracy"
                ],

            "reproduction":
                row[
                    "cv_accuracy"
                ],

            "difference":
                row[
                    "cv_accuracy"
                ]
                -
                row[
                    "paper_cv_accuracy"
                ],
        })

        if (
            not pd.isna(
                row[
                    "paper_mcc"
                ]
            )
        ):
            comparison_rows.append({
                "dataset":
                    row["dataset"],

                "metric":
                    "MCC",

                "paper":
                    row[
                        "paper_mcc"
                    ],

                "reproduction":
                    row[
                        "test_mcc"
                    ],

                "difference":
                    row[
                        "test_mcc"
                    ]
                    -
                    row[
                        "paper_mcc"
                    ],
            })

        if (
            not pd.isna(
                row[
                    "paper_cohen_kappa"
                ]
            )
        ):
            comparison_rows.append({
                "dataset":
                    row["dataset"],

                "metric":
                    "Cohen Kappa",

                "paper":
                    row[
                        "paper_cohen_kappa"
                    ],

                "reproduction":
                    row[
                        "test_cohen_kappa"
                    ],

                "difference":
                    row[
                        "test_cohen_kappa"
                    ]
                    -
                    row[
                        "paper_cohen_kappa"
                    ],
            })

    paper_comparison = pd.DataFrame(
        comparison_rows
    )

    paper_comparison.to_csv(
        output_dir
        / "paper_comparison.csv",
        index=False,
    )

    # ============================================================
    # PRINT FINAL OVERVIEW
    # ============================================================

    print_title(
        "FINAL REPRODUCTION SUMMARY"
    )

    show_cols = [
        "dataset",
        "cv_accuracy",
        "cv_recall",
        "cv_f1",
        "cv_roc_auc",
        "test_accuracy",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_mcc",
        "test_cohen_kappa",
    ]

    print(
        final_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    print_title(
        "PAPER VS REPRODUCTION"
    )

    print(
        paper_comparison.to_string(
            index=False
        )
    )

    if benchmark is not None:

        print_title(
            "OPENMP BENCHMARK"
        )

        show_benchmark_cols = [
            "n",
            "serial_seconds",
            "openmp_seconds",
            "speedup",
            "relative_eigenvalue_error",
            "openmp_orthogonality_error",
        ]

        print(
            benchmark[
                show_benchmark_cols
            ].to_string(
                index=False
            )
        )

    print_title(
        "OUTPUT FILES"
    )

    for path in sorted(
        output_dir.iterdir()
    ):
        print(
            f"- {path.name}"
        )

    print()
    print(
        f"Saved to: "
        f"{output_dir.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "10 - Final reproduction summary"
        )
    )

    parser.parse_args()

    run_final_summary()


if __name__ == "__main__":
    main()