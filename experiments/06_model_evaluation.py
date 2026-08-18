import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.evaluation import (
    evaluate_predictions,
    run_cross_validation,
)
from src.pca import load_npz_dataset
from src.tuning import build_tuning_pipeline


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
            f"Không tìm thấy {path}\n"
            "Hãy chạy bước 05 trước."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    best_params = (
        metadata[
            "tuning"
        ][
            "best_params"
        ]
    )

    return (
        best_params,
        metadata,
    )


def apply_best_params(
    pipeline,
    best_params,
):
    pipeline_params = {
        f"model__{key}": value
        for key, value
        in best_params.items()
    }

    pipeline.set_params(
        **pipeline_params
    )

    return pipeline


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    title,
):
    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        colorbar=False,
        ax=ax,
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_roc_curve(
    y_true,
    y_prob,
    output_path,
    title,
):
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def load_previous_results(
    dataset_name,
):
    frames = []

    step04 = (
        Path("results")
        / dataset_name
        / "04_xgboost_baseline"
        / "baseline_summary.csv"
    )

    if step04.exists():
        df04 = pd.read_csv(
            step04
        )

        keep_cols = [
            "variant",
            "features",
            "train_rows",
            "test_rows",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "mcc",
            "cohen_kappa",
            "tn",
            "fp",
            "fn",
            "tp",
            "train_time",
            "predict_time",
        ]

        df04 = df04[
            [
                col
                for col in keep_cols
                if col in df04.columns
            ]
        ].copy()

        df04["stage"] = "04_baseline"

        frames.append(
            df04
        )

    step05 = (
        Path("results")
        / dataset_name
        / "05_hyperparameter_search"
        / "summary.csv"
    )

    if step05.exists():
        df05 = pd.read_csv(
            step05
        )

        keep_cols = [
            "variant",
            "features",
            "train_rows",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "mcc",
            "cohen_kappa",
            "tn",
            "fp",
            "fn",
            "tp",
            "train_time",
            "predict_time",
        ]

        df05 = df05[
            [
                col
                for col in keep_cols
                if col in df05.columns
            ]
        ].copy()

        df05["stage"] = "05_balancing_tuning"

        frames.append(
            df05
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def run_evaluation(
    dataset_name,
    cv_splits=10,
):
    output_dir = (
        Path("results")
        / dataset_name
        / "06_model_evaluation"
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

    X_test = data[
        "X_test"
    ]

    y_train = data[
        "y_train"
    ]

    y_test = data[
        "y_test"
    ]

    print_title(
        f"{dataset_name.upper()} - LOAD DATA"
    )

    print(
        f"X_train       : "
        f"{X_train.shape}"
    )

    print(
        f"X_test        : "
        f"{X_test.shape}"
    )

    print(
        f"Train positive: "
        f"{y_train.sum():,}"
    )

    print(
        f"Test positive : "
        f"{y_test.sum():,}"
    )

    # ============================================================
    # LOAD BEST PARAMS
    # ============================================================

    (
        best_params,
        tuning_metadata,
    ) = load_best_params(
        dataset_name
    )

    print_title(
        "BEST PARAMETERS FROM STEP 05"
    )

    for key, value in best_params.items():
        print(
            f"{key:20s}: "
            f"{value}"
        )

    # ============================================================
    # BUILD FINAL PIPELINE
    # ============================================================

    pipeline = build_tuning_pipeline(
        dataset_name=dataset_name,
        random_state=42,
    )

    pipeline = apply_best_params(
        pipeline,
        best_params,
    )

    # ============================================================
    # 10-FOLD CROSS VALIDATION
    # ============================================================

    print_title(
        f"{cv_splits}-FOLD CROSS VALIDATION"
    )

    (
        fold_results,
        cv_summary,
    ) = run_cross_validation(
        pipeline=pipeline,
        X=X_train,
        y=y_train,
        n_splits=cv_splits,
        random_state=42,
        n_jobs=-1,
    )

    print(
        fold_results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Average accuracy : "
        f"{cv_summary['accuracy_mean']:.6f}"
        f" ± "
        f"{cv_summary['accuracy_std']:.6f}"
    )

    print(
        f"Average precision: "
        f"{cv_summary['precision_mean']:.6f}"
        f" ± "
        f"{cv_summary['precision_std']:.6f}"
    )

    print(
        f"Average recall   : "
        f"{cv_summary['recall_mean']:.6f}"
        f" ± "
        f"{cv_summary['recall_std']:.6f}"
    )

    print(
        f"Average F1       : "
        f"{cv_summary['f1_mean']:.6f}"
        f" ± "
        f"{cv_summary['f1_std']:.6f}"
    )

    print(
        f"Average ROC-AUC  : "
        f"{cv_summary['roc_auc_mean']:.6f}"
        f" ± "
        f"{cv_summary['roc_auc_std']:.6f}"
    )

    print(
        f"Total CV time    : "
        f"{cv_summary['total_cv_time']:.2f} s"
    )

    # ============================================================
    # FIT FINAL PIPELINE ON FULL ORIGINAL TRAIN
    # ============================================================

    print_title(
        "FIT FINAL MODEL"
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    tuned_sampler = (
        pipeline
        .named_steps[
            "sampler"
        ]
    )

    tuned_pca = (
        pipeline
        .named_steps[
            "pca"
        ]
    )

    print(
        f"Sampler            : "
        f"{tuned_sampler.__class__.__name__}"
    )

    print(
        f"PCA components     : "
        f"{tuned_pca.n_components_}"
    )

    print(
        f"Explained variance : "
        f"{tuned_pca.explained_variance_ratio_.sum():.6f}"
    )

    # ============================================================
    # TEST SET
    # ============================================================

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    (
        test_metrics,
        cm,
    ) = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    print_title(
        "FINAL TEST PERFORMANCE"
    )

    print(
        f"Accuracy      : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision     : "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Recall        : "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"F1            : "
        f"{test_metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC       : "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"MCC           : "
        f"{test_metrics['mcc']:.6f}"
    )

    print(
        f"Cohen Kappa   : "
        f"{test_metrics['cohen_kappa']:.6f}"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    # ============================================================
    # SAVE CV
    # ============================================================

    fold_results.to_csv(
        output_dir
        / "cross_validation_folds.csv",
        index=False,
    )

    with open(
        output_dir
        / "cross_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cv_summary,
            f,
            ensure_ascii=False,
            indent=4,
        )

    # ============================================================
    # SAVE TEST METRICS
    # ============================================================

    with open(
        output_dir
        / "final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            test_metrics,
            f,
            ensure_ascii=False,
            indent=4,
        )

    pd.DataFrame(
        cm,
        index=[
            "actual_0",
            "actual_1",
        ],
        columns=[
            "predicted_0",
            "predicted_1",
        ],
    ).to_csv(
        output_dir
        / "final_confusion_matrix.csv"
    )

    # ============================================================
    # FIGURES
    # ============================================================

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=(
            output_dir
            / "final_confusion_matrix.png"
        ),
        title=(
            f"{dataset_name} - "
            "Final Confusion Matrix"
        ),
    )

    save_roc_curve(
        y_true=y_test,
        y_prob=y_prob,
        output_path=(
            output_dir
            / "final_roc_curve.png"
        ),
        title=(
            f"{dataset_name} - "
            "Final ROC Curve"
        ),
    )

    # ============================================================
    # ALL MODEL COMPARISON
    # ============================================================

    comparison = load_previous_results(
        dataset_name
    )

    if not comparison.empty:
        comparison.to_csv(
            output_dir
            / "all_model_comparison.csv",
            index=False,
        )

        print_title(
            "ALL MODEL COMPARISON"
        )

        compare_cols = [
            "stage",
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
            comparison[
                [
                    col
                    for col
                    in compare_cols
                    if col
                    in comparison.columns
                ]
            ].to_string(
                index=False
            )
        )

    # ============================================================
    # FINAL METADATA
    # ============================================================

    metadata = {
        "dataset":
            dataset_name,

        "cv_folds":
            cv_splits,

        "best_params":
            best_params,

        "sampler":
            tuned_sampler
            .__class__
            .__name__,

        "pca_components":
            int(
                tuned_pca
                .n_components_
            ),

        "pca_explained_variance":
            float(
                tuned_pca
                .explained_variance_ratio_
                .sum()
            ),

        "cv":
            cv_summary,

        "test":
            test_metrics,
    }

    with open(
        output_dir
        / "evaluation_metadata.json",
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
            "06 - Final model evaluation "
            "and leakage-safe cross-validation"
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
        help=(
            "Number of stratified CV folds."
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        dataset_name=args.dataset,
        cv_splits=args.cv,
    )


if __name__ == "__main__":
    main()