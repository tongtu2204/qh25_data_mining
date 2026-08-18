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

from src.pca import load_npz_dataset
from src.xgboost_model import (
    build_xgboost_baseline,
    evaluate_binary_classifier,
    fit_model,
    predict_model,
)


def print_title(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def load_pca_dataset(
    dataset_name,
):
    path = (
        Path("data/processed")
        / f"{dataset_name}_pca95_train_test.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}\n"
            "Hãy chạy bước 03_pca_analysis trước."
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    return {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "path": path,
    }


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
        cmap=None,
        ax=ax,
        colorbar=False,
    )

    ax.set_title(title)

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

    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_classification_report(
    report,
    output_path,
):
    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class"
            }
        )
    )

    report_df.to_csv(
        output_path,
        index=False,
    )


def run_single_experiment(
    dataset_name,
    variant_name,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir,
):
    print_title(
        f"{dataset_name.upper()} - {variant_name.upper()}"
    )

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"Train positive rate : "
        f"{y_train.mean():.6f}"
    )

    print(
        f"Test positive rate  : "
        f"{y_test.mean():.6f}"
    )

    model = build_xgboost_baseline(
        random_state=42,
        n_jobs=-1,
    )

    model, train_time = fit_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    (
        y_pred,
        y_prob,
        predict_time,
    ) = predict_model(
        model=model,
        X_test=X_test,
    )

    (
        metrics,
        cm,
        report,
    ) = evaluate_binary_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    metrics["dataset"] = dataset_name
    metrics["variant"] = variant_name
    metrics["train_rows"] = int(
        X_train.shape[0]
    )
    metrics["test_rows"] = int(
        X_test.shape[0]
    )
    metrics["features"] = int(
        X_train.shape[1]
    )
    metrics["train_time"] = float(
        train_time
    )
    metrics["predict_time"] = float(
        predict_time
    )

    print()
    print(
        f"Accuracy      : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision     : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall        : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1            : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC       : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"MCC           : "
        f"{metrics['mcc']:.6f}"
    )

    print(
        f"Cohen Kappa   : "
        f"{metrics['cohen_kappa']:.6f}"
    )

    print(
        f"Train time    : "
        f"{train_time:.4f} s"
    )

    print(
        f"Predict time  : "
        f"{predict_time:.4f} s"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    variant_dir = (
        output_dir
        / variant_name
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        variant_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=[
            "predicted_0",
            "predicted_1",
        ],
    ).to_csv(
        variant_dir
        / "confusion_matrix.csv"
    )

    save_classification_report(
        report=report,
        output_path=(
            variant_dir
            / "classification_report.csv"
        ),
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=(
            variant_dir
            / "confusion_matrix.png"
        ),
        title=(
            f"{dataset_name} - "
            f"{variant_name}"
        ),
    )

    save_roc_curve(
        y_true=y_test,
        y_prob=y_prob,
        output_path=(
            variant_dir
            / "roc_curve.png"
        ),
        title=(
            f"{dataset_name} - "
            f"{variant_name}"
        ),
    )

    return metrics


def run_baseline(
    dataset_name,
):
    output_dir = (
        Path("results")
        / dataset_name
        / "04_xgboost_baseline"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data = load_npz_dataset(
        dataset_name
    )

    pca_data = load_pca_dataset(
        dataset_name
    )

    results = []

    metrics_no_pca = (
        run_single_experiment(
            dataset_name=dataset_name,
            variant_name="without_pca",
            X_train=raw_data[
                "X_train"
            ],
            X_test=raw_data[
                "X_test"
            ],
            y_train=raw_data[
                "y_train"
            ],
            y_test=raw_data[
                "y_test"
            ],
            output_dir=output_dir,
        )
    )

    results.append(
        metrics_no_pca
    )

    metrics_pca = (
        run_single_experiment(
            dataset_name=dataset_name,
            variant_name="with_pca",
            X_train=pca_data[
                "X_train"
            ],
            X_test=pca_data[
                "X_test"
            ],
            y_train=pca_data[
                "y_train"
            ],
            y_test=pca_data[
                "y_test"
            ],
            output_dir=output_dir,
        )
    )

    results.append(
        metrics_pca
    )

    summary = pd.DataFrame(
        results
    )

    columns = [
        "dataset",
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

    summary = summary[
        columns
    ]

    summary.to_csv(
        output_dir
        / "baseline_summary.csv",
        index=False,
    )

    print_title(
        f"{dataset_name.upper()} - BASELINE SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: "
        f"{output_dir.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "04 - XGBoost baseline "
            "with and without PCA"
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

    args = parser.parse_args()

    run_baseline(
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()