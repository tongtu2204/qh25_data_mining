import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.balancing import (
    balance_training_data,
)
from src.pca import load_npz_dataset
from src.tuning import run_grid_search
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


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
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

    return (
        metrics,
        cm,
        report,
        train_time,
        predict_time,
    )


def get_tuning_subset(
    X,
    y,
    max_rows,
    random_state=42,
):
    if max_rows is None:
        return X, y

    if len(y) <= max_rows:
        return X, y

    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        stratify=y,
        random_state=random_state,
    )

    return (
        X_subset,
        y_subset,
    )


def run_experiment(
    dataset_name,
    max_tuning_rows=None,
):
    output_dir = (
        Path("results")
        / dataset_name
        / "05_hyperparameter_search"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_npz_dataset(
        dataset_name
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print_title(
        f"{dataset_name.upper()} - ORIGINAL TRAIN"
    )

    print(
        f"X_train             : {X_train.shape}"
    )

    print(
        f"Positive            : {y_train.sum():,}"
    )

    print(
        f"Negative            : "
        f"{len(y_train) - y_train.sum():,}"
    )

    print(
        f"Positive rate       : "
        f"{y_train.mean():.6f}"
    )

    print_title(
        "CLASS BALANCING"
    )

    start = time.perf_counter()

    (
        X_balanced,
        y_balanced,
        sampler,
    ) = balance_training_data(
        X_train=X_train,
        y_train=y_train,
        dataset_name=dataset_name,
        random_state=42,
    )

    balance_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Method              : "
        f"{sampler.__class__.__name__}"
    )

    print(
        f"Rows before         : "
        f"{len(y_train):,}"
    )

    print(
        f"Rows after          : "
        f"{len(y_balanced):,}"
    )

    print(
        f"Positive after      : "
        f"{y_balanced.sum():,}"
    )

    print(
        f"Negative after      : "
        f"{len(y_balanced) - y_balanced.sum():,}"
    )

    print(
        f"Positive rate after : "
        f"{y_balanced.mean():.6f}"
    )

    print(
        f"Balance time        : "
        f"{balance_time:.4f} s"
    )

    summary_rows = []

    # ============================================================
    # VARIANT 1 - BALANCE ONLY
    # ============================================================

    print_title(
        f"{dataset_name.upper()} - BALANCE ONLY"
    )

    baseline_model = build_xgboost_baseline(
        random_state=42,
        n_jobs=-1,
    )

    (
        metrics_balance,
        cm_balance,
        _,
        train_time_balance,
        predict_time_balance,
    ) = evaluate_model(
        model=baseline_model,
        X_train=X_balanced,
        y_train=y_balanced,
        X_test=X_test,
        y_test=y_test,
    )

    metrics_balance.update({
        "variant": "balance_only",
        "features": int(
            X_balanced.shape[1]
        ),
        "train_rows": int(
            len(y_balanced)
        ),
        "train_time": float(
            train_time_balance
        ),
        "predict_time": float(
            predict_time_balance
        ),
    })

    summary_rows.append(
        metrics_balance
    )

    print_metrics(
        metrics_balance,
        cm_balance,
    )

    # ============================================================
    # PCA AFTER BALANCING
    # ============================================================

    print_title(
        "PCA AFTER BALANCING"
    )

    pca = PCA(
        n_components=0.95,
        svd_solver="full",
    )

    start = time.perf_counter()

    X_balanced_pca = pca.fit_transform(
        X_balanced
    )

    X_test_pca = pca.transform(
        X_test
    )

    pca_time = (
        time.perf_counter()
        - start
    )

    X_balanced_pca = np.asarray(
        X_balanced_pca,
        dtype=np.float32,
    )

    X_test_pca = np.asarray(
        X_test_pca,
        dtype=np.float32,
    )

    print(
        f"Input features      : "
        f"{X_balanced.shape[1]}"
    )

    print(
        f"PCA components      : "
        f"{pca.n_components_}"
    )

    print(
        f"Explained variance  : "
        f"{pca.explained_variance_ratio_.sum():.6f}"
    )

    print(
        f"PCA time            : "
        f"{pca_time:.4f} s"
    )

    # ============================================================
    # VARIANT 2 - BALANCE + PCA
    # ============================================================

    print_title(
        f"{dataset_name.upper()} - BALANCE + PCA"
    )

    pca_model = build_xgboost_baseline(
        random_state=42,
        n_jobs=-1,
    )

    (
        metrics_pca,
        cm_pca,
        _,
        train_time_pca,
        predict_time_pca,
    ) = evaluate_model(
        model=pca_model,
        X_train=X_balanced_pca,
        y_train=y_balanced,
        X_test=X_test_pca,
        y_test=y_test,
    )

    metrics_pca.update({
        "variant": "balance_pca",
        "features": int(
            X_balanced_pca.shape[1]
        ),
        "train_rows": int(
            len(y_balanced)
        ),
        "train_time": float(
            train_time_pca
        ),
        "predict_time": float(
            predict_time_pca
        ),
    })

    summary_rows.append(
        metrics_pca
    )

    print_metrics(
        metrics_pca,
        cm_pca,
    )

    # ============================================================
    # GRID SEARCH
    # ============================================================

    print_title(
        "GRID SEARCH - BALANCE + PCA"
    )

    (
        X_tune,
        y_tune,
    ) = get_tuning_subset(
        X=X_balanced_pca,
        y=y_balanced,
        max_rows=max_tuning_rows,
        random_state=42,
    )

    print(
        f"Full balanced rows  : "
        f"{len(y_balanced):,}"
    )

    print(
        f"Tuning rows         : "
        f"{len(y_tune):,}"
    )

    (
        best_model,
        best_params,
        best_cv_auc,
        tuning_time,
        grid_results,
    ) = run_grid_search(
        X_train=X_tune,
        y_train=y_tune,
        cv_splits=3,
        random_state=42,
        n_jobs=-1,
    )

    print()
    print(
        f"Best CV ROC-AUC     : "
        f"{best_cv_auc:.6f}"
    )

    print(
        f"Tuning time         : "
        f"{tuning_time:.4f} s"
    )

    print()
    print("Best parameters:")

    for key, value in best_params.items():
        print(
            f"  {key:20s}: {value}"
        )

    # GridSearch best estimator mới fit trên tuning subset.
    # Fit lại trên FULL balanced PCA train.

    print_title(
        "REFIT BEST MODEL ON FULL BALANCED TRAIN"
    )

    start = time.perf_counter()

    best_model.fit(
        X_balanced_pca,
        y_balanced,
    )

    refit_time = (
        time.perf_counter()
        - start
    )

    (
        y_pred,
        y_prob,
        predict_time,
    ) = predict_model(
        model=best_model,
        X_test=X_test_pca,
    )

    (
        metrics_tuned,
        cm_tuned,
        _,
    ) = evaluate_binary_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    metrics_tuned.update({
        "variant": "balance_pca_tuned",
        "features": int(
            X_balanced_pca.shape[1]
        ),
        "train_rows": int(
            len(y_balanced)
        ),
        "train_time": float(
            refit_time
        ),
        "predict_time": float(
            predict_time
        ),
        "best_cv_auc": float(
            best_cv_auc
        ),
        "tuning_time": float(
            tuning_time
        ),
    })

    summary_rows.append(
        metrics_tuned
    )

    print_metrics(
        metrics_tuned,
        cm_tuned,
    )

    # ============================================================
    # SAVE
    # ============================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    grid_results.to_csv(
        output_dir / "grid_search_results.csv",
        index=False,
    )

    with open(
        output_dir / "best_params.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "dataset": dataset_name,
                "best_cv_auc": best_cv_auc,
                "best_params": best_params,
                "balance_method":
                    sampler.__class__.__name__,
                "rows_before_balance":
                    int(len(y_train)),
                "rows_after_balance":
                    int(len(y_balanced)),
                "pca_components":
                    int(pca.n_components_),
                "pca_explained_variance":
                    float(
                        pca
                        .explained_variance_ratio_
                        .sum()
                    ),
                "max_tuning_rows":
                    max_tuning_rows,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print_title(
        f"{dataset_name.upper()} - SUMMARY"
    )

    show_cols = [
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

    print(
        summary[
            show_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: "
        f"{output_dir.resolve()}"
    )


def print_metrics(
    metrics,
    cm,
):
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

    print()
    print("Confusion matrix:")
    print(cm)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "05 - Class balancing, PCA "
            "and XGBoost Grid Search"
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
        "--max-tuning-rows",
        type=int,
        default=None,
        help=(
            "Giới hạn số dòng dùng Grid Search. "
            "Model tốt nhất vẫn refit trên full "
            "balanced training data."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        max_tuning_rows=args.max_tuning_rows,
    )


if __name__ == "__main__":
    main()