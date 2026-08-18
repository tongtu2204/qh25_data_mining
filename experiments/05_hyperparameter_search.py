import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.balancing import balance_training_data
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


def print_metrics(metrics, cm):
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

    # ============================================================
    # LOAD ORIGINAL TRAIN / TEST
    # ============================================================

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
        f"X_train             : "
        f"{X_train.shape}"
    )

    print(
        f"X_test              : "
        f"{X_test.shape}"
    )

    print(
        f"Positive            : "
        f"{y_train.sum():,}"
    )

    print(
        f"Negative            : "
        f"{len(y_train) - y_train.sum():,}"
    )

    print(
        f"Positive rate       : "
        f"{y_train.mean():.6f}"
    )

    summary_rows = []

    # ============================================================
    # CLASS BALANCING
    # ============================================================

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
    # GRID SEARCH - LEAKAGE SAFE
    #
    # IMPORTANT:
    # Grid Search receives ORIGINAL training data.
    #
    # Each CV fold performs:
    # sampler -> PCA -> XGBoost
    #
    # This prevents SMOTE / undersampling / PCA leakage.
    # ============================================================

    print_title(
        "GRID SEARCH - LEAKAGE SAFE PIPELINE"
    )

    (
        X_tune,
        y_tune,
    ) = get_tuning_subset(
        X=X_train,
        y=y_train,
        max_rows=max_tuning_rows,
        random_state=42,
    )

    print(
        f"Original train rows : "
        f"{len(y_train):,}"
    )

    print(
        f"Tuning rows         : "
        f"{len(y_tune):,}"
    )

    print(
        "CV pipeline         : "
        "Sampler -> PCA -> XGBoost"
    )

    (
        best_pipeline,
        best_params,
        best_cv_auc,
        tuning_time,
        grid_results,
    ) = run_grid_search(
        X_train=X_tune,
        y_train=y_tune,
        dataset_name=dataset_name,
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
            f"  {key:20s}: "
            f"{value}"
        )

    # ============================================================
    # REFIT BEST PIPELINE ON FULL ORIGINAL TRAIN
    #
    # best_pipeline contains:
    # sampler -> PCA -> XGBoost
    #
    # fit() on original training data means:
    # 1. sampler balances the full training set
    # 2. PCA fits only on balanced training data
    # 3. XGBoost fits on PCA output
    # ============================================================

    print_title(
        "REFIT BEST PIPELINE ON FULL ORIGINAL TRAIN"
    )

    start = time.perf_counter()

    best_pipeline.fit(
        X_train,
        y_train,
    )

    refit_time = (
        time.perf_counter()
        - start
    )

    # ============================================================
    # TEST PREDICTION
    # ============================================================

    start = time.perf_counter()

    y_pred = best_pipeline.predict(
        X_test
    )

    y_prob = best_pipeline.predict_proba(
        X_test
    )[:, 1]

    predict_time = (
        time.perf_counter()
        - start
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

    tuned_sampler = (
        best_pipeline
        .named_steps["sampler"]
    )

    tuned_pca = (
        best_pipeline
        .named_steps["pca"]
    )

    tuned_model = (
        best_pipeline
        .named_steps["model"]
    )

    metrics_tuned.update({
        "variant":
            "balance_pca_tuned",
        "features":
            int(
                tuned_pca.n_components_
            ),
        "train_rows":
            int(
                len(y_train)
            ),
        "train_time":
            float(
                refit_time
            ),
        "predict_time":
            float(
                predict_time
            ),
        "best_cv_auc":
            float(
                best_cv_auc
            ),
        "tuning_time":
            float(
                tuning_time
            ),
    })

    summary_rows.append(
        metrics_tuned
    )

    print(
        f"Sampler             : "
        f"{tuned_sampler.__class__.__name__}"
    )

    print(
        f"PCA components      : "
        f"{tuned_pca.n_components_}"
    )

    print(
        f"Explained variance  : "
        f"{tuned_pca.explained_variance_ratio_.sum():.6f}"
    )

    print(
        f"Refit time          : "
        f"{refit_time:.4f} s"
    )

    print(
        f"Predict time        : "
        f"{predict_time:.4f} s"
    )

    print()

    print_metrics(
        metrics_tuned,
        cm_tuned,
    )

    # ============================================================
    # SAVE GRID SEARCH RESULTS
    # ============================================================

    grid_results.to_csv(
        output_dir
        / "grid_search_results.csv",
        index=False,
    )

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    # ============================================================
    # SAVE METADATA
    # ============================================================

    metadata = {
        "dataset": dataset_name,

        "original_train": {
            "rows": int(
                len(y_train)
            ),
            "positive": int(
                y_train.sum()
            ),
            "negative": int(
                len(y_train)
                - y_train.sum()
            ),
            "positive_rate": float(
                y_train.mean()
            ),
        },

        "manual_balancing": {
            "method":
                sampler.__class__.__name__,

            "rows_after_balance":
                int(
                    len(y_balanced)
                ),

            "positive_after":
                int(
                    y_balanced.sum()
                ),

            "negative_after":
                int(
                    len(y_balanced)
                    - y_balanced.sum()
                ),

            "balance_time":
                float(
                    balance_time
                ),
        },

        "manual_pca": {
            "input_features":
                int(
                    X_balanced.shape[1]
                ),

            "pca_components":
                int(
                    pca.n_components_
                ),

            "explained_variance":
                float(
                    pca
                    .explained_variance_ratio_
                    .sum()
                ),

            "pca_time":
                float(
                    pca_time
                ),
        },

        "tuning": {
            "leakage_safe":
                True,

            "pipeline":
                "Sampler -> PCA -> XGBoost",

            "max_tuning_rows":
                max_tuning_rows,

            "actual_tuning_rows":
                int(
                    len(y_tune)
                ),

            "best_cv_auc":
                float(
                    best_cv_auc
                ),

            "tuning_time":
                float(
                    tuning_time
                ),

            "best_params":
                best_params,
        },

        "tuned_pipeline": {
            "sampler":
                tuned_sampler.__class__.__name__,

            "pca_components":
                int(
                    tuned_pca.n_components_
                ),

            "pca_explained_variance":
                float(
                    tuned_pca
                    .explained_variance_ratio_
                    .sum()
                ),

            "model":
                tuned_model.__class__.__name__,

            "refit_time":
                float(
                    refit_time
                ),

            "predict_time":
                float(
                    predict_time
                ),
        },
    }

    with open(
        output_dir
        / "best_params.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "05 - Class balancing, PCA "
            "and leakage-safe XGBoost Grid Search"
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
            "Giới hạn số ORIGINAL training rows "
            "được dùng cho Grid Search. "
            "Sampler và PCA vẫn chạy bên trong "
            "mỗi CV fold. "
            "Best pipeline sau đó được refit "
            "trên toàn bộ original training set."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        max_tuning_rows=args.max_tuning_rows,
    )


if __name__ == "__main__":
    main()