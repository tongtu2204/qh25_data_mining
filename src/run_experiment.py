from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_output_dir, get_raw_path, load_config
from src.modeling import cross_validate_auc, evaluate_model, make_pipeline
from src.plots import (
    save_confusion_matrix,
    save_pca_loadings,
    save_pca_plots,
    save_roc_curve,
    save_shap_plots,
)
from src.preprocess import build_preprocessor, cap_outliers_iqr_train_test, load_stroke_source

EXPERIMENTS = [
    ("unbalanced_no_pca", "none", False),
    ("balanced_no_pca", None, False),
    ("balanced_pca95", None, True),
]


def default_balance(dataset: str) -> str:
    return "smote" if dataset == "dataset1" else "undersample"


def run(dataset: str, source: str | None = None, balance: str | None = None, skip_cv: bool = False) -> None:
    cfg = load_config()
    source_path = Path(source) if source else get_raw_path(dataset)
    out_dir = get_output_dir() / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = cfg.get("experiment", {})
    test_size = float(exp_cfg.get("test_size", 0.20))
    random_state = int(exp_cfg.get("random_state", 42))

    print(f"Dataset : {dataset}")
    print(f"Source  : {source_path}")
    if source_path.suffix.lower() == ".zip":
        print("Mode    : đọc CSV trực tiếp trong ZIP (không giải nén thủ công)")

    data = load_stroke_source(source_path, dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=test_size,
        stratify=data.y,
        random_state=random_state,
    )

    X_train, X_test = cap_outliers_iqr_train_test(X_train, X_test)
    balance = balance or default_balance(dataset)

    summary: dict = {
        "dataset": dataset,
        "source_path": str(source_path),
        "target": data.target,
        "n_total": int(len(data.X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_total": float(data.y.mean()),
        "default_balance": balance,
    }

    rows = []
    fitted = {}

    for name, exp_balance, use_pca in EXPERIMENTS:
        strategy = exp_balance if exp_balance is not None else balance
        preprocessor, _, _ = build_preprocessor(X_train)
        model = make_pipeline(preprocessor, strategy, use_pca=use_pca, random_state=random_state)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        metrics, report = evaluate_model(model, X_test, y_test)
        metrics["fit_seconds"] = float(fit_seconds)
        metrics["balance_strategy"] = strategy
        metrics["use_pca"] = use_pca

        if use_pca:
            metrics["pca_components"] = int(model.named_steps["pca"].n_components_)
            metrics["pca_explained_variance"] = float(model.named_steps["pca"].explained_variance_ratio_.sum())
        else:
            metrics["pca_components"] = None
            metrics["pca_explained_variance"] = None

        summary[name] = metrics
        rows.append({"experiment": name, **metrics})
        fitted[name] = model

        report.to_csv(out_dir / f"classification_report_{name}.csv", index=True)
        save_confusion_matrix(model, X_test, y_test, out_dir / f"{name}_confusion_matrix.png")
        save_roc_curve(model, X_test, y_test, out_dir / f"{name}_roc_curve.png")

    pca_model = fitted["balanced_pca95"]
    save_pca_plots(
        pca_model,
        out_dir / "balanced_pca95_cumulative_variance.png",
        out_dir / "balanced_pca95_scree_plot.png",
    )
    save_pca_loadings(
        pca_model,
        out_dir / "balanced_pca95_loadings_pc1.png",
        out_dir / "balanced_pca95_loadings_pc2.png",
    )

    shap_model = fitted["balanced_no_pca"]
    X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=random_state)
    try:
        save_shap_plots(shap_model, X_shap, out_dir, "balanced_no_pca")
    except Exception as exc:
        summary["shap_warning"] = str(exc)
        print(f"Không tạo được SHAP: {exc}")

    if not skip_cv:
        preprocessor, _, _ = build_preprocessor(X_train)
        cv_model = make_pipeline(preprocessor, balance, use_pca=True, random_state=random_state)
        summary["cross_validation_pca_auc"] = cross_validate_auc(cv_model, X_train, y_train, n_splits=10)

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--source", default=None, help="Tùy chọn override path trong config/config.yaml")
    parser.add_argument("--balance", choices=["smote", "undersample"], default=None)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.source, args.balance, args.skip_cv)
