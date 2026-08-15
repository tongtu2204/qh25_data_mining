from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_confusion_matrix(model, X_test, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_roc_curve(model, X_test, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC-AUC Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_pca_plots(model, cumulative_path: Path, scree_path: Path) -> None:
    if "pca" not in model.named_steps:
        return

    pca = model.named_steps["pca"]
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    x = np.arange(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(x, cumulative, marker="o")
    ax.axhline(0.95, linestyle="--")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA cumulative explained variance")
    fig.tight_layout()
    fig.savefig(cumulative_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(x, explained * 100)
    ax.plot(x, explained * 100, marker="o")
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot")
    fig.tight_layout()
    fig.savefig(scree_path, dpi=220)
    plt.close(fig)


def save_pca_loadings(model, path_pc1: Path, path_pc2: Path, top_n: int = 20) -> None:
    if "pca" not in model.named_steps:
        return

    pre = model.named_steps["preprocess"]
    pca = model.named_steps["pca"]
    feature_names = np.asarray(pre.get_feature_names_out())

    for pc_idx, path in [(0, path_pc1), (1, path_pc2)]:
        if pca.components_.shape[0] <= pc_idx:
            continue
        weights = pca.components_[pc_idx]
        order = np.argsort(np.abs(weights))[-top_n:]
        vals = weights[order]
        names = feature_names[order]

        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=75, ha="right")
        ax.set_ylabel("Loading")
        ax.set_title(f"Feature weights in PC{pc_idx + 1}")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


def save_shap_plots(model, X_sample: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    import shap

    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    X_trans = pre.transform(X_sample)
    names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_trans)

    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_shap_bar.png", dpi=220, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Waterfall cho 1 quan sát. Gắn tên biến để đồ thị đọc được.
    one = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_trans[0],
        feature_names=names,
    )
    shap.plots.waterfall(one, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_shap_waterfall.png", dpi=220, bbox_inches="tight")
    plt.close()
