import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.explainability import (
    approximate_original_shap,
    compute_shap_values,
    get_global_importance,
    get_projected_global_importance,
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
            f"Không tìm thấy {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return metadata["tuning"]["best_params"]


def apply_best_params(
    pipeline,
    best_params,
):
    params = {
        f"model__{key}": value
        for key, value
        in best_params.items()
    }

    pipeline.set_params(
        **params
    )

    return pipeline


def sample_for_shap(
    X,
    max_rows,
    random_state=42,
):
    if max_rows is None:
        return X

    if len(X) <= max_rows:
        return X

    rng = np.random.default_rng(
        random_state
    )

    idx = rng.choice(
        len(X),
        size=max_rows,
        replace=False,
    )

    return X[idx]


def save_shap_bar(
    explanation,
    output_path,
    max_display=15,
):
    plt.figure()

    shap.plots.bar(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_shap_beeswarm(
    explanation,
    output_path,
    max_display=15,
):
    plt.figure()

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_shap_waterfall(
    explanation,
    output_path,
    instance_index=0,
    max_display=15,
):
    plt.figure()

    shap.plots.waterfall(
        explanation[
            instance_index
        ],
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_projected_bar(
    importance_df,
    output_path,
    top_n=15,
):
    plot_df = (
        importance_df
        .head(top_n)
        .sort_values(
            "mean_abs_projected_shap",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df[
            "mean_abs_projected_shap"
        ],
    )

    ax.set_xlabel(
        "Mean |projected SHAP|"
    )

    ax.set_ylabel(
        "Original feature"
    )

    ax.set_title(
        "Approximate SHAP importance "
        "projected to original features"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_projected_beeswarm(
    projected_shap,
    original_X,
    feature_names,
    output_path,
    max_display=15,
):
    explanation = shap.Explanation(
        values=projected_shap,
        data=original_X,
        feature_names=feature_names,
    )

    plt.figure()

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def run_shap_analysis(
    dataset_name,
    max_rows=2000,
):
    output_dir = (
        Path("results")
        / dataset_name
        / "07_shap_analysis"
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

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    original_feature_names = (
        data["feature_names"]
    )

    print_title(
        f"{dataset_name.upper()} - LOAD DATA"
    )

    print(
        f"X_train          : "
        f"{X_train.shape}"
    )

    print(
        f"X_test           : "
        f"{X_test.shape}"
    )

    print(
        f"Original features: "
        f"{len(original_feature_names)}"
    )

    # ============================================================
    # BUILD FINAL PIPELINE
    # ============================================================

    best_params = load_best_params(
        dataset_name
    )

    pipeline = build_tuning_pipeline(
        dataset_name=dataset_name,
        random_state=42,
    )

    pipeline = apply_best_params(
        pipeline,
        best_params,
    )

    print_title(
        "FIT FINAL PIPELINE"
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    sampler = (
        pipeline
        .named_steps["sampler"]
    )

    pca = (
        pipeline
        .named_steps["pca"]
    )

    model = (
        pipeline
        .named_steps["model"]
    )

    print(
        f"Sampler           : "
        f"{sampler.__class__.__name__}"
    )

    print(
        f"PCA components    : "
        f"{pca.n_components_}"
    )

    print(
        f"Explained variance: "
        f"{pca.explained_variance_ratio_.sum():.6f}"
    )

    # ============================================================
    # SAMPLE TEST DATA
    # ============================================================

    X_test_sample = sample_for_shap(
        X_test,
        max_rows=max_rows,
        random_state=42,
    )

    print(
        f"SHAP sample rows  : "
        f"{len(X_test_sample):,}"
    )

    # XGBoost actually receives PCA transformed data.

    X_test_pca = pca.transform(
        X_test_sample
    )

    X_test_pca = np.asarray(
        X_test_pca,
        dtype=np.float32,
    )

    pc_names = np.asarray(
        [
            f"PC{i}"
            for i in range(
                1,
                pca.n_components_ + 1
            )
        ]
    )

    # ============================================================
    # EXACT SHAP IN PCA SPACE
    # ============================================================

    print_title(
        "EXACT SHAP - PCA SPACE"
    )

    shap_explanation = compute_shap_values(
        model=model,
        X=X_test_pca,
        feature_names=pc_names,
    )

    pc_importance = get_global_importance(
        shap_explanation
    )

    print(
        pc_importance
        .head(15)
        .to_string(
            index=False
        )
    )

    pc_importance.to_csv(
        output_dir
        / "shap_pc_importance.csv",
        index=False,
    )

    save_shap_bar(
        explanation=
            shap_explanation,
        output_path=(
            output_dir
            / "shap_pc_bar.png"
        ),
    )

    save_shap_beeswarm(
        explanation=
            shap_explanation,
        output_path=(
            output_dir
            / "shap_pc_beeswarm.png"
        ),
    )

    save_shap_waterfall(
        explanation=
            shap_explanation,
        output_path=(
            output_dir
            / "shap_pc_waterfall.png"
        ),
        instance_index=0,
    )

    # ============================================================
    # APPROXIMATE PROJECTION BACK TO ORIGINAL FEATURES
    # ============================================================

    print_title(
        "APPROXIMATE ORIGINAL-FEATURE CONTRIBUTIONS"
    )

    projected_df = approximate_original_shap(
        shap_pc_values=
            shap_explanation.values,
        pca_components=
            pca.components_,
        original_feature_names=
            original_feature_names,
    )

    projected_importance = (
        get_projected_global_importance(
            projected_df
        )
    )

    print(
        projected_importance
        .head(15)
        .to_string(
            index=False
        )
    )

    projected_importance.to_csv(
        output_dir
        / "shap_projected_original_importance.csv",
        index=False,
    )

    # Sample-level projected SHAP can be large.
    # Save only first 1000 rows.

    projected_df.head(
        1000
    ).to_csv(
        output_dir
        / "shap_projected_original_sample.csv",
        index=False,
    )

    save_projected_bar(
        importance_df=
            projected_importance,
        output_path=(
            output_dir
            / "shap_projected_original_bar.png"
        ),
    )

    save_projected_beeswarm(
        projected_shap=
            projected_df.values,
        original_X=
            X_test_sample,
        feature_names=
            original_feature_names,
        output_path=(
            output_dir
            / "shap_projected_original_beeswarm.png"
        ),
    )

    # ============================================================
    # SAVE PCA LOADINGS USED FOR PROJECTION
    # ============================================================

    loading_df = pd.DataFrame(
        pca.components_.T,
        columns=pc_names,
    )

    loading_df.insert(
        0,
        "feature",
        original_feature_names,
    )

    loading_df.to_csv(
        output_dir
        / "pca_loadings_final_model.csv",
        index=False,
    )

    # ============================================================
    # METADATA
    # ============================================================

    metadata = {
        "dataset":
            dataset_name,

        "shap_sample_rows":
            int(
                len(X_test_sample)
            ),

        "pca_components":
            int(
                pca.n_components_
            ),

        "pca_explained_variance":
            float(
                pca
                .explained_variance_ratio_
                .sum()
            ),

        "exact_explanation_space":
            "PCA components",

        "projected_original_features":
            True,

        "projection_warning":
            (
                "Projected original-feature SHAP "
                "is an approximation obtained by "
                "multiplying SHAP values in PCA "
                "space by PCA loadings. It is not "
                "exact SHAP for the nonlinear "
                "composite model."
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
            "07 - SHAP analysis for final "
            "PCA + XGBoost pipeline"
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
        "--max-rows",
        type=int,
        default=2000,
        help=(
            "Maximum number of test rows "
            "used for SHAP plots."
        ),
    )

    args = parser.parse_args()

    run_shap_analysis(
        dataset_name=args.dataset,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()