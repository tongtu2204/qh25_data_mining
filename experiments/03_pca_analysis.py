import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pca import (
    fit_pca,
    get_explained_variance_table,
    get_loading_table,
    get_top_loadings,
    load_npz_dataset,
    save_pca_npz,
    transform_pca,
)


def print_title(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def plot_explained_variance(
    variance_table,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        variance_table["component"],
        variance_table[
            "explained_variance_ratio"
        ],
    )

    ax.set_xlabel(
        "Principal Component"
    )

    ax.set_ylabel(
        "Explained Variance Ratio"
    )

    ax.set_title(
        "Explained Variance by Principal Component"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_cumulative_variance(
    variance_table,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        variance_table["component"],
        variance_table[
            "cumulative_explained_variance"
        ],
        marker="o",
    )

    ax.axhline(
        0.95,
        linestyle="--",
    )

    ax.set_xlabel(
        "Number of Components"
    )

    ax.set_ylabel(
        "Cumulative Explained Variance"
    )

    ax.set_title(
        "Cumulative Explained Variance"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_component_loadings(
    top_loadings,
    component,
    output_path,
):
    plot_df = (
        top_loadings
        .sort_values(
            component,
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df[component],
    )

    ax.set_xlabel(
        "Loading"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"Top Feature Loadings - {component}"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_pca_analysis(
    dataset_name,
    variance_threshold=0.95,
):
    data = load_npz_dataset(
        dataset_name
    )

    print_title(
        f"{dataset_name.upper()} - LOAD NPZ"
    )

    print(
        f"Source          : {data['path']}"
    )

    print(
        f"X_train shape   : "
        f"{data['X_train'].shape}"
    )

    print(
        f"X_test shape    : "
        f"{data['X_test'].shape}"
    )

    print(
        f"Input features  : "
        f"{len(data['feature_names'])}"
    )

    print_title(
        "FIT PCA"
    )

    pca, X_train_pca = fit_pca(
        X_train=data["X_train"],
        variance_threshold=variance_threshold,
    )

    X_test_pca = transform_pca(
        pca=pca,
        X_test=data["X_test"],
    )

    variance_table = (
        get_explained_variance_table(
            pca
        )
    )

    loading_table = (
        get_loading_table(
            pca=pca,
            feature_names=data[
                "feature_names"
            ],
        )
    )

    print(
        f"Variance threshold : "
        f"{variance_threshold:.2%}"
    )

    print(
        f"Input features     : "
        f"{len(data['feature_names'])}"
    )

    print(
        f"PCA components     : "
        f"{pca.n_components_}"
    )

    print(
        f"Explained variance : "
        f"{pca.explained_variance_ratio_.sum():.6f}"
    )

    print(
        f"X_train PCA shape  : "
        f"{X_train_pca.shape}"
    )

    print(
        f"X_test PCA shape   : "
        f"{X_test_pca.shape}"
    )

    print_title(
        "TOP LOADINGS - PC1"
    )

    top_pc1 = get_top_loadings(
        loading_table=loading_table,
        component="PC1",
        top_n=10,
    )

    print(
        top_pc1.to_string(
            index=False
        )
    )

    top_pc2 = None

    if pca.n_components_ >= 2:
        print_title(
            "TOP LOADINGS - PC2"
        )

        top_pc2 = get_top_loadings(
            loading_table=loading_table,
            component="PC2",
            top_n=10,
        )

        print(
            top_pc2.to_string(
                index=False
            )
        )

    result_dir = (
        Path("results")
        / dataset_name
        / "03_pca_analysis"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir = Path(
        "data/processed"
    )

    pca_npz_path = (
        processed_dir
        / f"{dataset_name}_pca95_train_test.npz"
    )

    save_pca_npz(
        output_path=pca_npz_path,
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        y_train=data["y_train"],
        y_test=data["y_test"],
    )

    variance_table.to_csv(
        result_dir
        / "explained_variance.csv",
        index=False,
    )

    loading_table.to_csv(
        result_dir
        / "pca_loadings.csv",
        index=False,
    )

    top_pc1.to_csv(
        result_dir
        / "top_loadings_pc1.csv",
        index=False,
    )

    if top_pc2 is not None:
        top_pc2.to_csv(
            result_dir
            / "top_loadings_pc2.csv",
            index=False,
        )

    metadata = {
        "dataset": dataset_name,
        "variance_threshold": variance_threshold,
        "input_features": int(
            len(data["feature_names"])
        ),
        "pca_components": int(
            pca.n_components_
        ),
        "explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "train_rows": int(
            X_train_pca.shape[0]
        ),
        "test_rows": int(
            X_test_pca.shape[0]
        ),
    }

    with open(
        result_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )

    plot_explained_variance(
        variance_table,
        result_dir
        / "explained_variance.png",
    )

    plot_cumulative_variance(
        variance_table,
        result_dir
        / "cumulative_variance.png",
    )

    plot_component_loadings(
        top_loadings=top_pc1,
        component="PC1",
        output_path=(
            result_dir
            / "pc1_loadings.png"
        ),
    )

    if top_pc2 is not None:
        plot_component_loadings(
            top_loadings=top_pc2,
            component="PC2",
            output_path=(
                result_dir
                / "pc2_loadings.png"
            ),
        )

    print_title(
        "OUTPUT"
    )

    print(
        f"PCA NPZ : "
        f"{pca_npz_path.resolve()}"
    )

    print(
        f"Results : "
        f"{result_dir.resolve()}"
    )

    print()
    print("Files:")

    for file in sorted(
        result_dir.iterdir()
    ):
        print(
            f"  - {file.name}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "03 - PCA analysis "
            "with 95% explained variance"
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
        "--variance",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    run_pca_analysis(
        dataset_name=args.dataset,
        variance_threshold=args.variance,
    )


if __name__ == "__main__":
    main()