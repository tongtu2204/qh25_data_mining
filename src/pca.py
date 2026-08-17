from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_npz_dataset(
    dataset_name,
    processed_dir="data/processed",
):
    path = (
        Path(processed_dir)
        / f"{dataset_name}_train_test.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            "Hãy chạy bước 02_prepare_data trước."
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
        "feature_names": data["feature_names"].astype(str),
        "path": path,
    }


def fit_pca(
    X_train,
    variance_threshold=0.95,
):
    pca = PCA(
        n_components=variance_threshold,
        svd_solver="full",
    )

    X_train_pca = pca.fit_transform(
        X_train
    )

    return pca, X_train_pca


def transform_pca(
    pca,
    X_test,
):
    return pca.transform(
        X_test
    )


def get_explained_variance_table(
    pca,
):
    explained = (
        pca.explained_variance_ratio_
    )

    cumulative = np.cumsum(
        explained
    )

    return pd.DataFrame({
        "component": np.arange(
            1,
            len(explained) + 1,
        ),
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative,
    })


def get_loading_table(
    pca,
    feature_names,
):
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[
            f"PC{i}"
            for i in range(
                1,
                pca.n_components_ + 1,
            )
        ],
    )

    loadings.index.name = "feature"

    return loadings.reset_index()


def get_top_loadings(
    loading_table,
    component,
    top_n=10,
):
    result = loading_table[
        ["feature", component]
    ].copy()

    result["abs_loading"] = (
        result[component].abs()
    )

    result = (
        result
        .sort_values(
            "abs_loading",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return result


def save_pca_npz(
    output_path,
    X_train_pca,
    X_test_pca,
    y_train,
    y_test,
):
    np.savez_compressed(
        output_path,
        X_train=X_train_pca.astype(
            np.float32
        ),
        X_test=X_test_pca.astype(
            np.float32
        ),
        y_train=y_train.astype(
            np.int8
        ),
        y_test=y_test.astype(
            np.int8
        ),
    )