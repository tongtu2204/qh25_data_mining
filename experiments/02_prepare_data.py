import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_raw_path
from src.preprocessing import (
    load_raw_dataset,
    prepare_dataset,
)


def print_title(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def save_npz(
    output_path,
    prepared,
):
    np.savez_compressed(
        output_path,
        X_train=prepared["X_train"],
        X_test=prepared["X_test"],
        y_train=prepared["y_train"],
        y_test=prepared["y_test"],
        feature_names=prepared["feature_names"],
    )


def save_feature_names(
    output_path,
    feature_names,
):
    pd.DataFrame({
        "feature": feature_names
    }).to_csv(
        output_path,
        index=False,
    )


def save_metadata(
    output_path,
    metadata,
):
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )


def run_prepare_data(
    dataset_name,
    nrows=None,
):
    raw_path = get_raw_path(
        dataset_name
    )

    print_title(
        f"{dataset_name.upper()} - LOAD RAW DATA"
    )

    print(f"Path       : {raw_path}")
    print(
        "Mode       : "
        + (
            f"sample {nrows:,} rows"
            if nrows is not None
            else "full dataset"
        )
    )

    df = load_raw_dataset(
        path=raw_path,
        dataset_name=dataset_name,
        nrows=nrows,
    )

    print(f"Raw rows   : {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")
    print(
        f"Raw missing: "
        f"{df.isna().sum().sum():,}"
    )

    print_title("PREPARE DATA")

    prepared = prepare_dataset(
        df=df,
        dataset_name=dataset_name,
        test_size=0.20,
        random_state=42,
    )

    metadata = prepared["metadata"]

    print(
        f"Rows after cleaning : "
        f"{metadata['rows_after_basic_cleaning']:,}"
    )

    print(
        f"Train rows          : "
        f"{metadata['train_rows']:,}"
    )

    print(
        f"Test rows           : "
        f"{metadata['test_rows']:,}"
    )

    print(
        f"Processed features  : "
        f"{metadata['processed_features']:,}"
    )

    print(
        f"Train positive      : "
        f"{metadata['train_positive']:,}"
    )

    print(
        f"Train positive rate : "
        f"{metadata['train_positive_rate']:.6f}"
    )

    print(
        f"Test positive       : "
        f"{metadata['test_positive']:,}"
    )

    print(
        f"Test positive rate  : "
        f"{metadata['test_positive_rate']:.6f}"
    )

    if metadata["bmi_train_mean"] is not None:
        print(
            f"BMI train mean      : "
            f"{metadata['bmi_train_mean']:.6f}"
        )

    print()
    print(
        f"X_train shape       : "
        f"{prepared['X_train'].shape}"
    )

    print(
        f"X_test shape        : "
        f"{prepared['X_test'].shape}"
    )

    print(
        f"X_train dtype       : "
        f"{prepared['X_train'].dtype}"
    )

    print(
        f"y_train dtype       : "
        f"{prepared['y_train'].dtype}"
    )

    processed_dir = Path(
        "data/processed"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir = (
        Path("results")
        / dataset_name
        / "02_prepare_data"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"_sample_{nrows}"
        if nrows is not None
        else ""
    )

    npz_path = (
        processed_dir
        / f"{dataset_name}_train_test{suffix}.npz"
    )

    metadata_path = (
        result_dir
        / f"metadata{suffix}.json"
    )

    features_path = (
        result_dir
        / f"feature_names{suffix}.csv"
    )

    save_npz(
        output_path=npz_path,
        prepared=prepared,
    )

    save_metadata(
        output_path=metadata_path,
        metadata=metadata,
    )

    save_feature_names(
        output_path=features_path,
        feature_names=prepared["feature_names"],
    )

    print_title("OUTPUT")

    print(f"NPZ      : {npz_path.resolve()}")
    print(f"Metadata : {metadata_path.resolve()}")
    print(f"Features : {features_path.resolve()}")

    npz_size_mb = (
        npz_path.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"NPZ size : {npz_size_mb:.2f} MB"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "02 - Prepare train/test data "
            "and save compressed NPZ"
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
        "--nrows",
        type=int,
        default=None,
        help=(
            "Đọc mẫu N dòng. "
            "Bỏ trống để chạy full."
        ),
    )

    args = parser.parse_args()

    run_prepare_data(
        dataset_name=args.dataset,
        nrows=args.nrows,
    )


if __name__ == "__main__":
    main()