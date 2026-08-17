import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_raw_path


TARGET = "stroke"

CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

BINARY_COLS = [
    "hypertension",
    "heart_disease",
    "stroke",
]

NUMERIC_COLS = [
    "age",
    "avg_glucose_level",
    "bmi",
]

BINARY_FEATURE_COLS = [
    "hypertension",
    "heart_disease",
]

def load_raw_data(dataset_name, nrows=None):
    path = get_raw_path(dataset_name)

    compression = "zip" if path.suffix.lower() == ".zip" else None

    df = pd.read_csv(
        path,
        compression=compression,
        nrows=nrows,
    )

    return df, path


def print_title(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def get_basic_summary(df, dataset_name, path):
    summary = {
        "dataset": dataset_name,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
    }

    if TARGET in df.columns:
        summary["target_nunique_raw"] = int(df[TARGET].nunique(dropna=True))
        summary["target_min_raw"] = float(df[TARGET].min())
        summary["target_max_raw"] = float(df[TARGET].max())
        summary["target_mean_raw"] = float(df[TARGET].mean())

    return summary


def get_missing_summary(df):
    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_rate": df.isna().mean().values,
    })

    result = result.sort_values(
        ["missing_count", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return result


def get_outlier_summary(df):
    rows = []

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(
            ((series < lower) | (series > upper)).sum()
        )

        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_count / len(series),
        })

    return pd.DataFrame(rows)


def get_categorical_summary(df):
    frames = []

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        counts = (
            df[col]
            .fillna("MISSING")
            .value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="count")
        )

        counts["rate"] = counts["count"] / len(df)
        counts.insert(0, "column", col)

        frames.append(counts)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def normalize_dataset2_binary_columns(df):
    df = df.copy()

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce") >= 0.5
            ).astype("int8")

    return df


def get_cleaning_preview(df, dataset_name):
    cleaned = df.copy()

    raw_rows = len(cleaned)
    raw_missing = int(cleaned.isna().sum().sum())

    if dataset_name == "dataset1":
        if "bmi" in cleaned.columns:
            bmi_mean = cleaned["bmi"].mean()
            cleaned["bmi"] = cleaned["bmi"].fillna(bmi_mean)

    elif dataset_name == "dataset2":
        cleaned = normalize_dataset2_binary_columns(cleaned)
        cleaned = cleaned.dropna().reset_index(drop=True)

    summary = {
        "dataset": dataset_name,
        "raw_rows": raw_rows,
        "clean_rows": len(cleaned),
        "removed_rows": raw_rows - len(cleaned),
        "raw_missing": raw_missing,
        "clean_missing": int(cleaned.isna().sum().sum()),
    }

    if TARGET in cleaned.columns:
        summary["positive_count"] = int(cleaned[TARGET].sum())
        summary["negative_count"] = int(
            len(cleaned) - cleaned[TARGET].sum()
        )
        summary["positive_rate"] = float(cleaned[TARGET].mean())

    return cleaned, summary


def plot_target_distribution(df, dataset_name, output_dir):
    if TARGET not in df.columns:
        return

    target = df[TARGET].copy()

    if dataset_name == "dataset2":
        target = (
            pd.to_numeric(target, errors="coerce") >= 0.5
        ).astype("int8")

    counts = target.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(f"Phân phối nhãn stroke - {dataset_name}")
    ax.set_xlabel("Stroke")
    ax.set_ylabel("Số quan sát")
    ax.tick_params(axis="x", rotation=0)

    total = counts.sum()

    for i, value in enumerate(counts.values):
        rate = value / total

        ax.text(
            i,
            value,
            f"{value:,}\n({rate:.2%})",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        output_dir / "target_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_data_check(dataset_name, nrows=None):
    df, path = load_raw_data(
        dataset_name=dataset_name,
        nrows=nrows,
    )

    output_dir = Path("results") / dataset_name / "01_data_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_title(f"{dataset_name.upper()} - RAW DATA")

    print(f"Path              : {path}")
    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns)}")
    print(f"Duplicates        : {df.duplicated().sum():,}")
    print(f"Total missing     : {df.isna().sum().sum():,}")

    print()
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    print()
    print("Dtypes:")
    print(df.dtypes)

    print()
    print("Head:")
    print(df.head())

    if TARGET in df.columns:
        print_title("RAW TARGET CHECK")

        print(f"Target            : {TARGET}")
        print(f"Raw unique values : {df[TARGET].nunique():,}")
        print(f"Raw min           : {df[TARGET].min()}")
        print(f"Raw max           : {df[TARGET].max()}")
        print(f"Raw mean          : {df[TARGET].mean():.6f}")

    missing_summary = get_missing_summary(df)

    print_title("MISSING VALUES")
    print(
        missing_summary[
            missing_summary["missing_count"] > 0
        ].to_string(index=False)
    )

    if missing_summary["missing_count"].sum() == 0:
        print("Không có missing value.")

    outlier_summary = get_outlier_summary(df)

    print_title("IQR OUTLIER CHECK")
    print(outlier_summary.to_string(index=False))

    categorical_summary = get_categorical_summary(df)

    print_title("CATEGORICAL DISTRIBUTIONS")

    for col in CATEGORICAL_COLS:
        temp = categorical_summary[
            categorical_summary["column"] == col
        ]

        if len(temp) == 0:
            continue

        print()
        print(f"[{col}]")
        print(
            temp[
                ["value", "count", "rate"]
            ].to_string(index=False)
        )

    cleaned, cleaning_summary = get_cleaning_preview(
        df,
        dataset_name,
    )

    print_title("BASIC CLEANING PREVIEW")

    for key, value in cleaning_summary.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
        elif isinstance(value, int):
            print(f"{key:20s}: {value:,}")
        else:
            print(f"{key:20s}: {value}")

    basic_summary = pd.DataFrame([
        get_basic_summary(
            df=df,
            dataset_name=dataset_name,
            path=path,
        )
    ])

    cleaning_summary_df = pd.DataFrame([
        cleaning_summary
    ])

    basic_summary.to_csv(
        output_dir / "basic_summary.csv",
        index=False,
    )

    missing_summary.to_csv(
        output_dir / "missing_summary.csv",
        index=False,
    )

    outlier_summary.to_csv(
        output_dir / "outlier_summary.csv",
        index=False,
    )

    categorical_summary.to_csv(
        output_dir / "categorical_summary.csv",
        index=False,
    )

    cleaning_summary_df.to_csv(
        output_dir / "cleaning_preview.csv",
        index=False,
    )

    plot_target_distribution(
        df=df,
        dataset_name=dataset_name,
        output_dir=output_dir,
    )

    print_title("OUTPUT")

    print(f"Saved to: {output_dir.resolve()}")

    print()
    print("Files:")
    for file in sorted(output_dir.iterdir()):
        print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="01 - Raw dataset validation and EDA"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dataset1", "dataset2"],
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Số dòng đọc mẫu. Bỏ trống để đọc toàn bộ.",
    )

    args = parser.parse_args()

    run_data_check(
        dataset_name=args.dataset,
        nrows=args.nrows,
    )


if __name__ == "__main__":
    main()