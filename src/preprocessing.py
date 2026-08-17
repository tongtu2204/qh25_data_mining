from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "stroke"

CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

BINARY_FEATURE_COLS = [
    "hypertension",
    "heart_disease",
]

BINARY_COLS = BINARY_FEATURE_COLS + [TARGET]

NUMERIC_COLS = [
    "age",
    "avg_glucose_level",
    "bmi",
]

FEATURE_COLS = (
    NUMERIC_COLS
    + BINARY_FEATURE_COLS
    + CATEGORICAL_COLS
)


DTYPE_DATASET1 = {
    "gender": "category",
    "age": "float32",
    "hypertension": "int8",
    "heart_disease": "int8",
    "ever_married": "category",
    "work_type": "category",
    "Residence_type": "category",
    "avg_glucose_level": "float32",
    "bmi": "float32",
    "smoking_status": "category",
    "stroke": "int8",
}


DTYPE_DATASET2 = {
    "gender": "category",
    "age": "float32",
    "hypertension": "float32",
    "heart_disease": "float32",
    "ever_married": "category",
    "work_type": "category",
    "Residence_type": "category",
    "avg_glucose_level": "float32",
    "bmi": "float32",
    "smoking_status": "category",
    "stroke": "float32",
}


def load_raw_dataset(
    path,
    dataset_name,
    nrows=None,
):
    path = Path(path)

    compression = (
        "zip"
        if path.suffix.lower() == ".zip"
        else None
    )

    dtype_map = (
        DTYPE_DATASET1
        if dataset_name == "dataset1"
        else DTYPE_DATASET2
    )

    usecols = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    ]

    df = pd.read_csv(
        path,
        compression=compression,
        usecols=usecols,
        dtype=dtype_map,
        nrows=nrows,
    )

    return df


def normalize_dataset2_binary_columns(df):
    df = df.copy()

    for col in BINARY_COLS:
        df[col] = (
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
            >= 0.5
        ).astype("int8")

    return df


def clean_before_split(
    df,
    dataset_name,
):
    df = df.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    if dataset_name == "dataset2":
        df = normalize_dataset2_binary_columns(df)

        df = (
            df
            .dropna()
            .reset_index(drop=True)
        )

    return df


def split_train_test(
    df,
    test_size=0.20,
    random_state=42,
):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype("int8").copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def fit_bmi_mean(
    X_train,
):
    return float(
        X_train["bmi"].mean()
    )


def apply_bmi_imputation(
    X,
    bmi_mean,
):
    X = X.copy()

    X["bmi"] = X["bmi"].fillna(
        bmi_mean
    )

    return X


def fit_iqr_bounds(
    X_train,
):
    bounds = {}

    for col in NUMERIC_COLS:
        series = pd.to_numeric(
            X_train[col],
            errors="coerce",
        )

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bounds[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
        }

    return bounds


def apply_iqr_clipping(
    X,
    bounds,
):
    X = X.copy()

    for col, values in bounds.items():
        X[col] = X[col].clip(
            lower=values["lower"],
            upper=values["upper"],
        )

    return X


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLS,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURE_COLS,
            ),
            (
                "categorical",
                make_one_hot_encoder(),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def fit_transform_features(
    X_train,
    X_test,
):
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32,
    )

    feature_names = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=str,
    )

    return (
        X_train_processed,
        X_test_processed,
        feature_names,
        preprocessor,
    )


def prepare_dataset(
    df,
    dataset_name,
    test_size=0.20,
    random_state=42,
):
    df = clean_before_split(
        df=df,
        dataset_name=dataset_name,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(
        df=df,
        test_size=test_size,
        random_state=random_state,
    )

    bmi_mean = None

    if dataset_name == "dataset1":
        bmi_mean = fit_bmi_mean(
            X_train
        )

        X_train = apply_bmi_imputation(
            X_train,
            bmi_mean,
        )

        X_test = apply_bmi_imputation(
            X_test,
            bmi_mean,
        )

    iqr_bounds = fit_iqr_bounds(
        X_train
    )

    X_train = apply_iqr_clipping(
        X_train,
        iqr_bounds,
    )

    X_test = apply_iqr_clipping(
        X_test,
        iqr_bounds,
    )

    (
        X_train_processed,
        X_test_processed,
        feature_names,
        preprocessor,
    ) = fit_transform_features(
        X_train,
        X_test,
    )

    y_train_array = y_train.to_numpy(
        dtype=np.int8
    )

    y_test_array = y_test.to_numpy(
        dtype=np.int8
    )

    metadata = {
        "dataset": dataset_name,
        "rows_after_basic_cleaning": len(df),
        "train_rows": len(y_train_array),
        "test_rows": len(y_test_array),
        "input_features": len(FEATURE_COLS),
        "processed_features": len(feature_names),
        "train_positive": int(y_train_array.sum()),
        "test_positive": int(y_test_array.sum()),
        "train_positive_rate": float(y_train_array.mean()),
        "test_positive_rate": float(y_test_array.mean()),
        "bmi_train_mean": bmi_mean,
        "iqr_bounds": iqr_bounds,
    }

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train_array,
        "y_test": y_test_array,
        "feature_names": feature_names,
        "metadata": metadata,
        "preprocessor": preprocessor,
    }