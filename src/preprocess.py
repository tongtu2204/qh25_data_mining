from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = ["stroke", "Stroke", "diagnosis", "Diagnosis", "target", "Target"]
ID_CANDIDATES = ["id", "ID", "Id"]
BINARY_LIKE = ["hypertension", "heart_disease", "stroke"]

# Dtype tối ưu để Dataset 2 (~5.77 triệu dòng) có thể đọc trực tiếp từ ZIP
# mà không cần giải nén ra ổ đĩa và giảm đáng kể RAM so với dtype mặc định.
STROKE_DTYPES = {
    "id": "float32",
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


@dataclass
class StrokeData:
    X: pd.DataFrame
    y: pd.Series
    target: str


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out


def find_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Không tìm thấy cột nhãn. Các cột hiện có: {list(df.columns)}")


def read_source(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Đọc CSV hoặc CSV.ZIP trực tiếp. ZIP không được giải nén thủ công.

    File ZIP của Dataset 2 chứa một CSV nên pandas có thể stream-decompress trực tiếp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    kwargs = {
        "dtype": STROKE_DTYPES,
        "low_memory": False,
        "nrows": nrows,
    }
    if path.suffix.lower() == ".zip":
        kwargs["compression"] = "zip"
    return pd.read_csv(path, **kwargs)


def normalize_binary_like_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Dataset 2 synthetic có các biến nhị phân bị jitter quanh 0 và 1.

    Ngưỡng 0.5 đưa hypertension, heart_disease và stroke về đúng dạng 0/1.
    Dataset 1 được giữ nguyên vì các cột này vốn đã là 0/1.
    """
    out = df.copy()
    if dataset == "dataset2":
        for col in BINARY_LIKE:
            if col in out.columns:
                out[col] = (pd.to_numeric(out[col], errors="coerce") >= 0.5).astype("int8")
    return out


def load_stroke_source(path: str | Path, dataset: str, nrows: int | None = None) -> StrokeData:
    df = read_source(path, nrows=nrows)
    df = clean_columns(df)
    df = normalize_binary_like_columns(df, dataset)
    df = df.drop_duplicates().reset_index(drop=True)

    # Theo bài báo: Dataset 2 xóa các bản ghi thiếu; Dataset 1 giữ BMI thiếu
    # để mean imputation được học ở preprocessing.
    if dataset == "dataset2":
        df = df.dropna().reset_index(drop=True)

    target = find_target_column(df)
    y = pd.to_numeric(df[target], errors="raise").astype("int8")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Nhãn {target} không phải nhị phân 0/1: {sorted(y.unique())}")

    X = df.drop(columns=[target])
    drop_ids = [c for c in ID_CANDIDATES if c in X.columns]
    if drop_ids:
        X = X.drop(columns=drop_ids)

    return StrokeData(X=X, y=y, target=target)


def cap_outliers_iqr_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Winsorize numeric columns using bounds learned only from training data."""
    train = X_train.copy()
    test = X_test.copy()
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        train[col] = train[col].clip(lower=lower, upper=upper)
        test[col] = test[col].clip(lower=lower, upper=upper)

    return train, test


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_features, categorical_features
