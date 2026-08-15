from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import get_processed_path, get_raw_path, load_config
from src.preprocess import build_preprocessor, cap_outliers_iqr_train_test, load_stroke_source


def main(dataset: str) -> None:
    cfg = load_config()
    exp_cfg = cfg.get("experiment", {})
    test_size = float(exp_cfg.get("test_size", 0.20))
    random_state = int(exp_cfg.get("random_state", 42))

    source = get_raw_path(dataset)
    output = get_processed_path(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Đọc {dataset}: {source}")
    if source.suffix.lower() == ".zip":
        print("Đọc trực tiếp CSV bên trong ZIP; không tạo file CSV giải nén.")

    data = load_stroke_source(source, dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=test_size,
        stratify=data.y,
        random_state=random_state,
    )
    X_train, X_test = cap_outliers_iqr_train_test(X_train, X_test)

    preprocessor, _, _ = build_preprocessor(X_train)
    X_train_num = preprocessor.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_num = preprocessor.transform(X_test).astype(np.float32, copy=False)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)

    np.savez_compressed(
        output,
        X_train=X_train_num,
        X_test=X_test_num,
        y_train=np.asarray(y_train, dtype=np.int8),
        y_test=np.asarray(y_test, dtype=np.int8),
        feature_names=feature_names,
    )

    size_mb = output.stat().st_size / (1024 ** 2)
    print(f"Saved: {output}")
    print(f"X_train: {X_train_num.shape}, X_test: {X_test_num.shape}")
    print(f"NPZ size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    args = parser.parse_args()
    main(args.dataset)
