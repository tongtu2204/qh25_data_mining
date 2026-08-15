from __future__ import annotations

import argparse

from src.config import get_raw_path
from src.preprocess import load_stroke_source


def main(dataset: str, nrows: int) -> None:
    path = get_raw_path(dataset)
    data = load_stroke_source(path, dataset, nrows=nrows)
    print(f"Dataset       : {dataset}")
    print(f"Path          : {path}")
    print(f"Sample rows   : {len(data.X):,}")
    print(f"Features      : {data.X.shape[1]}")
    print(f"Target        : {data.target}")
    print(f"Positive rate : {data.y.mean():.6f}")
    print("\nDtypes:")
    print(data.X.dtypes)
    print("\nHead:")
    print(data.X.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--nrows", type=int, default=10000)
    args = parser.parse_args()
    main(args.dataset, args.nrows)
