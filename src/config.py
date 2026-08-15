from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {path}\n"
            "Hãy copy config/config.example.yaml thành config/config.yaml rồi sửa path dữ liệu."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str | Path) -> Path:
    """Giữ nguyên path tuyệt đối; path tương đối được tính từ project root."""
    value = str(value)
    # Path trên Windows có drive, ví dụ C:/..., nhưng khi code được kiểm tra trên Linux
    # pathlib không xem đó là absolute. Ta vẫn giữ nguyên chuỗi để chạy đúng trên Windows.
    if len(value) >= 3 and value[1:3] in {":/", ":\\"}:
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_raw_path(dataset: str, config_path: str | Path = DEFAULT_CONFIG) -> Path:
    cfg = load_config(config_path)
    try:
        path = resolve_path(cfg["data_raw_source"][dataset])
    except KeyError as exc:
        raise KeyError(f"Thiếu data_raw_source.{dataset} trong config") from exc
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu {dataset}: {path}\n"
            "Hãy sửa data_raw_source trong config/config.yaml."
        )
    return path


def get_processed_path(dataset: str, config_path: str | Path = DEFAULT_CONFIG) -> Path:
    cfg = load_config(config_path)
    try:
        return resolve_path(cfg["processed_data"][dataset])
    except KeyError as exc:
        raise KeyError(f"Thiếu processed_data.{dataset} trong config") from exc


def get_output_dir(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    cfg = load_config(config_path)
    return resolve_path(cfg.get("output_dir", "results"))
