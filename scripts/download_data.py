from pathlib import Path
import shutil

import kagglehub

DATASETS = {
    "dataset1": "fedesoriano/stroke-prediction-dataset",
    "dataset2": "pranavp1999/stroke-prediction-health-care-synthetic-dataset",
}

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

for name, slug in DATASETS.items():
    print(f"Downloading {name}: {slug}")
    src = Path(kagglehub.dataset_download(slug))
    dst = RAW_DIR / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Saved to {dst}")
