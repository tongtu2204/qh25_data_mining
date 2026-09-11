"""Experiment 14: algorithm-level challengers under the frozen strict protocol.

Goal
----
Compare new model families against Experiment 13 without changing the outer
80/20 split or touching/resampling the test set.

Phase 1 focuses on Dataset 1:
  - CatBoost with native categorical features + automatic class weights.
  - LightGBM with class weighting on one-hot encoded features.
  - Balanced Random Forest.

Dataset 2 is intentionally staged separately because it has millions of rows;
LightGBM on the full training set is the primary challenger there.

Run from repository root, e.g.
python experiments/14_algorithm_improvements.py --dataset dataset1 --data data/healthcare-dataset-stroke-data.csv
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import (
    CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, BINARY_FEATURE_COLS,
    clean_before_split, load_raw_dataset,
)

SEED_SPLIT = 42
SEED_INNER = 2026
SPEC_TARGET = 0.85


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=float(threshold), n=int(len(y)), positive=int(np.sum(y)),
        prevalence=float(np.mean(y)), accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        specificity=float(tn / (tn + fp)), f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
        mcc=float(matthews_corrcoef(y, pred)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def specificity_threshold(y, p, target=SPEC_TARGET):
    order = np.unique(p)
    best = 1.0
    # Quantile grid is deterministic and avoids scanning millions of unique probabilities.
    grid = np.unique(np.quantile(order, np.linspace(0, 1, 1001)))
    for t in grid:
        pred = p >= t
        neg = y == 0
        spec = 1.0 - float(pred[neg].mean())
        if spec >= target:
            best = float(t)
            break
    return best


def encode_categories(df):
    x = df.copy()
    for c in CATEGORICAL_COLS:
        x[c] = x[c].astype("string").fillna("__MISSING__")
    for c in NUMERIC_COLS + BINARY_FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def catboost_d1(xfit, yfit, xval, yval, xt, yt, xe):
    from catboost import CatBoostClassifier
    # Native categorical processing: no one-hot, no SMOTE, no PCA.
    fit = encode_categories(xfit); val = encode_categories(xval)
    train = encode_categories(xt); test = encode_categories(xe)
    params = dict(
        iterations=1200, depth=6, learning_rate=0.03, loss_function="Logloss",
        eval_metric="AUC", auto_class_weights="Balanced", random_seed=SEED_INNER,
        l2_leaf_reg=5.0, random_strength=0.5, verbose=False, allow_writing_files=False,
    )
    probe = CatBoostClassifier(**params)
    probe.fit(fit, yfit, cat_features=CATEGORICAL_COLS, eval_set=(val, yval), early_stopping_rounds=100)
    best_iter = max(50, int(probe.get_best_iteration()) + 1)
    final_params = dict(params, iterations=best_iter)
    final = CatBoostClassifier(**final_params)
    final.fit(train, yt, cat_features=CATEGORICAL_COLS)
    return final, final.predict_proba(test)[:, 1], final_params


def sklearn_preprocessor():
    numeric = NUMERIC_COLS + BINARY_FEATURE_COLS
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])


def lightgbm_model(xt, yt, xe):
    from lightgbm import LGBMClassifier
    ratio = float((yt == 0).sum() / (yt == 1).sum())
    model = Pipeline([
        ("prep", sklearn_preprocessor()),
        ("model", LGBMClassifier(
            objective="binary", n_estimators=900, learning_rate=0.035, num_leaves=31,
            max_depth=-1, min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, scale_pos_weight=ratio, random_state=SEED_INNER, n_jobs=-1,
        )),
    ])
    model.fit(xt, yt)
    return model, model.predict_proba(xe)[:, 1], {"scale_pos_weight": ratio}


def balanced_rf_d1(xt, yt, xe):
    from imblearn.ensemble import BalancedRandomForestClassifier
    model = Pipeline([
        ("prep", sklearn_preprocessor()),
        ("model", BalancedRandomForestClassifier(
            n_estimators=700, max_depth=None, min_samples_leaf=2,
            sampling_strategy="all", replacement=True, random_state=SEED_INNER, n_jobs=-1,
        )),
    ])
    model.fit(xt, yt)
    return model, model.predict_proba(xe)[:, 1], {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["dataset1", "dataset2"])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=None,
                    choices=["catboost", "lightgbm", "balanced_rf"])
    args = ap.parse_args()

    if args.models is None:
        args.models = ["catboost", "lightgbm", "balanced_rf"] if args.dataset == "dataset1" else ["lightgbm"]

    root = Path(__file__).resolve().parents[1]
    out = root / "results" / args.dataset / "14_algorithm_improvements"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset(args.data, args.dataset)
    df = clean_before_split(raw, args.dataset)
    X = df[FEATURE_COLS].copy(); y = df.stroke.to_numpy(dtype="int8")
    itr, ite = train_test_split(np.arange(len(y)), test_size=.2, random_state=SEED_SPLIT, stratify=y)
    xt, xe, yt, ye = X.iloc[itr], X.iloc[ite], y[itr], y[ite]
    fit_idx, val_idx = train_test_split(np.arange(len(yt)), test_size=.2, random_state=SEED_INNER, stratify=yt)
    xfit, xval, yfit, yval = xt.iloc[fit_idx], xt.iloc[val_idx], yt[fit_idx], yt[val_idx]

    # Frozen split identity makes comparison with Experiment 13 auditable.
    split_meta = dict(dataset=args.dataset, cleaned_rows=len(y), train_rows=len(yt), test_rows=len(ye),
                      train_positive=int(yt.sum()), test_positive=int(ye.sum()), seed_split=SEED_SPLIT,
                      seed_inner=SEED_INNER, untouched_test=True, models=args.models,
                      python=platform.python_version())
    (out / "metadata.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    rows = []
    for name in args.models:
        start = time.perf_counter()
        if name == "catboost":
            if args.dataset != "dataset1":
                raise ValueError("CatBoost is staged for D1 first; use --models lightgbm for D2.")
            model, prob, params = catboost_d1(xfit, yfit, xval, yval, xt, yt, xe)
            # Threshold must come from inner validation, not test.
            pv = model.predict_proba(encode_categories(xval))[:, 1]
        elif name == "lightgbm":
            model, prob, params = lightgbm_model(xt, yt, xe)
            # Separate inner model to choose threshold without reading test labels.
            inner, pv, _ = lightgbm_model(xfit, yfit, xval)
            del inner
        elif name == "balanced_rf":
            if args.dataset != "dataset1":
                raise ValueError("Balanced RF full D2 run is disabled by default due to scale.")
            model, prob, params = balanced_rf_d1(xt, yt, xe)
            inner, pv, _ = balanced_rf_d1(xfit, yfit, xval)
            del inner
        else:
            raise AssertionError(name)

        thresholds = {"default_0.5": .5, "specificity_0.85": specificity_threshold(yval, pv, .85)}
        for policy, threshold in thresholds.items():
            rows.append(dict(model=name, policy=policy, seconds=time.perf_counter()-start,
                             params=json.dumps(params, sort_keys=True), **evaluate(ye, prob, threshold)))
        pd.DataFrame(rows).to_csv(out / "test_comparison.csv", index=False)
        print(json.dumps(rows[-2:], indent=2), flush=True)

    # Selection is reported transparently; primary ranking is MCC, then F1, with Recall as guardrail.
    result = pd.DataFrame(rows)
    result["eligible_recall_0.70"] = result.recall >= .70
    result.sort_values(["eligible_recall_0.70", "mcc", "f1", "recall"], ascending=False).to_csv(
        out / "ranked_comparison.csv", index=False)
    split_meta["status"] = "completed"
    (out / "metadata.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
