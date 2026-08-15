from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def make_classifier(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=1,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def make_sampler(strategy: str, random_state: int = 42):
    if strategy == "none":
        return "passthrough"
    if strategy == "smote":
        return SMOTE(random_state=random_state)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=random_state)
    raise ValueError(f"Unknown balance strategy: {strategy}")


def make_pipeline(
    preprocessor,
    balance_strategy: str,
    use_pca: bool,
    random_state: int = 42,
) -> ImbPipeline:
    # Dataset 2 rất lớn: random undersampling được đặt TRƯỚC one-hot/scaling
    # để không tạo ma trận dense hàng triệu dòng rồi mới bỏ mẫu.
    # SMOTE cần dữ liệu numeric nên vẫn đặt SAU preprocessing.
    if balance_strategy == "undersample":
        steps: list[tuple[str, Any]] = [
            ("sampler", make_sampler(balance_strategy, random_state)),
            ("preprocess", preprocessor),
        ]
    else:
        steps = [
            ("preprocess", preprocessor),
            ("sampler", make_sampler(balance_strategy, random_state)),
        ]

    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    steps.append(("model", make_classifier(random_state)))
    return ImbPipeline(steps)


def make_numeric_pipeline(balance_strategy: str, use_pca: bool, random_state: int = 42) -> ImbPipeline:
    """Pipeline dùng cho cache NPZ đã được preprocessing thành ma trận số."""
    steps: list[tuple[str, Any]] = [
        ("sampler", make_sampler(balance_strategy, random_state)),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    steps.append(("model", make_classifier(random_state)))
    return ImbPipeline(steps)


def evaluate_model(model, X_test, y_test) -> tuple[dict[str, Any], pd.DataFrame]:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "tn": int(cm.ravel()[0]),
        "fp": int(cm.ravel()[1]),
        "fn": int(cm.ravel()[2]),
        "tp": int(cm.ravel()[3]),
    }

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    return metrics, report


def cross_validate_auc(model, X, y, n_splits: int = 10) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=1)
    return {
        "cv_auc_mean": float(np.mean(scores)),
        "cv_auc_std": float(np.std(scores)),
        "cv_auc_scores": [float(v) for v in scores],
    }
