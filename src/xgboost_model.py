import time

import numpy as np
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
from xgboost import XGBClassifier


def build_xgboost_baseline(
    random_state=42,
    n_jobs=-1,
):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",

        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,

        min_child_weight=1,
        gamma=0.0,

        subsample=0.9,
        colsample_bytree=0.9,

        tree_method="hist",

        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model


def fit_model(
    model,
    X_train,
    y_train,
):
    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.perf_counter()
        - start
    )

    return model, train_time


def predict_model(
    model,
    X_test,
):
    start = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    predict_time = (
        time.perf_counter()
        - start
    )

    return (
        y_pred.astype(np.int8),
        y_prob.astype(np.float32),
        predict_time,
    )


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_prob,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    return (
        metrics,
        cm,
        report,
    )