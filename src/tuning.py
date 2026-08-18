import time

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from xgboost import XGBClassifier


def get_xgboost_param_grid():
    return {
        "n_estimators": [
            200,
            400,
        ],
        "max_depth": [
            3,
            5,
        ],
        "learning_rate": [
            0.03,
            0.1,
        ],
        "min_child_weight": [
            1,
            5,
        ],
        "gamma": [
            0.0,
            0.2,
        ],
        "subsample": [
            0.8,
        ],
        "colsample_bytree": [
            0.8,
        ],
    }


def build_tuning_estimator(
    random_state=42,
):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )


def run_grid_search(
    X_train,
    y_train,
    cv_splits=3,
    random_state=42,
    n_jobs=-1,
):
    model = build_tuning_estimator(
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=get_xgboost_param_grid(),
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
        refit=True,
    )

    start = time.perf_counter()

    grid.fit(
        X_train,
        y_train,
    )

    tuning_time = (
        time.perf_counter()
        - start
    )

    results = pd.DataFrame(
        grid.cv_results_
    )

    results = results.sort_values(
        "rank_test_score"
    ).reset_index(drop=True)

    return (
        grid.best_estimator_,
        grid.best_params_,
        float(grid.best_score_),
        tuning_time,
        results,
    )