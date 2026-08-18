import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


def get_sampler(
    dataset_name,
    random_state=42,
):
    if dataset_name == "dataset1":
        return SMOTE(
            random_state=random_state,
        )

    if dataset_name == "dataset2":
        return RandomUnderSampler(
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


def build_tuning_pipeline(
    dataset_name,
    random_state=42,
):
    sampler = get_sampler(
        dataset_name=dataset_name,
        random_state=random_state,
    )

    pca = PCA(
        n_components=0.95,
        svd_solver="full",
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    pipeline = Pipeline([
        (
            "sampler",
            sampler,
        ),
        (
            "pca",
            pca,
        ),
        (
            "model",
            model,
        ),
    ])

    return pipeline


def get_xgboost_param_grid():
    return {
        "model__n_estimators": [
            200,
            400,
        ],
        "model__max_depth": [
            3,
            5,
        ],
        "model__learning_rate": [
            0.03,
            0.1,
        ],
        "model__min_child_weight": [
            1,
            5,
        ],
        "model__gamma": [
            0.0,
            0.2,
        ],
        "model__subsample": [
            0.8,
        ],
        "model__colsample_bytree": [
            0.8,
        ],
    }


def run_grid_search(
    X_train,
    y_train,
    dataset_name,
    cv_splits=3,
    random_state=42,
    n_jobs=-1,
):
    pipeline = build_tuning_pipeline(
        dataset_name=dataset_name,
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    grid = GridSearchCV(
        estimator=pipeline,
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

    results = (
        results
        .sort_values(
            "rank_test_score"
        )
        .reset_index(drop=True)
    )

    best_params = {
        key.replace(
            "model__",
            ""
        ): value
        for key, value
        in grid.best_params_.items()
    }

    return (
        grid.best_estimator_,
        best_params,
        float(
            grid.best_score_
        ),
        tuning_time,
        results,
    )