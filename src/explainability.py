import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def compute_shap_values(
    model,
    X,
    feature_names,
):
    """
    Compute exact Tree SHAP values using XGBoost's
    native pred_contribs implementation.

    This avoids compatibility issues between
    shap.TreeExplainer and newer XGBoost versions.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        X,
        feature_names=list(feature_names),
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
    )

    # Last column is the bias / expected value.
    shap_values = contributions[:, :-1]

    base_values = contributions[:, -1]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X,
        feature_names=list(feature_names),
    )

    return explanation


def get_global_importance(
    explanation,
):
    mean_abs_shap = np.abs(
        explanation.values
    ).mean(axis=0)

    result = pd.DataFrame({
        "feature":
            explanation.feature_names,
        "mean_abs_shap":
            mean_abs_shap,
    })

    result = (
        result
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


def approximate_original_shap(
    shap_pc_values,
    pca_components,
    original_feature_names,
):
    """
    Approximate projection of SHAP values from PCA
    space back to original feature space.

    Parameters
    ----------
    shap_pc_values:
        shape = (n_samples, n_components)

    pca_components:
        shape = (n_components, n_original_features)

    Notes
    -----
    This projection is useful for interpretation,
    but it is NOT exact SHAP for the nonlinear
    composite model XGBoost(PCA(X)).
    """

    projected = (
        shap_pc_values
        @ pca_components
    )

    return pd.DataFrame(
        projected,
        columns=original_feature_names,
    )


def get_projected_global_importance(
    projected_shap,
):
    result = pd.DataFrame({
        "feature":
            projected_shap.columns,
        "mean_abs_projected_shap":
            np.abs(
                projected_shap.values
            ).mean(axis=0),
    })

    result = (
        result
        .sort_values(
            "mean_abs_projected_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result