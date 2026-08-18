import numpy as np
import pandas as pd
import shap


def build_tree_explainer(
    model,
):
    return shap.TreeExplainer(
        model
    )


def compute_shap_values(
    model,
    X,
    feature_names,
):
    explainer = build_tree_explainer(
        model
    )

    explanation = explainer(
        X
    )

    explanation.feature_names = list(
        feature_names
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
    Approximate projection of SHAP values from PCA space
    back to original feature space.

    SHAP_PC shape:
        n_samples x n_components

    PCA components shape:
        n_components x n_original_features

    Approximation:
        contribution_original
        = SHAP_PC @ PCA_components

    This is NOT exact SHAP for the nonlinear
    XGBoost(PCA(X)) composite model.
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