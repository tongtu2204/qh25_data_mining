import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_training_data(
    X_train,
    y_train,
    dataset_name,
    random_state=42,
):
    if dataset_name == "dataset1":
        sampler = SMOTE(
            random_state=random_state,
        )

    elif dataset_name == "dataset2":
        sampler = RandomUnderSampler(
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    X_balanced, y_balanced = sampler.fit_resample(
        X_train,
        y_train,
    )

    X_balanced = np.asarray(
        X_balanced,
        dtype=np.float32,
    )

    y_balanced = np.asarray(
        y_balanced,
        dtype=np.int8,
    )

    return (
        X_balanced,
        y_balanced,
        sampler,
    )