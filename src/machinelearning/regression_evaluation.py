from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from datasets_util.naming_conventions import DatasetConfig

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor


DATASET_CONFIG_FILE = "multimodal_dataset_folders.json"
PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES = "split_id1"

ModelConfigDict = dict[str, Any]

# -----------------------------
# Metrics
# -----------------------------


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# -----------------------------
# Model + Hyperparameter Space
# -----------------------------
def get_model_configs(random_state: int | None = 42) -> dict[str, ModelConfigDict]:
    configs: dict[str, ModelConfigDict] = {}

    # Ridge
    configs["ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        "params": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

    # Lasso
    configs["lasso"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=10000))
        ]),
        "params": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0]
        }
    }

    # ElasticNet
    configs["elasticnet"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=10000))
        ]),
        "params": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
    }

    # SVR
    configs["svr"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.01, 0.1, 0.5],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale", "auto"]
        }
    }

    # Gaussian Process
    configs["gpr"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(normalize_y=True))
            # This internally standardizes y during training, BUT:
            # It automatically reverses the normalization when predicting
            # So y_pred is still in the original scale
        ]),
        "params": {
            "model__alpha": [1e-6, 1e-4, 1e-2],
            "model__kernel": [
                C(1.0) * RBF(),
                C(1.0) * Matern(nu=1.5),
                C(1.0) * Matern(nu=2.5)
            ]
        }
    }

    # KNN
    configs["knn"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "params": {
            "model__n_neighbors": [2, 3, 5, 7],
            "model__weights": ["uniform", "distance"]
        }
    }

    # Random Forest
    configs["rf"] = {
        "pipeline": RandomForestRegressor(random_state=random_state),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5]
        }
    }

    # Gradient Boosting
    configs["gbr"] = {
        "pipeline": GradientBoostingRegressor(random_state=random_state),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4]
        }
    }

    return configs


# -----------------------------
# Nested CV
# -----------------------------

def nested_cv_grouped(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = "participant_id",
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> pd.DataFrame:

    X = df.drop(columns=[target_col, group_col])
    y = df[target_col]
    groups = df[group_col]

    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)

    configs = get_model_configs()

    results = []

    for model_name, config in configs.items():
        print(f"\n=== Model: {model_name} ===")

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups)
        ):
            print(f"Outer fold {fold_idx+1}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            groups_train = groups.iloc[train_idx]

            grid = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            # CRITICAL: pass groups to inner CV
            grid.fit(X_train, y_train, groups=groups_train)

            fitted_model = grid.best_estimator_

            y_pred = fitted_model.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)

            fold_metrics.append(metrics)

        results.append({
            "model": model_name,
            "rmse_mean": np.mean([m["rmse"] for m in fold_metrics]),
            "rmse_std": np.std([m["rmse"] for m in fold_metrics]),
            "mae_mean": np.mean([m["mae"] for m in fold_metrics]),
            "mae_std": np.std([m["mae"] for m in fold_metrics]),
        })

    return pd.DataFrame(results).sort_values(by="rmse_mean")


def choose_feature_modalities(df: pd.DataFrame, modalities: list, always_keep=["GLC", "participant_id"]):
    cols_to_keep = [
        c for c in df.columns
        if c.startswith(tuple(modalities)) or c in always_keep
    ]

    df_filtered = df[cols_to_keep]
    return df_filtered


def remove_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''We will keep participant_id to allow grouped CV'''
    # return df.drop(columns=["participant_id", "session_id", "datetime"])
    return df.drop(columns=["session_id", "datetime"])


def train_with_train_validation_sets(
    results: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str,
    inner_splits: int = 3,
) -> BaseEstimator:
    # recover configuration from CV results:
    # nested_cv_grouped function only returns aggregated metrics, not the fitted estimators nor their
    # selected hyperparameters.
    # We will refit a GridSearchCV on the full training set (train + validation), using the same grouped
    # CV strategy, and then extract best_estimator_.
    best_name = str(results.iloc[0]["model"])
    configs = get_model_configs()
    config = configs[best_name]

    X_full = train_df.drop(columns=[target_col, "participant_id"])
    y_full = train_df[target_col]
    groups_full = train_df["participant_id"]

    inner_cv = GroupKFold(n_splits=inner_splits)

    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_full, y_full, groups=groups_full)

    fitted_model = grid.best_estimator_

    print(f"Best model: {best_name}")
    print(f"Best hyperparameters: {grid.best_params_}")

    return fitted_model


def regression_evaluation(modalities: list[str]) -> None:
    """
    Implement the whole procedure of hyperparameter tuning, training and testing.
    """
    np.random.seed(42)

    datasetConfig = DatasetConfig(DATASET_CONFIG_FILE)

    # read the split_id1_train.csv file
    train_set = pd.read_csv(os.path.join(datasetConfig.get_dataset_machine_learning_path(
    ), PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_train.csv"))
    test_set = pd.read_csv(os.path.join(datasetConfig.get_dataset_machine_learning_path(
    ), PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_test.csv"))
    validation_set = pd.read_csv(os.path.join(datasetConfig.get_dataset_machine_learning_path(
    ), PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_validation.csv"))

    # remove metadata but keep participant_id column
    train_set = remove_metadata_columns(train_set)
    test_set = remove_metadata_columns(test_set)
    validation_set = remove_metadata_columns(validation_set)

    # keep only the columns (features) whose names start with the prefixes provided in modalities
    train_set = choose_feature_modalities(train_set, modalities)
    test_set = choose_feature_modalities(test_set, modalities)
    validation_set = choose_feature_modalities(validation_set, modalities)

    print(f"Train set shape: {train_set.shape}")
    # put together train and validation sets
    train_df = pd.concat([train_set, validation_set])

    print(f"Train set shape: {train_set.shape}")

    target_col = "GLC"
    results = nested_cv_grouped(
        train_df, target_col=target_col, outer_splits=5, inner_splits=3)

    print("\n=== Nested CV Results ===")
    print(results)

    # Train the regressor: results only name the best model type; refit with the same
    # grouped GridSearchCV as nested_cv_grouped to recover hyperparameters.
    # create an instance of this object with the given hyperparameters
    fitted_model = train_with_train_validation_sets(
        results, train_df, target_col=target_col, inner_splits=3
    )

    # Test the classifier
    # Show the participants in test set
    print("Testing with participants: ", np.unique(test_set["participant_id"]))
    # now we drop the participant_id column
    X_test = test_set.drop(columns=[target_col, "participant_id"])

    print("Columns in test set:", X_test.columns)

    y_pred = fitted_model.predict(X_test)
    metrics = regression_metrics(test_set[target_col], y_pred)
    print(f"Test set metrics (best model): {metrics}")

    # Baseline for comparison: ignore features, predict training-set mean (dummy regressor)
    X_train_full = train_df.drop(columns=[target_col, "participant_id"])
    y_train_full = train_df[target_col]
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_full, y_train_full)
    y_dummy = dummy.predict(X_test)
    dummy_metrics = regression_metrics(test_set[target_col], y_dummy)
    print(f"Test set metrics (DummyRegressor mean baseline): {dummy_metrics}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Choose list among modalities = ["ppg_", "ecg_", "bioimp_"]
    modalities = ["ppg_"]
    regression_evaluation(modalities)
