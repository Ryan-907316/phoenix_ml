# model_training.py
# This module trains a set of regression models with default hyperparameters and then displays their metrics.

import os
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Users are free to add more regression models if they wish (though efficient integration is not guaranteed!)

from sklearn.metrics import mean_squared_error, r2_score

# Suppress logs
# This gets rid of most of the messages
os.environ['LIGHTGBM_VERBOSE'] = '0'
os.environ['XGBOOST_VERBOSITY'] = '0'

# Define metrics and keep consistency
metrics_dict = {
    "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
    "R^2": lambda y_true, y_pred: r2_score(y_true, y_pred),
    "ADJUSTED R^2": lambda y_true, y_pred, n, p: 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - p - 1),
    "Q^2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
}

# Default hyperparameters
# These are reasonable starting points, HPO will override later.
default_hyperparameters = {
    "SVR (RBF)": {"C": 1.0, "epsilon": 0.01, "gamma": 0.01},
    "Random Forest Regressor": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2, "min_samples_leaf": 1},
    "Gaussian Process Regressor": {"alpha": 1e-6},
    "XGBoost Regressor": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0},
    "HistGradientBoosting Regressor": {"max_iter": 100, "learning_rate": 0.1, "max_depth": None, "min_samples_leaf": 20},
    "LGBM Regressor": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": -1, "num_leaves": 31},
    "Bagging Regressor": {"n_estimators": 10, "max_samples": 1.0},
    "MLP Regressor": {"hidden_layer_sizes": (100,), "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 1000},
    "KNeighbors Regressor": {"n_neighbors": 5, "p": 2},
    "Extra Trees Regressor": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
}

# Model initialiser
def reset_model_to_defaults(model_name, override_params=None):
    model_map = {
        "SVR (RBF)": SVR(kernel='rbf'),
        "Random Forest Regressor": RandomForestRegressor(random_state=0),
        "Gaussian Process Regressor": GaussianProcessRegressor(),
        "XGBoost Regressor": XGBRegressor(objective="reg:squarederror", verbosity=0),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(),
        "LGBM Regressor": LGBMRegressor(verbosity=-1),
        "Bagging Regressor": BaggingRegressor(),
        "MLP Regressor": MLPRegressor(),
        "KNeighbors Regressor": KNeighborsRegressor(),
        "Extra Trees Regressor": ExtraTreesRegressor(),
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    base_params = default_hyperparameters[model_name].copy()
    if override_params:
        base_params.update(override_params)

    return model_map[model_name].set_params(**base_params)

# Train and evaluate
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_var, selected_metrics):
    start_time = time.time()
    model.fit(X_train, y_train[target_var])
    y_pred = model.predict(X_test)

    # For Adjusted R^2 we need n (rows) and p (features)
    n, p = X_test.shape
    if n <= 1 or p < 1:
        raise ValueError(f"Invalid dimensions for Adjusted R^2: n={n}, p={p}.")

    metrics_results = {
        metric_name: (
            metric_func(y_test[target_var], y_pred, n, p)
            if metric_name == "ADJUSTED R^2"
            else metric_func(y_test[target_var], y_pred)
        )
        for metric_name, metric_func in selected_metrics.items()
    }

    metrics_results["Time Elapsed (s)"] = time.time() - start_time
    return metrics_results

# 5. Run models
def run_models(
    selected_model_names,
    X_train, X_test, y_train, y_test,
    target_columns,
    selected_metrics,
    custom_hyperparams=None
):
    results = []
    default_metrics = {}
    default_params = {}
    model_results_dict = {}

    for model_name in selected_model_names:
        print(f"Training {model_name}...")
        override_params = custom_hyperparams.get(model_name, {}) if custom_hyperparams else None
        model = reset_model_to_defaults(model_name, override_params)

        default_metrics[model_name] = {}
        default_params[model_name] = {}
        model_results_dict[model_name] = {}

        for target_var in target_columns:
            metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_var, selected_metrics)
            params = default_hyperparameters.get(model_name, {})

            default_metrics[model_name][target_var] = {
            k: metrics[k] for k in selected_metrics.keys()
        }
            default_metrics[model_name][target_var]["elapsed_time"] = metrics["Time Elapsed (s)"]

            default_params[model_name][target_var] = params

            model_results_dict[model_name][target_var] = {
                **default_metrics[model_name][target_var],
                "params": params
            }

            results.append({
                "Model": model_name,
                "Target Variable": target_var,
                **metrics
            })

    return pd.DataFrame(results), default_metrics, default_params, model_results_dict

# Run the models
def run_model_training_workflow(
    X_train,
    X_test,
    y_train,
    y_test,
    target_columns,
    selected_model_names=None,
    custom_hyperparams=None,
    selected_metrics=None,
    verbose=True
):
    if selected_model_names is None:
        selected_model_names = list(default_hyperparameters.keys())

    if selected_metrics is None:
        selected_metrics = {key: metrics_dict[key] for key in ["MSE", "R^2", "ADJUSTED R^2", "Q^2"]}

    results_df, default_metrics, default_params, model_results_dict = run_models(
        selected_model_names,
        X_train, X_test, y_train, y_test,
        target_columns,
        selected_metrics,
        custom_hyperparams
    )

    if verbose:
        print("\nModel Performance Summary:")
        print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))

    return results_df, default_metrics, default_params, model_results_dict