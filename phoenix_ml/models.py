# models.py
# This is where the user is free to add, remove, or modify regression models and their hyperparameter search spaces

from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Dictionary of model instances
# Disable verbosity whenever possible, default kernels
models_dict = {
    "SVR (RBF)": SVR(kernel="rbf"),
    "Random Forest Regressor": RandomForestRegressor(verbose=0),
    "Gaussian Process Regressor": GaussianProcessRegressor(),
    "XGBoost Regressor": XGBRegressor(verbosity=0),
    "HistGradientBoosting Regressor": HistGradientBoostingRegressor(),
    "LGBM Regressor": LGBMRegressor(verbose=-1),
    "Bagging Regressor": BaggingRegressor(),
    "MLP Regressor": MLPRegressor(max_iter=1000, verbose=False),
    "KNeighbors Regressor": KNeighborsRegressor(),
    "Extra Trees Regressor": ExtraTreesRegressor(verbose=0),
}

# Define hyperparameter search spaces
# "type": {"int" | "uniform" | "loguniform"} determines sampling strategy.
# "bounds": (low, high)
# "int" → inclusive integer range
# "uniform" → continuous range [low, high]
# "loguniform" → continuous range on log scale (strictly positive)
param_spaces = {
    "SVR (RBF)": {
        "C": {"type": "loguniform", "bounds": (1e-3, 1e3)},
        "epsilon": {"type": "uniform", "bounds": (0.01, 0.1)},
        "gamma": {"type": "loguniform", "bounds": (1e-4, 1e1)},
    },
    "Random Forest Regressor": {
        "n_estimators": {"type": "int", "bounds": (100, 1000)},
        "max_depth": {"type": "int", "bounds": (10, 100)},
        "max_features": {"type": "uniform", "bounds": (0.1, 1.0)},
        "min_samples_split": {"type": "int", "bounds": (2, 20)},
        "min_samples_leaf": {"type": "int", "bounds": (1, 20)},
    },
    "Gaussian Process Regressor": {
        "alpha": {"type": "loguniform", "bounds": (1e-10, 1e-1)},
    },
    "XGBoost Regressor": {
        "learning_rate": {"type": "uniform", "bounds": (0.01, 0.2)},
        "n_estimators": {"type": "int", "bounds": (50, 200)},
        "max_depth": {"type": "int", "bounds": (3, 30)},
        "subsample": {"type": "uniform", "bounds": (0.5, 1.0)},
        "colsample_bytree": {"type": "uniform", "bounds": (0.5, 1.0)},
    },
    "HistGradientBoosting Regressor": {
        "max_iter": {"type": "int", "bounds": (50, 300)},
        "learning_rate": {"type": "uniform", "bounds": (0.01, 0.2)},
        "max_depth": {"type": "int", "bounds": (3, 30)},
        "min_samples_leaf": {"type": "int", "bounds": (1, 20)},
    },
    "LGBM Regressor": {
        "learning_rate": {"type": "uniform", "bounds": (0.01, 0.2)},
        "n_estimators": {"type": "int", "bounds": (50, 300)},
        "max_depth": {"type": "int", "bounds": (3, 30)},
        "num_leaves": {"type": "int", "bounds": (10, 50)},
    },
    "Bagging Regressor": {
        "n_estimators": {"type": "int", "bounds": (10, 200)},
        "max_samples": {"type": "uniform", "bounds": (0.5, 1.0)},
    },
    "MLP Regressor": {
        "hidden_layer_sizes": {"type": "int", "bounds": (50, 300)},
        "alpha": {"type": "loguniform", "bounds": (1e-6, 1e-2)},
        "learning_rate_init": {"type": "loguniform", "bounds": (1e-4, 1e-1)},
        "max_iter": {"type": "int", "bounds": (500, 2000)},
    },
    "KNeighbors Regressor": {
        "n_neighbors": {"type": "int", "bounds": (1, 50)},
        "p": {"type": "int", "bounds": (1, 5)},
    },
    "Extra Trees Regressor": {
        "n_estimators": {"type": "int", "bounds": (50, 500)},
        "max_depth": {"type": "int", "bounds": (10, 100)},
        "min_samples_split": {"type": "int", "bounds": (2, 20)},
    },
}

# Refer to the documentation for more information concerning hyperparameter search spaces.
# In general, wider range is better but takes much longer.

def _check_dicts_in_sync(models: dict, spaces: dict) -> None:
    """models_dict and param_spaces are hand-synced dictionaries with nothing else
    enforcing it — a model added to one but not the other used to fail later with a
    confusing KeyError deep inside HPO. Raise immediately, naming exactly which
    model(s) are missing from which dict, since this file is explicitly meant to be
    user-editable."""
    missing_spaces = set(models) - set(spaces)
    missing_models = set(spaces) - set(models)
    if missing_spaces or missing_models:
        raise AssertionError(
            "models.py: models_dict and param_spaces are out of sync — "
            f"missing from param_spaces: {sorted(missing_spaces)}; "
            f"missing from models_dict: {sorted(missing_models)}"
        )


_check_dicts_in_sync(models_dict, param_spaces)