# model_training.py
# This module trains a set of regression models with default hyperparameters and then displays their metrics.

import os
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from sklearn.base import clone

from phoenix_ml.models import models_dict
# Users are free to add more regression models if they wish (though efficient integration is not guaranteed!)

from sklearn.metrics import mean_squared_error, r2_score

# Suppress logs
# This gets rid of most of the messages
os.environ['LIGHTGBM_VERBOSE'] = '0'
os.environ['XGBOOST_VERBOSITY'] = '0'

# Define metrics and keep consistency
metrics_dict = {
    "MSE":          lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
    "RMSE":         lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    # NRMSE: range-normalised RMSE; undefined when all targets are identical (returns NaN)
    "NRMSE":        lambda y_true, y_pred: (
                        np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))
                        if (np.max(y_true) - np.min(y_true)) > 0 else float("nan")
                    ),
    # MAPE: epsilon guard avoids division-by-zero when y_true passes through or near zero
    "MAPE":         lambda y_true, y_pred: float(
                        np.mean(np.abs(
                            (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) /
                            np.where(np.abs(np.asarray(y_true, dtype=float)) > 1e-10,
                                     np.abs(np.asarray(y_true, dtype=float)), 1e-10)
                        )) * 100
                    ),
    "R^2":          lambda y_true, y_pred: r2_score(y_true, y_pred),
    "ADJUSTED R^2": lambda y_true, y_pred, n, p: 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - p - 1),
    # Q^2: same zero-variance failure mode as NRMSE above (a constant y_true — a genuinely
    # constant column, or just an unlucky small test fold — makes the denominator 0);
    # unlike R^2 above (sklearn's r2_score has its own internal zero-variance handling),
    # this hand-rolled formula divided by zero silently until this guard was added.
    "Q^2":          lambda y_true, y_pred: (
                        1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                        if np.sum((y_true - np.mean(y_true)) ** 2) > 0 else float("nan")
                    ),
    # KGE: guards against zero variance (std) and zero-mean targets (mean ratio) which cause division-by-zero
    "KGE":          lambda y_true, y_pred: (
                        1 - np.sqrt(
                            (np.corrcoef(y_true, y_pred)[0, 1] - 1) ** 2 +
                            (np.std(y_pred) / np.std(y_true) - 1) ** 2 +
                            (np.mean(y_pred) / np.mean(y_true) - 1) ** 2
                        )
                        if np.std(y_true) > 0 and np.mean(y_true) != 0 else float("nan")
                    ),
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

_MONOTONIC_CAPABLE_MODELS = ("XGBoost Regressor", "LGBM Regressor")


def derive_seed(base_seed, offset):
    """Derive a distinct-but-reproducible seed for one stochastic stage from the single
    session-level random_seed, rather than reusing the same literal integer everywhere
    (e.g. train/test split, HPO samplers, and SHAP background sampling all seeded 0 would
    be an unnecessary — if largely theoretical — correlation risk). Offsets are assigned
    per call site (see SEED_OFFSET_*); any fixed base_seed always reproduces the same
    derived seeds.

    The result is reduced modulo 2**32: numpy's RandomState (which sklearn
    estimators use) only accepts seeds in [0, 2**32 - 1], so a negative or very
    large base seed used to sail through here and crash much later, at fit
    time, with an error nowhere near the setting that caused it. Any integer
    base seed is valid input; the same value always maps to the same seed.
    """
    return (int(base_seed) + int(offset)) % (2 ** 32)


# Named offsets for derive_seed(), one per independent stochastic stage in the
# pipeline — centralised here so callers across files don't collide on the same
# literal offset by accident.
SEED_OFFSET_MODEL_CONSTRUCTION = 0
SEED_OFFSET_TRAIN_TEST_SPLIT   = 1
SEED_OFFSET_CV                 = 2
# One shared offset covers all three HPO backends (Random/Hyperopt/Skopt) — they're
# entirely different algorithms/libraries, so reusing one derived seed across them
# carries no real correlation risk, unlike reusing it within the same sampler.
SEED_OFFSET_HPO                = 3
SEED_OFFSET_UQ_BOOTSTRAP       = 6
SEED_OFFSET_PERMUTATION_IMPORTANCE = 9
SEED_OFFSET_OUTLIER_DETECTION  = 10
SEED_OFFSET_SHAP_BACKGROUND    = 11


def build_monotone_constraints_kwarg(model_name, feature_names, constraints):
    """Convert a {feature_name: -1/0/+1} dict into the kwarg each library expects.

    XGBoost *can* take a {feature_name: +/-1} dict, but only when the training data
    carries matching column names at fit time — this pipeline fits on X_train_scaled,
    a plain ndarray from scaler.fit_transform() (data_preprocessing.py), not a
    DataFrame, so the dict form raises "Constrained features are not a subset of
    training data feature names". Using the same full-length, feature-order list for
    both libraries sidesteps that entirely (XGBoost accepts a plain tuple/list, purely
    positional, no name matching needed). Returns {} when there's nothing to apply, so
    callers can unconditionally do base_params.update(...).
    """
    if not constraints or model_name not in _MONOTONIC_CAPABLE_MODELS:
        return {}
    # Only -1/0/+1 are meaningful monotonicity directions; anything else (a 2,
    # a "up") used to pass straight through to LGBM/XGBoost/HGB, which either
    # crash with their own message or silently accept an undefined value.
    for f, v in constraints.items():
        try:
            iv = int(v)
            integral = (float(v) == iv)  # rejects 0.5 -> silently becoming "no constraint"
        except (TypeError, ValueError):
            raise ValueError(
                f"Monotonic constraint for feature {f!r} must be -1, 0 or 1, got {v!r}."
            ) from None
        if not integral or iv not in (-1, 0, 1):
            raise ValueError(
                f"Monotonic constraint for feature {f!r} must be -1, 0 or 1, got {v!r}."
            )
    values = [int(constraints.get(f, 0)) for f in feature_names]
    if not any(values):
        return {}
    if model_name == "XGBoost Regressor":
        return {"monotone_constraints": tuple(values)}
    return {"monotone_constraints": values}


def apply_monotone_constraints_for_target(model, model_name, feature_names, target_constraints):
    """Apply one target's monotonicity constraint to a model instance that is SHARED
    and refit once per target in a loop — baseline training, before-HPO UQ, before-HPO
    interpretability, and HPO itself all reuse one instance this way, since constructing
    a fresh instance per target would lose the point of reusing the same search/CV state.

    Must be called for every target in the loop, not only ones with an active
    constraint: build_monotone_constraints_kwarg() alone returns {} both when the model
    can't accept monotone_constraints at all AND when this target's own constraints are
    empty/all-zero — a caller that only calls set_params() when that dict is truthy
    (the bug this function replaces, found via a systematic failure-mode sweep) silently
    leaves a PREVIOUS target's constraint active for any model that does support the
    parameter, since nothing else ever resets it between targets. No-op for models that
    don't accept the parameter at all.

    Capability is checked against _MONOTONIC_CAPABLE_MODELS, not model.get_params()
    reflectively: LightGBM's sklearn wrapper only surfaces monotone_constraints in
    get_params() after it has been set at least once, so a freshly-constructed
    LGBMRegressor that's never had a constraint applied would otherwise look
    (incorrectly) like a model that doesn't support the parameter at all.
    """
    if feature_names is None or model_name not in _MONOTONIC_CAPABLE_MODELS:
        return
    kwargs = build_monotone_constraints_kwarg(model_name, feature_names, target_constraints or {})
    if not kwargs:
        neutral = [0] * len(feature_names)
        kwargs = {"monotone_constraints": tuple(neutral) if model_name == "XGBoost Regressor" else neutral}
    model.set_params(**kwargs)


# Model initialiser
def reset_model_to_defaults(model_name, override_params=None, feature_names=None,
                            monotonic_constraints=None, random_state=None):
    """Fresh, unfitted instance of `model_name` with default hyperparameters applied.

    Clones from models.py's models_dict rather than maintaining a second hardcoded
    constructor list — models_dict is the single source of truth for each model's
    class and base constructor kwargs (e.g. verbosity flags), so a model only ever
    needs to be registered in one place.
    """
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}")

    base_params = default_hyperparameters.get(model_name, {}).copy()
    if override_params:
        base_params.update(override_params)
    if feature_names is not None and monotonic_constraints:
        base_params.update(build_monotone_constraints_kwarg(model_name, feature_names, monotonic_constraints))

    model = clone(models_dict[model_name])
    # Not every model accepts random_state (e.g. SVR, KNeighbors are deterministic given
    # fixed data) — check reflectively rather than hardcoding which model classes support it.
    # None resolves to seed 0 (package-wide convention, same as the HPO/UQ modules) so
    # results are reproducible by default even when no seed is supplied. setdefault keeps
    # any explicit random_state arriving via override_params.
    if "random_state" in model.get_params():
        base_params.setdefault("random_state", 0 if random_state is None else random_state)

    # warm_start=True carries a previous .fit() call's state into the next one — every
    # caller in this codebase refits ONE shared instance across multiple targets in a
    # loop (run_models, permutation/LOFO importance, HPO), so a user-set warm_start=True
    # (e.g. RandomForest/MLP via custom hyperparameters) would silently build target 2's
    # model on top of target 1's fitted state. This function promises a fresh instance,
    # so warm_start is never allowed through regardless of what override_params requests.
    if "warm_start" in model.get_params() and base_params.get("warm_start"):
        print(f"[WARN] '{model_name}': warm_start is not supported here (models are "
              f"refit per target from one shared instance) - forcing warm_start=False")
        base_params["warm_start"] = False

    return model.set_params(**base_params)

# Train and evaluate
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_var, selected_metrics):
    start_time = time.time()
    model.fit(X_train, y_train[target_var])
    y_pred = model.predict(X_test)

    # For Adjusted R^2 we need n (rows) and p (features)
    n, p = X_test.shape
    if n <= 1 or n <= p + 1:
        # Too few test samples to compute Adjusted R² reliably — return NaN for that metric
        metrics_results = {}
        for metric_name, metric_func in selected_metrics.items():
            if metric_name == "ADJUSTED R^2":
                metrics_results[metric_name] = float("nan")
            else:
                metrics_results[metric_name] = metric_func(y_test[target_var], y_pred)
        metrics_results["Time Elapsed (s)"] = time.time() - start_time
        return metrics_results

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

# GPR is O(n^2) memory / O(n^3) time in training-set rows; SVR is O(n^2)-O(n^3)
# depending on kernel/data — both can hang or OOM on a realistically large dataset
# with zero warning otherwise (HPO subsamples internally, but a baseline .fit() here
# does not).
_QUADRATIC_OR_WORSE_MODELS = {"SVR (RBF)", "Gaussian Process Regressor"}
_LARGE_DATASET_WARN_ROWS = 3000


def _warn_if_large_dataset_for_model(model_name, n_rows):
    if model_name in _QUADRATIC_OR_WORSE_MODELS and n_rows > _LARGE_DATASET_WARN_ROWS:
        tqdm.write(
            f"  [WARN] '{model_name}' scales poorly with training set size (O(n^2) or "
            f"worse) and {n_rows} training rows may be slow or memory-heavy."
        )


# 5. Run models
def run_models(
    selected_model_names,
    X_train, X_test, y_train, y_test,
    target_columns,
    selected_metrics,
    custom_hyperparams=None,
    feature_names=None,
    monotonic_constraints=None,
    random_state=None,
    checkpoint_fn=None,
):
    results = []
    default_metrics = {}
    default_params = {}
    model_results_dict = {}

    for model_name in tqdm(selected_model_names, desc="Training models", unit="model", leave=False):
        # Optional cooperative pause/cancel hook (e.g. from the UI): checked once per
        # model, mirroring run_all_models_optimisation's own checkpoint granularity.
        if checkpoint_fn is not None:
            checkpoint_fn()
        tqdm.write(f"  Training {model_name}...")
        _warn_if_large_dataset_for_model(model_name, len(X_train))
        override_params = custom_hyperparams.get(model_name, {}) if custom_hyperparams else None
        # monotonic_constraints may be per-target ({target: {feature: +-1}}); the shared
        # model instance below is constructed unconstrained and re-constrained per target
        # in the loop, since a single instance is fit once per target_var in turn.
        per_target_constraints = bool(monotonic_constraints) and all(
            isinstance(v, dict) for v in monotonic_constraints.values()
        )
        model = reset_model_to_defaults(
            model_name, override_params, feature_names=feature_names,
            monotonic_constraints=None if per_target_constraints else monotonic_constraints,
            random_state=random_state,
        )

        default_metrics[model_name] = {}
        default_params[model_name] = {}
        model_results_dict[model_name] = {}

        for target_var in target_columns:
            if per_target_constraints:
                apply_monotone_constraints_for_target(
                    model, model_name, feature_names, monotonic_constraints.get(target_var, {})
                )
            metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_var, selected_metrics)
            # Record what was actually trained with: defaults merged with any custom
            # overrides, as a fresh copy per target so later mutation can't leak.
            params = default_hyperparameters.get(model_name, {}).copy()
            if override_params:
                params.update(override_params)

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
    verbose=True,
    feature_names=None,
    monotonic_constraints=None,
    random_state=None,
    checkpoint_fn=None,
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
        custom_hyperparams,
        feature_names=feature_names,
        monotonic_constraints=monotonic_constraints,
        random_state=random_state,
        checkpoint_fn=checkpoint_fn,
    )

    if verbose:
        print("\nModel Performance Summary:")
        print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))

    return results_df, default_metrics, default_params, model_results_dict