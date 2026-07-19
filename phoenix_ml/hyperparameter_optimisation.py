# hyperparameter_optimisation.py
# This module takes all the regression models and tries to optimise the hyperparameters of all models via different methods in order to obtain more accurate results.
# The user is allowed to choose between three HPO methods: Random search (of which they can specify a sampling method), Hyperopt, and scikit-optimize.
# Random search allows for the choice of random sampling (aka Monte Carlo), Sobol, Halton, or Latin Hypercube sampling.
# Hyperopt uses a technique called Tree-based Parzen Estimators (TPE), and scikit-optimize uses Gaussian Process Minimisation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import time

from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    _SKOPT_AVAILABLE = True
except Exception:
    _SKOPT_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
    from hyperopt.exceptions import AllTrialsFailed
    _HYPEROPT_AVAILABLE = True
except Exception:
    _HYPEROPT_AVAILABLE = False
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

from phoenix_ml.models import param_spaces
from phoenix_ml.model_training import metrics_dict, apply_monotone_constraints_for_target
from phoenix_ml.validation import require_int_at_least

import os
import warnings
import contextlib
from sklearn.exceptions import ConvergenceWarning

# Suppress logs
os.environ['LIGHTGBM_VERBOSE'] = '0'
os.environ['XGBOOST_VERBOSITY'] = '0'

# Suppress general Python warnings from skopt and others
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Patch joblib so parallel batch completions update a tqdm bar."""
    import joblib
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

# skopt's gp_minimize has no "failed trial" concept — a bad hyperparameter
# combination (e.g. KNN's n_neighbors exceeding the training set size) must
# still return a real float. This value is bad in both the raw (minimising
# metrics) and negated (maximising metrics) directions the caller uses it in.
_TRIAL_FAILURE_SENTINEL = 1e10


def format_elapsed_time(seconds):
    minutes = int(seconds) // 60
    rem_seconds = seconds - (minutes * 60)
    return f"{minutes}m {rem_seconds:.2f}s"

# Centralised metrics so it's consistent across Random/Hyperopt/Skopt. ADJUSTED R^2 needs (n, p), validated here to avoid silent misuse.
_HPO_LOWER_IS_BETTER = {"MSE", "RMSE", "MAE", "NRMSE", "MAPE"}

_METRIC_PLOT_LABEL = {
    "MSE":         "MSE",
    "RMSE":        "RMSE",
    "R^2":         "R²",
    "ADJUSTED R^2":"Adjusted R²",
    "Q^2":         "Q²/NSE",
    "NRMSE":       "NRMSE",
    "MAPE":        "MAPE (%)",
    "MAE":         "MAE",
    "KGE":         "KGE",
}

def _metric_label(key: str) -> str:
    return _METRIC_PLOT_LABEL.get(key, key)

def _compute_metric(y_true, y_pred, metric_name, n=None, p=None):
    if metric_name == "MSE":
        return mean_squared_error(y_true, y_pred)
    elif metric_name == "RMSE":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric_name == "MAE":
        return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
    elif metric_name == "R^2":
        return r2_score(y_true, y_pred)
    elif metric_name == "ADJUSTED R^2":
        if n is None or p is None:
            raise ValueError("Adjusted R^2 requires n and p.")
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    elif metric_name == "Q^2":
        # Same zero-variance guard as NRMSE/KGE below — a constant y_true divided by
        # zero silently until this was added (found via a systematic failure-mode sweep).
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - np.sum((y_true - y_pred) ** 2) / denom if denom > 0 else float("nan")
    elif metric_name == "NRMSE":
        r = np.max(y_true) - np.min(y_true)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)) / r) if r > 0 else float("nan")
    elif metric_name == "MAPE":
        denom = np.where(np.abs(y_true) > 1e-10, np.abs(y_true), 1e-10)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    elif metric_name == "KGE":
        if np.std(y_true) == 0 or np.mean(y_true) == 0:
            return float("nan")
        return 1 - np.sqrt(
            (np.corrcoef(y_true, y_pred)[0, 1] - 1) ** 2 +
            (np.std(y_pred) / np.std(y_true) - 1) ** 2 +
            (np.mean(y_pred) / np.mean(y_true) - 1) ** 2
        )
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

def _cast_params(params: dict, model_name: str) -> dict:
    """Cast HPO-sampled values to types expected by sklearn/xgboost/lgbm."""
    result = params.copy()
    for key, value in params.items():
        if key == "hidden_layer_sizes":
            result[key] = (int(value),) if not isinstance(value, tuple) else value
        elif key in param_spaces.get(model_name, {}) and param_spaces[model_name][key]["type"] == "int":
            result[key] = int(value)
    return result

def run_random_search(
    model, param_space, X_train, X_test, y_train, y_test, sample_size, n_iter, n_jobs, target_var, metric, sampling_method,
    patience=None, min_delta=1e-4, model_name="", random_state=None
):
    # 1) Build a low-discrepancy sampler (Sobol/Halton/LHS) or PRNG ("Random").
    # 2) Map samples in [0,1]^d onto each hyperparameter's domain:
    #    - "uniform"/"loguniform": linear/log scaling
    #    - "int": round to nearest int
    #    - "choice": index into options
    #    - special-case "hidden_layer_sizes": produce (units,) tuple for MLP
    # 3) (Optional) Subsample training set for speed on large data.
    # 4) Fit/evaluate each config; parallel when ES is off, sequential when ES is on.
    # 5) Return a DataFrame of all trials + the best row, runtime, and early-stopping info.
    # n_iter=0 used to produce an empty tracking DataFrame and crash with a bare
    # KeyError on the metric column; negative values crashed inside the sampler
    # with "negative dimensions are not allowed".
    require_int_at_least("n_iter", n_iter, minimum=1)
    require_int_at_least("sample_size", sample_size, minimum=1)
    start_time = time.time()
    def generate_param_samples(param_space, sampler):
        param_samples = {}
        for i, (param, bounds) in enumerate(param_space.items()):
            if bounds["type"] in ["loguniform", "uniform"]:
                low, high = bounds["bounds"]
                param_samples[param] = low + (high - low) * sampler[:, i]
            elif bounds["type"] == "choice":
                opts = bounds["options"]
                idx = np.floor(sampler[:, i] * len(opts)).astype(int).clip(0, len(opts)-1)
                param_samples[param] = [opts[j] for j in idx]
            elif bounds["type"] == "int":
                low, high = bounds["bounds"]
                vals = np.round(low + (high - low) * sampler[:, i]).astype(int)
                if param == "hidden_layer_sizes":
                    param_samples[param] = [(int(v),) for v in vals]
                else:
                    param_samples[param] = vals
        return param_samples

    # Generate parameter samples
    # Map a unit hypercube sample matrix 'sampler' -> dict[param] -> list of typed values.
    # Keep lengths aligned so we can zip into a list of param dicts.
    d = len(param_space)
    _seed = 0 if random_state is None else random_state
    if sampling_method == "Random":
        sampler = np.random.RandomState(_seed).rand(n_iter, d)
    elif sampling_method == "Sobol":
        # Use next power of two, then slice to n_iter
        m = int(np.ceil(np.log2(max(1, n_iter))))
        sob = Sobol(d=d, scramble=True, seed=_seed)
        sampler = sob.random_base2(m)[:n_iter]
    elif sampling_method == "Halton":
        sampler = Halton(d=d, scramble=True, seed=_seed).random(n=n_iter)
    elif sampling_method == "Latin Hypercube":
        sampler = LatinHypercube(d=d, seed=_seed).random(n=n_iter)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    param_samples = generate_param_samples(param_space, sampler)
    param_combinations = [dict(zip(param_space.keys(), values)) for values in zip(*param_samples.values())]

    # Subsample training data if needed
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=_seed)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # When patience is None, evaluate() runs inside joblib.Parallel below — a model
    # with its own internal n_jobs (XGBoost/LGBM default to using all cores) then
    # oversubscribes CPU cores multiplicatively against the outer n_jobs, making the
    # search far slower wall-clock than the requested n_jobs/n_iter would imply.
    # Force single-threaded models in that branch only; the sequential
    # (early-stopping) branch has no outer parallelism to conflict with.
    _parallel_dispatch = patience is None
    _supports_n_jobs = "n_jobs" in model.get_params()

    # Function to evaluate a single configuration
    def evaluate(params):
        if _parallel_dispatch and _supports_n_jobs:
            model.set_params(n_jobs=1)
        # Same RF constraint fix hyperopt_objective/skopt_objective both apply —
        # without it, Random Search silently searches a slightly different RF
        # space than the other two backends for the same nominal bounds.
        if model_name == "Random Forest Regressor":
            if "min_samples_split" in params and "min_samples_leaf" in params:
                if params["min_samples_split"] <= params["min_samples_leaf"]:
                    params["min_samples_split"] = params["min_samples_leaf"] + 1
        model.set_params(**params)
        try:
            model.fit(X_train_sample, y_train_sample[target_var])
            y_pred = model.predict(X_test)
            n, p = X_test.shape
            metric_value = _compute_metric(y_test[target_var], y_pred, metric, n, p)
        except Exception as exc:
            # E.g. a sampled n_neighbors exceeding the (sub)training set size for
            # KNN — one bad hyperparameter combination must not crash the whole
            # search; penalise it to the worst possible score and move on.
            tqdm.write(f"  [WARN] Random Search trial failed for '{model_name}' "
                       f"with {params}: {exc} - skipping")
            metric_value = np.inf if is_minimised else -np.inf
        return params, metric_value

    # Run evaluations: sequential with early stopping, or parallel without
    is_minimised = metric in _HPO_LOWER_IS_BETTER or metric == "MAE"

    if patience is not None:
        best_es_score = np.inf if is_minimised else -np.inf
        no_improve_count = 0
        actual_iters = 0
        results = []
        desc = f"Random Search [{model_name}] {target_var}"
        with tqdm(total=n_iter, desc=desc, unit="config", leave=False) as pbar:
            for params in param_combinations:
                params_out, metric_value = evaluate(params)
                results.append((params_out, metric_value))
                actual_iters += 1
                pbar.update(1)
                improved = (metric_value < best_es_score - min_delta) if is_minimised else (metric_value > best_es_score + min_delta)
                if improved:
                    best_es_score = metric_value
                    no_improve_count = 0
                else:
                    # Only a non-improving step can trigger the stop — patience=0
                    # means "stop at the first evaluation that fails to improve",
                    # never "stop after an improvement" (an unconditional check
                    # here used to break immediately at patience=0 even when the
                    # first evaluation had just improved on -inf).
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        tqdm.write(f"  Early stopping at {actual_iters}/{n_iter} iterations (no {metric} improvement > {min_delta} for {no_improve_count} steps)")
                        break
        stopped_early = actual_iters < n_iter
    else:
        desc = f"Random Search [{model_name}] {target_var}"
        with _tqdm_joblib(tqdm(total=n_iter, desc=desc, unit="config", leave=False)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate)(params) for params in param_combinations
            )
        actual_iters = n_iter
        stopped_early = False

    # Compile results into a DataFrame
    metric_name = metric.upper()
    results_df = pd.DataFrame([{**params, metric_name: value} for params, value in results])

    # Determine whether to maximise or minimise the metric
    if metric in _HPO_LOWER_IS_BETTER:
        best_row = results_df.loc[results_df[metric_name].idxmin()]
    elif metric in ["R^2", "ADJUSTED R^2", "Q^2", "KGE"]:
        best_row = results_df.loc[results_df[metric_name].idxmax()]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Extract the best parameters and metric value
    best_params = best_row.drop([metric_name]).to_dict()
    best_metric_value = best_row[metric_name]

    elapsed_time = time.time() - start_time
    print(f"Random Search completed in: {format_elapsed_time(elapsed_time)}")

    es_info = {
        "stopped_early": stopped_early,
        "actual_iters": actual_iters,
        "max_iters": n_iter,
        "patience": patience,
        "min_delta": min_delta,
    }
    return results_df, best_params, best_metric_value, elapsed_time, es_info


def plot_random_search_results(results_by_target, param_space, model_name, metric, sampling_method, es_info_by_target=None):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Build early-stopping annotation for the suptitle
    es_info_by_target = es_info_by_target or {}
    es_notes = []
    for tv, es in es_info_by_target.items():
        if es.get("stopped_early"):
            es_notes.append(f"{tv}: {es['actual_iters']}/{es['max_iters']} iters")
    es_suffix = f"\nEarly Stopped — {', '.join(es_notes)}" if es_notes else ""

    # Create subplots for each target variable. The y-axis is shared per ROW (per
    # target), not across the whole grid: different targets can live on entirely
    # different metric scales, and a grid-wide share left the better-scaled target's
    # row mostly empty space.
    fig, axes = plt.subplots(num_targets, num_params, figsize=(5 * num_params, 4 * num_targets))
    _mlabel = _metric_label(metric_column)
    fig.suptitle(f"Random Search ({sampling_method}) Results for {model_name} - {_mlabel}{es_suffix}",
                 fontsize=16, fontweight="bold")

    if num_targets == 1:
        axes = [axes]  # Make sure axes is iterable if there's only one target
    if num_params == 1:
        axes = [[ax] for ax in axes]  # Adjust for single parameter

    # Plot results for each target variable
    for row_idx, (target_var, results_df) in enumerate(results_by_target.items()):
        for col_idx, (param, config) in enumerate(param_space.items()):
            ax = axes[row_idx][col_idx]
            if col_idx > 0:
                ax.sharey(axes[row_idx][0])
            data = results_df[param].apply(lambda x: x[0] if isinstance(x, tuple) else x)

            ax.scatter(data, results_df[metric_column], alpha=0.7, c="blue")
            ax.set_xlabel(param)
            ax.set_ylabel(_mlabel if col_idx == 0 else "")
            ax.set_title(f"{target_var}: {param} vs {_mlabel}")
            ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

# Tracking lists to store the best values
def tracking():
    return [], []

# Objective Function for Hyperopt
  # Cast types expected by the estimator (ints/tuples).
    # Enforce simple constraints (e.g., min_samples_split > min_samples_leaf).
    # Subsample training for speed; evaluate selected metric and
    # return {"loss": value, "status": STATUS_OK, ...params}
    # Convention: Hyperopt minimises 'loss' => we negate metrics we aim to maximise.
def hyperopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size,
                       random_state=None):
    params = _cast_params(params, model_name)

    # Ensure constraints (e.g., min_samples_split > min_samples_leaf)
    if model_name == "Random Forest Regressor":
        if "min_samples_split" in params and "min_samples_leaf" in params:
            if params["min_samples_split"] <= params["min_samples_leaf"]:
                params["min_samples_split"] = params["min_samples_leaf"] + 1

    # Perform dataset subsampling
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size,
                                                   random_state=0 if random_state is None else random_state)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Set model parameters and fit
    model.set_params(**params)
    try:
        model.fit(X_train_sample, y_train_sample[target_var])
        y_pred = model.predict(X_test)
        n, p = X_test.shape
        metric_value = _compute_metric(y_test[target_var], y_pred, metric, n, p)
    except Exception as exc:
        # E.g. a sampled n_neighbors exceeding the (sub)training set size for KNN —
        # one bad hyperparameter combination must not crash the whole search.
        # STATUS_FAIL is hyperopt's own mechanism for this: TPE just treats the
        # trial as uninformative and keeps going, rather than needing a fake
        # worst-case loss value threaded through here.
        print(f"  [WARN] Hyperopt trial failed for '{model_name}' with {params}: {exc} - skipping")
        return {"loss": None, "status": STATUS_FAIL, **params}

    # For maximising metrics, return the negative value; for minimising, return as-is
    return {"loss": metric_value if metric in _HPO_LOWER_IS_BETTER else -metric_value, "status": STATUS_OK, **params}

# Update best values for tracking
def update_best_values(tracking_lists, params, metric_value, metric_type):
    best_params, best_metric = tracking_lists

    # Revert the negative metric to its original value for maximisation metrics
    metric_value = -metric_value if metric_type in ["R^2", "ADJUSTED R^2", "Q^2", "KGE"] else metric_value

    if not best_metric:
        best_params.append(params)
        best_metric.append(metric_value)
    else:
        # Update the best value based on the metric type
        if metric_type in ["R^2", "ADJUSTED R^2", "Q^2", "KGE"]:
            if metric_value > max(best_metric):
                best_params.append(params)
                best_metric.append(metric_value)
            else:
                best_params.append(best_params[-1])
                best_metric.append(max(best_metric))
        else: # This is for MSE
            if metric_value < min(best_metric):
                best_params.append(params)
                best_metric.append(metric_value)
            else:
                best_params.append(best_params[-1])
                best_metric.append(min(best_metric))

# Define Hyperopt-compatible parameter spaces
    # Translate our neutral param_space schema -> Hyperopt search space:
    # - loguniform/uniform -> hp.loguniform/hp.uniform
    # - int -> hp.quniform (step=1)
    # - choice -> hp.choice
    # - hidden_layer_sizes -> quniform then cast to (int,)
def convert_to_hyperopt_space(param_space):
    hyperopt_space = {}
    for param, config in param_space.items():
        if config["type"] == "loguniform":
            hyperopt_space[param] = hp.loguniform(param, np.log(config["bounds"][0]), np.log(config["bounds"][1]))
        elif config["type"] == "uniform":
            hyperopt_space[param] = hp.uniform(param, config["bounds"][0], config["bounds"][1])
        elif config["type"] == "int":
            hyperopt_space[param] = hp.quniform(param, config["bounds"][0], config["bounds"][1], 1)
        elif config["type"] == "choice":
            hyperopt_space[param] = hp.choice(param, config["options"])
        elif param == "hidden_layer_sizes":
            hyperopt_space[param] = hp.quniform(param, config["bounds"][0], config["bounds"][1], 1)
    return hyperopt_space


# Drive TPE, collect Trials, and build a running "best so far" trace:
# 'tracking_lists' = (list_of_best_params_over_time, list_of_best_metric_over_time)
# This is used to plot convergence later.
def run_hyperopt_optimisation(model_name, model, param_space, evals, X_train, X_test, y_train, y_test, target_var, metric, sample_size,
                              patience=None, min_delta=1e-4, random_state=None):
    if not _HYPEROPT_AVAILABLE:
        raise ImportError(
            "hyperopt could not be imported (likely due to a missing pkg_resources). "
            "Fix: pip install \"setuptools<80\""
        )
    # evals=0 used to crash inside hyperopt with "There are no evaluation
    # tasks, cannot return argmin of task losses".
    require_int_at_least("evals", evals, minimum=1)
    start_time = time.time()
    trials = Trials()
    tracking_lists = tracking()  # Tracking best parameters and metric

    # Convert param_space to Hyperopt-compatible format
    hyperopt_space = convert_to_hyperopt_space(param_space)

    # Build early-stopping function for fmin if patience is set.
    # Hyperopt always minimises loss, so improvement = loss decreased by >= min_delta.
    early_stop_fn = None
    if patience is not None:
        es_state = {"best": None, "count": 0}
        def early_stop_fn(trials_obj, *args):
            losses = [l for l in trials_obj.losses() if l is not None]
            if not losses:
                return False, {}
            current_best = min(losses)
            if es_state["best"] is None or current_best < es_state["best"] - min_delta:
                es_state["best"] = current_best
                es_state["count"] = 0
                # An improving evaluation can never trigger the stop (matters
                # for patience=0, which stops at the first NON-improving one).
                return False, {}
            es_state["count"] += 1
            return (es_state["count"] >= patience), {}

    # Run optimisation. If every trial fails (STATUS_FAIL), fmin's own internal
    # trials.argmin raises AllTrialsFailed even though each individual failure was
    # already handled gracefully inside hyperopt_objective — treat that the same
    # way as "no usable results" rather than letting it propagate as a crash.
    try:
        best_params = fmin(
            fn=lambda params: hyperopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size,
                                                 random_state=random_state),
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials,
            early_stop_fn=early_stop_fn,
            rstate=np.random.default_rng(0 if random_state is None else random_state),
        )
    except AllTrialsFailed:
        print(f"  [WARN] Every Hyperopt trial failed for '{model_name}' - no usable result")
        best_params = {}

    # Convert integer parameters back from floats if needed
    for key, value in best_params.items():
        if param_space[key]["type"] == "int":
            best_params[key] = int(value)

    # Track the best values during optimisation — skip STATUS_FAIL trials (loss is
    # None for those, which would blow up update_best_values' numeric comparisons).
    for trial in trials.trials:
        if trial["result"].get("status") != STATUS_OK:
            continue
        trial_result = {key: val for key, val in trial["result"].items() if key not in ["loss", "status"]}
        update_best_values(tracking_lists, trial_result, trial["result"]["loss"], metric)

    actual_iters = len(trials.trials)
    stopped_early = actual_iters < evals
    if stopped_early:
        print(f"  Early stopping at {actual_iters}/{evals} evaluations (no improvement > {min_delta} for {patience} steps)")

    elapsed_time = time.time() - start_time
    print(f"Hyperopt completed in: {format_elapsed_time(elapsed_time)}")

    es_info = {
        "stopped_early": stopped_early,
        "actual_iters": actual_iters,
        "max_iters": evals,
        "patience": patience,
        "min_delta": min_delta,
    }
    return best_params, tracking_lists, elapsed_time, es_info

# Plotting for Hyperopt
def plot_hyperopt_results(results_by_target, param_space, model_name, metric, es_info_by_target=None):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)
    es_info_by_target = es_info_by_target or {}

    # Create subplots for each target variable
    fig, axes = plt.subplots(
        num_targets,
        num_params + 1,
        figsize=(5 * (num_params + 1), 4 * num_targets),
        squeeze=False,  # Always returns a 2D array
        sharey=False
    )
    _mlabel = _metric_label(metric_column)
    fig.suptitle(f"Hyperopt Results for {model_name} - {_mlabel}", fontsize=16)

    # Plot results for each target variable
    for row_idx, (target_var, tracking_lists) in enumerate(results_by_target.items()):
        best_params, best_metric = tracking_lists
        if not best_params:
            continue
        param_names = list(best_params[0].keys())  # Hyperparameter names
        es = es_info_by_target.get(target_var, {})

        # Plot hyperparameter evolution
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            param_values = [params[param_name] for params in best_params]
            ax.plot(range(1, len(param_values) + 1), param_values, marker=".")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel(f"{param_name}")
            ax.set_title(f"{target_var}: {param_name}")
            ax.grid(True)

        # Plot metric evolution; annotate with ES info if triggered
        ax = axes[row_idx, -1]
        ax.plot(range(1, len(best_metric) + 1), best_metric, color="green", marker=".")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel(_mlabel)
        iter_label = f" [{es['actual_iters']}/{es['max_iters']} iters, ES]" if es.get("stopped_early") else ""
        ax.set_title(f"{target_var}: {_mlabel}{iter_label}")
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

# Same casting/constraints as Hyperopt but arg signature is a flat list of values.
# Return raw metric (positive if maximising, MSE if minimising).
def skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size,
                    random_state=None):
    param_dict = {param_name: param_value for param_name, param_value in zip(param_spaces[model_name].keys(), params)}
    param_dict = _cast_params(param_dict, model_name)

    # Enforce constraints
    if model_name == "Random Forest Regressor":
        if "min_samples_split" in param_dict and "min_samples_leaf" in param_dict:
            if param_dict["min_samples_split"] <= param_dict["min_samples_leaf"]:
                param_dict["min_samples_split"] = param_dict["min_samples_leaf"] + 1

    # Subsample the dataset if needed
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size,
                                                   random_state=0 if random_state is None else random_state)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    model.set_params(**param_dict)
    try:
        model.fit(X_train_sample, y_train_sample[target_var])
        y_pred = model.predict(X_test)
        n, p = X_test.shape
        return _compute_metric(y_test[target_var], y_pred, metric, n, p)
    except Exception as exc:
        # E.g. a sampled n_neighbors exceeding the (sub)training set size for KNN.
        # skopt has no "failed trial" concept like hyperopt's STATUS_FAIL — gp_minimize
        # always needs a real float back. A large sentinel is bad in both directions
        # the caller uses it in: as-is (minimising metrics) or negated (maximising ones).
        print(f"  [WARN] Skopt trial failed for '{model_name}' with {param_dict}: {exc} - skipping")
        return _TRIAL_FAILURE_SENTINEL

 # Build skopt space (Real/Integer/Categorical).
    # Use callbacks to:
    #   - update tqdm progress
    #   - append current best (param_dict, metric_value) to tracking lists
    #   - trigger early stopping (returning True from a callback stops gp_minimize)
    # After optimisation, extract best params and best metric value.
def run_skopt_optimisation(model_name, model, param_space, calls, X_train, X_test, y_train, y_test, target_var, metric, sample_size, n_jobs,
                           patience=None, min_delta=1e-4, random_state=None):
    if not _SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize could not be imported.")
    start_time = time.time()
    tracking_lists = tracking()

    # Define Scikit-Optimize's search space
    scikit_opt_space = [
        Real(bounds["bounds"][0], bounds["bounds"][1], "log-uniform" if bounds["type"] == "loguniform" else "uniform", name=param)
        if bounds["type"] in ["loguniform", "uniform"] else
        Integer(bounds["bounds"][0], bounds["bounds"][1], name=param)
        if bounds["type"] == "int" else
        Categorical(bounds["options"], name=param)
        for param, bounds in param_space.items()
    ]

    # Early stopping state; gp_minimize stops when a callback returns True
    es_state = {"best": None, "count": 0, "stopped_at": None}

    with tqdm(total=calls, desc=f"Scikit-Optimize Progress for {model_name} (Target: {target_var})", unit="call") as pbar:
        def callback(res):
            pbar.update()
            metric_value = -res.fun if metric in ["R^2", "ADJUSTED R^2", "Q^2", "KGE"] else res.fun
            param_dict = {param_name: param_value for param_name, param_value in zip(param_space.keys(), res.x)}
            tracking_lists[0].append(param_dict)
            tracking_lists[1].append(metric_value)

            if patience is not None:
                # res.fun is the best loss seen so far (skopt always minimises)
                current_best = res.fun
                if es_state["best"] is None or current_best < es_state["best"] - min_delta:
                    es_state["best"] = current_best
                    es_state["count"] = 0
                else:
                    # Only non-improving calls count towards the stop (matters
                    # for patience=0: first NON-improving call, not first call).
                    es_state["count"] += 1
                    if es_state["count"] >= patience:
                        es_state["stopped_at"] = len(res.func_vals)
                        return True  # signal gp_minimize to stop
            return False

        results = gp_minimize(
            lambda params: -skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size, random_state=random_state)
            if metric in ["R^2", "ADJUSTED R^2", "Q^2", "KGE"] else
            skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size, random_state=random_state),
            scikit_opt_space,
            n_calls=calls,
            callback=[callback],
            random_state=0 if random_state is None else random_state,
        )

    actual_iters = es_state["stopped_at"] if es_state["stopped_at"] is not None else len(results.func_vals)
    stopped_early = actual_iters < calls
    if stopped_early:
        print(f"  Early stopping at {actual_iters}/{calls} calls (no improvement > {min_delta} for {patience} steps)")

    best_params = {param_name: param_value for param_name, param_value in zip(param_space.keys(), results.x)}
    best_metric_value = -results.fun if metric in ["R^2", "ADJUSTED R^2", "Q^2", "KGE"] else results.fun

    elapsed_time = time.time() - start_time
    print(f"Scikit-Optimize completed in: {format_elapsed_time(elapsed_time)}")

    es_info = {
        "stopped_early": stopped_early,
        "actual_iters": actual_iters,
        "max_iters": calls,
        "patience": patience,
        "min_delta": min_delta,
    }
    return best_params, tracking_lists, best_metric_value, elapsed_time, es_info


def plot_skopt_results(results_by_target, param_space, model_name, metric, es_info_by_target=None):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)
    es_info_by_target = es_info_by_target or {}

    # Create subplots: one row per target variable, columns for hyperparameters and the metric
    fig, axes = plt.subplots(
        num_targets,
        num_params + 1,
        figsize=(5 * (num_params + 1), 4 * num_targets),
        squeeze=False,  # Always create a 2D array
        sharey=False
    )
    _mlabel = _metric_label(metric_column)
    fig.suptitle(f"Scikit-Optimize Results for {model_name} - {_mlabel}", fontsize=16)

    # Plot results for each target variable
    for row_idx, (target_var, tracking_lists) in enumerate(results_by_target.items()):
        best_params, best_metric = tracking_lists
        if not best_params:
            continue
        param_names = list(best_params[0].keys())  # Hyperparameter names
        es = es_info_by_target.get(target_var, {})

        # Plot hyperparameter evolution
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            param_values = [params[param_name] for params in best_params]
            ax.plot(range(1, len(param_values) + 1), param_values, marker=".")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel(f"{param_name}")
            ax.set_title(f"{target_var}: {param_name}")
            ax.grid(True)

        # Plot metric evolution; annotate with ES info if triggered
        ax = axes[row_idx, -1]
        ax.plot(range(1, len(best_metric) + 1), best_metric, color="green", marker=".")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel(_mlabel)
        iter_label = f" [{es['actual_iters']}/{es['max_iters']} iters, ES]" if es.get("stopped_early") else ""
        ax.set_title(f"{target_var}: {_mlabel}{iter_label}")
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

def run_hyperparameter_optimisation_workflow(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    target_columns: list,
    methods_to_run: list = ["random", "hyperopt", "skopt"],
    metric: str = "Q^2",
    sampling_method: str = "Sobol",
    sample_size: int = 10000,
    n_iter: int = 100,
    evals: int = 100,
    calls: int = 100,
    n_jobs: int = -1,
    plot: bool = True,
    output_dir: str = "images",
    early_stopping: dict = None,
    random_state=None,
    feature_names=None,
    monotonic_constraints=None,
):
    # output_dir is only ever used to save plot PNGs below (every reference to
    # it lives inside an `if plot:` block) -- creating it unconditionally used
    # to leave a stray, always-empty "images/" directory behind for any
    # caller (test or script) that disabled plotting but didn't override the
    # default relative path.
    if plot:
        os.makedirs(output_dir, exist_ok=True)
    param_space = param_spaces[model_name]
    results_summary = {}

    def _apply_target_constraints(target_var):
        # HPO mutates `model` in place via .set_params() across every trial and never
        # touches monotone_constraints again (it's not part of param_space) — so the
        # per-target constraint must be (re-)applied once, here, before that target's
        # search begins, since the same shared `model` instance is reused across targets.
        apply_monotone_constraints_for_target(
            model, model_name, feature_names, (monotonic_constraints or {}).get(target_var, {})
        )

    # Extract per-method early stopping settings
    early_stopping = early_stopping or {}
    es_rs = early_stopping.get("random_search", {})
    es_ho = early_stopping.get("hyperopt", {})
    es_sk = early_stopping.get("skopt", {})

    # Each backend is contained individually: one failing (e.g. skopt's
    # n_calls >= 10 requirement, a hyperopt import failure) must not discard
    # the completed results of the backends that already succeeded for this
    # model. A failed method is simply absent from results_summary, which the
    # caller already handles per-method.
    if "random" in methods_to_run:
        try:
            print(f"\nRunning Random Search for {model_name}...")
            rs_results_by_target = {}
            for target_var in target_columns:
                _apply_target_constraints(target_var)
                rs_df, best_params, best_score, elapsed_time, es_info = run_random_search(
                    model, param_space, X_train, X_test, y_train, y_test,
                    sample_size, n_iter, n_jobs, target_var, metric, sampling_method,
                    patience=es_rs.get("patience"),
                    min_delta=es_rs.get("min_delta", 1e-4),
                    model_name=model_name,
                    random_state=random_state,
                )
                rs_results_by_target[target_var] = {
                    "tracking": rs_df,
                    "elapsed_time": elapsed_time,
                    "es_info": es_info,
                }
                print(f"Best for {target_var}: {best_params} | {metric} = {best_score:.4f}")
            if plot:
                es_info_by_target = {tv: rs_results_by_target[tv]["es_info"] for tv in target_columns}
                fig = plot_random_search_results(
                    {k: v["tracking"] for k, v in rs_results_by_target.items()},
                    param_space, model_name, metric, sampling_method,
                    es_info_by_target=es_info_by_target,
                )
                plot_path = os.path.join(output_dir, f"HPO_{model_name}_Random.png")
                fig.savefig(plot_path, bbox_inches="tight")
                rs_results_by_target["plot"] = plot_path
            results_summary["random"] = rs_results_by_target
        except Exception as exc:
            print(f"[WARN] Random Search failed for '{model_name}': {exc} - "
                  f"results from the other HPO methods are kept")

    if "hyperopt" in methods_to_run:
        try:
            print(f"\nRunning Hyperopt for {model_name}...")
            hyperopt_results_by_target = {}
            for target_var in target_columns:
                _apply_target_constraints(target_var)
                best_params, tracking_lists, elapsed_time, es_info = run_hyperopt_optimisation(
                    model_name, model, param_space, evals, X_train, X_test,
                    y_train, y_test, target_var, metric, sample_size,
                    patience=es_ho.get("patience"),
                    min_delta=es_ho.get("min_delta", 1e-4),
                    random_state=random_state,
                )
                hyperopt_results_by_target[target_var] = {
                    "tracking": tracking_lists,
                    "elapsed_time": elapsed_time,
                    "es_info": es_info,
                }
                print(f"Best for {target_var}: {best_params}")
            if plot:
                es_info_by_target = {tv: hyperopt_results_by_target[tv]["es_info"] for tv in target_columns}
                fig = plot_hyperopt_results(
                    {k: v["tracking"] for k, v in hyperopt_results_by_target.items()},
                    param_space, model_name, metric,
                    es_info_by_target=es_info_by_target,
                )
                plot_path = os.path.join(output_dir, f"HPO_{model_name}_Hyperopt.png")
                fig.savefig(plot_path, bbox_inches="tight")
                hyperopt_results_by_target["plot"] = plot_path
            results_summary["hyperopt"] = hyperopt_results_by_target
        except Exception as exc:
            print(f"[WARN] Hyperopt failed for '{model_name}': {exc} - "
                  f"results from the other HPO methods are kept")

    if "skopt" in methods_to_run:
        try:
            print(f"\nRunning Scikit-Optimize for {model_name}...")
            skopt_results_by_target = {}
            for target_var in target_columns:
                _apply_target_constraints(target_var)
                best_params, tracking_lists, best_score, elapsed_time, es_info = run_skopt_optimisation(
                    model_name, model, param_space, calls, X_train, X_test,
                    y_train, y_test, target_var, metric, sample_size, n_jobs,
                    patience=es_sk.get("patience"),
                    min_delta=es_sk.get("min_delta", 1e-4),
                    random_state=random_state,
                )
                skopt_results_by_target[target_var] = {
                    "tracking": tracking_lists,
                    "elapsed_time": elapsed_time,
                    "es_info": es_info,
                }
                print(f"Best for {target_var}: {best_params} | {metric} = {best_score:.4f}")
            if plot:
                es_info_by_target = {tv: skopt_results_by_target[tv]["es_info"] for tv in target_columns}
                fig = plot_skopt_results(
                    {k: v["tracking"] for k, v in skopt_results_by_target.items()},
                    param_space, model_name, metric,
                    es_info_by_target=es_info_by_target,
                )
                plot_path = os.path.join(output_dir, f"HPO_{model_name}_Skopt.png")
                fig.savefig(plot_path, bbox_inches="tight")
                skopt_results_by_target["plot"] = plot_path
            results_summary["skopt"] = skopt_results_by_target
        except Exception as exc:
            print(f"[WARN] Scikit-Optimize failed for '{model_name}': {exc} - "
                  f"results from the other HPO methods are kept")

    return results_summary

# From this point onwards I barely have any clue on what's going on. This section caused me the most grief, and is a lot of mismash-ing together to make sure it all works.
# It's a miracle that this code even works, so if you end up here- good luck! 

# Fan-out workflow across multiple models.
    # Aggregates into:
    #   - all_metrics[method][model][target] = {metric_name: best, "elapsed_time": s}
    #   - all_params[method][model][target]  = {param: value, ...}
    #   - all_times[method][model][target]   = seconds (duplicate of elapsed_time)
    #   - all_plots[method][model]           = path (if emitted)
    # Guard: if no trials ran, store {} to avoid .items() errors downstream.
def run_all_models_optimisation(
    models_dict: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    target_columns: list,
    selected_model_names: list = None,
    methods_to_run: list = ["random", "hyperopt", "skopt"],
    metric: str = "Q^2",
    sampling_method: str = "Sobol",
    sample_size: int = 10000,
    n_iter: int = 1000,
    evals: int = 100,
    calls: int = 100,
    n_jobs: int = -1,
    plot: bool = True,
    output_dir: str = "images",
    early_stopping: dict = None,
    checkpoint_fn=None,
    random_state=None,
    feature_names=None,
    monotonic_constraints=None,
):
    # If user specified a subset of models, restrict to those
    if selected_model_names:
        models_dict = {name: models_dict[name] for name in selected_model_names}

    # Initialise storage dictionaries for each optimisation method
    all_metrics = {method: {} for method in methods_to_run}
    all_params = {method: {} for method in methods_to_run}
    all_times = {method: {} for method in methods_to_run}
    all_plots = {method: {} for method in methods_to_run}

    # Loop over each model in the supplied dictionary
    for model_name, model in models_dict.items():
        # Optional cooperative pause/cancel hook (e.g. from the UI): checked once per
        # model, since a single model's optimisation run isn't itself interruptible.
        if checkpoint_fn is not None:
            checkpoint_fn()
        print(f"\n=== Optimising {model_name} ===")

        # One model failing (e.g. a hyperopt import failure, documented above as
        # happening on modern setuptools) must not abort the whole multi-model sweep
        # and lose every already-computed result for the other models.
        try:
            results = run_hyperparameter_optimisation_workflow(
                model_name=model_name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                target_columns=target_columns,
                methods_to_run=methods_to_run,
                metric=metric,
                sampling_method=sampling_method,
                sample_size=sample_size,
                n_iter=n_iter,
                evals=evals,
                calls=calls,
                n_jobs=n_jobs,
                plot=plot,
                output_dir=output_dir,
                early_stopping=early_stopping,
                random_state=random_state,
                feature_names=feature_names,
                monotonic_constraints=monotonic_constraints,
            )
        except Exception as exc:
            print(f"[WARN] Optimisation failed for '{model_name}': {exc} - skipping this model")
            for method in methods_to_run:
                all_metrics[method][model_name] = {}
                all_params[method][model_name] = {}
                all_times[method][model_name] = {}
            continue

        # Process and store the best results for each HPO method
        for method in methods_to_run:
            if method in results:
                all_metrics[method][model_name] = {}
                all_params[method][model_name] = {}
                all_times[method][model_name] = {}

                # Grab plot if present
                if "plot" in results[method]:
                    all_plots[method][model_name] = results[method]["plot"]

                for target_var, tracking_data in results[method].items():
                    if target_var == "plot":
                        continue 

                    elapsed_time = tracking_data["elapsed_time"]
                    
                    # Extract best-performing hyperparameters and score
                    _lower_is_best = metric in _HPO_LOWER_IS_BETTER or metric == "MAE"
                    if method == "random":
                        tracking_df = tracking_data["tracking"]
                        if isinstance(tracking_df, pd.DataFrame) and not tracking_df.empty:
                            best_row = tracking_df.sort_values(metric.upper(), ascending=_lower_is_best).iloc[0]
                            best_score = best_row[metric.upper()]
                            best_params = best_row.drop(labels=[metric.upper()]).to_dict()
                        else:
                            best_score, best_params = None, None

                    elif method in ["hyperopt", "skopt"]:
                        best_params_list, best_scores_list = tracking_data["tracking"]
                        if best_scores_list:
                            best_idx = np.argmin(best_scores_list) if _lower_is_best else np.argmax(best_scores_list)
                            best_score = best_scores_list[best_idx]
                            best_params = best_params_list[best_idx]
                        else:
                            best_score, best_params = None, None

                    else:
                        best_score, best_params = None, None

                    es_info = tracking_data.get("es_info", {})
                    all_metrics[method][model_name][target_var] = {
                        metric: best_score,
                        "elapsed_time": elapsed_time,
                        "stopped_early": es_info.get("stopped_early", False),
                        "actual_iters": es_info.get("actual_iters"),
                        "max_iters": es_info.get("max_iters"),
                        "patience": es_info.get("patience"),
                        "min_delta": es_info.get("min_delta"),
                    }

                    optimised_keys = list(param_spaces[model_name].keys())
                    if best_params is None:
                        filtered_params = {}
                    else:
                        filtered_params = {k: v for k, v in best_params.items() if k in optimised_keys}
                    all_params[method][model_name][target_var] = filtered_params
                    all_times[method][model_name][target_var] = elapsed_time

    # Return all results grouped by optimisation method
    return all_metrics, all_params, all_times, all_plots 

# Flatten nested dict results into a DataFrame for easy CSV export or reporting.
# Each row = (Model, Target, HPO Method, Hyperparameters, Metric, Runtime)
# Skips rows with no data to keep the table compact.
def collect_results_as_dataframe(
    models_dict,
    target_columns,
    default_metrics, default_params,
    random_metrics, random_params, random_sampling_method,
    hyperopt_metrics, hyperopt_params,
    skopt_metrics, skopt_params,
    metric_name
):
    methods = [
        "Default",
        f"Random Search ({random_sampling_method})",
        "Hyperopt (TPE)",
        "Scikit-Optimize (GP_minimize)"
    ]
    metrics_list = [default_metrics, random_metrics, hyperopt_metrics, skopt_metrics]
    params_list = [default_params, random_params, hyperopt_params, skopt_params]

    results_list = []

    for model_name in models_dict.keys():
        for target_var in target_columns:
            for method, metrics, params in zip(methods, metrics_list, params_list):
                metric_value = metrics.get(model_name, {}).get(target_var, {}).get(metric_name)
                param_values = params.get(model_name, {}).get(target_var)
                elapsed_time = metrics.get(model_name, {}).get(target_var, {}).get("elapsed_time")

                if metric_value is None and param_values is None:
                    continue

                results_list.append({
                    "Model": model_name,
                    "Target Variable": target_var,
                    "Hyperparameter Optimisation Method": method,
                    "Value of Hyperparameters": str(param_values),
                    metric_name: metric_value,
                    "Elapsed Time (s)": elapsed_time
                })

    return pd.DataFrame(results_list)

# Normalise or coerce hyperparameter types before fitting models:
def process_hyperparameters(hyperparams, model_name):
    processed_hyperparams = hyperparams.copy()
    param_space = param_spaces.get(model_name, {})

    for key, value in hyperparams.items():
        if value is None:
            processed_hyperparams[key] = None

        elif isinstance(value, tuple):
            # Preserve tuples as-is (likely intended for bounds, shapes, etc.)
            processed_hyperparams[key] = value

        elif isinstance(value, (float, int)) and key in [
            "n_estimators", "max_depth", "num_iterations", "num_leaves",
            "n_neighbors", "min_samples_split", "min_samples_leaf"
        ]:
            processed_hyperparams[key] = int(value)

        elif key in param_space:
            param_type = param_space[key]["type"]
            if param_type == "int":
                try:
                    processed_hyperparams[key] = int(float(value))
                except (ValueError, TypeError):
                    processed_hyperparams[key] = value
            elif param_type in ["uniform", "loguniform"]:
                try:
                    processed_hyperparams[key] = float(value)
                except (ValueError, TypeError):
                    # String-valued params like 'sqrt', 'log2' — pass through
                    processed_hyperparams[key] = value
            else:
                processed_hyperparams[key] = value

        elif isinstance(value, str):
            processed_hyperparams[key] = value

        else:
            try:
                processed_hyperparams[key] = ast.literal_eval(str(value))
            except (ValueError, SyntaxError):
                processed_hyperparams[key] = value

    return processed_hyperparams

def find_best_model_and_hyperparams(collected_results_df, metric, verbose=True):
    """verbose: print the per-target "Best model" blocks. True only for the HPO
    step's own announcement; internal lookups (interpretability's visual-model
    choice, CV/PERL/report fallbacks when HPO didn't run) pass False so their
    bookkeeping doesn't masquerade as step output in the progress log."""
    if metric not in metrics_dict:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {list(metrics_dict.keys())}")

    is_minimized = metric in _HPO_LOWER_IS_BETTER or metric in ("MAE",)
    best_models = {}

    # Guard: ensure required columns exist
    required_cols = {"Model", "Target Variable", "Hyperparameter Optimisation Method", "Value of Hyperparameters", metric}
    missing = required_cols - set(collected_results_df.columns)
    if missing:
        raise ValueError(f"collected_results_df missing columns: {missing}")

    for target_var in collected_results_df["Target Variable"].unique():
        target_results = collected_results_df[collected_results_df["Target Variable"] == target_var]
        target_results = target_results.dropna(subset=[metric])
        if target_results.empty:
            continue

        idx = target_results[metric].idxmin() if is_minimized else target_results[metric].idxmax()
        best_row = target_results.loc[idx].copy()

        # Normalise keys so downstream code is stable
        best_row = best_row.rename({
            "Model": "model_name",
            "Target Variable": "target_variable",
            "Hyperparameter Optimisation Method": "hpo_method",
            "Value of Hyperparameters": "hyperparameters",
        })

        if verbose:
            print(f"\n=== Best model for {best_row['target_variable']} ===")
            print(f"  Model           : {best_row['model_name']}")
            print(f"  HPO Method      : {best_row['hpo_method']}")
            print(f"  Hyperparameters : {best_row['hyperparameters']}")
            print(f"  {metric:<15} : {best_row[metric]:.4f}")

        best_models[target_var] = best_row

    return best_models

def coerce_numeric_hyperparams(params_dict):
    return {
        k: int(v) if isinstance(v, float) and v.is_integer() else v
        for k, v in params_dict.items()
    }

# For every selected model AND every target, finds that (model, target) pair's own best
# HPO method/hyperparameters — unlike the old get_best_models_by_method_across_targets
# (removed: averaged method choice across targets, then reused only the first target's
# hyperparameters for every target), this never lets one target's tuning leak into
# another's. Returns dict[model_name -> dict[target -> unfit instance, tuned params set]].
def get_all_models_tuned_per_target(
    selected_models, target_columns, metrics, params, metric_name, models_dict
):
    best_model_instances = {}
    is_minimised = metric_name in _HPO_LOWER_IS_BETTER

    for model_name in selected_models:
        per_target = {}
        for target in target_columns:
            best_score = np.inf if is_minimised else -np.inf
            best_hyperparams = None

            for method in ["random", "hyperopt", "skopt"]:
                model_metrics = metrics[method].get(model_name)
                model_params = params[method].get(model_name)
                if not model_metrics or not model_params:
                    continue
                target_metrics = model_metrics.get(target)
                target_params = model_params.get(target)
                if not target_metrics or metric_name not in target_metrics or target_params is None:
                    continue

                score = target_metrics[metric_name]
                is_better = score < best_score if is_minimised else score > best_score
                if is_better:
                    best_score = score
                    best_hyperparams = target_params

            if best_hyperparams is not None:
                raw_params = best_hyperparams
                if isinstance(raw_params, str):
                    raw_params = ast.literal_eval(raw_params)
                raw_params = coerce_numeric_hyperparams(raw_params)
                model = deepcopy(models_dict[model_name])
                per_target[target] = model.set_params(**raw_params)
            else:
                print(f"[WARN] No optimised hyperparameters found for {model_name} / {target}")

        if per_target:
            best_model_instances[model_name] = per_target

    return best_model_instances

