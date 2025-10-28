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
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

from phoenix_ml.models import param_spaces
from phoenix_ml.model_training import metrics_dict

import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress logs
os.environ['LIGHTGBM_VERBOSE'] = '0'
os.environ['XGBOOST_VERBOSITY'] = '0'

# Suppress general Python warnings from skopt and others
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def format_elapsed_time(seconds):
    minutes = int(seconds) // 60
    rem_seconds = seconds - (minutes * 60)
    return f"{minutes}m {rem_seconds:.2f}s"

# Centralised metrics so it's consistent across Random/Hyperopt/Skopt. ADJUSTED R^2 needs (n, p), validated here to avoid silent misuse.
def _compute_metric(y_true, y_pred, metric_name, n=None, p=None):
    if metric_name == "MSE":
        return mean_squared_error(y_true, y_pred)
    elif metric_name == "R^2":
        return r2_score(y_true, y_pred)
    elif metric_name == "ADJUSTED R^2":
        if n is None or p is None:
            raise ValueError("Adjusted R^2 requires n and p.")
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    elif metric_name == "Q^2":
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

def run_random_search(
    model, param_space, X_train, X_test, y_train, y_test, sample_size, n_iter, n_jobs, target_var, metric, sampling_method
):
    # 1) Build a low-discrepancy sampler (Sobol/Halton/LHS) or PRNG ("Random").
    # 2) Map samples in [0,1]^d onto each hyperparameter's domain:
    #    - "uniform"/"loguniform": linear/log scaling
    #    - "int": round to nearest int
    #    - "choice": index into options
    #    - special-case "hidden_layer_sizes": produce (units,) tuple for MLP
    # 3) (Optional) Subsample training set for speed on large data.
    # 4) Fit/evaluate each config in parallel (joblib.Parallel).
    # 5) Return a DataFrame of all trials + the best row and runtime.
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
                param_samples[param] = vals
            elif param == "hidden_layer_sizes":
                low, high = bounds["bounds"]
                layer_size = np.round(low + (high - low) * sampler[:, i]).astype(int)
                # one tuple per iteration
                param_samples[param] = [(int(ls),) for ls in layer_size]
        return param_samples
    
    # Generate parameter samples
    # Map a unit hypercube sample matrix 'sampler' -> dict[param] -> list of typed values.
    # Keep lengths aligned so we can zip into a list of param dicts.
    d = len(param_space)
    if sampling_method == "Random":
        sampler = np.random.RandomState(0).rand(n_iter, d)
    elif sampling_method == "Sobol":
        # Use next power of two, then slice to n_iter
        m = int(np.ceil(np.log2(max(1, n_iter))))
        sob = Sobol(d=d, scramble=True, seed=0)
        sampler = sob.random_base2(m)[:n_iter]
    elif sampling_method == "Halton":
        sampler = Halton(d=d, scramble=True, seed=0).random(n=n_iter)
    elif sampling_method == "Latin Hypercube":
        sampler = LatinHypercube(d=d, seed=0).random(n=n_iter)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    param_samples = generate_param_samples(param_space, sampler)
    param_combinations = [dict(zip(param_space.keys(), values)) for values in zip(*param_samples.values())]

    # Subsample training data if needed
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=0)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Function to evaluate a single configuration
    def evaluate(params):
        model.set_params(**params)
        model.fit(X_train_sample, y_train_sample[target_var])
        y_pred = model.predict(X_test)
        n, p = X_test.shape
        metric_value = _compute_metric(y_test[target_var], y_pred, metric, n, p)
        return params, metric_value  # ← add this

    # Parallelise evaluations
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(params) for params in param_combinations
    )

    # Compile results into a DataFrame
    metric_name = metric.upper()
    results_df = pd.DataFrame([{**params, metric_name: value} for params, value in results])

    # Determine whether to maximise or minimise the metric
    if metric in ["MSE"]:
        best_row = results_df.loc[results_df[metric_name].idxmin()]
    elif metric in ["R^2", "ADJUSTED R^2", "Q^2"]:
        best_row = results_df.loc[results_df[metric_name].idxmax()]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Extract the best parameters and metric value
    best_params = best_row.drop([metric_name]).to_dict()
    best_metric_value = best_row[metric_name]

    elapsed_time = time.time() - start_time
    print(f"Random Search completed in: {format_elapsed_time(elapsed_time)}")

    return results_df, best_params, best_metric_value, elapsed_time


def plot_random_search_results(results_by_target, param_space, model_name, metric, sampling_method):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Create subplots for each target variable
    fig, axes = plt.subplots(num_targets, num_params, figsize=(5 * num_params, 4 * num_targets), sharey=True)
    fig.suptitle(f"Random Search ({sampling_method}) Results for {model_name} - {metric_column} ", fontsize=16)

    if num_targets == 1:
        axes = [axes]  # Make sure axes is iterable if there's only one target
    if num_params == 1:
        axes = [[ax] for ax in axes]  # Adjust for single parameter

    # Plot results for each target variable
    for row_idx, (target_var, results_df) in enumerate(results_by_target.items()):
        for col_idx, (param, config) in enumerate(param_space.items()):
            ax = axes[row_idx][col_idx]
            if config["type"] in ["int", "loguniform", "uniform"]:
                data = results_df[param]
            else:
                data = results_df[param].apply(lambda x: x[0] if isinstance(x, tuple) else x)

            ax.scatter(data, results_df[metric_column], alpha=0.7, c="blue")
            ax.set_xlabel(param)
            ax.set_ylabel(metric_column if col_idx == 0 else "")
            ax.set_title(f"{target_var}: {param} vs {metric_column}")
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
def hyperopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size):
    # Convert float parameters to integers or tuples where necessary
    for key, value in params.items():
        if param_spaces[model_name][key]["type"] == "int":
            params[key] = int(value)
        elif key == "hidden_layer_sizes":  # Special handling for tuple parameters
            params[key] = (int(value),)

    # Ensure constraints (e.g., min_samples_split > min_samples_leaf)
    if model_name == "Random Forest Regressor":
        if "min_samples_split" in params and "min_samples_leaf" in params:
            if params["min_samples_split"] <= params["min_samples_leaf"]:
                params["min_samples_split"] = params["min_samples_leaf"] + 1

    # Perform dataset subsampling
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=0)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Set model parameters and fit
    model.set_params(**params)
    model.fit(X_train_sample, y_train_sample[target_var])
    y_pred = model.predict(X_test)

    # Evaluate the chosen metric
    n, p = X_test.shape
    metric_value = (
        mean_squared_error(y_test[target_var], y_pred) if metric == "MSE" else
        r2_score(y_test[target_var], y_pred) if metric == "R^2" else
        1 - (1 - r2_score(y_test[target_var], y_pred)) * (n - 1) / (n - p - 1) if metric == "ADJUSTED R^2" else
        1 - np.sum((y_test[target_var] - y_pred) ** 2) / np.sum((y_test[target_var] - np.mean(y_test[target_var])) ** 2) if metric == "Q^2" else
        None
    )
    if metric_value is None:
        raise ValueError(f"Unsupported metric: {metric}")

    # For maximising metrics, return the negative value; for minimising, return as-is
    return {"loss": metric_value if metric == "MSE" else -metric_value, "status": STATUS_OK, **params}

# Update best values for tracking
def update_best_values(tracking_lists, params, metric_value, metric_type):
    best_params, best_metric = tracking_lists

    # Revert the negative metric to its original value for maximisation metrics
    metric_value = -metric_value if metric_type in ["R^2", "ADJUSTED R^2", "Q^2"] else metric_value

    if not best_metric:
        best_params.append(params)
        best_metric.append(metric_value)
    else:
        # Update the best value based on the metric type
        if metric_type in ["R^2", "ADJUSTED R^2", "Q^2"]:
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
def run_hyperopt_optimisation(model_name, model, param_space, evals, X_train, X_test, y_train, y_test, target_var, metric, sample_size):
    start_time = time.time()
    trials = Trials()
    tracking_lists = tracking()  # Tracking best parameters and metric

    # Convert param_space to Hyperopt-compatible format
    hyperopt_space = convert_to_hyperopt_space(param_space)

    # Run optimisation
    best_params = fmin(
        fn=lambda params: hyperopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size),
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)

    # Convert integer parameters back from floats if needed
    for key, value in best_params.items():
        if param_space[key]["type"] == "int":
            best_params[key] = int(value)

    # Track the best values during optimisation
    for trial in trials.trials:
        trial_result = {key: val for key, val in trial["result"].items() if key not in ["loss", "status"]}
        update_best_values(tracking_lists, trial_result, trial["result"]["loss"], metric)

    elapsed_time = time.time() - start_time
    print(f"Hyperopt completed in: {format_elapsed_time(elapsed_time)}")
 
    return best_params, tracking_lists, elapsed_time

# Plotting for Hyperopt
def plot_hyperopt_results(results_by_target, param_space, model_name, metric):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Create subplots for each target variable
    fig, axes = plt.subplots(
        num_targets,
        num_params + 1,
        figsize=(5 * (num_params + 1), 4 * num_targets),
        squeeze=False,  # Always returns a 2D array
        sharey=False
    )
    fig.suptitle(f"Hyperopt Results for {model_name} - {metric_column}", fontsize=16)

    # Plot results for each target variable
    for row_idx, (target_var, tracking_lists) in enumerate(results_by_target.items()):
        best_params, best_metric = tracking_lists
        param_names = list(best_params[0].keys())  # Hyperparameter names

        # Plot hyperparameter evolution
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            param_values = [params[param_name] for params in best_params]
            ax.plot(range(1, len(param_values) + 1), param_values, marker=".")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel(f"{param_name}")
            ax.set_title(f"{target_var}: {param_name}")
            ax.grid(True)

        # Plot metric evolution
        ax = axes[row_idx, -1]
        ax.plot(range(1, len(best_metric) + 1), best_metric, color="green", marker=".")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel(metric_column)
        ax.set_title(f"{target_var}: {metric_column}")
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

# Same casting/constraints as Hyperopt but arg signature is a flat list of values.
# Return raw metric (positive if maximising, MSE if minimising).
def skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size):
    # Convert params to dictionary
    param_dict = {param_name: param_value for param_name, param_value in zip(param_spaces[model_name].keys(), params)}

    # Handle parameter types
    for key, value in param_dict.items():
        if param_spaces[model_name][key]["type"] == "int":
            param_dict[key] = int(value)
        elif key == "hidden_layer_sizes":
            param_dict[key] = (int(value),)

    # Enforce constraints
    if model_name == "Random Forest Regressor":
        if "min_samples_split" in param_dict and "min_samples_leaf" in param_dict:
            if param_dict["min_samples_split"] <= param_dict["min_samples_leaf"]:
                param_dict["min_samples_split"] = param_dict["min_samples_leaf"] + 1

    # Subsample the dataset if needed
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=0)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Fit the model with the sampled parameters
    model.set_params(**param_dict)
    model.fit(X_train_sample, y_train_sample[target_var])
    y_pred = model.predict(X_test)

    # Calculate metrics
    n, p = X_test.shape
    metric_value = (
        mean_squared_error(y_test[target_var], y_pred) if metric == "MSE" else
        r2_score(y_test[target_var], y_pred) if metric == "R^2" else
        1 - (1 - r2_score(y_test[target_var], y_pred)) * (n - 1) / (n - p - 1) if metric == "ADJUSTED R^2" else
        1 - np.sum((y_test[target_var] - y_pred) ** 2) / np.sum((y_test[target_var] - np.mean(y_test[target_var])) ** 2) if metric == "Q^2" else
        None
    )

    if metric_value is None:
        raise ValueError(f"Unsupported metric: {metric}")

    return metric_value

 # Build skopt space (Real/Integer/Categorical).
    # Use a callback to:
    #   - update tqdm progress
    #   - append current best (param_dict, metric_value) to tracking lists
    # After optimisation, extract best params and best metric value.
    # Note: gp_minimize is stochastic unless you set random_state in newer versions; add it if you need strict reproducibility.
def run_skopt_optimisation(model_name, model, param_space, calls, X_train, X_test, y_train, y_test, target_var, metric, sample_size, n_jobs):
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

    with tqdm(total=calls, desc=f"Scikit-Optimize Progress for {model_name} (Target: {target_var})", unit="call") as pbar:
        def callback(res):
            pbar.update()
            metric_value = -res.fun if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else res.fun
            param_dict = {param_name: param_value for param_name, param_value in zip(param_space.keys(), res.x)}
            tracking_lists[0].append(param_dict)
            tracking_lists[1].append(metric_value)

        results = gp_minimize(
            lambda params: -skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size)
            if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else
            skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size),
            scikit_opt_space,
            n_calls=calls,
            callback=[callback]
        )

    best_params = {param_name: param_value for param_name, param_value in zip(param_space.keys(), results.x)}
    best_metric_value = -results.fun if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else results.fun

    elapsed_time = time.time() - start_time
    print(f"Scikit-Optimize completed in: {format_elapsed_time(elapsed_time)}")

    return best_params, tracking_lists, best_metric_value, elapsed_time


def plot_skopt_results(results_by_target, param_space, model_name, metric):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Create subplots: one row per target variable, columns for hyperparameters and the metric
    fig, axes = plt.subplots(
        num_targets,
        num_params + 1,
        figsize=(5 * (num_params + 1), 4 * num_targets),
        squeeze=False,  # Always create a 2D array
        sharey=False
    )
    fig.suptitle(f"Scikit-Optimize Results for {model_name} - {metric_column}", fontsize=16)

    # Plot results for each target variable
    for row_idx, (target_var, tracking_lists) in enumerate(results_by_target.items()):
        best_params, best_metric = tracking_lists
        param_names = list(best_params[0].keys())  # Hyperparameter names

        # Plot hyperparameter evolution
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            param_values = [params[param_name] for params in best_params]
            ax.plot(range(1, len(param_values) + 1), param_values, marker=".")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel(f"{param_name}")
            ax.set_title(f"{target_var}: {param_name}")
            ax.grid(True)

        # Plot metric evolution
        ax = axes[row_idx, -1]
        ax.plot(range(1, len(best_metric) + 1), best_metric, color="green", marker=".")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel(metric_column)
        ax.set_title(f"{target_var}: {metric_column}")
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
    output_dir: str = "images"
):
    os.makedirs(output_dir, exist_ok=True)
    param_space = param_spaces[model_name]
    results_summary = {}

    if "random" in methods_to_run:
        print(f"\nRunning Random Search for {model_name}...")
        rs_results_by_target = {}
        for target_var in target_columns:
            rs_df, best_params, best_score, elapsed_time = run_random_search(
                model, param_space, X_train, X_test, y_train, y_test,
                sample_size, n_iter, n_jobs, target_var, metric, sampling_method
            )
            rs_results_by_target[target_var] = {
                "tracking": rs_df,
                "elapsed_time": elapsed_time
            }
            print(f"Best for {target_var}: {best_params} | {metric} = {best_score:.4f}")
        if plot:
            fig = plot_random_search_results(
                {k: v["tracking"] for k, v in rs_results_by_target.items()},
                param_space, model_name, metric, sampling_method
            )
            plot_path = os.path.join(output_dir, f"HPO_{model_name}_Random.png")
            fig.savefig(plot_path, bbox_inches="tight")
            rs_results_by_target["plot"] = plot_path
        results_summary["random"] = rs_results_by_target

    if "hyperopt" in methods_to_run:
        print(f"\nRunning Hyperopt for {model_name}...")
        hyperopt_results_by_target = {}
        for target_var in target_columns:
            best_params, tracking_lists, elapsed_time = run_hyperopt_optimisation(
                model_name, model, param_space, evals, X_train, X_test,
                y_train, y_test, target_var, metric, sample_size
            )
            hyperopt_results_by_target[target_var] = {
                "tracking": tracking_lists,
                "elapsed_time": elapsed_time
            }
            print(f"Best for {target_var}: {best_params}")
        if plot:
            fig = plot_hyperopt_results(
                {k: v["tracking"] for k, v in hyperopt_results_by_target.items()},
                param_space, model_name, metric
            )
            plot_path = os.path.join(output_dir, f"HPO_{model_name}_Hyperopt.png")
            fig.savefig(plot_path, bbox_inches="tight")
            hyperopt_results_by_target["plot"] = plot_path
        results_summary["hyperopt"] = hyperopt_results_by_target

    if "skopt" in methods_to_run:
        print(f"\nRunning Scikit-Optimize for {model_name}...")
        skopt_results_by_target = {}
        for target_var in target_columns:
            best_params, tracking_lists, best_score, elapsed_time = run_skopt_optimisation(
                model_name, model, param_space, calls, X_train, X_test,
                y_train, y_test, target_var, metric, sample_size, n_jobs
            )
            skopt_results_by_target[target_var] = {
                "tracking": tracking_lists,
                "elapsed_time": elapsed_time
            }
            print(f"Best for {target_var}: {best_params} | {metric} = {best_score:.4f}")
        if plot:
            fig = plot_skopt_results(
                {k: v["tracking"] for k, v in skopt_results_by_target.items()},
                param_space, model_name, metric
            )
            plot_path = os.path.join(output_dir, f"HPO_{model_name}_Skopt.png")
            fig.savefig(plot_path, bbox_inches="tight")
            skopt_results_by_target["plot"] = plot_path
        results_summary["skopt"] = skopt_results_by_target

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
    output_dir: str = "images"
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
        print(f"\n{'='*30}\nOptimising {model_name}\n{'='*30}")
        
        # Run all selected optimisation methods for this model
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
            output_dir=output_dir
        )

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
                    if method == "random":
                        tracking_df = tracking_data["tracking"]
                        if isinstance(tracking_df, pd.DataFrame) and not tracking_df.empty:
                            if metric == "MSE":
                                best_row = tracking_df.sort_values(metric.upper()).iloc[0]
                            else:
                                best_row = tracking_df.sort_values(metric.upper(), ascending=False).iloc[0]
                            best_score = best_row[metric.upper()]
                            best_params = best_row.drop(labels=[metric.upper()]).to_dict()
                        else:
                            best_score, best_params = None, None

                    elif method in ["hyperopt", "skopt"]:
                        best_params_list, best_scores_list = tracking_data["tracking"]
                        if best_scores_list:
                            best_idx = np.argmin(best_scores_list) if metric == "MSE" else np.argmax(best_scores_list)
                            best_score = best_scores_list[best_idx]
                            best_params = best_params_list[best_idx]
                        else:
                            best_score, best_params = None, None

                    else:
                        best_score, best_params = None, None

                    all_metrics[method][model_name][target_var] = {
                        metric: best_score,
                        "elapsed_time": elapsed_time
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
def compare_and_plot_optimisation_methods(
    model_name, target_var,
    default_metrics, default_params,
    random_metrics, random_params, random_sampling_method,
    hyperopt_metrics, hyperopt_params,
    skopt_metrics, skopt_params,
    param_names,
    metric_name
):
    methods = [
        "Default",
        f"Random Search ({random_sampling_method})",
        "Hyperopt (TPE)",
        "Scikit-Optimize (GP Minimisation)"
    ]
    metrics = [default_metrics, random_metrics, hyperopt_metrics, skopt_metrics]
    params = [default_params, random_params, hyperopt_params, skopt_params]

    metric_values = [
    m.get(metric_name, float('inf')) if isinstance(m, dict) else float('inf')
    for m in metrics
    ]

    param_data = {
        param: [
            p.get(param, None) if isinstance(p, dict) else None 
            for p in params
        ]
        for param in param_names
    }

    x = np.arange(len(methods))
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, param in enumerate(param_names):
        values = [
            np.log10(value) if isinstance(value, (int, float)) and value > 0 else 0
            for value in param_data[param]
        ]
        bar_positions = x + (i - len(param_names) / 2) * bar_width
        bars = ax.bar(bar_positions, values, bar_width,
                      label=f'Log({param})' if any(v > 0 for v in values) else param)
        for bar, orig_value in zip(bars, param_data[param]):
            if orig_value is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{orig_value:.2e}" if isinstance(orig_value, (float, int)) else str(orig_value),
                        ha='center', va='bottom', fontsize=8)

    metric_positions = x + (len(param_names) / 2) * bar_width
    metric_bars = ax.bar(metric_positions, metric_values, bar_width,
                         label=metric_name.upper(), color='lightgreen')

    for bar, value in zip(metric_bars, metric_values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.4f}", ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Optimisation Method')
    ax.set_ylabel('Values (log scale for parameters)')
    ax.set_title(f'{model_name} (Target: {target_var}, Metric: {metric_name.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.show()

    optimal_idx = np.argmin(metric_values) if metric_name.lower() in ["mse", "mae"] else np.argmax(metric_values)
    return params[optimal_idx], metrics[optimal_idx]

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

def export_results_to_csv(results_df, directory_path):
    os.makedirs(directory_path, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H%M%S")
    file_path = os.path.join(directory_path, f"Hyperparameter Optimisation Results {current_time}.csv")
    results_df.to_csv(file_path, index=False, na_rep="N/A", float_format="%.10f")
    print(f"DataFrame exported to {file_path}")

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
                processed_hyperparams[key] = int(value)
            elif param_type in ["uniform", "loguniform"]:
                processed_hyperparams[key] = float(value)
            else:
                processed_hyperparams[key] = value

        elif isinstance(value, str):
            processed_hyperparams[key] = value

        else:
            try:
                processed_hyperparams[key] = ast.literal_eval(str(value))
            except:
                processed_hyperparams[key] = value

    return processed_hyperparams

def find_best_model_and_hyperparams(collected_results_df, metric):
    if metric not in metrics_dict:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {list(metrics_dict.keys())}")

    is_minimized = metric in ["MSE", "MAE"]
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

        print(f"\nBest Results for Target Variable: {best_row['target_variable']}")
        print(f"Model: {best_row['model_name']}")
        print(f"HPO Method: {best_row['hpo_method']}")
        print(f"Hyperparameters: {best_row['hyperparameters']}")
        print(f"{metric}: {best_row[metric]:.4f}")

        best_models[target_var] = best_row

    return best_models

def coerce_numeric_hyperparams(params_dict):
    return {
        k: int(v) if isinstance(v, float) and v.is_integer() else v
        for k, v in params_dict.items()
    }

# Selects the best overall method per model (averaged over targets).
# Reconstructs each model with its optimal hyperparameters and returns a dict of ready-to-use model instances.
def get_best_models_by_method_across_targets(
    selected_models, target_columns, metrics, params, metric_name, models_dict
):
    best_model_instances = {}

    for model_name in selected_models:
        best_score = -np.inf
        best_method = None
        best_hyperparams = None

        for method in ["random", "hyperopt", "skopt"]:
            model_metrics = metrics[method].get(model_name)
            model_params = params[method].get(model_name)

            if not model_metrics or not model_params:
                continue

            scores = [
                model_metrics[target][metric_name]
                for target in target_columns
                if model_metrics.get(target) and metric_name in model_metrics[target]
            ]

            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_method = method
                    best_hyperparams = model_params

        if best_hyperparams:
            first_target = next(iter(best_hyperparams))
            raw_params = best_hyperparams[first_target]
            if isinstance(raw_params, str):
                raw_params = ast.literal_eval(raw_params)
            raw_params = coerce_numeric_hyperparams(raw_params)
            model = deepcopy(models_dict[model_name])
            best_model_instances[model_name] = model.set_params(**raw_params)
        else:
            print(f"[WARNING] No optimised hyperparameters found for {model_name}")

    return best_model_instances

# If this code breaks... cry
