# interpretability.py
# This module generates ICE/PDP and SHAP-based interpretability visualisations for the model chosen (either the preferred or fastest)

# Notes:
# Plotting APIs are simple and consistent for report generation, at times the one provided in the library is used.
# SHAP explainers differ depending on the model family, a "sensible" one is chosen most of the time.
# SHAP can be slow for KernelExplainer, solution: keep background_sample_size small.

import time
import numpy as np
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import pandas as pd

# Get Fastest Model (timed fitting)
def get_fastest_model(models_dict, X_train, y_train, target_var, sample_size=1000):
    fastest_model = None
    min_time = float("inf")
    X_sample = X_train[:sample_size]
    y_sample = y_train.iloc[:sample_size]

    for model_name, model in models_dict.items():
        try:
            start_time = time.time()
            model.fit(X_sample, y_sample[target_var])
            elapsed_time = time.time() - start_time
            if elapsed_time < min_time:
                fastest_model = model_name
                min_time = elapsed_time
        except Exception as e:
            # If a model fails on this subset, just skip it
            print(f"Model '{model_name}' failed during training: {str(e)}")
            continue

    if not fastest_model:
        raise ValueError("No valid models available for ICE, PDP, and SHAP plots.")
    
    print(f"Fastest model selected: {fastest_model} ({min_time:.4f}s)")
    return fastest_model

# Max subplot rows per figure — keeps each figure within one A4 content-height page.
# ICE/PDP and SHAP Dependence use 12"-wide 2-col layouts with 4.5" rows.
# 3 rows × 4.5" = 13.5" figure height → ~542pt display — safely under the 652pt page limit.
_MAX_ROWS_PER_FIG = 3

# ICE and PDP
# Plot ICE + PDP for every feature, split across multiple figures so each fits on one PDF page.
def plot_ice_and_pdp(model, X_train, feature_names, target_var, model_name,
                     subsample=250, grid_resolution=20):
    num_features = len(feature_names)
    num_cols = min(num_features, 2)
    subplot_w, subplot_h = 6.0, 4.5
    features_per_fig = _MAX_ROWS_PER_FIG * num_cols

    chunks = [list(range(i, min(i + features_per_fig, num_features)))
              for i in range(0, num_features, features_per_fig)]
    n_parts = len(chunks)
    figs = []

    for part_idx, chunk in enumerate(chunks):
        chunk_n = len(chunk)
        num_rows = (chunk_n + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(num_cols * subplot_w, num_rows * subplot_h),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for i, feature_idx in enumerate(chunk):
            ax = axes_flat[i]
            PartialDependenceDisplay.from_estimator(
                model, X_train, [feature_idx],
                feature_names=feature_names,
                kind="both", subsample=subsample,
                grid_resolution=grid_resolution,
                percentiles=(0.1, 0.9), ax=ax,
            )
            ax.set_title(f"Feature: {feature_names[feature_idx]}", fontsize=11)
            ax.set_xlabel(feature_names[feature_idx], fontsize=10)
            ax.set_ylabel(target_var, fontsize=10)
            ax.tick_params(labelsize=9)

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        part_label = f" (Part {part_idx + 1} of {n_parts})" if n_parts > 1 else ""
        fig.suptitle(f"ICE and PDP Plots for Target: {target_var} ({model_name}){part_label}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.close(fig)
        figs.append(fig)

    return figs

# SHAP Explainer Selection
def select_shap_explainer(model, X_train, background_sample_size, feature_names=None):
    """
    Pick an appropriate SHAP explainer based on model class.
    Falls back to KernelExplainer with a small background sample if TreeExplainer fails.

    Implementation details:
    • TreeExplainer for tree-based models (fastest/common).
    • GradientExplainer for MLP (requires differentiability).
    • KernelExplainer otherwise (model-agnostic, slower).
    """

    import shap
    import numpy as np
    import pandas as pd

    # Always ensure DataFrame
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)

    try:
        # Preferred: TreeExplainer for tree-based models
        if isinstance(model, (RandomForestRegressor, XGBRegressor, BaggingRegressor, ExtraTreesRegressor)):
            return shap.TreeExplainer(model)
        elif isinstance(model, (MLPRegressor,)):
            return shap.GradientExplainer(model, X_train)
        else:
            background = X_train.sample(n=min(background_sample_size, len(X_train)), replace=False)
            return shap.KernelExplainer(model.predict, background)

    except Exception as e:
        # 🔄 Fallback logic for SHAP/XGBoost parsing bug
        print(f"SHAP TreeExplainer failed ({e}). Falling back to KernelExplainer...")

        try:
            background = X_train.sample(n=min(background_sample_size, len(X_train)), replace=False)
            return shap.KernelExplainer(model.predict, background)
        except Exception as e2:
            raise ValueError(f"SHAP explainer fallback error: {e2}")

# SHAP summary (dot) plot showing feature importance and impact distribution.
def plot_shap_summary(model, X_train, feature_names, background_sample_size, target_var, model_name):
    explainer = select_shap_explainer(model, X_train, background_sample_size)
    shap_values = explainer.shap_values(X_train)

    # SHAP draws its own figure; we wrap it in a controlled fig for the report
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False, plot_type='dot')
    
    # Attach to current figure explicitly
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(f"SHAP Summary for Target: {target_var} ({model_name})", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

# SHAP Dependence — split across multiple figures so each fits on one PDF page.
def plot_shap_dependence(model, X_train, feature_names, background_sample_size, target_var, model_name):
    explainer = select_shap_explainer(model, X_train, background_sample_size, feature_names)
    shap_values = explainer.shap_values(X_train)

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    num_features = len(feature_names)
    num_cols = min(num_features, 2)
    subplot_w, subplot_h = 6.0, 4.5
    features_per_fig = _MAX_ROWS_PER_FIG * num_cols

    norm = plt.Normalize(vmin=X_train.min().min(), vmax=X_train.max().max())
    cmap = plt.cm.coolwarm

    chunks = [list(range(i, min(i + features_per_fig, num_features)))
              for i in range(0, num_features, features_per_fig)]
    n_parts = len(chunks)
    figs = []

    for part_idx, chunk in enumerate(chunks):
        chunk_n = len(chunk)
        num_rows = (chunk_n + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(num_cols * subplot_w, num_rows * subplot_h),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for i, feature_idx in enumerate(chunk):
            ax = axes_flat[i]
            shap.dependence_plot(
                feature_idx, shap_values, X_train,
                feature_names=feature_names, ax=ax, show=False,
                color=cmap(norm(X_train.iloc[:, feature_idx])),
            )
            ax.set_title(f"SHAP Dependence: {feature_names[feature_idx]}", fontsize=11)
            ax.set_xlabel(feature_names[feature_idx], fontsize=10)
            ax.set_ylabel(f"SHAP value ({target_var})", fontsize=10)
            ax.tick_params(labelsize=9)

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        part_label = f" (Part {part_idx + 1} of {n_parts})" if n_parts > 1 else ""
        fig.suptitle(f"SHAP Dependence Plots for Target: {target_var} ({model_name}){part_label}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.close(fig)
        figs.append(fig)

    return figs

# Run All Interpretability Plots
def run_interpretability_analysis(
    models_dict,
    X_train,
    y_train,
    target_columns,
    feature_names,
    preferred_model_name="XGBoost Regressor",
    test_sample_size=1000,
    background_sample_size=20,
    subsample=250,
    grid_resolution=20,
    show_plots=True
):
    """
    Orchestrate ICE/PDP + SHAP Summary + SHAP Dependence for each target.

    Strategy:
      1) Use preferred model if available; otherwise benchmark and pick the fastest.
      2) Fit on a small sample (test_sample_size) to keep plots snappy.
      3) Produce three figures per target and return them in a dict.

    Returns
    dict[str, matplotlib.figure.Figure]
        Keys like "ICE_PDP_<target>", "SHAP_Summary_<target>", "SHAP_Dependence_<target>"
    """

    target_var = target_columns[0]
    model_name = preferred_model_name if preferred_model_name in models_dict else get_fastest_model(
        models_dict, X_train, y_train, target_var
    )
    model = models_dict[model_name]

    X_sampled = X_train[:test_sample_size]
    y_sampled = y_train.iloc[:test_sample_size]

    tqdm.write(f"\n  Running Interpretability Analysis for: {model_name}")
    figures = {}

    for target_var in tqdm(target_columns, desc=f"Interpretability ({model_name})", unit="target", leave=False):
        tqdm.write(f"  Target: {target_var}")
        model.fit(X_sampled, y_sampled[target_var])

        if show_plots:
            tqdm.write("  Plotting ICE + PDP...")
            fig1 = plot_ice_and_pdp(model, X_sampled, feature_names, target_var, model_name,
                                    subsample=subsample, grid_resolution=grid_resolution)
            figures[f"ICE_PDP_{target_var}"] = fig1

            tqdm.write("  Plotting SHAP Summary...")
            fig2 = plot_shap_summary(model, X_sampled, feature_names, background_sample_size, target_var, model_name)
            figures[f"SHAP_Summary_{target_var}"] = fig2

            tqdm.write("  Plotting SHAP Dependence...")
            fig3 = plot_shap_dependence(model, X_sampled, feature_names, background_sample_size, target_var, model_name)
            figures[f"SHAP_Dependence_{target_var}"] = fig3

    return figures


