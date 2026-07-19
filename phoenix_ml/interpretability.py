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
from sklearn.utils import resample
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd

from phoenix_ml.sensitivity_analysis import (
    compute_morris_indices, compute_sobol_indices, compute_fast_indices,
    plot_sensitivity_indices, plot_sobol_fast_comparison,
)
from phoenix_ml.model_training import build_monotone_constraints_kwarg, apply_monotone_constraints_for_target
from phoenix_ml.validation import require_int_at_least

# Max subplot rows per figure — keeps each figure within one A4 content-height page.
# ICE/PDP, ALE, and SHAP Dependence use 18"-wide 3-col layouts with 4.5" rows.
# 3 rows × 4.5" = 13.5" figure height → ~542pt display — safely under the 652pt page limit.
# (Height budget is row-count-driven, so this holds regardless of column count.)
_MAX_ROWS_PER_FIG = 3

# ICE and PDP
# Plot ICE + PDP for every feature, split across multiple figures so each fits on one PDF page.
def plot_ice_and_pdp(model, X_train, feature_names, target_var, model_name,
                     subsample=250, grid_resolution=20):
    num_features = len(feature_names)
    num_cols = min(num_features, 3)
    subplot_w, subplot_h = 6.0, 4.5
    features_per_fig = _MAX_ROWS_PER_FIG * num_cols

    chunks = [list(range(i, min(i + features_per_fig, num_features)))
              for i in range(0, num_features, features_per_fig)]
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
            # Recolour the PDP average line to the brand orange so it stands out over ICE lines
            if ax.lines:
                ax.lines[-1].set_color("#E07818")
                ax.lines[-1].set_linewidth(2.0)
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f"Feature: {feature_names[feature_idx]}", fontsize=11)
            ax.set_xlabel(feature_names[feature_idx], fontsize=10)
            ax.set_ylabel(target_var, fontsize=10)
            ax.tick_params(labelsize=9)

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.suptitle(f"ICE and PDP Plots: {target_var} ({model_name})",
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.close(fig)
        figs.append(fig)

    return figs


# ALE (Accumulated Local Effects)
# Unlike PDP, which averages predictions over the *marginal* distribution of the other
# features (holding them at their real values while sweeping one feature — this can
# create unrealistic, never-observed feature combinations when features are
# correlated), ALE only ever compares nearby, observed values of the target feature
# within narrow quantile bins, then accumulates the local differences. This makes ALE
# far less misleading than PDP when features are correlated — exactly the situation
# phoenix_ml's own multicollinearity diagnostics frequently flag.
def _compute_ale_1d(model, X, feature_idx, grid_resolution=20):
    """Centered 1D ALE curve for one feature. Returns (edges, ale_values), each of
    length K+1, or None if the feature has fewer than 2 distinct quantile-bin edges
    (a constant/near-constant column — shouldn't normally occur post-preprocessing).
    """
    x = np.asarray(X[:, feature_idx], dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, grid_resolution + 1)))
    if len(edges) < 2:
        return None
    K = len(edges) - 1

    # Bin k (0-indexed) covers (edges[k], edges[k+1]]; the leftmost bin also includes
    # edges[0] itself.
    bin_idx = np.searchsorted(edges, x, side="left") - 1
    bin_idx = np.clip(bin_idx, 0, K - 1)

    local_effects = np.zeros(K)
    for k in range(K):
        mask = bin_idx == k
        if not mask.any():
            continue
        X_lo, X_hi = X[mask].copy(), X[mask].copy()
        X_lo[:, feature_idx] = edges[k]
        X_hi[:, feature_idx] = edges[k + 1]
        local_effects[k] = np.mean(model.predict(X_hi) - model.predict(X_lo))

    uncentered = np.concatenate([[0.0], np.cumsum(local_effects)])
    # Centre by the sample-weighted mean: each of the n samples contributes the
    # accumulated value of its own bin, not a uniform mean over the K bins.
    sample_values = uncentered[bin_idx + 1]
    return edges, uncentered - sample_values.mean()


def plot_ale(model, X_train, feature_names, target_var, model_name, grid_resolution=20):
    """ALE plots for every feature, split across multiple figures (same chunking as
    plot_ice_and_pdp so each figure fits one PDF page)."""
    num_features = len(feature_names)
    num_cols = min(num_features, 3)
    subplot_w, subplot_h = 6.0, 4.5
    features_per_fig = _MAX_ROWS_PER_FIG * num_cols

    chunks = [list(range(i, min(i + features_per_fig, num_features)))
              for i in range(0, num_features, features_per_fig)]
    figs = []

    for chunk in chunks:
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
            result = _compute_ale_1d(model, X_train, feature_idx, grid_resolution=grid_resolution)
            if result is None:
                ax.set_title(f"Feature: {feature_names[feature_idx]} (constant, skipped)", fontsize=11)
                ax.axis("off")
                continue
            edges, ale_values = result
            ax.plot(edges, ale_values, color="#E07818", linewidth=2.0, marker="o", markersize=3)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_title(f"Feature: {feature_names[feature_idx]}", fontsize=11)
            ax.set_xlabel(feature_names[feature_idx], fontsize=10)
            ax.set_ylabel(f"ALE ({target_var})", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, linewidth=0.4)

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.suptitle(f"Accumulated Local Effects (ALE): {target_var} ({model_name})",
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.close(fig)
        figs.append(fig)

    return figs


# SHAP Explainer Selection
def select_shap_explainer(model, X_train, background_sample_size, feature_names=None, random_state=None):
    """
    Pick an appropriate SHAP explainer based on model class.
    Falls back to KernelExplainer with a small background sample if the
    preferred explainer fails.

    Implementation details:
    • TreeExplainer for tree-based models (fastest/common) — this includes
      LightGBM and HistGradientBoosting, not just sklearn's RandomForest/
      ExtraTrees/Bagging and XGBoost; TreeExplainer reads the tree structure
      directly, so it also sidesteps the KernelExplainer issue below.
    • KernelExplainer otherwise (model-agnostic, slower). MLPRegressor is
      NOT routed to shap.GradientExplainer here: that explainer requires a
      real TensorFlow/PyTorch autodiff graph and always fails on a plain
      scikit-learn MLPRegressor, so special-casing it just wastes one
      guaranteed-failing attempt before falling back to KernelExplainer anyway.
    """

    import shap
    import numpy as np
    import pandas as pd

    # None resolves to seed 0 (package-wide convention) so the KernelExplainer
    # background sample is reproducible by default.
    random_state = 0 if random_state is None else random_state

    # Always ensure DataFrame
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)

    _TREE_MODELS = (
        RandomForestRegressor, XGBRegressor, BaggingRegressor, ExtraTreesRegressor,
        LGBMRegressor, HistGradientBoostingRegressor,
    )

    def _kernel_explainer():
        background = X_train.sample(n=min(background_sample_size, len(X_train)), replace=False,
                                    random_state=random_state)
        return shap.KernelExplainer(model.predict, background)

    try:
        if isinstance(model, _TREE_MODELS):
            return shap.TreeExplainer(model)
        return _kernel_explainer()

    except Exception as e:
        attempted = "TreeExplainer" if isinstance(model, _TREE_MODELS) else "KernelExplainer"
        print(f"SHAP {attempted} failed ({e}). Falling back to KernelExplainer...")

        try:
            return _kernel_explainer()
        except Exception as e2:
            raise ValueError(f"SHAP explainer fallback error: {e2}")

def _shap_values(explainer, X):
    """explainer.shap_values(X), suppressing KernelExplainer's own internal per-row
    tqdm progress bar (one iteration per row of X — e.g. 800 for an 800-row sample).
    That bar nests inside this module's own per-model progress bar, and two live tqdm
    bars at once is what floods a non-terminal log capture (the UI's captured-stdout
    stream) with cursor-repositioning escape codes. TreeExplainer/GradientExplainer
    don't have an internal progress bar and don't accept a `silent` kwarg at all, so
    only pass it for KernelExplainer.
    """
    if isinstance(explainer, shap.KernelExplainer):
        return explainer.shap_values(X, silent=True)
    return explainer.shap_values(X)


# SHAP summary (dot) plot showing feature importance and impact distribution.
def plot_shap_summary(model, X_train, feature_names, background_sample_size, target_var, model_name,
                      random_state=None):
    explainer = select_shap_explainer(model, X_train, background_sample_size, random_state=random_state)
    shap_values = _shap_values(explainer, X_train)

    # shap.summary_plot() calls plt.gcf() internally rather than creating its own figure,
    # so plt.close('all') is needed to clear stale preprocessing figures from pyplot's
    # registry first. Stored Python Figure objects are unaffected and remain valid for savefig().
    plt.close('all')
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False, plot_type='dot')
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(f"SHAP Summary: {target_var} ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

# SHAP Dependence — split across multiple figures so each fits on one PDF page.
def plot_shap_dependence(model, X_train, feature_names, background_sample_size, target_var, model_name,
                         random_state=None):
    explainer = select_shap_explainer(model, X_train, background_sample_size, feature_names, random_state=random_state)
    shap_values = _shap_values(explainer, X_train)

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    num_features = len(feature_names)
    num_cols = min(num_features, 3)
    subplot_w, subplot_h = 6.0, 4.5
    features_per_fig = _MAX_ROWS_PER_FIG * num_cols

    norm = plt.Normalize(vmin=X_train.min().min(), vmax=X_train.max().max())
    cmap = plt.cm.coolwarm

    chunks = [list(range(i, min(i + features_per_fig, num_features)))
              for i in range(0, num_features, features_per_fig)]
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

        fig.suptitle(f"SHAP Dependence Plots: {target_var} ({model_name})",
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.close(fig)
        figs.append(fig)

    return figs


# SHAP Waterfall — one figure per representative sample (worst/spread/best error)
def plot_shap_waterfall(model, X_test, feature_names, target_var, model_name,
                        background_sample_size, y_test, n_samples=3, percentiles=None,
                        random_state=None):
    """
    SHAP waterfall plots for n_samples representative predictions spread from worst
    to best prediction error.  Returns a list of matplotlib figures.

    Explainer routing:
      TreeExplainer  → explainer(X) returns Explanation directly.
      KernelExplainer → shap_values() + manual Explanation construction.
    (select_shap_explainer never returns a GradientExplainer — see its docstring —
    so there is no third case here.)
    """
    if isinstance(X_test, np.ndarray):
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = X_test.reset_index(drop=True)

    y_arr  = np.asarray(y_test).ravel()
    y_pred = model.predict(X_test_df)
    errors = np.abs(y_arr - y_pred)
    n      = len(errors)

    sorted_idx = np.argsort(errors)  # ascending: index 0 = smallest error

    if percentiles is not None:
        # Explicit percentile fractions supplied (0.0 = best, 1.0 = worst)
        pct_fracs = [max(0.0, min(1.0, p)) for p in percentiles]
    else:
        k = max(n_samples, 1)
        pct_fracs = [1.0 - i / max(k - 1, 1) for i in range(k)]

    positions      = [int(round(p * (n - 1))) for p in pct_fracs]
    sample_indices = [int(sorted_idx[pos]) for pos in positions]

    labels = []
    for p_frac, pos in zip(pct_fracs, positions):
        if pos >= n - 1:
            labels.append("Worst Error")
        elif pos <= 0:
            labels.append("Best Error")
        else:
            labels.append(f"{int(round(p_frac * 100))}th-percentile Error")

    X_selected = X_test_df.iloc[sample_indices].reset_index(drop=True)

    explainer = select_shap_explainer(model, X_test_df, background_sample_size, feature_names, random_state=random_state)

    # Build shap.Explanation — routing by explainer type
    if isinstance(explainer, shap.TreeExplainer):
        explanation = explainer(X_selected)
    else:
        # KernelExplainer: shap_values() returns numpy, build Explanation manually
        sv = _shap_values(explainer, X_selected)
        ev = explainer.expected_value
        explanation = shap.Explanation(
            values=sv,
            base_values=np.full(len(X_selected), float(ev)),
            data=X_selected.values,
            feature_names=feature_names,
        )

    figs = []
    for i, label in enumerate(labels):
        # shap.plots.waterfall() may draw on plt.gcf() rather than creating its own figure.
        # plt.close('all') clears stale preprocessing figures from pyplot's registry so SHAP
        # always gets a fresh figure. Python Figure objects in session remain valid for savefig().
        plt.close('all')
        shap.plots.waterfall(explanation[i], show=False)
        fig = plt.gcf()
        # shap.plots.waterfall() already sizes the figure height to fit one row per
        # feature (num_features * 0.5in + 1.5in); forcing a fixed height here would
        # squash the labels back together for anything beyond a handful of features,
        # so only the width is overridden and a little headroom is added for the
        # suptitle below.
        _, auto_height = fig.get_size_inches()
        fig.set_size_inches(10, auto_height + 0.4)
        fig.suptitle(
            f"SHAP Waterfall ({label}): {target_var}: {model_name}",
            fontsize=11, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.close(fig)
        figs.append(fig)

    return figs


def _rank_spearman(a, b):
    """Spearman rank correlation between two equal-length importance vectors,
    without adding a scipy.stats dependency beyond what SALib already pulls in."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_interpretability_metrics(model_name, target_var, feature_names,
                                     morris_results, sobol_results,
                                     X_sampled, y_sampled_target, target_constraints,
                                     fast_results=None):
    """One comparable summary row per (model, target) — the metrics_df half of
    run_interpretability_analysis()'s return, mirroring the Model Training
    Results / UQ Summary tables. Self-contained: draws only on Morris/Sobol results
    already computed in this module (not postprocessing.py's permutation/LOFO output),
    so turning off the CV/Postprocessing step can't silently break this table.

    Column set (user-selected "cross-method agreement" design): top feature per method
    plus a single rank-agreement score, rather than each method's full margin/
    concentration/interaction-share detail — the point is spotting when Morris and
    Sobol disagree on a model, not reproducing every statistic from the (single best
    model's) sensitivity plots here.
    """
    row = {"Model": model_name, "Target Variable": target_var}

    def _top_feature(values, names):
        # Returns "FeatureName (value)" so the summary table shows not just which
        # feature won but by how much, without needing a separate value column.
        if values is None or len(values) == 0:
            return None
        idx = int(np.argmax(values))
        return f"{names[idx]} ({values[idx]:.3f})"

    row["Morris Top Feature"] = _top_feature(
        morris_results["mu_star"] if morris_results is not None else None,
        morris_results["feature_names"] if morris_results is not None else None,
    )
    row["Sobol Top Feature"] = _top_feature(
        sobol_results["ST"] if sobol_results is not None else None,
        sobol_results["feature_names"] if sobol_results is not None else None,
    )
    if fast_results is not None:
        row["FAST Top Feature"] = _top_feature(fast_results["ST"],
                                               fast_results["feature_names"])

    if morris_results is not None and sobol_results is not None:
        # Both rank the same feature list — align on morris_results' order.
        names = morris_results["feature_names"]
        sobol_by_name = dict(zip(sobol_results["feature_names"], sobol_results["ST"]))
        aligned_sobol = [sobol_by_name.get(n, np.nan) for n in names]
        row["Rank Agreement"] = _rank_spearman(morris_results["mu_star"], aligned_sobol)

    if sobol_results is not None and fast_results is not None:
        # Sobol and FAST estimate the SAME total-order indices via different
        # mechanisms — their rank agreement is direct cross-estimator
        # corroboration (low agreement = don't trust either decomposition yet;
        # likely undersampled).
        names = sobol_results["feature_names"]
        fast_by_name = dict(zip(fast_results["feature_names"], fast_results["ST"]))
        aligned_fast = [fast_by_name.get(n, np.nan) for n in names]
        row["Sobol-FAST Agreement"] = _rank_spearman(sobol_results["ST"], aligned_fast)

    # Monotonicity sanity check: does the enforced constraint direction agree with the
    # feature's raw (unconstrained) correlation sign against the target in this sample?
    if target_constraints:
        X_arr = np.asarray(X_sampled)
        y_arr = np.asarray(y_sampled_target, dtype=float)
        mismatches = []
        for feat, direction in target_constraints.items():
            if direction == 0 or feat not in feature_names:
                continue
            idx = feature_names.index(feat)
            col = X_arr[:, idx].astype(float)
            if np.std(col) == 0 or np.std(y_arr) == 0:
                continue
            raw_sign = 1 if np.corrcoef(col, y_arr)[0, 1] >= 0 else -1
            if raw_sign != direction:
                mismatches.append(feat)
        row["Monotonicity Mismatches"] = ", ".join(mismatches) if mismatches else None

    return row


# Run All Interpretability Plots
class FailedPlot:
    """Sentinel stored in the figures dict when a plot type fails to generate for
    a (model, target) pair. Without this, a failed plot was only logged to the
    console (tqdm.write) — the PDF report simply had no image for that slot, with
    nothing telling the reader a plot was attempted and failed vs. never
    requested. report_generation.add_interpretability_section renders this as an
    explicit note instead of an image."""

    def __init__(self, reason: str):
        self.reason = reason


def run_interpretability_analysis(
    models_dict,
    X_train,
    y_train,
    target_columns,
    feature_names,
    model_names_to_run=None,
    test_sample_size=1000,
    background_sample_size=20,
    subsample=250,
    grid_resolution=20,
    show_ice_pdp=True,
    show_ale=True,
    show_shap_summary=True,
    show_shap_dependence=True,
    show_shap_waterfall=True,
    n_waterfall_samples=3,
    waterfall_percentiles=None,
    show_sensitivity_morris=True,
    show_sensitivity_sobol=False,
    show_sensitivity_fast=False,
    sensitivity_morris_trajectories=10,
    sensitivity_morris_levels=4,
    sensitivity_sobol_n=512,
    sensitivity_fast_n=512,
    show_plots=True,
    random_state=None,
    monotonic_constraints=None,
    visual_model_by_target=None,
    checkpoint_fn=None,
):
    """
    Orchestrate Sensitivity analysis (+ the comparable metrics table) for every selected
    model and target, mirroring run_uncertainty_quantification's every-model loop shape —
    but the expensive per-feature visuals (ICE/PDP, ALE, SHAP) only render for each
    target's own best model, not every model, since those are what actually dominate
    report size/runtime (~10 pages per model). See run_step_interpretability_before/after
    in workflow_steps.py for how this is invoked before vs. after HPO.

    visual_model_by_target: dict[target -> model_name] | None. When given, ICE/PDP/ALE/
    SHAP are only generated for the (model, target) pairs it names; every other selected
    model still gets its Morris/Sobol-derived metrics row (cheap: batched .predict()
    calls, no extra plotting). None means "generate visuals for every model" (used by
    workflow.py's simpler single-pass standalone entry point).

    Returns (metrics_df, figures) — the same (df, figs) order as
    run_uncertainty_quantification, for the same before/after-HPO session-tuple shape.
      figures: dict[str, Figure | list[Figure]], keyed "{PREFIX}_{target}__{model_name}"
      metrics_df: one row per (model, target) — see compute_interpretability_metrics().
    """
    # Bad sizes used to surface as sklearn's own InvalidParameterError about
    # "resample n_samples" (test_sample_size=0) or fail later inside SHAP with
    # an empty background set — named, immediate errors instead. Oversized
    # values are fine: sampling clamps to the data actually available.
    require_int_at_least("test_sample_size", test_sample_size, minimum=1)
    require_int_at_least("background_sample_size", background_sample_size, minimum=1)
    require_int_at_least("sensitivity_morris_trajectories", sensitivity_morris_trajectories, minimum=1)
    require_int_at_least("sensitivity_sobol_n", sensitivity_sobol_n, minimum=1)
    # The authoritative N > 4M^2 FAST constraint lives in compute_fast_indices
    # (which knows M); this just catches type/sign nonsense at the entry.
    require_int_at_least("sensitivity_fast_n", sensitivity_fast_n, minimum=1)

    if model_names_to_run is None:
        model_names_to_run = list(models_dict.keys())

    # None resolves to seed 0 (package-wide convention) so the subsample below is
    # reproducible by default. Kept fixed across every (model, target) pair in the
    # loop below so all of them are still trained/explained on the SAME subsample —
    # the same apples-to-apples comparison the old head-slice gave, just no longer
    # biased toward the start of the training set.
    _seed = 0 if random_state is None else random_state

    figures = {}
    metrics_rows = []

    for model_name in tqdm(model_names_to_run, desc="Interpretability", unit="model", leave=False):
        # Optional cooperative pause/cancel hook (e.g. from the UI): checked once per
        # model, mirroring run_all_models_optimisation's own checkpoint granularity.
        if checkpoint_fn is not None:
            checkpoint_fn()
        entry = models_dict[model_name]
        # entry is either a single shared estimator ({model_name: instance} — used
        # before HPO, refit per target below) or a per-target dict ({target: instance}
        # — used after HPO via get_all_models_tuned_per_target, where each target
        # already has its own correctly-tuned instance and constraints are already baked
        # in via the same session.selected -> HPO -> tuned-instance path).
        per_target_instances = entry if isinstance(entry, dict) else None
        tqdm.write(f"\n=== Interpretability: {model_name} ===")

        # Plain inner loop, not a second tqdm bar — matches the single-tqdm-level
        # pattern already used by run_uncertainty_quantification/run_models elsewhere
        # in this codebase. Nesting two live bars floods a non-terminal log capture
        # (the UI's captured-stdout stream) with cursor-repositioning escape codes.
        for target_var in target_columns:
            if per_target_instances is not None and target_var not in per_target_instances:
                continue
            tqdm.write(f"  --- Target: {target_var} ---")
            if len(X_train) > test_sample_size:
                # A contiguous head-slice only describes an early chunk of the
                # operating envelope for training sets larger than test_sample_size —
                # every SHAP/ICE/PDP/ALE/Morris/Sobol result (and the sensitivity
                # bounds built from the same sample) would silently be biased toward
                # whatever generated the start of the dataset.
                X_sampled, y_sampled = resample(
                    X_train, y_train, n_samples=test_sample_size,
                    replace=False, random_state=_seed,
                )
            else:
                X_sampled, y_sampled = X_train, y_train
            target_constraints = (monotonic_constraints or {}).get(target_var, {})

            if per_target_instances is not None:
                model = per_target_instances[target_var]
            else:
                # model is shared/reused across every target for this model_name — a
                # per-target monotonic constraint must be (re-)applied here, since it's
                # never touched again once fitting begins.
                model = entry
                apply_monotone_constraints_for_target(model, model_name, feature_names, target_constraints)
            model.fit(X_sampled, y_sampled[target_var])

            morris_results = None
            sobol_results = None
            fast_results = None

            # ICE/PDP/ALE/SHAP are the expensive, page-heavy visuals — only generate
            # them for each target's actual best model, not every selected model.
            # visual_model_by_target=None means "every model" (workflow.py's simpler
            # single-pass path, which doesn't have a per-target best-model concept).
            is_visual_model = (
                visual_model_by_target is None
                or visual_model_by_target.get(target_var) == model_name
            )

            if show_plots and is_visual_model:
                key_suffix = f"{target_var}__{model_name}"

                # Every plot type below is wrapped individually: with every selected
                # model now profiled (not just one "preferred" model), a failure specific
                # to one model/plot-type combination (e.g. a SHAP/library version
                # incompatibility for one model class) must not wipe out that model's
                # other analyses, let alone every other model's results.
                if show_ice_pdp:
                    tqdm.write("  Plotting ICE + PDP...")
                    try:
                        figures[f"ICE_PDP_{key_suffix}"] = plot_ice_and_pdp(
                            model, X_sampled, feature_names, target_var, model_name,
                            subsample=subsample, grid_resolution=grid_resolution,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] ICE/PDP failed for {model_name}/{target_var}: {e}")
                        figures[f"ICE_PDP_{key_suffix}"] = FailedPlot(str(e))

                if show_ale:
                    tqdm.write("  Plotting ALE...")
                    try:
                        figures[f"ALE_{key_suffix}"] = plot_ale(
                            model, X_sampled, feature_names, target_var, model_name,
                            grid_resolution=grid_resolution,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] ALE failed for {model_name}/{target_var}: {e}")
                        figures[f"ALE_{key_suffix}"] = FailedPlot(str(e))

                if show_shap_summary:
                    tqdm.write("  Plotting SHAP Summary...")
                    try:
                        figures[f"SHAP_Summary_{key_suffix}"] = plot_shap_summary(
                            model, X_sampled, feature_names, background_sample_size, target_var, model_name,
                            random_state=random_state,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] SHAP Summary failed for {model_name}/{target_var}: {e}")
                        figures[f"SHAP_Summary_{key_suffix}"] = FailedPlot(str(e))

                if show_shap_dependence:
                    tqdm.write("  Plotting SHAP Dependence...")
                    try:
                        figures[f"SHAP_Dependence_{key_suffix}"] = plot_shap_dependence(
                            model, X_sampled, feature_names, background_sample_size, target_var, model_name,
                            random_state=random_state,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] SHAP Dependence failed for {model_name}/{target_var}: {e}")
                        figures[f"SHAP_Dependence_{key_suffix}"] = FailedPlot(str(e))

                if show_shap_waterfall:
                    tqdm.write("  Plotting SHAP Waterfall...")
                    try:
                        figures[f"SHAP_Waterfall_{key_suffix}"] = plot_shap_waterfall(
                            model, X_sampled, feature_names, target_var, model_name,
                            background_sample_size,
                            y_test=y_sampled[target_var],
                            n_samples=n_waterfall_samples,
                            percentiles=waterfall_percentiles,
                            random_state=random_state,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] SHAP Waterfall failed for {model_name}/{target_var}: {e}")
                        figures[f"SHAP_Waterfall_{key_suffix}"] = FailedPlot(str(e))

            # Sensitivity analysis reuses this same already-fitted `model` (no retraining)
            # — every evaluation below is just a batched .predict() call, even though the
            # evaluation *count* runs into the hundreds/thousands. Unlike the visuals
            # above, this always runs when show_plots is set (not gated by
            # is_visual_model): it's cheap, and it's what feeds this (model, target)'s
            # row in the metrics table — the whole point of comparing every model there
            # instead of showing every model's full visual page-set.
            if show_plots:
                key_suffix = f"{target_var}__{model_name}"

                if show_sensitivity_morris:
                    tqdm.write("  Computing Morris sensitivity...")
                    try:
                        morris_results = compute_morris_indices(
                            model, X_sampled, feature_names,
                            n_trajectories=sensitivity_morris_trajectories,
                            num_levels=sensitivity_morris_levels,
                            random_state=random_state,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] Morris sensitivity failed for {model_name}/{target_var}: {e}")
                        morris_results = None
                        if is_visual_model:
                            figures[f"Sensitivity_Morris_{key_suffix}"] = FailedPlot(str(e))
                    if morris_results is not None and is_visual_model:
                        figures[f"Sensitivity_Morris_{key_suffix}"] = plot_sensitivity_indices(
                            morris_results, target_var, model_name, "Morris"
                        )

                if show_sensitivity_sobol:
                    tqdm.write("  Computing Sobol sensitivity...")
                    try:
                        sobol_results = compute_sobol_indices(
                            model, X_sampled, feature_names, n_base=sensitivity_sobol_n,
                            random_state=random_state,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] Sobol sensitivity failed for {model_name}/{target_var}: {e}")
                        sobol_results = None
                        if is_visual_model:
                            figures[f"Sensitivity_Sobol_{key_suffix}"] = FailedPlot(str(e))

                if show_sensitivity_fast:
                    tqdm.write("  Computing FAST sensitivity...")
                    try:
                        fast_results = compute_fast_indices(
                            model, X_sampled, feature_names, n_samples=sensitivity_fast_n,
                            random_state=random_state,
                        )
                    except Exception as e:
                        tqdm.write(f"  [WARN] FAST sensitivity failed for {model_name}/{target_var}: {e}")
                        fast_results = None
                        if is_visual_model:
                            figures[f"Sensitivity_FAST_{key_suffix}"] = FailedPlot(str(e))

                # Sobol and FAST estimate the SAME indices (S1/ST), so when both
                # succeeded they render as ONE paired-bar comparison figure — the
                # point of running both is checking their agreement, not printing
                # two redundant charts. If only one ran (or one failed above),
                # whichever succeeded gets its own standalone figure.
                if is_visual_model:
                    if sobol_results is not None and fast_results is not None:
                        figures[f"Sensitivity_Sobol_FAST_{key_suffix}"] = plot_sobol_fast_comparison(
                            sobol_results, fast_results, target_var, model_name
                        )
                    else:
                        if sobol_results is not None:
                            figures[f"Sensitivity_Sobol_{key_suffix}"] = plot_sensitivity_indices(
                                sobol_results, target_var, model_name, "Sobol"
                            )
                        if fast_results is not None:
                            figures[f"Sensitivity_FAST_{key_suffix}"] = plot_sensitivity_indices(
                                fast_results, target_var, model_name, "FAST"
                            )

            # Only flag mismatches for models the constraint was actually applied to —
            # monotone_constraints is XGBoost/LGBM-only (build_monotone_constraints_kwarg
            # returns {} for every other model type), so target_constraints being
            # non-empty for this *target* doesn't mean it was enforced on this *model*.
            effective_constraints = (
                target_constraints
                if feature_names is not None
                and build_monotone_constraints_kwarg(model_name, feature_names, target_constraints)
                else {}
            )
            metrics_rows.append(compute_interpretability_metrics(
                model_name, target_var, feature_names, morris_results, sobol_results,
                X_sampled, y_sampled[target_var], effective_constraints,
                fast_results=fast_results,
            ))

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, figures


