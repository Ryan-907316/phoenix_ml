# postprocessing.py
# This module provides post-training diagnostics and validation.
# The user can choose a method of cross-validation: K-fold, repeated K-fold, Group K-fold, LOO, LpO, and Shuffle Split, and provide parameters.
# Additionally, the module provides visualisations on influential points via Cook's Distance, residual analysis (scatter diagrams, histograms, Q-Q)
# There are also residual normalisation transforms and normality ranking using Anderson-Darling statistic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from scipy.stats import skew, kurtosis, anderson, probplot, norm, boxcox
from statsmodels.stats.diagnostic import lilliefors as _lilliefors
from sklearn.model_selection import (
    KFold, RepeatedKFold, GroupKFold, LeaveOneOut, LeavePOut, ShuffleSplit
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    explained_variance_score, r2_score
)
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score
import ast

from phoenix_ml.model_training import reset_model_to_defaults
from phoenix_ml.hyperparameter_optimisation import process_hyperparameters

def _pick(row, *candidates):
    # Return the first column present in 'row' among candidates.
    for c in candidates:
        if c in row.index:
            return row[c]
    raise KeyError(f"None of the expected columns found: {candidates}")

def _get_model_name(row):
    return _pick(row, 'model_name', 'Model')

def _get_hyperparams(row):
    # Return hyperparameters as a dict (parse string safely, allow dict).
    raw = _pick(row, 'hyperparameters', 'Value of Hyperparameters')
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw if isinstance(raw, dict) else {}

# Metrics Dictionary
metrics_dict = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "Explained Variance": explained_variance_score,
    "R^2": r2_score,
    "ADJUSTED R^2": lambda y_true, y_pred, n, p: 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - p - 1),
    "Q^2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
}

# CV methods
cv_methods = {
    "K-Fold": KFold,
    "Repeated K-Fold": RepeatedKFold,
    "LOO": LeaveOneOut,
    "LpO": LeavePOut,
    "Shuffle Split": ShuffleSplit,
}

# Parameters accepted by each CV method (used to filter the generic cv_args dict)
_CV_VALID_PARAMS = {
    "K-Fold":          {"n_splits", "shuffle", "random_state"},
    "Repeated K-Fold": {"n_splits", "n_repeats", "random_state"},
    "LOO":             set(),
    "LpO":             {"p"},
    "Shuffle Split":   {"n_splits", "test_size", "random_state"},
}

def _build_cv(cv_method: str, cv_args: dict):
    """Construct the appropriate CV splitter, stripping incompatible kwargs."""
    args = cv_args or {}
    if cv_method == "LOO":
        return LeaveOneOut()
    if cv_method == "LpO":
        p = int(args.get("p", args.get("n_splits", 2)))
        return LeavePOut(p=p)
    valid = _CV_VALID_PARAMS.get(cv_method, set())
    filtered = {k: v for k, v in args.items() if k in valid}
    # KFold only uses random_state when shuffle is enabled
    if cv_method == "K-Fold" and "random_state" in filtered:
        filtered["shuffle"] = True
    return cv_methods[cv_method](**filtered)

# Run cross_val_score with a scikit-learn splitter and return summary stats.
def perform_cross_validation_with_summary(model, X_train, y_train, target_var, cv_method, cv_args, scoring_metric, verbose: bool = False):
    cv = _build_cv(cv_method, cv_args)
    scoring_map = {
        "MSE": "neg_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "Explained Variance": "explained_variance",
        "R^2": "r2",
    }

    if scoring_metric in scoring_map:
        scoring = scoring_map[scoring_metric]
    elif scoring_metric in ["Q^2", "ADJUSTED R^2"]:
        def custom_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            n, p = X.shape
            if scoring_metric == "Q^2":
                denom = np.sum((y - np.mean(y))**2)
                return 1 - np.sum((y - y_pred)**2) / denom if denom > 0 else 0.0
            else:
                if n <= p + 1:  # guard against divide-by-zero (e.g. LOO with many features)
                    return float("nan")
                r2 = r2_score(y, y_pred)
                return 1 - (1 - r2) * (n - 1) / (n - p - 1)
        scoring = custom_scorer
    else:
        raise ValueError(f"Unsupported scoring metric: {scoring_metric}")

    scores = cross_val_score(model, X_train, y_train[target_var], cv=cv, scoring=scoring)
    if scoring_metric in ["MSE", "MAE"]:
        scores = -scores

    return {
        "method": cv_method,
        "params": cv_args,
        "mean_score": scores.mean(),
        "std_dev": scores.std(),
        "scores": scores
    }

def create_cv_summary_df(cv_summary, scoring_metric):
    df = pd.DataFrame([{
        "Target Variable": target,
        "CV Method": result["method"],
        "CV Parameters": result["params"],
        "Mean Score": result["mean_score"],
        "Std Deviation": result["std_dev"]
    } for target, result in cv_summary.items()])

    # Print formatted output
    print("\nCross-Validation Summary:\n")
    for _, row in df.iterrows():
        print(f"Target Variable: {row['Target Variable']}")
        print(f"  CV Method: {row['CV Method']}")
        print(f"  CV Parameters: {row['CV Parameters']}")
        print(f"  Mean {scoring_metric}: {row['Mean Score']:.4f}")
        print(f"  Std Deviation: {row['Std Deviation']:.4f}\n")

    return df

# Influence Analysis
def calculate_cooks_distance(X, residuals):
    X_const = sm.add_constant(X)
    influence = sm.OLS(residuals, X_const).fit().get_influence()
    return influence.cooks_distance[0]

# Stem plot of Cook's distance per sample for each target.
# Threshold line is 4/n (common rule-of-thumb). Y-axis on log scale to surface small/large differences.
def plot_cooks_distance_all_targets(best_models, X_train, X_test, y_train, y_test):
    num_targets = len(best_models)
    fig, axes = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
    axes = [axes] if num_targets == 1 else axes

    for idx, (target, row) in enumerate(best_models.items()):
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = y_test[target] - y_pred
        cooks_d = calculate_cooks_distance(X_test, residuals)
        threshold = 4 / len(cooks_d)

        n_exceed = int(np.sum(cooks_d > threshold))
        pct_exceed = 100.0 * n_exceed / len(cooks_d)
        axes[idx].stem(range(len(cooks_d)), cooks_d, basefmt=" ", markerfmt=".", linefmt="b")
        axes[idx].axhline(
            y=threshold, color="red", linestyle="--",
            label=f"Threshold 4/n = {threshold:.4f}  |  {n_exceed} points ({pct_exceed:.1f}%) exceed threshold"
        )
        axes[idx].set_yscale("log")
        axes[idx].set_title(f"Cook's Distance: {model_name} ({target})", fontweight='bold')
        axes[idx].set_xlabel("Testing Sample Index")
        axes[idx].set_ylabel("Cook's Distance (log scale)")
        axes[idx].legend(loc='best')

    plt.tight_layout()
    return fig

# Apply a residual transform to assess normality.
'''
Supported:
      "None": identity
      "Log":  sign(log(|r| + eps))        # preserves sign, compresses tails
      "Sqrt": sign(sqrt(|r|))             # milder than log
      "Box-Cox": boxcox(r - min(r) + eps) # requires strictly positive input
      "Yeo-Johnson": works with zero/negative residuals

    Note:
      - We shift residuals for Box-Cox to ensure positivity.
      - eps avoids log(0) and boxcox(0) errors.'''

def apply_transformation(residuals, transformation):
    if transformation == "Log":
        return np.log(np.abs(residuals) + 1e-6) * np.sign(residuals)
    elif transformation == "Sqrt":
        return np.sqrt(np.abs(residuals)) * np.sign(residuals)
    elif transformation == "Box-Cox":
        return boxcox(residuals - residuals.min() + 1e-6)[0]
    elif transformation == "Yeo-Johnson":
        return PowerTransformer(method="yeo-johnson").fit_transform(residuals.values.reshape(-1, 1)).flatten()
    return residuals

# Score each transform per target with simple normality indicators.
def evaluate_transformations(best_models, X_train, X_test, y_train, y_test):
    results = []
    for target, row in best_models.items():
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = y_test[target] - y_pred

        for t in ["None", "Log", "Sqrt", "Box-Cox", "Yeo-Johnson"]:
            try:
                transformed = apply_transformation(residuals, t)
                tr_arr = np.asarray(transformed, dtype=float)

                # Lilliefors test (correct KS for estimated parameters)
                try:
                    lf_stat, lf_p = _lilliefors(tr_arr, dist='norm', pvalmethod='table')
                    lf_p = float(lf_p)
                except Exception:
                    lf_p = None

                # Filiben coefficient (Q-Q Pearson correlation)
                try:
                    osm, osr = probplot(tr_arr, dist="norm")[0]
                    filiben = float(np.corrcoef(osm, osr)[0, 1])
                except Exception:
                    filiben = None

                results.append({
                    "Target Variable": target,
                    "Model": model_name,
                    "Transformation": t,
                    "Skewness": skew(tr_arr),
                    "Excess Kurtosis": kurtosis(tr_arr, fisher=True),
                    "AD Statistic": anderson(tr_arr)[0],
                    "Lilliefors p": lf_p,
                    "Filiben": filiben,
                })
            except Exception as e:
                results.append({
                    "Target Variable": target,
                    "Model": model_name,
                    "Transformation": t,
                    "Skewness": None,
                    "Excess Kurtosis": None,
                    "AD Statistic": None,
                    "Lilliefors p": None,
                    "Filiben": None,
                    "Error": str(e)
                })

    return pd.DataFrame(results)

# Plot residual diagnostics using the best transform per target.
# Returns three figures: scatter+histogram (side-by-side), Q-Q.
# "Best" is the transform with the minimum AD statistic per target.
def plot_all_transformations(results_df, best_models, X_train, X_test, y_train, y_test):
    results_df = results_df.dropna(subset=["AD Statistic"])
    best_rows = results_df.loc[results_df.groupby("Target Variable")["AD Statistic"].idxmin()]
    num_targets = len(best_rows)

    fig_rh, ax_rh = plt.subplots(num_targets, 2, figsize=(12, 5 * num_targets), squeeze=False)
    fig_q,  ax_q  = plt.subplots(num_targets, 1, figsize=(6,  5 * num_targets))
    ax_q = [ax_q] if num_targets == 1 else list(ax_q)

    for plot_idx, (_, row) in enumerate(best_rows.iterrows()):
        target = row["Target Variable"]
        trans = row["Transformation"]
        model_name = _get_model_name(row)
        best_row = best_models[target]
        params = process_hyperparameters(_get_hyperparams(best_row), model_name)
        model = reset_model_to_defaults(model_name)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = y_test[target] - y_pred
        transformed = apply_transformation(residuals, trans)
        tr_arr = np.asarray(transformed, dtype=float)

        # Left: residuals scatter
        ax_s = ax_rh[plot_idx, 0]
        ax_s.scatter(y_pred, tr_arr, alpha=0.5, s=8, edgecolors="none", label="Data Points")
        ax_s.axhline(0, color="green", linestyle="--", linewidth=1, label="Zero Line")
        ax_s.set_title(f"Residuals ({trans}): {target}", fontweight='bold')
        ax_s.set_xlabel("Predicted Values")
        ax_s.set_ylabel("Transformed Residuals")
        ax_s.legend(loc='best')
        ax_s.grid(True, linewidth=0.4)

        # Right: histogram
        ax_h = ax_rh[plot_idx, 1]
        ax_h.hist(tr_arr, bins=30, density=True, alpha=0.6, edgecolor="k")
        x = np.linspace(tr_arr.min(), tr_arr.max(), 100)
        ax_h.plot(x, norm.pdf(x, np.mean(tr_arr), np.std(tr_arr)), 'r--', label="Normal fit")
        ax_h.axvline(0, color='black', linestyle='--', linewidth=1, label="Zero")
        ax_h.set_title(f"Histogram of Residuals ({trans}): {target}", fontweight='bold')
        ax_h.set_xlabel("Residuals")
        ax_h.set_ylabel("Density")
        ax_h.legend(loc='best')
        ax_h.grid(True, linewidth=0.4)

        # Q-Q
        res = probplot(tr_arr, dist="norm")
        ax_q[plot_idx].scatter(res[0][0], res[0][1], alpha=0.6, s=8, edgecolors="none", label="Data Points")
        ax_q[plot_idx].plot(res[0][0], res[1][0] * res[0][0] + res[1][1], 'r--', label="Fitted Line")
        ax_q[plot_idx].set_title(f"Q-Q Plot ({trans}): {target}", fontweight='bold')
        ax_q[plot_idx].set_xlabel("Theoretical Quantiles")
        ax_q[plot_idx].set_ylabel("Actual Quantiles")
        ax_q[plot_idx].legend(loc='best')
        ax_q[plot_idx].grid(True, linewidth=0.4)

    fig_rh.suptitle("Transformed Residual Diagnostics", fontsize=13, fontweight='bold')
    fig_rh.tight_layout(rect=[0, 0, 1, 0.97])
    fig_q.tight_layout()
    return fig_rh, None, fig_q

# Run cross-validation summary for each target with its best model.
def process_all_targets_with_summary(best_models, X_train, y_train, cv_method, cv_args, scoring_metric):
    summary = {}
    for target, row in tqdm(best_models.items(), desc="Cross-Validation", unit="target", leave=False):
        tqdm.write(f"  CV for target: {target}")
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        result = perform_cross_validation_with_summary(model, X_train, y_train, target, cv_method, cv_args, scoring_metric)
        summary[target] = result
    return summary

def plot_residuals_with_influential_points_all_targets(best_models, X_train, X_test, y_train, y_test):
    num_targets = len(best_models)
    fig, axes = plt.subplots(num_targets, 2, figsize=(12, 5 * num_targets), squeeze=False)

    for idx, (target, row) in enumerate(best_models.items()):
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = np.asarray(y_test[target] - y_pred, dtype=float)

        cooks_d = calculate_cooks_distance(X_test, residuals)
        threshold = 4 / len(cooks_d)
        influential = cooks_d > threshold

        # Left: scatter
        ax_s = axes[idx, 0]
        ax_s.scatter(y_pred, residuals, s=8, alpha=0.5, edgecolors="none", label="Residuals")
        ax_s.scatter(y_pred[influential], residuals[influential], s=20, color='red',
                     edgecolors='k', linewidths=0.5, label="Influential")
        ax_s.axhline(0, color="green", linestyle="--", linewidth=1)
        ax_s.set_title(f"Residuals: {model_name} ({target})", fontweight='bold')
        ax_s.set_xlabel("Predicted Values")
        ax_s.set_ylabel("Residuals")
        ax_s.legend(loc='best')
        ax_s.grid(True, linewidth=0.4)

        # Right: histogram
        ax_h = axes[idx, 1]
        ax_h.hist(residuals, bins=30, density=True, alpha=0.6, edgecolor="k")
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax_h.plot(x, norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r--', label="Normal fit")
        ax_h.axvline(0, color='black', linestyle='--', linewidth=1, label="Zero")
        ax_h.set_title(f"Histogram of Residuals: {target}", fontweight='bold')
        ax_h.set_xlabel("Residuals")
        ax_h.set_ylabel("Density")
        ax_h.legend(loc='best')
        ax_h.grid(True, linewidth=0.4)

    fig.suptitle("Residuals with Influential Points (Pre-Transformation)", fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

# End-to-end postprocessing pipeline for selected best models.
# Steps (controlled by flags):
    #  1) Cross-validation summary (per target)
    #  2) Cook's distance plots (per target)
    #  3) Residual scatter with influential point highlighting (per target)
    #  4) Residual transforms → pick best by AD statistic → plot diagnostic trio
def run_postprocessing_analysis(
    best_models,
    X_train,
    X_test,
    y_train,
    y_test,
    cv_method,
    cv_args,
    scoring_metric="R^2",
    show_cv_summary=True,
    show_cooks_distance=True,
    show_residuals=True,
    show_transformation_plots=True,
    image_output_dir: str = "report_images"
):
    cooks_fig = None
    residuals_fig = None
    fig_r = fig_h = fig_q = None

    if show_cv_summary:
        print("\nRunning Cross-Validation Summary...")
        cv_summary = process_all_targets_with_summary(
            best_models,
            X_train,
            y_train,
            cv_method,
            cv_args,
            scoring_metric,
        )
        cv_summary_df = create_cv_summary_df(cv_summary, scoring_metric)
    else:
        cv_summary_df = pd.DataFrame()

    if show_cooks_distance:
        print("\nRunning Cook's Distance Analysis...")
        cooks_fig = plot_cooks_distance_all_targets(
            best_models, X_train, X_test, y_train, y_test
        )

    if show_residuals:
        print("\nRunning Residual Analysis...")
        residuals_fig = plot_residuals_with_influential_points_all_targets(
            best_models, X_train, X_test, y_train, y_test
        )

    print("\nEvaluating Residual Transformations...")
    transformation_results_df = evaluate_transformations(
        best_models, X_train, X_test, y_train, y_test
    )
    print(transformation_results_df)

    if show_transformation_plots:
        print("\nPlotting Transformed Residuals...")
        fig_r, fig_h, fig_q = plot_all_transformations(
            transformation_results_df,
            best_models,
            X_train,
            X_test,
            y_train,
            y_test
        )

    return {
        "cv_summary_df": cv_summary_df,
        "scoring_metric": scoring_metric,
        "transformation_df": transformation_results_df,
        "cooks_fig": cooks_fig if show_cooks_distance else None,
        "residuals_fig": residuals_fig if show_residuals else None,
        "transformation_figs": {
            "residual": fig_r,
            "histogram": fig_h,
            "qq": fig_q
        } if show_transformation_plots else {}
    }


