# postprocessing.py
# This module provides post-training diagnostics and validation.
# The user can choose a method of cross-validation: K-fold, repeated K-fold, Group K-fold, LOO, LpO, and Shuffle Split, and provide parameters.
# Additionally, the module provides visualisations on influential points via Cook's Distance, residual analysis (scatter diagrams, histograms, Q-Q)
# There are also residual normalisation transforms and normality ranking using Anderson-Darling statistic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, anderson, probplot, norm, boxcox
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
    "Group K-Fold": GroupKFold,
    "LOO": LeaveOneOut,
    "LpO": LeavePOut,
    "Shuffle Split": ShuffleSplit,
}

# Run cross_val_score with a scikit-learn splitter and return summary stats.
def perform_cross_validation_with_summary(model, X_train, y_train, target_var, cv_method, cv_args, scoring_metric, verbose: bool = False):
    cv = cv_methods[cv_method](**cv_args) if cv_args else cv_methods[cv_method]()
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
                return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            else:
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

        axes[idx].stem(range(len(cooks_d)), cooks_d, basefmt=" ", markerfmt=".", linefmt="b")
        axes[idx].axhline(y=threshold, color="red", linestyle="--", label=f"Threshold (4/n = {threshold:.4f})")
        axes[idx].set_yscale("log")
        axes[idx].set_title(f"Cook's Distance for {model_name} ({target})")
        axes[idx].set_xlabel("Testing Sample Index #")
        axes[idx].set_ylabel("Cook's Distance (log scale)")
        axes[idx].legend()

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
                results.append({
                    "Target Variable": target,
                    "Model": model_name,
                    "Transformation": t,
                    "Skewness": skew(transformed),
                    "Excess Kurtosis": kurtosis(transformed, fisher=True),
                    "AD Statistic": anderson(transformed)[0]
                })
            except Exception as e:
                results.append({
                    "Target Variable": target,
                    "Model": model_name,
                    "Transformation": t,
                    "Skewness": None,
                    "Excess Kurtosis": None,
                    "AD Statistic": None,
                    "Error": str(e)
                })

    return pd.DataFrame(results)

# Plot residual scatter, histogram, and Q–Q using the best transform per target.
# "Best" is the transform with the minimum AD statistic per target.
def plot_all_transformations(results_df, best_models, X_train, X_test, y_train, y_test):
    results_df = results_df.dropna(subset=["AD Statistic"])
    best_rows = results_df.loc[results_df.groupby("Target Variable")["AD Statistic"].idxmin()]
    num_targets = len(best_rows)

    fig_r, ax_r = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
    fig_h, ax_h = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
    fig_q, ax_q = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
    ax_r, ax_h, ax_q = map(lambda x: [x] if num_targets == 1 else x, [ax_r, ax_h, ax_q])

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

        # Residuals
        ax_r[plot_idx].scatter(y_pred, transformed, alpha=0.6, edgecolor="k", label="Data Points")
        ax_r[plot_idx].axhline(0, color="green", linestyle="--", linewidth=1, label="Zero Residual Line")
        ax_r[plot_idx].set_title(f"Residuals ({trans}) for {model_name} - {target}")
        ax_r[plot_idx].set_xlabel("Predicted Values")
        ax_r[plot_idx].set_ylabel("Transformed Residuals")
        ax_r[plot_idx].legend()
        ax_r[plot_idx].grid(True)

        # Histogram
        ax_h[plot_idx].hist(transformed, bins=30, density=True, alpha=0.6, edgecolor="k")
        x = np.linspace(transformed.min(), transformed.max(), 100)
        p = norm.pdf(x, np.mean(transformed), np.std(transformed))
        ax_h[plot_idx].plot(x, p, 'r--', label="Normal Distribution")
        ax_h[plot_idx].set_title(f"Histogram of Residuals ({trans}) for {target}")
        ax_h[plot_idx].set_xlabel("Residuals")
        ax_h[plot_idx].set_ylabel("Density")
        ax_h[plot_idx].legend()
        ax_h[plot_idx].grid(True)

        # Q-Q Plot
        res = probplot(transformed, dist="norm")
        ax_q[plot_idx].scatter(res[0][0], res[0][1], alpha=0.6, edgecolor="k", label="Data Points")
        ax_q[plot_idx].plot(res[0][0], res[1][0] * res[0][0] + res[1][1], 'r--', label="Fitted Line")
        ax_q[plot_idx].set_title(f"Q-Q Plot ({trans}) for {target}")
        ax_q[plot_idx].set_xlabel("Theoretical Quantiles")
        ax_q[plot_idx].set_ylabel("Actual Quantiles")
        ax_q[plot_idx].legend()
        ax_q[plot_idx].grid(True)

    plt.tight_layout()
    return fig_r, fig_h, fig_q

# Run cross-validation summary for each target with its best model.
def process_all_targets_with_summary(best_models, X_train, y_train, cv_method, cv_args, scoring_metric):
    summary = {}
    for target, row in best_models.items():
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
        influential = cooks_d > threshold

        axes[idx].scatter(y_pred, residuals, label="Residuals", edgecolor='k')
        axes[idx].scatter(y_pred[influential], residuals[influential], color='red', label="Influential", edgecolor='k')
        axes[idx].axhline(0, color="green", linestyle="--")
        axes[idx].set_title(f"Residuals with Influential Points: {model_name} ({target})")
        axes[idx].set_xlabel("Predicted Values")
        axes[idx].set_ylabel("Residuals")
        axes[idx].legend()

    plt.tight_layout()
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
        "transformation_df": transformation_results_df,
        "cooks_fig": cooks_fig if show_cooks_distance else None,
        "residuals_fig": residuals_fig if show_residuals else None,
        "transformation_figs": {
            "residual": fig_r,
            "histogram": fig_h,
            "qq": fig_q
        } if show_transformation_plots else {}
    }


