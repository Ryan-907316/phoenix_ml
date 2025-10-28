# uncertainty_quantification.py
# This module provides the user with two UQ methods (bootstrapping and conformal predictions)
# The user can also state the CI/PIs and then plots the specified intervals in the report.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import os
from datetime import datetime

#  Bootstrapping: refit on bootstrap samples, form percentile CIs
def bootstrap_uncertainty(model, X_train, y_train, X_test, target_var, n_bootstrap, confidence_interval):
    predictions = []
    for i in range(n_bootstrap):
        # Deterministic resampling per iteration (random_state=i)
        X_resampled, y_resampled = resample(X_train, y_train[target_var], random_state=i)
        model.fit(X_resampled, y_resampled)
        predictions.append(model.predict(X_test))
    predictions = np.array(predictions)
    predictions_mean = predictions.mean(axis=0)
    
    # Two-sided CI from bootstrap distribution
    lower_bound = np.percentile(predictions, (100 - confidence_interval) / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - confidence_interval) / 2, axis=0)
    return predictions_mean, lower_bound, upper_bound

# Conformal Prediction (split-conformal variant):
# Fit on residual fold, calibrate quantile on holdout residuals, then form symmetric prediction intervals on X_test.

def conformal_predictions(model, X_train, y_train, X_test, target_var, calibration_frac):
    X_calib, X_residual, y_calib, y_residual = train_test_split(
        X_train, y_train[target_var], test_size=calibration_frac)
    model.fit(X_residual, y_residual)
    residuals = np.abs(y_calib - model.predict(X_calib))
    quantile = np.quantile(residuals, 1 - calibration_frac)
    test_preds = model.predict(X_test)
    lower_bound = test_preds - quantile
    upper_bound = test_preds + quantile
    return test_preds, lower_bound, upper_bound

# Do NOT abbreviate conformal prediction to CP

# Utility coverage calculation
def calculate_coverage(y_true, lower, upper):
    within_interval = (y_true >= lower) & (y_true <= upper)
    return np.mean(within_interval) * 100

# Run UQ for a single model/target, optionally subsample test set for speed
def perform_uncertainty_quantification_for_model(
    uq_method, model, X_train, y_train, X_test, y_test, target_var, n_bootstrap,
    calibration_frac, subsample_test_size, confidence_interval
):
    if len(X_test) > subsample_test_size:
        test_indices = np.random.choice(len(X_test), subsample_test_size, replace=False)
        X_test_subsample = X_test[test_indices]
        y_test_subsample = y_test[target_var].iloc[test_indices]
    else:
        X_test_subsample = X_test
        y_test_subsample = y_test[target_var]

    results = {}

    # Bootstrapping
    if uq_method in ["Bootstrapping", "Both"]:
        predictions_mean, lower_bound_bs, upper_bound_bs = bootstrap_uncertainty(
            model, X_train, y_train, X_test_subsample, target_var, n_bootstrap, confidence_interval
        )
        avg_interval = np.mean(upper_bound_bs - lower_bound_bs)
        std_interval = np.std(upper_bound_bs - lower_bound_bs)
        coverage = calculate_coverage(y_test_subsample, lower_bound_bs, upper_bound_bs)
        results["Bootstrapping"] = {
            "mean": predictions_mean,
            "lower": lower_bound_bs,
            "upper": upper_bound_bs,
            "avg_range": avg_interval,
            "std_range": std_interval,
            "coverage": coverage
        }

    # Conformal Predictions
    if uq_method in ["Conformal", "Both"]:
        test_preds, lower_bound_cp, upper_bound_cp = conformal_predictions(
            model, X_train, y_train, X_test_subsample, target_var, calibration_frac
        )
        avg_interval = np.mean(upper_bound_cp - lower_bound_cp)
        std_interval = np.std(upper_bound_cp - lower_bound_cp)
        coverage = calculate_coverage(y_test_subsample, lower_bound_cp, upper_bound_cp)
        results["Conformal"] = {
            "mean": test_preds,
            "lower": lower_bound_cp,
            "upper": upper_bound_cp,
            "avg_range": avg_interval,
            "std_range": std_interval,
            "coverage": coverage
        }

    return results, X_test_subsample

# Plotting for each target and method, and show mean prediction and interval band
def plot_uncertainty_results_for_model(
    results_by_target, model_name, uq_method, target_columns,
    confidence_interval, calibration_frac, stage_label=""
):
    methods = ["Bootstrapping", "Conformal"] if uq_method == "Both" else [uq_method]
    num_targets = len(results_by_target)

    fig, axes = plt.subplots(
        num_targets, len(methods),
        figsize=(8 * len(methods), 5 * num_targets),
        squeeze=False
    )
    fig.suptitle(f"Uncertainty Quantification for {model_name}", fontsize=16)

    for row_idx, target_var in enumerate(results_by_target):
        result_data = results_by_target[target_var]
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            res = result_data[method]
            mean_pred, lb, ub = res["mean"], res["lower"], res["upper"]
            label = f"{confidence_interval}% CI" if method == "Bootstrapping" else f"{(1 - calibration_frac) * 100:.0f}% PI"
            ax.plot(mean_pred, label="Prediction", color="blue")
            ax.fill_between(range(len(mean_pred)), lb, ub, color="red", alpha=0.2, label=label)
            ax.set_title(f"{method}: {target_var}\nMean ± Std Range = {res['avg_range']:.2f} ± {res['std_range']:.2f}, Coverage = {res['coverage']:.2f}%")
            ax.set_xlabel("Sample #")
            ax.set_ylabel(f"Predicted {target_var}")
            ax.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Run UQ across selected models and targets; optionally plot and return a tidy DataFrame with summary stats and figure handles by label.
def run_uncertainty_quantification(
    models_dict,
    X_train,
    X_test,
    y_train,
    y_test,
    target_columns,
    model_names_to_run=None,
    uq_method="Both",
    n_bootstrap=10,
    confidence_interval=95,
    calibration_frac=0.05,
    subsample_test_size=50,
    stage_label="",
    show_plots=True
):

    uq_records = []
    uq_figures = {}

    if model_names_to_run is None:
        model_names_to_run = list(models_dict.keys())

    for model_name in model_names_to_run:
        model = models_dict[model_name]
        print(f"\n Performing UQ ({stage_label}) for {model_name}")
        results_by_target = {}

        for target_var in target_columns:
            results, _ = perform_uncertainty_quantification_for_model(
                uq_method, model, X_train, y_train, X_test, y_test, target_var,
                n_bootstrap, calibration_frac, subsample_test_size, confidence_interval
            )
            results_by_target[target_var] = results

            for method, res in results.items():
                avg = res["avg_range"]
                std = res["std_range"]
                cov = res["coverage"]

                print(f"  {method} - {target_var}:")
                print(f"    Mean ± Std Range: {avg:.2f} ± {std:.2f}")
                print(f"    Coverage: {cov:.2f}%")

                uq_records.append({
                    "Model": model_name,
                    "Target Variable": target_var,
                    "Stage": stage_label,
                    "UQ Method": method,
                    "Mean Range": avg,
                    "Std Range": std,
                    "Coverage (%)": cov
                })

        if show_plots:
            fig = plot_uncertainty_results_for_model(
                results_by_target, model_name, uq_method, target_columns,
                confidence_interval, calibration_frac,
                stage_label=stage_label
            )
            label = f"{model_name} - {stage_label}".strip()
            uq_figures[label] = fig

    return pd.DataFrame(uq_records), uq_figures

def save_uq_plots(figures, output_dir, prefix="uq"):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    for label, fig in figures.items():
        filename = f"{prefix}_{label.lower().replace(' ', '_').replace('-', '_')}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plot_paths[label] = path
        plt.close(fig)  # Prevent blank plots

    return plot_paths

def save_uncertainty_results(uq_df, results_dir, stage="Before HPO"):
    os.makedirs(results_dir, exist_ok=True)

    # Automatically clean up redundant words
    clean_stage = (
        stage.replace("Uncertainty Quantification", "")
             .replace("Uncertainty_Quantification", "")
             .replace("(", "")
             .replace(")", "")
             .strip()
    )

    # Timestamp format for clarity and compactness
    timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")

    # Final concise filename
    filename = f"UQ {clean_stage} {timestamp}.csv"
    path = os.path.join(results_dir, filename)

    # Save CSV
    uq_df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\nSaved {stage} results to: {path}")
    return path



