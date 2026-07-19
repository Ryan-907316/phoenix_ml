# uncertainty_quantification.py
# UQ methods: bootstrapping, conformal prediction, and GP posterior predictive intervals.
# Calibration reporting: reliability diagrams (embedded in UQ figures), CRPS, and RMSCE.

import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from joblib import Parallel, delayed
import pandas as pd
import os
from tqdm import tqdm

from phoenix_ml.model_training import apply_monotone_constraints_for_target
from phoenix_ml.progress import log_table
from phoenix_ml.validation import require_int_at_least, require_in_range, require_choice


def _bootstrap_worker(model, X_train, y_train_target, X_test, seed):
    """Single bootstrap iteration — module-level so multiprocessing can pickle it."""
    from sklearn.utils import resample as _resample
    X_res, y_res = _resample(X_train, y_train_target, random_state=seed)
    m = copy.deepcopy(model)
    try:
        if "n_jobs" in m.get_params():
            m.set_params(n_jobs=1)
    except Exception:
        pass
    m.fit(X_res, y_res)
    return m.predict(X_test)


from sklearn.gaussian_process import GaussianProcessRegressor as _GPR


def bootstrap_uncertainty(model, X_train, y_train, X_test, target_var,
                          n_bootstrap, confidence_interval, n_jobs=1, random_state=None):
    y_target = y_train[target_var]
    base_seed = 0 if random_state is None else random_state
    if n_jobs == 1:
        preds_list = [
            _bootstrap_worker(model, X_train, y_target, X_test, base_seed + i)
            for i in range(n_bootstrap)
        ]
    else:
        preds_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_worker)(model, X_train, y_target, X_test, base_seed + i)
            for i in range(n_bootstrap)
        )
    predictions = np.array(preds_list)                                   # (B, n_test)
    predictions_mean = predictions.mean(axis=0)
    lower_bound = np.percentile(predictions, (100 - confidence_interval) / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - confidence_interval) / 2, axis=0)
    return predictions_mean, lower_bound, upper_bound, predictions


# Do NOT abbreviate conformal prediction to CP
def conformal_predictions(model, X_train, y_train, X_test, target_var, calibration_frac,
                          confidence_interval=95, random_state=None):
    """Split-conformal prediction intervals.

    calibration_frac controls only how much training data is held out to measure
    residuals; the interval's coverage level comes from confidence_interval. These
    are independent knobs — earlier versions derived the quantile from
    calibration_frac, which silently changed the nominal coverage whenever the
    calibration split size was changed.

    Uses the finite-sample-corrected quantile ceil((n+1)*conf)/n, clipped to 1,
    so small calibration sets don't systematically under-cover. The model is
    deep-copied before fitting: the caller's instance is shared across UQ methods
    and later pipeline steps, and must not be silently refit on a subset here.
    """
    m = copy.deepcopy(model)
    X_proper, X_calib, y_proper, y_calib = train_test_split(
        X_train, y_train[target_var], test_size=calibration_frac,
        random_state=0 if random_state is None else random_state)
    m.fit(X_proper, y_proper)
    residuals = np.abs(y_calib - m.predict(X_calib))
    test_preds = m.predict(X_test)
    if len(residuals) == 0:
        return test_preds, test_preds, test_preds, np.array([])
    n_cal = len(residuals)
    level = min(1.0, np.ceil((n_cal + 1) * confidence_interval / 100.0) / n_cal)
    quantile = np.quantile(residuals, level)
    lower_bound = test_preds - quantile
    upper_bound = test_preds + quantile
    return test_preds, lower_bound, upper_bound, residuals


def gp_posterior_uncertainty(model, X_train, y_train, X_test, target_var, confidence_interval):
    if not isinstance(model, _GPR):
        return None
    from scipy.stats import norm as _norm
    m = copy.deepcopy(model)
    try:
        m.fit(X_train, y_train[target_var])
        y_pred, y_std = m.predict(X_test, return_std=True)
    except Exception as exc:
        # Unlike every other risky call in this module, this one had no guard —
        # a singular-kernel LinAlgError (plausible with duplicate rows) would
        # abort UQ for every remaining model/target instead of just skipping
        # this one method, the same way the caller already skips a None return
        # for a non-GPR model.
        print(f"[WARN] GP Posterior uncertainty failed for '{target_var}': {exc} - skipping")
        return None
    z = _norm.ppf(0.5 + confidence_interval / 200.0)
    return y_pred, y_pred - z * y_std, y_pred + z * y_std, y_std


def calculate_coverage(y_true, lower, upper):
    within_interval = (y_true >= lower) & (y_true <= upper)
    return np.mean(within_interval) * 100


# ── Calibration metrics ───────────────────────────────────────────────────────

def compute_calibration_metrics(result, y_test, method, n_levels=19):
    """
    Compute reliability curve, CRPS, and RMSCE for one UQ method on one target.

    result  : dict from results[method] — must contain "raw_preds" (bootstrap),
              "calibration_residuals" (conformal), or "y_std" (GP Posterior).
    y_test  : 1-D array of true target values.
    method  : "Bootstrapping" | "Conformal" | "GP Posterior"

    Returns dict: {"reliability_curve": [(nominal, actual), ...], "CRPS": float|None, "RMSCE": float}
    Returns None if required data are missing.
    """
    from scipy.stats import norm as _norm

    y = np.asarray(y_test, dtype=float)
    levels = np.linspace(0.05, 0.95, n_levels)
    reliability_curve = []
    crps = None

    if method == "Bootstrapping":
        raw_preds = result.get("raw_preds")
        if raw_preds is None or raw_preds.size == 0:
            return None
        B = raw_preds.shape[0]

        for conf in levels:
            alpha = 1.0 - conf
            lo = np.percentile(raw_preds, 50.0 * alpha, axis=0)
            hi = np.percentile(raw_preds, 100.0 - 50.0 * alpha, axis=0)
            actual = float(np.mean((y >= lo) & (y <= hi)))
            reliability_curve.append((round(float(conf), 4), round(actual, 4)))

        # CRPS via sorted-samples decomposition (vectorised over test points)
        energy = np.mean(np.abs(raw_preds - y[np.newaxis, :]), axis=0)          # (n,)
        sorted_p = np.sort(raw_preds, axis=0)                                    # (B, n)
        b_idx = np.arange(1, B + 1, dtype=float)[:, np.newaxis]
        spread = np.sum(sorted_p * (2.0 * b_idx - B - 1.0), axis=0) / (B ** 2)
        crps = float(np.mean(energy - spread))

    elif method == "Conformal":
        residuals = result.get("calibration_residuals")
        y_pred = result.get("mean")
        if residuals is None or len(residuals) == 0 or y_pred is None:
            return None

        n_cal = len(residuals)
        for conf in levels:
            # Same finite-sample-corrected quantile conformal_predictions() uses
            # for the actually-reported interval (see its docstring) — a plain
            # quantile here made this curve visibly disagree with the reported
            # interval's own coverage at small calibration sizes, since the two
            # used different quantile definitions for the same nominal level.
            level = min(1.0, np.ceil((n_cal + 1) * conf) / n_cal)
            q = np.quantile(residuals, level)
            actual = float(np.mean(np.abs(y - y_pred) <= q))
            reliability_curve.append((round(float(conf), 4), round(actual, 4)))

        # CRPS is not well-defined for conformal prediction (set-valued, not distributional)
        crps = None

    elif method == "GP Posterior":
        mu = result.get("mean")
        sigma = result.get("y_std")
        if mu is None or sigma is None:
            return None
        sigma = np.maximum(sigma, 1e-9)

        for conf in levels:
            z = _norm.ppf((1.0 + conf) / 2.0)
            lo = mu - z * sigma
            hi = mu + z * sigma
            actual = float(np.mean((y >= lo) & (y <= hi)))
            reliability_curve.append((round(float(conf), 4), round(actual, 4)))

        # Closed-form Gaussian CRPS
        z_vals = (y - mu) / sigma
        crps = float(np.mean(
            sigma * (2.0 * _norm.pdf(z_vals) + z_vals * (2.0 * _norm.cdf(z_vals) - 1.0)
                     - 1.0 / np.sqrt(np.pi))
        ))

    else:
        return None

    nominals = np.array([c for c, _ in reliability_curve])
    actuals  = np.array([a for _, a in reliability_curve])
    rmsce = float(np.sqrt(np.mean((nominals - actuals) ** 2)))

    return {"reliability_curve": reliability_curve, "CRPS": crps, "RMSCE": rmsce}


def _draw_reliability_diagram_on_ax(ax, calib_by_method, target_var, confidence_interval):
    """Draw a Reliability Diagram onto an existing Axes object (for embedding in UQ figures)."""
    METHOD_COLORS = {
        "Bootstrapping": "#4472C4",
        "Conformal":     "#E07818",
        "GP Posterior":  "#70AD47",
    }

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, zorder=2, label="Perfect")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.07, color="#D94F3D", zorder=1)
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.07, color="#4472C4", zorder=1)

    for method, calib in calib_by_method.items():
        if calib is None:
            continue
        curve = calib.get("reliability_curve", [])
        if not curve:
            continue
        nominal = [c for c, _ in curve]
        actual  = [a for _, a in curve]
        rmsce   = calib.get("RMSCE", float("nan"))
        crps    = calib.get("CRPS")
        color   = METHOD_COLORS.get(method, "#888888")
        crps_str = f"\nCRPS={crps:.3f}" if crps is not None else ""
        label = f"{method} (RMSCE={rmsce:.3f}{crps_str})"
        ax.plot(nominal, actual, color=color, linewidth=1.5, marker="o", markersize=3,
                zorder=3, label=label)

    nom_ci = confidence_interval / 100.0
    ax.axvline(nom_ci, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal Coverage", fontsize=8)
    ax.set_ylabel("Actual Coverage", fontsize=8)
    ax.set_title(f"Reliability Diagram (RD): {target_var}", fontweight="bold", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--")


# ── Per-model UQ ─────────────────────────────────────────────────────────────

def perform_uncertainty_quantification_for_model(
    uq_method, model, X_train, y_train, X_test, y_test, target_var, n_bootstrap,
    calibration_frac, subsample_test_size, confidence_interval, n_jobs=1,
    include_gp_posterior=False, calibration_enabled=False, random_state=None,
):
    """
    Run UQ for a single model/target. Returns (results, y_test_subsample).
    Calibration data is stored inside each method's result dict when calibration_enabled=True.
    """
    _seed = 0 if random_state is None else random_state
    if len(X_test) > subsample_test_size:
        # Seeded: every model (and both UQ stages) is evaluated on the SAME test
        # subsample, so rows of the UQ Summary table are directly comparable.
        test_indices = np.random.default_rng(_seed).choice(len(X_test), subsample_test_size, replace=False)
        X_test_subsample = X_test[test_indices]
        y_test_subsample = y_test[target_var].iloc[test_indices]
    else:
        X_test_subsample = X_test
        y_test_subsample = y_test[target_var]

    results = {}

    # Bootstrapping
    if uq_method in ["Bootstrapping", "Both"]:
        predictions_mean, lower_bound_bs, upper_bound_bs, raw_preds = bootstrap_uncertainty(
            model, X_train, y_train, X_test_subsample, target_var,
            n_bootstrap, confidence_interval, n_jobs=n_jobs, random_state=_seed,
        )
        results["Bootstrapping"] = {
            "mean":       predictions_mean,
            "lower":      lower_bound_bs,
            "upper":      upper_bound_bs,
            "avg_range":  float(np.mean(upper_bound_bs - lower_bound_bs)),
            "std_range":  float(np.std(upper_bound_bs - lower_bound_bs)),
            "coverage":   calculate_coverage(y_test_subsample, lower_bound_bs, upper_bound_bs),
            "raw_preds":  raw_preds,
        }

    # Conformal Predictions
    if uq_method in ["Conformal", "Both"]:
        test_preds, lower_bound_cp, upper_bound_cp, residuals = conformal_predictions(
            model, X_train, y_train, X_test_subsample, target_var, calibration_frac,
            confidence_interval=confidence_interval, random_state=_seed,
        )
        results["Conformal"] = {
            "mean":                  test_preds,
            "lower":                 lower_bound_cp,
            "upper":                 upper_bound_cp,
            "avg_range":             float(np.mean(upper_bound_cp - lower_bound_cp)),
            "std_range":             float(np.std(upper_bound_cp - lower_bound_cp)),
            "coverage":              calculate_coverage(y_test_subsample, lower_bound_cp, upper_bound_cp),
            "calibration_residuals": residuals,
        }

    # GP Posterior
    if include_gp_posterior:
        gp_result = gp_posterior_uncertainty(
            model, X_train, y_train, X_test_subsample, target_var, confidence_interval
        )
        if gp_result is not None:
            gp_pred, gp_lower, gp_upper, gp_std = gp_result
            results["GP Posterior"] = {
                "mean":      gp_pred,
                "lower":     gp_lower,
                "upper":     gp_upper,
                "avg_range": float(np.mean(gp_upper - gp_lower)),
                "std_range": float(np.std(gp_upper - gp_lower)),
                "coverage":  calculate_coverage(y_test_subsample, gp_lower, gp_upper),
                "y_std":     gp_std,
            }

    # Calibration metrics stored inside each method's result dict
    if calibration_enabled and results:
        y_arr = (y_test_subsample.values
                 if hasattr(y_test_subsample, "values")
                 else np.asarray(y_test_subsample))
        for method, res in results.items():
            cal = compute_calibration_metrics(res, y_arr, method)
            if cal is not None:
                res["calibration"] = cal

    return results, y_test_subsample


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_uncertainty_results_for_model(
    results_by_target, model_name, uq_method, target_columns,
    confidence_interval, calibration_frac, stage_label="",
    y_true_by_target=None, calib_by_target=None,
):
    """
    Grid of UQ line plots for all targets and methods.
    When calib_by_target is provided, appends a Reliability Diagram column
    on the right of each row so everything stays in one figure per model.
    """
    base_methods = ["Bootstrapping", "Conformal"] if uq_method == "Both" else [uq_method]
    first_res = next(iter(results_by_target.values()), {})
    methods = base_methods + (["GP Posterior"] if "GP Posterior" in first_res else [])
    num_targets = len(results_by_target)

    has_calib = (
        calib_by_target is not None
        and any(
            v is not None
            for calib in calib_by_target.values()
            for v in calib.values()
        )
    )
    n_cols = len(methods) + (1 if has_calib else 0)

    # UQ columns at 7 inches wide; RD column at 5 inches wide
    width_ratios = [7] * len(methods) + ([5] if has_calib else [])
    fig, axes = plt.subplots(
        num_targets, n_cols,
        figsize=(sum(width_ratios), 5 * num_targets),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    fig.suptitle(f"Uncertainty Quantification: {model_name}", fontsize=14, fontweight="bold")

    for row_idx, target_var in enumerate(results_by_target):
        result_data = results_by_target[target_var]
        y_true = (y_true_by_target or {}).get(target_var)

        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            res = result_data[method]
            mean_pred, lb, ub = res["mean"], res["lower"], res["upper"]
            x_range = range(len(mean_pred))
            if method == "Bootstrapping":
                label = f"{confidence_interval}% CI"
            elif method == "GP Posterior":
                label = f"{confidence_interval}% GP PI"
            else:
                label = f"{confidence_interval}% PI"
            ax.fill_between(x_range, lb, ub, color="red", alpha=0.2, label=label)
            ax.plot(mean_pred, label="Prediction", color="blue", linewidth=1.2)
            if y_true is not None:
                y_vals = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
                ax.plot(y_vals, linestyle="--", color="black", linewidth=1.0, label="Ground Truth")
            if method in ("Bootstrapping", "GP Posterior"):
                stat_str = (f"Mean +/- Std Width = {res['avg_range']:.2f} +/- {res['std_range']:.2f}"
                            f", Coverage = {res['coverage']:.2f}%")
            else:
                stat_str = f"Interval Width = {res['avg_range']:.2f}, Coverage = {res['coverage']:.2f}%"
            ax.set_title(f"{method}: {target_var}\n{stat_str}", fontweight="bold")
            ax.set_xlabel("Sample #")
            ax.set_ylabel(f"Predicted {target_var}")
            ax.legend(loc="upper right")

        # Reliability Diagram column (last column when calibration enabled)
        if has_calib:
            ax_rd = axes[row_idx][-1]
            calib = (calib_by_target or {}).get(target_var, {})
            _draw_reliability_diagram_on_ax(ax_rd, calib, target_var, confidence_interval)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig


# ── Top-level runner ──────────────────────────────────────────────────────────

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
    n_jobs=1,
    stage_label="",
    show_plots=True,
    include_gp_posterior=False,
    calibration_enabled=True,
    random_state=None,
    feature_names=None,
    monotonic_constraints=None,
    checkpoint_fn=None,
):
    # Validated once here at the public entry, with the parameter's own name --
    # bad values used to surface as cryptic crashes deep inside numpy/sklearn
    # (n_bootstrap=0 -> IndexError in percentile; calibration_frac=0 -> sklearn
    # InvalidParameterError about "test_size") or, worse, produce silently
    # nonsensical intervals (conformal confidence_interval=150 clamped the
    # quantile to the max residual with no error at all).
    require_int_at_least("n_bootstrap", n_bootstrap, minimum=1)
    require_in_range("confidence_interval", confidence_interval, 0, 100,
                     inclusive_low=False, inclusive_high=False)
    require_in_range("calibration_frac", calibration_frac, 0, 1,
                     inclusive_low=False, inclusive_high=False)
    require_int_at_least("subsample_test_size", subsample_test_size, minimum=1)
    require_choice("uq_method", uq_method, ("Both", "Bootstrapping", "Conformal"))

    uq_records = []
    uq_figures = {}

    if model_names_to_run is None:
        model_names_to_run = list(models_dict.keys())

    for model_name in tqdm(model_names_to_run, desc=f"UQ ({stage_label})", unit="model", leave=False):
        # Optional cooperative pause/cancel hook (e.g. from the UI): checked once per
        # model, mirroring run_all_models_optimisation's own checkpoint granularity.
        if checkpoint_fn is not None:
            checkpoint_fn()
        entry = models_dict[model_name]
        # entry is either a single shared estimator ({model_name: instance} — used
        # before HPO, when every target reuses the same default-hyperparameter model and
        # refits per target below) or a per-target dict ({target: instance} — used after
        # HPO via get_all_models_tuned_per_target, where each target already has its own
        # correctly-tuned instance and no further constraint application is needed).
        per_target_instances = entry if isinstance(entry, dict) else None
        model = None if per_target_instances is not None else entry
        tqdm.write(f"\n=== UQ ({stage_label}): {model_name} ===")
        results_by_target = {}
        y_true_by_target  = {}
        calib_by_target   = {}
        model_rows_start = len(uq_records)

        for target_var in target_columns:
            if per_target_instances is not None:
                if target_var not in per_target_instances:
                    continue
                target_model = per_target_instances[target_var]
            else:
                # model is shared/reused across every target for this model_name — a
                # per-target monotonic constraint must be (re-)applied here, since it's
                # never touched again once training/fitting begins downstream.
                apply_monotone_constraints_for_target(
                    model, model_name, feature_names, (monotonic_constraints or {}).get(target_var, {})
                )
                target_model = model
            results, y_sub = perform_uncertainty_quantification_for_model(
                uq_method, target_model, X_train, y_train, X_test, y_test, target_var,
                n_bootstrap, calibration_frac, subsample_test_size, confidence_interval,
                n_jobs=n_jobs, include_gp_posterior=include_gp_posterior,
                calibration_enabled=calibration_enabled, random_state=random_state,
            )
            results_by_target[target_var] = results
            y_true_by_target[target_var]  = y_sub

            if calibration_enabled:
                calib_by_target[target_var] = {
                    method: res.get("calibration")
                    for method, res in results.items()
                }

            for method, res in results.items():
                avg = res["avg_range"]
                std = res["std_range"]
                cov = res["coverage"]

                record = {
                    "Model":           model_name,
                    "Target Variable": target_var,
                    "Stage":           stage_label,
                    "UQ Method":       method,
                    "Mean Range":      avg,
                    "Std Range":       std if method != "Conformal" else None,
                    "Coverage (%)":    cov,
                }
                if calibration_enabled:
                    calib = res.get("calibration") or {}
                    record["CRPS"]  = calib.get("CRPS")
                    record["RMSCE"] = calib.get("RMSCE")
                uq_records.append(record)

        # One compact table per model, instead of the old loose per-method prose
        # lines. "Interval / Range" folds the Conformal fixed-width vs mean+/-std
        # distinction into one readable column.
        model_records = uq_records[model_rows_start:]
        if model_records:
            display_df = pd.DataFrame([{
                "Target Variable": rec["Target Variable"],
                "UQ Method":       rec["UQ Method"],
                "Interval / Range": (
                    f"{rec['Mean Range']:.2f} (fixed width)"
                    if rec["Std Range"] is None
                    else f"{rec['Mean Range']:.2f} +/- {rec['Std Range']:.2f}"
                ),
                "Coverage (%)":    f"{rec['Coverage (%)']:.2f}",
            } for rec in model_records])
            log_table(display_df, floatfmt=".2f")

        if show_plots:
            fig = plot_uncertainty_results_for_model(
                results_by_target, model_name, uq_method, target_columns,
                confidence_interval, calibration_frac,
                stage_label=stage_label,
                y_true_by_target=y_true_by_target,
                calib_by_target=calib_by_target if calibration_enabled else None,
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
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plot_paths[label] = path
        plt.close(fig)

    return plot_paths
