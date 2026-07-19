# postprocessing.py
# This module provides post-training diagnostics and validation.
# The user can choose a method of cross-validation: K-fold, repeated K-fold, LOO, LpO, and Shuffle Split, and provide parameters.
# Additionally, the module provides visualisations on influential points via Cook's Distance, residual analysis (scatter diagrams, histograms, Q-Q)
# There are also residual normalisation transforms and normality ranking using Anderson-Darling statistic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from scipy.stats import (
    skew, kurtosis, anderson, probplot, norm,
    shapiro, jarque_bera, normaltest,
)
from scipy.optimize import minimize_scalar
from statsmodels.stats.diagnostic import (
    lilliefors as _lilliefors,
    het_breuschpagan,
    het_white,
    acorr_ljungbox,
)
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import (
    KFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    explained_variance_score, r2_score
)
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance as _permutation_importance
import ast
import math

from phoenix_ml.model_training import reset_model_to_defaults
from phoenix_ml.hyperparameter_optimisation import process_hyperparameters
from phoenix_ml.progress import log_table, log_warn
from phoenix_ml.validation import require_choice

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
    # Zero-variance y_true (constant target, or an unlucky small CV fold) previously
    # divided by zero silently; guarded the same way KGE below already is.
    "Q^2": lambda y_true, y_pred: (
        1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        if np.sum((y_true - np.mean(y_true)) ** 2) > 0 else float("nan")
    ),
    "KGE": lambda y_true, y_pred: (
        1 - np.sqrt(
            (np.corrcoef(y_true, y_pred)[0, 1] - 1) ** 2 +
            (np.std(y_pred) / np.std(y_true) - 1) ** 2 +
            (np.mean(y_pred) / np.mean(y_true) - 1) ** 2
        )
        if np.std(y_true) > 0 and np.mean(y_true) != 0 else float("nan")
    ),
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

# LeavePOut produces C(n_train, p) splits, each a full model refit — this grows
# combinatorially, not linearly. Even the default p=2 on a few-hundred-row training
# set means tens of thousands of full refits; an easy way to hang the app for hours
# with zero warning if left unbounded.
_MAX_LPO_SPLITS = 10_000

def _build_cv(cv_method: str, cv_args: dict, n_train: int | None = None):
    """Construct the appropriate CV splitter, stripping incompatible kwargs.

    n_train, when given, lets LpO refuse combinatorially explosive configurations
    up front with a clear message instead of silently hanging for hours.
    """
    args = cv_args or {}
    # Fail with a named error listing the valid options, not a bare KeyError
    # from the cv_methods dict lookup at the bottom.
    require_choice("cv_method", cv_method, set(cv_methods))
    if cv_method == "LOO":
        return LeaveOneOut()
    if cv_method == "LpO":
        p = int(args.get("p", args.get("n_splits", 2)))
        if n_train is not None and 0 < p <= n_train:
            n_splits = math.comb(n_train, p)
            if n_splits > _MAX_LPO_SPLITS:
                raise ValueError(
                    f"Leave-P-Out with p={p} on {n_train} training rows would run "
                    f"{n_splits:,} splits (each a full model refit) — more than the "
                    f"{_MAX_LPO_SPLITS:,} limit. Reduce p, or use a different CV method "
                    f"(K-Fold and Shuffle Split scale with n_splits, not C(n, p))."
                )
        return LeavePOut(p=p)
    valid = _CV_VALID_PARAMS.get(cv_method, set())
    filtered = {k: v for k, v in args.items() if k in valid}
    # KFold only uses random_state when shuffle is enabled
    if cv_method == "K-Fold" and "random_state" in filtered:
        filtered["shuffle"] = True
    return cv_methods[cv_method](**filtered)

# Run cross_val_score with a scikit-learn splitter and return summary stats.
# Metrics with no sklearn built-in scorer name, shared by CV, permutation importance,
# and LOFO importance — all three used to only support this set for R^2/MSE/MAE/
# Explained Variance and silently fell back to plain R^2 for everything else
# (Q^2/Adjusted R^2/NRMSE/MAPE/KGE), with nothing in the results or plot titles
# saying so. Extracted here (originally only implemented for CV) so permutation and
# LOFO importance now actually compute the metric requested, not a silent substitute.
_CUSTOM_SCORER_METRICS = ("Q^2", "ADJUSTED R^2", "NRMSE", "MAPE", "KGE")


def _custom_metric_scorer(scoring_metric):
    """Return a sklearn-compatible scorer callable (estimator, X, y) -> float,
    higher-is-better, for one of _CUSTOM_SCORER_METRICS."""
    def custom_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        n, p = X.shape
        if scoring_metric == "Q^2":
            denom = np.sum((y - np.mean(y))**2)
            return 1 - np.sum((y - y_pred)**2) / denom if denom > 0 else 0.0
        elif scoring_metric == "ADJUSTED R^2":
            if n <= p + 1:
                return float("nan")
            r2 = r2_score(y, y_pred)
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)
        elif scoring_metric == "NRMSE":
            r = np.max(y) - np.min(y)
            return -(np.sqrt(np.mean((y - y_pred) ** 2)) / r) if r > 0 else float("nan")
        elif scoring_metric == "KGE":
            if np.std(y) == 0 or np.mean(y) == 0:
                return float("nan")
            return 1 - np.sqrt(
                (np.corrcoef(y, y_pred)[0, 1] - 1) ** 2 +
                (np.std(y_pred) / np.std(y) - 1) ** 2 +
                (np.mean(y_pred) / np.mean(y) - 1) ** 2
            )
        else:  # MAPE — negated so higher (less negative) is better, matching neg_* scorers
            denom = np.where(np.abs(y) > 1e-10, np.abs(y), 1e-10)
            return -float(np.mean(np.abs((y - y_pred) / denom)) * 100)
    return custom_scorer


def perform_cross_validation_with_summary(model, X_train, y_train, target_var, cv_method, cv_args, scoring_metric, verbose: bool = False):
    cv = _build_cv(cv_method, cv_args, n_train=len(X_train))
    scoring_map = {
        "MSE": "neg_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "Explained Variance": "explained_variance",
        "R^2": "r2",
    }

    if scoring_metric in scoring_map:
        scoring = scoring_map[scoring_metric]
    elif scoring_metric in _CUSTOM_SCORER_METRICS:
        scoring = _custom_metric_scorer(scoring_metric)
    else:
        raise ValueError(f"Unsupported scoring metric: {scoring_metric}")

    scores = cross_val_score(model, X_train, y_train[target_var], cv=cv, scoring=scoring)
    if scoring_metric in ["MSE", "MAE", "NRMSE", "MAPE"]:
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

    # Same tabulate grid as every other console table (Model Performance
    # Summary, UQ, transformations) instead of loose per-target prose lines.
    print(f"\nCross-Validation Summary (Mean Score = mean {scoring_metric}):")
    log_table(df, max_col_width=48)

    return df

# Influence Analysis
def calculate_cooks_distance(X, residuals):
    X_const = sm.add_constant(X)
    X_const_arr = np.asarray(X_const, dtype=float)
    rank = np.linalg.matrix_rank(X_const_arr)
    n_cols = X_const_arr.shape[1]
    if rank < n_cols or len(X_const_arr) <= n_cols:
        print(
            f"[WARN] Cook's Distance: the OLS design matrix is rank-deficient "
            f"(rank {rank} of {n_cols} columns, {len(X_const_arr)} rows) - values "
            f"are numerically unreliable, not just noisy."
        )
    influence = sm.OLS(residuals, X_const).fit().get_influence()
    return influence.cooks_distance[0]


def _fit_and_get_influence(best_models, X_train, X_test, y_train, y_test,
                           feature_names=None, monotonic_constraints=None, random_state=None):
    """Fit each target's best model once and build its OLS influence object once.

    Both plot_influential_points_per_target() and compute_extended_diagnostics() need
    the same per-target (model fit, residuals, influence) triple; computing it here and
    sharing it avoids fitting every model twice when both are enabled (the default).
    """
    data = {}
    for target, row in best_models.items():
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = np.asarray(y_test[target] - y_pred, dtype=float)

        X_arr = np.asarray(X_test, dtype=float)
        X_const = sm.add_constant(X_arr, has_constant="add")
        n, p = X_arr.shape

        # statsmodels' OLS defaults to a pinv fit, which never crashes on a
        # rank-deficient design (e.g. test n small vs feature count, or
        # collinear features) — but the resulting Cook's Distance/DFFITS/
        # leverage/studentised residuals are then numerically meaningless, not
        # just noisier, with nothing here saying so.
        rank = np.linalg.matrix_rank(X_const)
        if rank < X_const.shape[1] or n <= X_const.shape[1]:
            print(
                f"[WARN] Influence diagnostics for '{target}' ({model_name}): the OLS "
                f"design matrix is rank-deficient (rank {rank} of {X_const.shape[1]} "
                f"columns, {n} test rows) - Cook's Distance/DFFITS/leverage/studentised "
                f"residuals for this target are numerically unreliable, not just noisy."
            )

        influence = sm.OLS(residuals, X_const).fit().get_influence()

        data[target] = {
            "model_name": model_name, "residuals": residuals,
            "X_const": X_const, "influence": influence, "n": n, "p": p,
        }
    return data

def _stem_with_threshold(ax, x, y, threshold, symmetric, title, xlabel, ylabel,
                         log_scale=False, ylim=None):
    """Draw a stem-style plot where points exceeding the threshold are red.

    If symmetric, points are flagged when |y| > threshold and dashed lines are drawn
    at +/- threshold. Otherwise points are flagged when y > threshold (metric is
    non-negative, e.g. Cook's Distance or Leverage) and a single dashed line is drawn.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    flagged = np.abs(y) > threshold if symmetric else y > threshold
    normal  = ~flagged

    ax.vlines(x[normal],  0, y[normal],  colors="steelblue", linewidth=0.6, zorder=2)
    ax.scatter(x[normal], y[normal], s=6, color="steelblue", zorder=3)
    if flagged.any():
        ax.vlines(x[flagged], 0, y[flagged], colors="red", linewidth=0.8, zorder=4)
        ax.scatter(x[flagged], y[flagged], s=12, color="red", zorder=5, label="Exceeds threshold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
              label=f"Threshold = {threshold:.4g}")
    if symmetric:
        ax.axhline(-threshold, color="black", linestyle="--", linewidth=1)
    if log_scale:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="best", fontsize=7)


def _cooks_ylim(cooks_d, threshold):
    """Readable log-scale y-limits for Cook's Distance.

    Near-zero Cook's Distance values (often <1e-10 for well-behaved points) would
    otherwise stretch the axis across many irrelevant orders of magnitude and
    compress the informative range near the threshold into a sliver at the top.
    """
    positive = cooks_d[cooks_d > 0]
    if positive.size == 0:
        return None
    top = max(positive.max(), threshold) * 3
    floor = np.percentile(positive, 5)
    floor = max(floor, top / 1e6)
    floor = min(floor, threshold)
    return floor, top


def plot_influential_points_per_target(best_models, X_train, X_test, y_train, y_test,
                                       influence_data=None, feature_names=None,
                                       monotonic_constraints=None):
    """Return {target: fig} where each fig is a 2x2 panel of influence diagnostics.

    influence_data: optional precomputed output of _fit_and_get_influence(), to avoid
    refitting when the caller (run_postprocessing_analysis) also needs
    compute_extended_diagnostics() for the same models.
    """
    if influence_data is None:
        influence_data = _fit_and_get_influence(best_models, X_train, X_test, y_train, y_test,
                                                 feature_names=feature_names,
                                                 monotonic_constraints=monotonic_constraints)

    figs = {}
    for target, data in influence_data.items():
        model_name = data["model_name"]
        influence  = data["influence"]
        n, p = data["n"], data["p"]
        idx  = np.arange(n)

        cooks_d   = influence.cooks_distance[0]
        dffits    = influence.dffits[0]
        leverage  = influence.hat_matrix_diag
        stud_res  = influence.resid_studentized_external

        cooks_thresh   = 4.0 / n
        dffits_thresh  = 2.0 * np.sqrt(p / n)
        lev_thresh     = 2.0 * (p + 1) / n

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Influential Points Analysis: {model_name} ({target})",
                     fontsize=11, fontweight="bold")

        _stem_with_threshold(
            axes[0, 0], idx, cooks_d,
            threshold=cooks_thresh, symmetric=False,
            title="Cook's Distance",
            xlabel="Test Sample Index", ylabel="Cook's Distance (log scale)",
            log_scale=True, ylim=_cooks_ylim(cooks_d, cooks_thresh),
        )
        _stem_with_threshold(
            axes[0, 1], idx, dffits,
            threshold=dffits_thresh, symmetric=True,
            title="DFFITS",
            xlabel="Test Sample Index", ylabel="DFFITS",
        )
        _stem_with_threshold(
            axes[1, 0], idx, leverage,
            threshold=lev_thresh, symmetric=False,
            title="Leverage",
            xlabel="Test Sample Index", ylabel="Leverage (h)",
        )
        _stem_with_threshold(
            axes[1, 1], idx, stud_res,
            threshold=3.0, symmetric=True,
            title="Studentised Residuals",
            xlabel="Test Sample Index", ylabel="Studentised Residual",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.close(fig)
        figs[target] = fig
    return figs

# Named points on Tukey's ladder of powers, annotated on the AD-vs-lambda curve
# and used as reference labels wherever a fitted lambda is reported. Box-Cox/
# Yeo-Johnson/Tukey's ladder are the same family (x^lambda with continuity
# patches), so these classic named transforms are special cases of the single
# Yeo-Johnson search below rather than separate candidates to try.
YEO_JOHNSON_NAMED_LAMBDAS = {
    -1.0: "Inverse",
    -0.5: "Inverse-sqrt",
    0.0: "Log",
    1 / 3: "Cube-root",
    0.5: "Sqrt",
    1.0: "Identity",
    2.0: "Square",
}

# Default search range for the AD-minimizing lambda: wide enough to cover
# every named point above with margin, narrow enough to keep the curve
# readable and the search well-behaved.
YEO_JOHNSON_LAMBDA_BOUNDS = (-2.0, 3.0)


def yeo_johnson_transform(residuals, lam, eps=1e-6):
    """Closed-form Yeo-Johnson transform (Yeo & Johnson, 2000) for an explicit
    lambda. sklearn's PowerTransformer always fits its own lambda via MLE
    internally and has no way to apply a caller-chosen lambda -- this exists
    so a lambda found by an external search (e.g. minimizing the AD statistic,
    see fit_yeo_johnson_lambda below) can actually be applied, and so the same
    lambda can be re-applied later for plotting without re-fitting."""
    x = np.asarray(residuals, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    if abs(lam) > eps:
        out[pos] = ((x[pos] + 1) ** lam - 1) / lam
    else:
        out[pos] = np.log1p(x[pos])
    if abs(lam - 2) > eps:
        out[neg] = -(((-x[neg]) + 1) ** (2 - lam) - 1) / (2 - lam)
    else:
        out[neg] = -np.log1p(-x[neg])
    return out


def _yeo_johnson_ad_statistic(residuals, lam):
    try:
        transformed = yeo_johnson_transform(residuals, lam)
        if not np.all(np.isfinite(transformed)):
            return np.inf
        return anderson(transformed)[0]
    except Exception:
        return np.inf


def fit_yeo_johnson_lambda(residuals, lam_bounds=YEO_JOHNSON_LAMBDA_BOUNDS):
    """Find the lambda (within lam_bounds) that minimizes the Anderson-Darling
    statistic of the Yeo-Johnson-transformed residuals. A bounded 1-D search
    (Brent's method) is deterministic -- same residuals always give the same
    lambda, no seeding needed."""
    result = minimize_scalar(
        lambda lam: _yeo_johnson_ad_statistic(residuals, lam),
        bounds=lam_bounds, method="bounded",
    )
    return float(result.x)


def sweep_yeo_johnson_ad_curve(residuals, lam_bounds=YEO_JOHNSON_LAMBDA_BOUNDS, n_points=100):
    """Evaluate the AD statistic across an evenly spaced lambda grid, for
    plotting the full curve shape -- separate from fit_yeo_johnson_lambda's
    own optimizer trajectory, which isn't guaranteed to look like a smooth
    curve. Returns (lambdas, ad_statistics)."""
    lams = np.linspace(lam_bounds[0], lam_bounds[1], n_points)
    ad_stats = np.array([_yeo_johnson_ad_statistic(residuals, lam) for lam in lams])
    return lams, ad_stats


# Apply a residual transform to assess normality.
'''
Supported:
      "None":        identity
      "Yeo-Johnson": sign(r)-aware power transform; lambda is fit by
                      minimizing the Anderson-Darling statistic (see
                      fit_yeo_johnson_lambda), not sklearn's default MLE.
                      Subsumes Log/Sqrt/Cube-root/etc as special-case lambdas
                      on the same curve (see YEO_JOHNSON_NAMED_LAMBDAS).
      "Arcsinh":     sign-preserving log alternative, smooth through zero --
                      a genuinely different shape family from Yeo-Johnson,
                      not a special case of it.
'''

SUPPORTED_TRANSFORMS = ("Yeo-Johnson", "Arcsinh")


def apply_transformation(residuals, transformation, lam=None):
    if transformation == "Yeo-Johnson":
        if lam is None:
            lam = fit_yeo_johnson_lambda(np.asarray(residuals, dtype=float))
        return yeo_johnson_transform(residuals, lam)
    elif transformation == "Arcsinh":
        return np.arcsinh(residuals)
    elif transformation == "None":
        return residuals
    # An unknown name used to silently return the input unchanged — so a stale
    # name from an old config (e.g. "Log"/"Sqrt", valid before the Yeo-Johnson
    # consolidation) produced an identity transform mislabelled as the
    # requested one, with metrics identical to "None" and nothing saying why.
    raise ValueError(
        f"Unknown transformation {transformation!r}. Supported: "
        f"{('None',) + SUPPORTED_TRANSFORMS}."
    )

# Score each transform per target with simple normality indicators.
def evaluate_transformations(best_models, X_train, X_test, y_train, y_test, transforms=None,
                             feature_names=None, monotonic_constraints=None, random_state=None):
    """transforms: which named transforms to evaluate in addition to the untransformed
    baseline (always included as "None"). Defaults to all supported transforms.
    Unknown names (e.g. from a config that predates the current transform set)
    are warned about and skipped rather than crashing the whole analysis — or,
    worse, silently masquerading as an applied transform.
    """
    if transforms is None:
        transforms = list(SUPPORTED_TRANSFORMS)
    unknown = [t for t in transforms if t not in SUPPORTED_TRANSFORMS and t != "None"]
    if unknown:
        log_warn(f"Unknown residual transformation(s) skipped: {unknown}. "
                 f"Supported: {list(SUPPORTED_TRANSFORMS)}.")
    transforms_to_try = ["None"] + [t for t in transforms
                                    if t in SUPPORTED_TRANSFORMS and t != "None"]

    results = []
    for target, row in best_models.items():
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = y_test[target] - y_pred

        for t in transforms_to_try:
            try:
                lam = None
                if t == "Yeo-Johnson":
                    lam = fit_yeo_johnson_lambda(np.asarray(residuals, dtype=float))
                transformed = apply_transformation(residuals, t, lam=lam)
                tr_arr = np.asarray(transformed, dtype=float)

                # Every normality test is ALWAYS computed here; which ones the PDF
                # displays is a report-level choice (normality_tests selection). The
                # Excel export always carries the full set. Each test is individually
                # guarded: one failing (e.g. Shapiro-Wilk's 3-observation minimum)
                # must not cost the row its other metrics.

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

                # Shapiro-Wilk: best power at small-to-moderate n (the 2011
                # Razali & Wah comparison's winner, with AD close behind).
                try:
                    sw_p = float(shapiro(tr_arr).pvalue)
                except Exception:
                    sw_p = None

                # Jarque-Bera: tests skewness/kurtosis jointly (asymptotic chi²(2);
                # unreliable below a few dozen samples).
                try:
                    jb_p = float(jarque_bera(tr_arr).pvalue)
                except Exception:
                    jb_p = None

                # D'Agostino's K²: skewness+kurtosis with small-sample corrections
                # (scipy requires n >= 8 for the kurtosis component).
                try:
                    dk_p = float(normaltest(tr_arr).pvalue)
                except Exception:
                    dk_p = None

                results.append({
                    "Target Variable": target,
                    "Model": model_name,
                    "Transformation": t,
                    "Lambda": lam,
                    "Skewness": skew(tr_arr),
                    "Excess Kurtosis": kurtosis(tr_arr, fisher=True),
                    "AD Statistic": anderson(tr_arr)[0],
                    "Shapiro-Wilk p": sw_p,
                    "Lilliefors p": lf_p,
                    "Filiben": filiben,
                    "Jarque-Bera p": jb_p,
                    "D'Agostino p": dk_p,
                })
            except Exception as e:
                results.append({
                    "Target Variable": target,
                    "Model": model_name,
                    "Transformation": t,
                    "Lambda": lam,
                    "Skewness": None,
                    "Excess Kurtosis": None,
                    "AD Statistic": None,
                    "Shapiro-Wilk p": None,
                    "Lilliefors p": None,
                    "Filiben": None,
                    "Jarque-Bera p": None,
                    "D'Agostino p": None,
                    "Error": str(e)
                })

    return pd.DataFrame(results)

# Shapiro-Wilk is the most powerful general-purpose normality test at the
# small-to-moderate sample sizes typical here; used as the parsimony gate below.
NORMALITY_P_THRESHOLD = 0.05


def select_best_transformation_indices(df, group_col="Target Variable", value_col="AD Statistic",
                                       transform_col="Transformation", none_label="None",
                                       p_col="Shapiro-Wilk p", p_threshold=NORMALITY_P_THRESHOLD):
    """Row label of the best transformation per group.

    Prefers leaving residuals untransformed ("None") when they already pass
    the normality check (p > p_threshold) -- no need to transform data that's
    already normal. Otherwise falls back to the transform with the lowest AD
    statistic, same as the original always-transform behaviour. A group that
    is entirely NaN (a degenerate residual target failing every fit) is
    skipped rather than raising.
    """
    valid = df[df[value_col].notna()]
    indices = []
    for _, group in valid.groupby(group_col):
        none_rows = group[group[transform_col] == none_label]
        if not none_rows.empty:
            none_row = none_rows.iloc[0]
            none_p = none_row.get(p_col)
            if none_p is not None and pd.notna(none_p) and none_p > p_threshold:
                indices.append(none_row.name)
                continue
        indices.append(group[value_col].idxmin())
    return pd.Index(indices)

# Plot residual diagnostics using the best transform per target.
# Returns three figures: scatter+histogram (side-by-side), Q-Q.
# "Best" is "None" if residuals already pass a normality check, otherwise the
# transform with the minimum AD statistic (see select_best_transformation_indices).
def plot_all_transformations(results_df, best_models, X_train, X_test, y_train, y_test,
                             feature_names=None, monotonic_constraints=None, random_state=None):
    best_rows = results_df.loc[select_best_transformation_indices(results_df)]
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
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = y_test[target] - y_pred
        # Reuse the lambda already fit in evaluate_transformations rather than
        # re-running the search, so the plot exactly matches the reported row.
        transformed = apply_transformation(residuals, trans, lam=row.get("Lambda"))
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

# Plot the Anderson-Darling statistic across the Yeo-Johnson lambda search
# range, one figure per target (not one multi-panel figure) so the report can
# lay each plot out beside its own reference table. Only meaningful when
# "Yeo-Johnson" was one of the transforms tried.
# Also returns a per-target reference table (named transform, lambda, AD
# statistic at that lambda) supplementing the plot's dotted-line annotations
# with exact numbers, plus a "Yeo-Johnson (optimum)" row per target. Which
# row (if any) represents the transform actually selected as best overall is
# a separate decision (select_best_transformation_indices) left to the
# caller -- this table only reports numbers, it doesn't highlight a winner.
def plot_yeo_johnson_lambda_curve(results_df, best_models, X_train, X_test, y_train, y_test,
                                  feature_names=None, monotonic_constraints=None, random_state=None,
                                  lam_bounds=YEO_JOHNSON_LAMBDA_BOUNDS):
    yj_rows = results_df[results_df["Transformation"] == "Yeo-Johnson"].dropna(subset=["Lambda"])
    targets = list(yj_rows["Target Variable"])
    if not targets:
        return {}, pd.DataFrame()

    figs = {}
    reference_rows = []

    for target in targets:
        row = yj_rows[yj_rows["Target Variable"] == target].iloc[0]
        model_name = _get_model_name(row)
        best_row = best_models[target]
        params = process_hyperparameters(_get_hyperparams(best_row), model_name)
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        residuals = y_test[target] - y_pred
        residuals_arr = np.asarray(residuals, dtype=float)

        lams, ad_stats = sweep_yeo_johnson_ad_curve(residuals_arr, lam_bounds=lam_bounds)
        finite = np.isfinite(ad_stats)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(lams[finite], ad_stats[finite], color="tab:blue", linewidth=1.5, label="AD statistic")

        chosen_lam = float(row["Lambda"])
        chosen_ad = float(row["AD Statistic"]) if pd.notna(row["AD Statistic"]) else None
        if chosen_ad is not None:
            ax.scatter([chosen_lam], [chosen_ad], color="red", s=60, zorder=5,
                      label=f"Chosen λ = {chosen_lam:.3f}")

        for named_lam, label in sorted(YEO_JOHNSON_NAMED_LAMBDAS.items()):
            if lam_bounds[0] <= named_lam <= lam_bounds[1]:
                ax.axvline(named_lam, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.annotate(label, xy=(named_lam, ax.get_ylim()[1]), xytext=(2, -10),
                           textcoords="offset points", rotation=90, fontsize=7,
                           color="dimgray", va="top", ha="left")
                ad_at_named = _yeo_johnson_ad_statistic(residuals_arr, named_lam)
                reference_rows.append({
                    "Target Variable": target,
                    "Transform": label,
                    "Lambda": named_lam,
                    "AD Statistic": ad_at_named if np.isfinite(ad_at_named) else None,
                })
        reference_rows.append({
            "Target Variable": target,
            "Transform": "Yeo-Johnson (optimum)",
            "Lambda": chosen_lam,
            "AD Statistic": chosen_ad,
        })

        ax.set_title(f"Anderson-Darling Statistic vs. Yeo-Johnson λ: {target}", fontweight='bold')
        ax.set_xlabel("λ")
        ax.set_ylabel("AD Statistic (lower is more normal)")
        ax.legend(loc='best')
        ax.grid(True, linewidth=0.4)
        fig.tight_layout()
        figs[target] = fig

    return figs, pd.DataFrame(reference_rows)

# Run cross-validation summary for each target with its best model.
def process_all_targets_with_summary(best_models, X_train, y_train, cv_method, cv_args, scoring_metric,
                                     feature_names=None, monotonic_constraints=None, random_state=None):
    summary = {}
    for target, row in tqdm(best_models.items(), desc="Cross-Validation", unit="target", leave=False):
        tqdm.write(f"  CV for target: {target}")
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        result = perform_cross_validation_with_summary(model, X_train, y_train, target, cv_method, cv_args, scoring_metric)
        summary[target] = result
    return summary

def plot_residuals_with_influential_points_all_targets(best_models, X_train, X_test, y_train, y_test,
                                                        feature_names=None, monotonic_constraints=None,
                                                        random_state=None):
    num_targets = len(best_models)
    fig, axes = plt.subplots(num_targets, 2, figsize=(12, 5 * num_targets), squeeze=False)

    for idx, (target, row) in enumerate(best_models.items()):
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
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
                     edgecolors='k', linewidths=0.5, label="Influential (Cook's)")
        ax_s.axhline(0, color="black", linestyle="--", linewidth=1)
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

def compute_permutation_importance(best_models, X_train, X_test, y_train, y_test,
                                   feature_names, scoring_metric="R^2", n_repeats=10,
                                   monotonic_constraints=None, random_state=None):
    """Compute permutation importance for each target's best model on the test set."""
    # None resolves to seed 0 (package-wide convention) so the shuffles are
    # reproducible by default.
    random_state = 0 if random_state is None else random_state
    _scorer_map = {
        "R^2":   "r2",
        "MSE":   "neg_mean_squared_error",
        "MAE":   "neg_mean_absolute_error",
    }
    if scoring_metric in _scorer_map:
        scorer = _scorer_map[scoring_metric]
    elif scoring_metric in _CUSTOM_SCORER_METRICS:
        # These used to silently fall back to R^2 with no indication anywhere
        # (results or plot title) that a different metric had been requested.
        scorer = _custom_metric_scorer(scoring_metric)
    else:
        scorer = "r2"

    results = {}
    for target, row in best_models.items():
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        model.set_params(**params)
        model.fit(X_train, y_train[target])
        pi = _permutation_importance(
            model, X_test, y_test[target],
            scoring=scorer, n_repeats=n_repeats, random_state=random_state
        )
        results[target] = {
            "importances_mean": pi.importances_mean,
            "importances_std":  pi.importances_std,
            "feature_names":    list(feature_names),
            "scoring_metric":   scoring_metric,
        }
    return results


def plot_permutation_importance(perm_results):
    """Bar chart of permutation importance with ± 1 std error bars, one subplot per target."""
    targets = list(perm_results.keys())
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False)

    for idx, target in enumerate(targets):
        res = perm_results[target]
        means = res["importances_mean"]
        stds  = res["importances_std"]
        names = res["feature_names"]
        order = np.argsort(means)[::-1]

        sorted_means = [means[i] for i in order]
        sorted_stds  = [stds[i]  for i in order]
        sorted_names = [names[i] for i in order]

        ax = axes[idx][0]
        bars = ax.barh(
            sorted_names, sorted_means,
            xerr=sorted_stds,
            color="#E07818", alpha=0.85, ecolor="gray", capsize=4,
        )
        metric_name = res.get("scoring_metric", "R^2")
        ax.set_xlabel(f"Mean decrease in {metric_name}")
        ax.set_title(f"Permutation Importance ({metric_name}): {target}", fontweight="bold")
        ax.invert_yaxis()

        # Extend x-axis to make room for numeric labels
        x_max = max((m + s) for m, s in zip(sorted_means, sorted_stds)) if sorted_means else 1.0
        ax.set_xlim(right=x_max * 1.55)

        for bar, mean_val, std_val in zip(bars, sorted_means, sorted_stds):
            label_x = mean_val + std_val + x_max * 0.025
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{mean_val:.3f} ± {std_val:.3f}",
                va="center", ha="left", fontsize=7, color="#333333",
            )

    plt.tight_layout()
    plt.close(fig)
    return fig


def compute_lofo_importance(best_models, X_train, X_test, y_train, y_test,
                            feature_names, scoring_metric="R^2", monotonic_constraints=None,
                            random_state=None, checkpoint_fn=None):
    """Leave-One-Feature-Out importance: for each feature, retrain without it and
    measure the drop in test-set score against a full-feature-set baseline.

    Unlike permutation importance (which shuffles a feature but leaves its correlated
    substitutes in place, understating importance under multicollinearity), LOFO
    actually removes the feature, so its correlated substitutes have to carry the full
    signal — a stronger, if far more expensive (N+1 fits per target), signal.

    Caveat carried into the report: hyperparameters are reused from `best_models` (tuned
    via HPO on the FULL feature set) and are not re-optimised per feature removed, so
    reported importances may be conservative relative to a fully-refit search.
    """
    from sklearn.metrics import get_scorer
    _scorer_map = {
        "R^2":   "r2",
        "MSE":   "neg_mean_squared_error",
        "MAE":   "neg_mean_absolute_error",
    }
    if scoring_metric in _scorer_map:
        scorer = get_scorer(_scorer_map[scoring_metric])
    elif scoring_metric in _CUSTOM_SCORER_METRICS:
        # These used to silently fall back to R^2 with no indication anywhere
        # (results or plot title) that a different metric had been requested.
        # get_scorer() returns an object callable as scorer(estimator, X, y);
        # _custom_metric_scorer already matches that exact signature.
        scorer = _custom_metric_scorer(scoring_metric)
    else:
        scorer = get_scorer("r2")

    results = {}
    for target, row in best_models.items():
        model_name = _get_model_name(row)
        params = process_hyperparameters(_get_hyperparams(row), model_name)

        X_train_arr = np.asarray(X_train)
        X_test_arr  = np.asarray(X_test)
        target_constraints = (monotonic_constraints or {}).get(target, {})

        baseline_model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                                 monotonic_constraints=target_constraints,
                                                 random_state=random_state)
        baseline_model.set_params(**params)
        baseline_model.fit(X_train_arr, y_train[target])
        baseline_score = scorer(baseline_model, X_test_arr, y_test[target])

        importances = np.zeros(len(feature_names))
        for idx, feat in enumerate(feature_names):
            # Optional cooperative pause/cancel hook (e.g. from the UI): checked once
            # per feature — this loop is the single most expensive part of CV
            # (N+1 full refits), so it's the one place in this module that gets
            # per-iteration (not just per-sub-analysis) granularity.
            if checkpoint_fn is not None:
                checkpoint_fn()
            X_train_reduced = np.delete(X_train_arr, idx, axis=1)
            X_test_reduced  = np.delete(X_test_arr, idx, axis=1)
            reduced_names = [f for i, f in enumerate(feature_names) if i != idx]

            model = reset_model_to_defaults(model_name, feature_names=reduced_names,
                                            monotonic_constraints=target_constraints,
                                            random_state=random_state)
            model.set_params(**params)
            model.fit(X_train_reduced, y_train[target])
            dropped_score = scorer(model, X_test_reduced, y_test[target])
            importances[idx] = baseline_score - dropped_score

        results[target] = {
            "importances": importances,
            "feature_names": list(feature_names),
            "baseline_score": baseline_score,
            "scoring_metric": scoring_metric,
        }
    return results


def plot_lofo_importance(lofo_results):
    """Bar chart of LOFO importance, one subplot per target. No error bars — unlike
    permutation importance, there's no repeated-sampling distribution here, each value
    is a single baseline-minus-dropped score difference."""
    targets = list(lofo_results.keys())
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False)

    for idx, target in enumerate(targets):
        res = lofo_results[target]
        importances = res["importances"]
        names = res["feature_names"]
        order = np.argsort(importances)[::-1]

        sorted_importances = [importances[i] for i in order]
        sorted_names = [names[i] for i in order]

        ax = axes[idx][0]
        bars = ax.barh(sorted_names, sorted_importances, color="#E07818", alpha=0.85)
        metric_name = res.get("scoring_metric", "R^2")
        ax.set_xlabel(f"Score drop ({metric_name}) when feature is removed (baseline − dropped)")
        ax.set_title(f"LOFO Importance ({metric_name}): {target}", fontweight="bold")
        ax.invert_yaxis()

        x_max = max(abs(v) for v in sorted_importances) if sorted_importances else 1.0
        ax.set_xlim(right=x_max * 1.55 if x_max > 0 else 1.0)

        for bar, val in zip(bars, sorted_importances):
            label_x = val + (x_max * 0.025 if val >= 0 else -x_max * 0.025)
            ax.text(
                label_x, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=7, color="#333333",
            )

    plt.tight_layout()
    plt.close(fig)
    return fig


def compute_extended_diagnostics(best_models, X_train, X_test, y_train, y_test,
                                 influence_data=None, feature_names=None,
                                 monotonic_constraints=None):
    """
    For each target's best model, compute influence diagnostics and statistical tests
    on the test-set residuals. Returns a DataFrame with one row per target.

    Metrics:
      - DFFITS Flagged: observations with |DFFITS| > 2*sqrt(p/n)
      - High Leverage: observations with h_ii > 2*(p+1)/n
      - Outliers (|t|>3): externally studentised residuals with |t| > 3
      - Breusch-Pagan: heteroscedasticity test (H0: homoscedastic)
      - White: heteroscedasticity test (H0: homoscedastic)
      - Durbin-Watson: autocorrelation test (near 2 = no autocorrelation)
      - Ljung-Box (lag 10): autocorrelation test (H0: no autocorrelation)

    influence_data: optional precomputed output of _fit_and_get_influence(), to avoid
    refitting when the caller (run_postprocessing_analysis) also needs
    plot_influential_points_per_target() for the same models.
    """
    if influence_data is None:
        influence_data = _fit_and_get_influence(best_models, X_train, X_test, y_train, y_test,
                                                 feature_names=feature_names,
                                                 monotonic_constraints=monotonic_constraints)

    rows = []
    for target, data in influence_data.items():
        model_name = data["model_name"]
        residuals  = data["residuals"]
        X_const    = data["X_const"]
        influence  = data["influence"]
        n, p = data["n"], data["p"]

        # Cook's distance (percentages of test set)
        cooks_d = influence.cooks_distance[0]
        cooks_thresh = 4.0 / n
        pct_cooks = round(100.0 * float(np.sum(cooks_d > cooks_thresh)) / n, 1)

        # Influence measures (as % of test set size)
        dffits_vals = influence.dffits[0]
        dffits_thresh = 2.0 * np.sqrt(p / n)
        pct_dffits = round(100.0 * float(np.sum(np.abs(dffits_vals) > dffits_thresh)) / n, 1)

        leverage = influence.hat_matrix_diag
        leverage_thresh = 2.0 * (p + 1) / n
        pct_leverage = round(100.0 * float(np.sum(leverage > leverage_thresh)) / n, 1)

        stud_resid = influence.resid_studentized_external
        pct_outliers = round(100.0 * float(np.sum(np.abs(stud_resid) > 3)) / n, 1)

        # Breusch-Pagan heteroscedasticity test
        try:
            _, bp_pval, _, _ = het_breuschpagan(residuals, X_const)
            bp_pval = round(float(bp_pval), 4)
        except Exception:
            bp_pval = float("nan")

        # White heteroscedasticity test
        try:
            _, white_pval, _, _ = het_white(residuals, X_const)
            white_pval = round(float(white_pval), 4)
        except Exception:
            white_pval = float("nan")

        # Durbin-Watson autocorrelation statistic
        dw_stat = round(float(durbin_watson(residuals)), 4)

        # Ljung-Box autocorrelation test (lag 10)
        try:
            lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            lb_pval = round(float(lb_result["lb_pvalue"].iloc[0]), 4)
        except Exception:
            lb_pval = float("nan")

        rows.append({
            "Target": target,
            "Model": model_name,
            "Cook's >4/n (%)": pct_cooks,
            "DFFITS (%)": pct_dffits,
            "High Leverage (%)": pct_leverage,
            "Outliers |t|>3 (%)": pct_outliers,
            "BP p-value": bp_pval,
            "White p-value": white_pval,
            "Durbin-Watson": dw_stat,
            "LB p-value (lag 10)": lb_pval,
        })

    return pd.DataFrame(rows)


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
    show_extended_diagnostics=True,
    show_residuals=True,
    show_transformation_plots=True,
    show_permutation_importance=True,
    show_lofo_importance=False,
    feature_names=None,
    image_output_dir: str = "report_images",
    transforms_to_run=None,
    monotonic_constraints=None,
    random_state=None,
    checkpoint_fn=None,
):
    influential_figs = {}
    residuals_fig = None
    fig_r = fig_h = fig_q = None
    lambda_curve_figs = {}
    transformation_results_df = pd.DataFrame()
    lambda_reference_df = pd.DataFrame()

    # Optional cooperative pause/cancel hook (e.g. from the UI). This orchestrator has
    # no single dominant per-model/per-target loop the way HPO/UQ/Interpretability do —
    # it runs several independent sub-analyses in sequence — so the checkpoint is
    # called once before each one, not inside their own internals.
    def _checkpoint():
        if checkpoint_fn is not None:
            checkpoint_fn()

    if show_cv_summary:
        _checkpoint()
        print("\nRunning Cross-Validation Summary...")
        cv_summary = process_all_targets_with_summary(
            best_models,
            X_train,
            y_train,
            cv_method,
            cv_args,
            scoring_metric,
            feature_names=feature_names,
            monotonic_constraints=monotonic_constraints,
            random_state=random_state,
        )
        cv_summary_df = create_cv_summary_df(cv_summary, scoring_metric)
    else:
        cv_summary_df = pd.DataFrame()

    extended_diagnostics_df = None
    if show_cooks_distance or show_extended_diagnostics:
        _checkpoint()
        # Shared across both: same per-target model fit and OLS influence object,
        # computed once regardless of how many of the two are enabled.
        influence_data = _fit_and_get_influence(best_models, X_train, X_test, y_train, y_test,
                                                 feature_names=feature_names,
                                                 monotonic_constraints=monotonic_constraints,
                                                 random_state=random_state)

        if show_cooks_distance:
            print("\nRunning Influential Points Analysis...")
            influential_figs = plot_influential_points_per_target(
                best_models, X_train, X_test, y_train, y_test, influence_data=influence_data
            )

        if show_extended_diagnostics:
            print("\nComputing Extended Regression Diagnostics...")
            extended_diagnostics_df = compute_extended_diagnostics(
                best_models, X_train, X_test, y_train, y_test, influence_data=influence_data
            )

    if show_residuals:
        _checkpoint()
        print("\nRunning Residual Analysis...")
        residuals_fig = plot_residuals_with_influential_points_all_targets(
            best_models, X_train, X_test, y_train, y_test,
            feature_names=feature_names, monotonic_constraints=monotonic_constraints,
            random_state=random_state,
        )

    if show_transformation_plots:
        _checkpoint()
        print("\nEvaluating Residual Transformations...")
        transformation_results_df = evaluate_transformations(
            best_models, X_train, X_test, y_train, y_test, transforms=transforms_to_run,
            feature_names=feature_names, monotonic_constraints=monotonic_constraints,
            random_state=random_state,
        )
        # Core columns only for the console (the raw DataFrame repr truncated
        # badly); every computed metric still goes to the Excel export/report.
        _core_cols = [c for c in ["Target Variable", "Model", "Transformation",
                                   "Lambda", "Skewness", "Excess Kurtosis",
                                   "AD Statistic", "Shapiro-Wilk p"]
                      if c in transformation_results_df.columns]
        _display = transformation_results_df[_core_cols].copy()
        if "Lambda" in _display.columns:
            # Blank, not "nan", for transforms that have no lambda
            _display["Lambda"] = _display["Lambda"].map(
                lambda v: "" if pd.isna(v) else f"{v:.4f}")
        log_table(_display, max_col_width=32)

        print("\nPlotting Transformed Residuals...")
        fig_r, fig_h, fig_q = plot_all_transformations(
            transformation_results_df,
            best_models,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names=feature_names,
            monotonic_constraints=monotonic_constraints,
            random_state=random_state,
        )

        # Only meaningful (and only shown in the report) when Yeo-Johnson was
        # actually one of the transforms evaluated above.
        if "Yeo-Johnson" in transformation_results_df["Transformation"].values:
            print("\nPlotting Yeo-Johnson lambda optimisation curve...")
            lambda_curve_figs, lambda_reference_df = plot_yeo_johnson_lambda_curve(
                transformation_results_df,
                best_models,
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names=feature_names,
                monotonic_constraints=monotonic_constraints,
                random_state=random_state,
            )

    perm_results = {}
    perm_fig = None
    if show_permutation_importance and feature_names is not None:
        _checkpoint()
        print("\nComputing Permutation Importance...")
        perm_results = compute_permutation_importance(
            best_models, X_train, X_test, y_train, y_test,
            feature_names, scoring_metric, monotonic_constraints=monotonic_constraints,
            random_state=random_state,
        )
        perm_fig = plot_permutation_importance(perm_results)

    lofo_results = {}
    lofo_fig = None
    if show_lofo_importance and feature_names is not None:
        _checkpoint()
        print("\nComputing LOFO Importance (this retrains once per feature - slower)...")
        lofo_results = compute_lofo_importance(
            best_models, X_train, X_test, y_train, y_test,
            feature_names, scoring_metric, monotonic_constraints=monotonic_constraints,
            random_state=random_state, checkpoint_fn=checkpoint_fn,
        )
        lofo_fig = plot_lofo_importance(lofo_results)

    return {
        "cv_summary_df": cv_summary_df,
        "scoring_metric": scoring_metric,
        "transformation_df": transformation_results_df,
        "lambda_reference_df": lambda_reference_df,
        "lambda_curve_figs": lambda_curve_figs,
        "influential_figs": influential_figs,
        "extended_diagnostics_df": extended_diagnostics_df,
        "residuals_fig": residuals_fig if show_residuals else None,
        "transformation_figs": {
            "residual": fig_r,
            "histogram": fig_h,
            "qq": fig_q,
        } if show_transformation_plots else {},
        "permutation_importance": perm_results,
        "permutation_importance_fig": perm_fig,
        "lofo_importance": lofo_results,
        "lofo_importance_fig": lofo_fig,
    }


