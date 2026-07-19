"""Tests for postprocessing.py — CV splitter construction and residual
transformations.

_build_cv silently filters a generic cv_args dict down to what each splitter
accepts; the failure mode being guarded against is a kwarg slipping through to
a splitter that rejects it (crash) or being dropped when it mattered
(silently different CV than the user configured).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    LeavePOut,
    RepeatedKFold,
    ShuffleSplit,
)

from phoenix_ml.postprocessing import (
    _build_cv,
    _fit_and_get_influence,
    _get_hyperparams,
    _get_model_name,
    _pick,
    apply_transformation,
    calculate_cooks_distance,
    compute_extended_diagnostics,
    compute_lofo_importance,
    compute_permutation_importance,
    evaluate_transformations,
    metrics_dict,
    perform_cross_validation_with_summary,
    plot_influential_points_per_target,
    plot_residuals_with_influential_points_all_targets,
    plot_yeo_johnson_lambda_curve,
    run_postprocessing_analysis,
    select_best_transformation_indices,
    YEO_JOHNSON_NAMED_LAMBDAS,
)


def test_q2_zero_variance_target_returns_nan_not_a_crash():
    # Same guard, same regression as model_training.py's copy of this metric —
    # postprocessing.py maintains its own metrics_dict independently.
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    assert np.isnan(metrics_dict["Q^2"](y_true, y_pred))


# ── _build_cv ─────────────────────────────────────────────────────────────────

def test_kfold_random_state_forces_shuffle_on():
    """sklearn's KFold raises at split time if random_state is set while
    shuffle=False — so passing a random_state through MUST also enable
    shuffle, otherwise every seeded K-Fold run would crash."""
    cv = _build_cv("K-Fold", {"n_splits": 3, "random_state": 7})
    assert isinstance(cv, KFold)
    assert cv.shuffle is True
    assert cv.random_state == 7
    # Must actually be usable.
    X = np.arange(24).reshape(12, 2)
    assert len(list(cv.split(X))) == 3


def test_kfold_without_random_state_stays_unshuffled():
    cv = _build_cv("K-Fold", {"n_splits": 4})
    assert cv.shuffle is False


def test_loo_ignores_all_args():
    # LeaveOneOut takes no constructor args — any leftover generic cv_args
    # (n_splits, random_state, ...) must be dropped, not passed through.
    cv = _build_cv("LOO", {"n_splits": 10, "random_state": 0})
    assert isinstance(cv, LeaveOneOut)


def test_lpo_reads_p_with_n_splits_fallback():
    cv = _build_cv("LpO", {"p": 3})
    assert isinstance(cv, LeavePOut)
    assert cv.p == 3
    # The UI's generic dialog only exposes n_splits — it doubles as p.
    cv = _build_cv("LpO", {"n_splits": 4})
    assert cv.p == 4


def test_lpo_refuses_combinatorially_explosive_split_counts():
    """Regression test for a real bug: LeavePOut produces C(n_train, p) splits,
    each a full model refit — even the default p=2 on a few-hundred-row
    training set means tens of thousands of refits, an easy way to hang the
    app for hours with zero warning. n_train=500, p=3 -> C(500,3) ~= 20.7M."""
    with pytest.raises(ValueError, match="Leave-P-Out"):
        _build_cv("LpO", {"p": 3}, n_train=500)


def test_lpo_allows_small_split_counts():
    # p=2 on a small training set is a legitimate, cheap configuration and
    # must not be blocked by the same guard.
    cv = _build_cv("LpO", {"p": 2}, n_train=10)  # C(10,2) = 45
    assert isinstance(cv, LeavePOut)
    assert cv.p == 2


def test_lpo_cap_is_a_noop_without_n_train():
    # Callers that don't know the training set size yet (or are just
    # constructing the splitter for inspection) must not be blocked.
    cv = _build_cv("LpO", {"p": 3})
    assert isinstance(cv, LeavePOut)


def test_shuffle_split_and_repeated_kfold_keep_their_own_params():
    cv = _build_cv("Shuffle Split", {"n_splits": 5, "test_size": 0.25,
                                     "random_state": 1, "n_repeats": 99})
    assert isinstance(cv, ShuffleSplit)
    assert cv.n_splits == 5 and cv.test_size == 0.25
    # n_repeats is not a ShuffleSplit parameter and must have been stripped.

    cv = _build_cv("Repeated K-Fold", {"n_splits": 3, "n_repeats": 2,
                                       "random_state": 1, "test_size": 0.5})
    assert isinstance(cv, RepeatedKFold)
    # test_size is not a RepeatedKFold parameter and must have been stripped.


# ── apply_transformation ──────────────────────────────────────────────────────

def _residuals(values):
    return pd.Series(np.asarray(values, dtype=float))


def test_arcsinh_is_sign_preserving_and_finite():
    res = _residuals([-4.0, -1.0, 0.0, 1.0, 4.0])
    out = np.asarray(apply_transformation(res, "Arcsinh"), dtype=float)
    assert np.all(np.isfinite(out))
    nonzero = np.asarray(res) != 0
    assert np.all(np.sign(out[nonzero]) == np.sign(np.asarray(res)[nonzero]))


def test_yeo_johnson_handles_zeros_and_negatives_without_shift():
    # Yeo-Johnson's whole reason for inclusion is native support for zero and
    # negative values — no shift, no NaN. lam=None triggers the AD-minimizing
    # search internally.
    res = _residuals([-2.0, 0.0, 0.0, 1.0, 3.0])
    out = apply_transformation(res, "Yeo-Johnson")
    assert np.all(np.isfinite(out))
    assert len(out) == len(res)


def test_yeo_johnson_transform_matches_sklearns_reference_at_fixed_lambda():
    """yeo_johnson_transform is hand-rolled (sklearn's PowerTransformer always
    fits its own lambda internally and has no way to apply a caller-chosen one)
    -- it must match scipy's independent implementation exactly at any given
    lambda, across both the positive and negative branches. Compared against
    scipy rather than sklearn's PowerTransformer because scipy.stats.yeojohnson
    accepts an explicit lambda via its public API, while sklearn only exposes
    this through a private method whose name has already changed across
    sklearn versions."""
    from scipy.stats import yeojohnson
    from phoenix_ml.postprocessing import yeo_johnson_transform

    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, 100) + rng.normal(0, 1, 100) ** 3  # signed, skewed

    for lam in (-1.5, -0.5, 0.0, 1 / 3, 0.5, 1.0, 1.7, 2.0, 2.8):
        mine = yeo_johnson_transform(x, lam)
        reference = yeojohnson(x, lmbda=lam)
        assert np.allclose(mine, reference), f"mismatch at lambda={lam}"


def test_yeo_johnson_transform_is_identity_at_lambda_one():
    res = _residuals([-4.0, -1.0, 0.0, 1.0, 4.0])
    from phoenix_ml.postprocessing import yeo_johnson_transform
    out = yeo_johnson_transform(res, lam=1.0)
    assert np.allclose(out, res)


def test_fit_yeo_johnson_lambda_is_deterministic():
    """The bounded 1-D search is a deterministic optimizer (Brent's method,
    no randomness) -- identical residuals must always yield the identical
    lambda, run to run."""
    from phoenix_ml.postprocessing import fit_yeo_johnson_lambda
    rng = np.random.default_rng(1)
    res = rng.normal(0, 2, 60) + rng.exponential(1, 60)
    a = fit_yeo_johnson_lambda(res)
    b = fit_yeo_johnson_lambda(res)
    assert a == b


def test_apply_transformation_reuses_a_supplied_lambda_instead_of_refitting():
    """evaluate_transformations fits lambda once and stores it; downstream
    consumers (the diagnostic plots) must be able to pass that exact lambda
    back in and get the identical transform, without re-running the search."""
    res = _residuals([-3.2, -0.5, 0.0, 1.4, 2.8, 5.5])
    out_auto = np.asarray(apply_transformation(res, "Yeo-Johnson"), dtype=float)
    from phoenix_ml.postprocessing import fit_yeo_johnson_lambda
    lam = fit_yeo_johnson_lambda(np.asarray(res, dtype=float))
    out_explicit = np.asarray(apply_transformation(res, "Yeo-Johnson", lam=lam), dtype=float)
    assert np.allclose(out_auto, out_explicit)


def test_sweep_yeo_johnson_ad_curve_returns_a_point_per_lambda():
    from phoenix_ml.postprocessing import sweep_yeo_johnson_ad_curve
    rng = np.random.default_rng(2)
    res = rng.normal(0, 1, 50) ** 3
    lams, ad_stats = sweep_yeo_johnson_ad_curve(res, lam_bounds=(-2.0, 3.0), n_points=25)
    assert len(lams) == len(ad_stats) == 25
    assert lams[0] == -2.0 and lams[-1] == 3.0
    assert np.all(np.isfinite(ad_stats))


def test_unknown_transformation_returns_input_unchanged():
    res = _residuals([1.0, -2.0])
    out = apply_transformation(res, "None")
    assert list(out) == [1.0, -2.0]


# ── plot_yeo_johnson_lambda_curve ──────────────────────────────────────────────

def test_plot_yeo_johnson_lambda_curve_returns_none_without_yeo_johnson_rows():
    """Regression test for the report-visibility rule: the curve must only
    appear when Yeo-Johnson was actually one of the evaluated transforms --
    if evaluate_transformations was run with only Arcsinh, there's no lambda
    to plot, and this must return ({}, empty df) rather than raise."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    results_df = evaluate_transformations(
        best_models, np.asarray(X_train), np.asarray(X_test), y_train, y_test,
        transforms=["Arcsinh"], feature_names=feats, random_state=0)
    figs, reference_df = plot_yeo_johnson_lambda_curve(
        results_df, best_models, np.asarray(X_train), np.asarray(X_test), y_train, y_test,
        feature_names=feats, random_state=0)
    assert figs == {}
    assert reference_df.empty


def test_plot_yeo_johnson_lambda_curve_returns_one_figure_per_target():
    """One standalone single-axis figure per target (not one multi-panel
    figure) -- the report lays each target's plot out beside its own
    reference table, which requires a separate image per target."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    results_df = evaluate_transformations(
        best_models, np.asarray(X_train), np.asarray(X_test), y_train, y_test,
        transforms=["Yeo-Johnson"], feature_names=feats, random_state=0)
    figs, reference_df = plot_yeo_johnson_lambda_curve(
        results_df, best_models, np.asarray(X_train), np.asarray(X_test), y_train, y_test,
        feature_names=feats, random_state=0)
    assert set(figs.keys()) == set(best_models.keys())
    for fig in figs.values():
        assert len(fig.axes) == 1

    # One row per named ladder point within bounds, plus a "Yeo-Johnson
    # (optimum)" row, for each target.
    for target in best_models:
        target_rows = reference_df[reference_df["Target Variable"] == target]
        assert "Yeo-Johnson (optimum)" in target_rows["Transform"].values
        assert set(YEO_JOHNSON_NAMED_LAMBDAS.values()) <= set(target_rows["Transform"].values)
        assert target_rows["AD Statistic"].notna().all()


# ── select_best_transformation_indices (parsimony/normality gate) ─────────────

def _transform_rows(rows):
    """rows: list of (target, transformation, ad_statistic, shapiro_p)."""
    return pd.DataFrame([
        {"Target Variable": t, "Transformation": tr, "AD Statistic": ad, "Shapiro-Wilk p": sw}
        for t, tr, ad, sw in rows
    ])


def test_select_best_prefers_none_when_residuals_already_pass_normality():
    """Regression test: selection must not always chase the lowest AD statistic
    -- if the untransformed residuals already look normal (Shapiro-Wilk p >
    0.05), transforming anyway is unnecessary complexity, even if some other
    transform happens to score a lower AD statistic."""
    df = _transform_rows([
        ("T", "None", 0.5, 0.90),   # already normal, but not the lowest AD
        ("T", "Log", 0.1, 0.01),    # lowest AD, but not actually normal
    ])
    best_idx = select_best_transformation_indices(df)
    assert list(df.loc[best_idx, "Transformation"]) == ["None"]


def test_select_best_falls_back_to_lowest_ad_when_none_fails_normality():
    df = _transform_rows([
        ("T", "None", 0.9, 0.01),   # fails normality
        ("T", "Log", 0.5, 0.20),
        ("T", "Sqrt", 0.1, 0.30),   # lowest AD among the real candidates
    ])
    best_idx = select_best_transformation_indices(df)
    assert list(df.loc[best_idx, "Transformation"]) == ["Sqrt"]


def test_select_best_handles_multiple_targets_independently():
    df = _transform_rows([
        ("A", "None", 0.5, 0.90),
        ("A", "Log", 0.1, 0.01),
        ("B", "None", 0.9, 0.01),
        ("B", "Sqrt", 0.2, 0.30),
    ])
    best_idx = select_best_transformation_indices(df)
    best = df.loc[best_idx].set_index("Target Variable")["Transformation"]
    assert best["A"] == "None"
    assert best["B"] == "Sqrt"


def test_select_best_skips_a_group_that_is_entirely_nan():
    """A degenerate residual target failing every transform's AD fit must not
    take down selection for the other, healthy targets."""
    df = _transform_rows([
        ("A", "None", np.nan, None),
        ("A", "Log", np.nan, None),
        ("B", "None", 0.9, 0.01),
        ("B", "Sqrt", 0.2, 0.30),
    ])
    best_idx = select_best_transformation_indices(df)
    best = df.loc[best_idx]
    assert list(best["Target Variable"]) == ["B"]


# ── CV scorer sign correctness ────────────────────────────────────────────────
#
# cross_val_score internally works with higher-is-better scores, so error
# metrics run through negated scorers (sklearn's neg_* for MSE/MAE, hand-rolled
# negation for NRMSE/MAPE) and must be flipped back positive before reporting,
# while Q²/KGE must come through unflipped. A sign bug here would report a
# near-perfect model's MSE as negative, or its Q² as ≈ -1.

def _cv_score(scoring_metric):
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 10, 40)                       # >= 1 keeps MAPE stable
    y = 3.0 * x + rng.normal(0, 0.05, 40)            # near-perfect linear fit
    X_train = pd.DataFrame({"x": x})
    y_train = pd.DataFrame({"T": y})
    result = perform_cross_validation_with_summary(
        LinearRegression(), X_train, y_train, "T",
        "Shuffle Split", {"n_splits": 4, "test_size": 0.25, "random_state": 0},
        scoring_metric,
    )
    return result["mean_score"]


def test_error_metrics_are_reported_positive_after_cv():
    # Imperfect predictions -> strictly positive error, never the raw
    # negated score cross_val_score works with internally.
    assert 0.0 < _cv_score("MSE") < 0.05
    assert 0.0 < _cv_score("MAE") < 0.2
    assert 0.0 < _cv_score("MAPE") < 5.0     # reported as a percentage


def test_goodness_metrics_are_not_accidentally_negated_by_cv():
    # A near-perfect model must report Q²/KGE near +1 — a sign slip in the
    # custom scorer would put these near -1 instead.
    assert _cv_score("Q^2") > 0.99
    assert _cv_score("KGE") > 0.9


# ── permutation / LOFO importance ─────────────────────────────────────────────

def _importance_inputs(n=40, seed=0):
    """y depends only on 'signal'; 'noise' is irrelevant — the informative
    feature must dominate both importance measures."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"signal": rng.uniform(0, 10, n), "noise": rng.normal(0, 1, n)})
    y = pd.DataFrame({"T": 3.0 * X["signal"] + rng.normal(0, 0.5, n)})
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train, y_test = y.iloc[:30], y.iloc[30:]
    # best_models rows arrive as pandas Series in the real workflow (rows of
    # the best-models DataFrame) — _get_model_name/_get_hyperparams look
    # fields up via row.index, so a plain dict is not a valid input here.
    best_models = {"T": pd.Series({"model_name": "Random Forest Regressor",
                                   "hyperparameters": {"n_estimators": 10}})}
    return best_models, X_train, X_test, y_train, y_test, ["signal", "noise"]


def test_permutation_importance_is_reproducible_with_a_fixed_seed():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()

    def run():
        return compute_permutation_importance(
            best_models, X_train, X_test, y_train, y_test, feats,
            scoring_metric="R^2", n_repeats=5, random_state=3)

    a, b = run(), run()
    assert np.array_equal(a["T"]["importances_mean"], b["T"]["importances_mean"])
    assert np.array_equal(a["T"]["importances_std"], b["T"]["importances_std"])


def test_permutation_importance_is_reproducible_even_without_a_seed():
    """random_state=None resolves to seed 0 (package-wide convention).
    Regression test for the earlier inconsistency where this module passed
    None straight through to the model and the shuffler, making unseeded
    results non-reproducible while HPO/UQ resolved None deterministically."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()

    def run():
        return compute_permutation_importance(
            best_models, X_train, X_test, y_train, y_test, feats,
            scoring_metric="R^2", n_repeats=5, random_state=None)

    a, b = run(), run()
    assert np.array_equal(a["T"]["importances_mean"], b["T"]["importances_mean"])


def test_evaluate_transformations_computes_every_normality_test():
    """All five normality tests (plus AD, skewness, kurtosis) are ALWAYS
    computed for every transformation row — the report's Normality Test
    Metrics checkboxes only control display, and the Excel export always
    carries the full set. Anderson-Darling must be populated since it selects
    the best transformation."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    df = evaluate_transformations(
        best_models, np.asarray(X_train), np.asarray(X_test), y_train, y_test,
        transforms=["Yeo-Johnson"], feature_names=feats, random_state=0)

    for col in ["AD Statistic", "Shapiro-Wilk p", "Lilliefors p", "Filiben",
                "Jarque-Bera p", "D'Agostino p"]:
        assert col in df.columns, f"missing column: {col}"
    # The untransformed baseline row ("None") on a healthy residual sample must
    # produce real values for every test (10 test rows > every test's minimum n).
    baseline = df[df["Transformation"] == "None"].iloc[0]
    for col in ["AD Statistic", "Shapiro-Wilk p", "Lilliefors p", "Filiben",
                "Jarque-Bera p", "D'Agostino p"]:
        assert pd.notna(baseline[col]), f"{col} unexpectedly NaN on the baseline row"


def test_permutation_importance_supports_custom_scoring_metrics():
    """Regression test for a real bug: permutation importance's scorer map only
    covered R^2/MSE/MAE and silently substituted plain R^2 for every other
    scoring metric offered elsewhere in the app (Q^2, Adjusted R^2, NRMSE, MAPE,
    KGE) — with nothing in the results or plot title indicating the substitution
    had happened. Must now actually use the requested metric and record it."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    result = compute_permutation_importance(
        best_models, X_train, X_test, y_train, y_test, feats,
        scoring_metric="NRMSE", n_repeats=5, random_state=3)
    assert result["T"]["scoring_metric"] == "NRMSE"
    # The informative feature must still dominate under a different metric.
    imp = dict(zip(result["T"]["feature_names"], result["T"]["importances_mean"]))
    assert imp["signal"] > imp["noise"]


def test_lofo_importance_is_reproducible_and_ranks_the_signal_feature_first():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()

    def run():
        return compute_lofo_importance(
            best_models, X_train, X_test, y_train, y_test, feats,
            scoring_metric="R^2", random_state=3)

    a, b = run(), run()
    assert np.array_equal(a["T"]["importances"], b["T"]["importances"])
    # Dropping the only informative feature must cost far more score than
    # dropping the pure-noise one.
    imp = dict(zip(a["T"]["feature_names"], a["T"]["importances"]))
    assert imp["signal"] > imp["noise"]
    assert imp["signal"] > 0.1


def test_lofo_importance_supports_custom_scoring_metrics():
    """Same regression as permutation importance's custom-metric fallback bug,
    for LOFO's independent (and differently-shaped) scorer lookup."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    result = compute_lofo_importance(
        best_models, X_train, X_test, y_train, y_test, feats,
        scoring_metric="KGE", random_state=3)
    assert result["T"]["scoring_metric"] == "KGE"
    imp = dict(zip(result["T"]["feature_names"], result["T"]["importances"]))
    assert imp["signal"] > imp["noise"]


# ── OLS influence diagnostics: rank-deficiency warning ───────────────────────
#
# Regression tests for a real risk: sm.OLS(...).fit() defaults to a pinv fit,
# which never crashes on a rank-deficient design (test n small vs feature
# count, or collinear features) — but Cook's Distance/DFFITS/leverage/
# studentised residuals are then numerically meaningless, not just noisier,
# with nothing saying so.

def test_calculate_cooks_distance_warns_on_rank_deficient_design(capsys):
    # 2 rows, 3 features -> with the intercept column, 4 columns but only 2
    # rows: guaranteed rank-deficient.
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    residuals = np.array([0.1, -0.1])
    calculate_cooks_distance(X, residuals)
    out = capsys.readouterr().out
    assert "[WARN]" in out and "rank-deficient" in out


def test_calculate_cooks_distance_no_warning_for_a_well_posed_design(capsys):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 2))
    residuals = rng.normal(0, 1, 30)
    calculate_cooks_distance(X, residuals)
    assert "[WARN]" not in capsys.readouterr().out


def test_fit_and_get_influence_warns_on_rank_deficient_test_set(capsys):
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    # 2 features + intercept = 3 columns, shrunk to 2 test rows -> rank-deficient.
    X_test_small = X_test.iloc[:2]
    y_test_small = y_test.iloc[:2]
    _fit_and_get_influence(best_models, X_train, X_test_small, y_train, y_test_small)
    out = capsys.readouterr().out
    assert "[WARN]" in out and "rank-deficient" in out


def test_fit_and_get_influence_no_warning_for_a_well_posed_test_set(capsys):
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    _fit_and_get_influence(best_models, X_train, X_test, y_train, y_test)
    assert "[WARN]" not in capsys.readouterr().out


# ── _pick / _get_model_name / _get_hyperparams: column-naming fallback ──────

def test_pick_falls_back_to_the_second_candidate_name():
    row = pd.Series({"Model": "SVR (RBF)"})
    assert _pick(row, "model_name", "Model") == "SVR (RBF)"


def test_pick_raises_when_no_candidate_is_present():
    row = pd.Series({"unrelated": 1})
    with pytest.raises(KeyError, match="model_name"):
        _pick(row, "model_name", "Model")


def test_get_model_name_accepts_either_column_naming_convention():
    assert _get_model_name(pd.Series({"model_name": "A"})) == "A"
    assert _get_model_name(pd.Series({"Model": "B"})) == "B"


def test_get_hyperparams_accepts_either_column_naming_convention():
    assert _get_hyperparams(pd.Series({"hyperparameters": {"C": 1}})) == {"C": 1}
    assert _get_hyperparams(pd.Series({"Value of Hyperparameters": {"C": 2}})) == {"C": 2}


def test_get_hyperparams_parses_a_stringified_dict():
    row = pd.Series({"hyperparameters": "{'n_estimators': 5}"})
    assert _get_hyperparams(row) == {"n_estimators": 5}


def test_get_hyperparams_treats_nan_as_empty_dict():
    row = pd.Series({"hyperparameters": float("nan")})
    assert _get_hyperparams(row) == {}


# ── Influence/residual plotting + the big orchestrator ──────────────────────

def test_plot_influential_points_per_target_returns_one_figure_per_target():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    figs = plot_influential_points_per_target(best_models, X_train, X_test, y_train, y_test)
    assert set(figs.keys()) == {"T"}
    import matplotlib.figure
    assert isinstance(figs["T"], matplotlib.figure.Figure)


def test_compute_extended_diagnostics_returns_one_row_per_target_with_every_test_column():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    df = compute_extended_diagnostics(best_models, X_train, X_test, y_train, y_test)
    assert len(df) == 1
    assert df.iloc[0]["Target"] == "T"
    for col in ["Cook's >4/n (%)", "DFFITS (%)", "High Leverage (%)",
                "Outliers |t|>3 (%)", "BP p-value", "White p-value",
                "Durbin-Watson", "LB p-value (lag 10)"]:
        assert col in df.columns


def test_plot_residuals_with_influential_points_returns_a_figure_with_a_row_per_target():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    fig = plot_residuals_with_influential_points_all_targets(
        best_models, X_train, X_test, y_train, y_test)
    # One row of (scatter, histogram) axes per target.
    assert len(fig.axes) == 2 * len(best_models)


def test_run_postprocessing_analysis_orchestrates_every_sub_analysis():
    """Integration test for the module's big orchestrator — locks in that
    every sub-result it's supposed to assemble (CV, influence figs/diagnostics,
    residuals fig, transformation df/figs, permutation AND LOFO importance)
    actually comes back populated for a realistic small config, not just that
    the function returns without crashing."""
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    result = run_postprocessing_analysis(
        best_models, X_train, X_test, y_train, y_test,
        cv_method="Shuffle Split", cv_args={"n_splits": 3, "test_size": 0.25, "random_state": 0},
        scoring_metric="R^2",
        show_lofo_importance=True,
        feature_names=feats,
        transforms_to_run=["Yeo-Johnson"],
        random_state=0,
    )
    assert not result["cv_summary_df"].empty
    assert set(result["influential_figs"].keys()) == {"T"}
    assert len(result["extended_diagnostics_df"]) == 1
    assert result["residuals_fig"] is not None
    assert not result["transformation_df"].empty
    assert "Lambda" in result["transformation_df"].columns
    assert result["transformation_figs"]["residual"] is not None
    # Yeo-Johnson was requested, so the lambda-optimisation curve (and its
    # reference table) must be produced too (gated on Yeo-Johnson actually
    # having been evaluated).
    assert result["lambda_curve_figs"]["T"] is not None
    assert not result["lambda_reference_df"].empty
    assert "T" in result["permutation_importance"]
    assert result["permutation_importance_fig"] is not None
    assert "T" in result["lofo_importance"]
    assert result["lofo_importance_fig"] is not None


# ── cooperative pause/cancel checkpoint ──────────────────────────────────────
#
# Regression test for a real QA bug: checkpoint_fn was only ever threaded into
# HPO, so Stop/Pause during CV/Postprocessing did nothing until it ended on its
# own. run_postprocessing_analysis has no single dominant per-model/per-target
# loop the way HPO/UQ/Interpretability do — it runs several independent
# sub-analyses in sequence — so it's checked once before each of those (6 here:
# CV summary, influence diagnostics, residuals, transformations, permutation,
# LOFO), plus once per feature inside LOFO specifically (the single most
# expensive part: N+1 full refits).

class _StopAfterN:
    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls >= self.n:
            raise RuntimeError("simulated Stop")


def test_run_postprocessing_analysis_checkpoints_every_sub_phase_and_lofo_feature():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    checkpoint = _StopAfterN(n=999)
    run_postprocessing_analysis(
        best_models, X_train, X_test, y_train, y_test,
        cv_method="Shuffle Split", cv_args={"n_splits": 3, "test_size": 0.25, "random_state": 0},
        scoring_metric="R^2", show_lofo_importance=True, feature_names=feats,
        transforms_to_run=["Yeo-Johnson"], random_state=0, checkpoint_fn=checkpoint,
    )
    # 6 sub-phases (CV, influence, residuals, transformations, permutation, LOFO)
    # + 1 per LOFO feature (2 features here).
    assert checkpoint.calls == 8


def test_run_postprocessing_analysis_stops_partway_when_checkpoint_raises():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    checkpoint = _StopAfterN(n=3)
    with pytest.raises(RuntimeError, match="simulated Stop"):
        run_postprocessing_analysis(
            best_models, X_train, X_test, y_train, y_test,
            cv_method="Shuffle Split", cv_args={"n_splits": 3, "test_size": 0.25, "random_state": 0},
            scoring_metric="R^2", show_lofo_importance=True, feature_names=feats,
            transforms_to_run=["Yeo-Johnson"], random_state=0, checkpoint_fn=checkpoint,
        )
    assert checkpoint.calls == 3


def test_compute_lofo_importance_checkpoints_once_per_feature():
    best_models, X_train, X_test, y_train, y_test, feats = _importance_inputs()
    checkpoint = _StopAfterN(n=999)
    compute_lofo_importance(
        best_models, X_train, X_test, y_train, y_test, feats,
        random_state=0, checkpoint_fn=checkpoint,
    )
    assert checkpoint.calls == len(feats)
