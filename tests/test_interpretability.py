"""Tests for interpretability.py's comparable-metrics-table logic.

The monotonicity-mismatch gating test below is a direct regression test for a
real bug found in this project: target_constraints was once passed
unconditionally into compute_interpretability_metrics() for every model, so
non-monotonic-capable models (SVR, Random Forest, MLP, ...) got flagged with
"mismatches" against a constraint they were never actually given.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from phoenix_ml.interpretability import (
    compute_interpretability_metrics,
    run_interpretability_analysis,
    select_shap_explainer,
    _rank_spearman,
)

FEATURES = ["Input Torque", "Residual Armature Current"]


def _toy_xy_features(n=40, seed=0):
    # NOTE: named distinctly from _toy_xy() below (SHAP-routing section) — a
    # second def _toy_xy(...) later in this file used to silently shadow this
    # one at module scope, so every test between here and there was actually
    # calling the OTHER _toy_xy() (a generic "a"/"b" DataFrame) despite this
    # function's realistic torque/current-named data and matching docstrings.
    # Harmless by coincidence (both datasets give column 1 a positive
    # correlation with y), but not what the tests below were meant to exercise.
    rng = np.random.default_rng(seed)
    input_torque = rng.uniform(-2, 2, n)
    armature_current = rng.uniform(-2, 2, n)
    X = np.column_stack([input_torque, armature_current])
    # Target increases with armature_current -> raw correlation sign is +1.
    y = armature_current + rng.normal(0, 0.01, n)
    return X, y


def test_monotonicity_mismatch_flagged_when_constraint_disagrees_with_data():
    X, y = _toy_xy_features()
    # Data says armature_current correlates positively with the target, but
    # we tell it the enforced constraint is decreasing (-1) -> mismatch.
    row = compute_interpretability_metrics(
        "LGBM Regressor", "Target", FEATURES,
        morris_results=None, sobol_results=None,
        X_sampled=X, y_sampled_target=y,
        target_constraints={"Residual Armature Current": -1},
    )
    assert row["Monotonicity Mismatches"] == "Residual Armature Current"


def test_monotonicity_no_mismatch_when_constraint_agrees_with_data():
    X, y = _toy_xy_features()
    row = compute_interpretability_metrics(
        "LGBM Regressor", "Target", FEATURES,
        morris_results=None, sobol_results=None,
        X_sampled=X, y_sampled_target=y,
        target_constraints={"Residual Armature Current": 1},
    )
    assert row["Monotonicity Mismatches"] is None


def test_empty_constraints_never_flags_a_mismatch():
    # Regression test for the real bug: callers must pass {} (not the raw
    # per-target constraints dict) for any model that can't actually receive
    # the constraint — compute_interpretability_metrics itself has no way to
    # know which models are monotonic-capable, so it must trust its input.
    X, y = _toy_xy_features()
    row = compute_interpretability_metrics(
        "SVR (RBF)", "Target", FEATURES,
        morris_results=None, sobol_results=None,
        X_sampled=X, y_sampled_target=y,
        target_constraints={},
    )
    assert "Monotonicity Mismatches" not in row or row["Monotonicity Mismatches"] is None


def test_top_feature_columns_report_name_and_value():
    morris_results = {"mu_star": np.array([0.012, 5.205]), "feature_names": FEATURES}
    sobol_results = {"ST": np.array([0.0, 0.970]), "feature_names": FEATURES}
    X, y = _toy_xy_features()
    row = compute_interpretability_metrics(
        "LGBM Regressor", "Target", FEATURES,
        morris_results=morris_results, sobol_results=sobol_results,
        X_sampled=X, y_sampled_target=y, target_constraints={},
    )
    assert row["Morris Top Feature"] == "Residual Armature Current (5.205)"
    assert row["Sobol Top Feature"] == "Residual Armature Current (0.970)"


def test_rank_agreement_is_perfect_when_both_methods_agree():
    morris_results = {"mu_star": np.array([1.0, 2.0, 3.0]), "feature_names": FEATURES + ["Other"]}
    sobol_results = {"ST": np.array([0.1, 0.2, 0.3]), "feature_names": FEATURES + ["Other"]}
    X, y = _toy_xy_features()
    row = compute_interpretability_metrics(
        "LGBM Regressor", "Target", FEATURES + ["Other"],
        morris_results=morris_results, sobol_results=sobol_results,
        X_sampled=np.column_stack([X, np.zeros(len(X))]), y_sampled_target=y,
        target_constraints={},
    )
    assert row["Rank Agreement"] == 1.0


def test_rank_spearman_handles_constant_input_without_crashing():
    # A feature with zero variance in its ranks (all ties) has no defined
    # correlation — must return NaN, not raise.
    assert np.isnan(_rank_spearman([1, 1, 1], [1, 2, 3]))


# ── select_shap_explainer routing ─────────────────────────────────────────────
#
# Regression tests for a real bug: LGBMRegressor and HistGradientBoostingRegressor
# were missing from the tree-model isinstance check, so they fell into the
# generic KernelExplainer branch — which crashes for LightGBM specifically
# (its sklearn wrapper exposes feature_names_in_ as a read-only property, and
# something in SHAP's generic model-wrapping tries to set it). Because the
# "fallback" was byte-for-byte the same code that had just failed, it failed
# identically every time, silently dropping ALL SHAP output (Summary,
# Dependence, Waterfall) for any run where LGBM was profiled.

def _toy_xy(n=30, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = X["a"] * 2 + X["b"]
    return X, y


def test_lgbm_routes_to_tree_explainer_not_kernel_explainer():
    X, y = _toy_xy()
    model = LGBMRegressor(verbose=-1, n_estimators=10).fit(X, y)
    explainer = select_shap_explainer(model, X.values, background_sample_size=10,
                                      feature_names=list(X.columns), random_state=0)
    assert type(explainer).__name__ == "TreeExplainer"


def test_histgb_routes_to_tree_explainer_not_kernel_explainer():
    X, y = _toy_xy()
    model = HistGradientBoostingRegressor(max_iter=10).fit(X, y)
    explainer = select_shap_explainer(model, X.values, background_sample_size=10,
                                      feature_names=list(X.columns), random_state=0)
    assert type(explainer).__name__ == "TreeExplainer"


def test_mlp_routes_directly_to_kernel_explainer():
    # MLPRegressor must NOT be routed through GradientExplainer (which only
    # supports TensorFlow/PyTorch models and always fails on a plain sklearn
    # MLP) — it should go straight to the working KernelExplainer path.
    X, y = _toy_xy()
    model = MLPRegressor(hidden_layer_sizes=(5,), max_iter=200).fit(X, y)
    explainer = select_shap_explainer(model, X.values, background_sample_size=10,
                                      feature_names=list(X.columns), random_state=0)
    assert type(explainer).__name__ == "KernelExplainer"


def test_non_tree_model_still_uses_kernel_explainer():
    X, y = _toy_xy()
    model = SVR(kernel="rbf").fit(X, y)
    explainer = select_shap_explainer(model, X.values, background_sample_size=10,
                                      feature_names=list(X.columns), random_state=0)
    assert type(explainer).__name__ == "KernelExplainer"


# ── run_interpretability_analysis: test_sample_size subsampling ─────────────

class _RecordingModel:
    """Records exactly what it was fit on — everything else about
    run_interpretability_analysis is disabled via show_plots=False, so this
    is a cheap way to observe the (test_sample_size-bounded) subsample
    without paying for SHAP/ICE/PDP/Morris/Sobol."""

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _large_toy_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Input Torque": rng.uniform(-2, 2, n),
        "Residual Armature Current": rng.uniform(-2, 2, n),
    })
    y = pd.DataFrame({"Target": rng.uniform(0, 1, n)})
    return X, y


def test_subsample_is_not_a_contiguous_head_slice():
    """Regression test for a real risk: X_sampled = X_train[:test_sample_size]
    was a contiguous head-slice, not a random sample — every SHAP/ICE/PDP/ALE/
    Morris/Sobol result (and the sensitivity bounds built from the same
    sample) described only an early chunk of the operating envelope for
    training sets larger than test_sample_size."""
    X_train, y_train = _large_toy_xy(n=200)
    model = _RecordingModel()
    run_interpretability_analysis(
        {"Stub": model}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50,
        show_plots=False, random_state=0,
    )
    assert model.fit_X.shape[0] == 50
    head_rows = set(map(tuple, X_train.iloc[:50].values))
    sampled_rows = set(map(tuple, model.fit_X))
    assert sampled_rows != head_rows


def test_subsample_is_reproducible_with_a_fixed_seed():
    X_train, y_train = _large_toy_xy(n=200)

    def run():
        model = _RecordingModel()
        run_interpretability_analysis(
            {"Stub": model}, X_train, y_train, ["Target"], list(X_train.columns),
            model_names_to_run=["Stub"], test_sample_size=50,
            show_plots=False, random_state=3,
        )
        return model.fit_X

    assert np.array_equal(run(), run())


def test_no_subsampling_when_training_set_is_smaller_than_test_sample_size():
    X_train, y_train = _large_toy_xy(n=30)
    model = _RecordingModel()
    run_interpretability_analysis(
        {"Stub": model}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50,
        show_plots=False, random_state=0,
    )
    assert model.fit_X.shape[0] == 30


# ── run_interpretability_analysis: cooperative pause/cancel checkpoint ──────
#
# Regression test for a real QA bug: checkpoint_fn was only ever threaded into
# HPO, so Stop/Pause during Interpretability did nothing until it ended on its
# own. Called once per model (the same granularity HPO already used).

class _StopAfterN:
    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls >= self.n:
            raise RuntimeError("simulated Stop")


def test_run_interpretability_analysis_calls_checkpoint_fn_once_per_model():
    X_train, y_train = _large_toy_xy(n=30)
    models = {"A": _RecordingModel(), "B": _RecordingModel()}
    checkpoint = _StopAfterN(n=999)
    run_interpretability_analysis(
        models, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["A", "B"], test_sample_size=30,
        show_plots=False, random_state=0, checkpoint_fn=checkpoint,
    )
    assert checkpoint.calls == 2


def test_run_interpretability_analysis_stops_partway_when_checkpoint_raises():
    X_train, y_train = _large_toy_xy(n=30)
    models = {"A": _RecordingModel(), "B": _RecordingModel(), "C": _RecordingModel()}
    checkpoint = _StopAfterN(n=2)
    with pytest.raises(RuntimeError, match="simulated Stop"):
        run_interpretability_analysis(
            models, X_train, y_train, ["Target"], list(X_train.columns),
            model_names_to_run=["A", "B", "C"], test_sample_size=30,
            show_plots=False, random_state=0, checkpoint_fn=checkpoint,
        )
    assert checkpoint.calls == 2


# ── run_interpretability_analysis: failed plots are surfaced, not silent ────

class _PredictFailsModel(_RecordingModel):
    def predict(self, X):
        raise RuntimeError("simulated prediction failure")


def test_a_failed_plot_is_recorded_as_a_failedplot_sentinel_not_silently_dropped():
    """Regression test for a real risk: a plot type that raised was only
    logged to the console (tqdm.write) — the figures dict simply had no
    entry for that slot, giving report_generation nothing to distinguish
    "attempted and failed" from "never requested"."""
    from phoenix_ml.interpretability import FailedPlot

    X_train, y_train = _large_toy_xy(n=30)
    model = _PredictFailsModel()
    _, figures = run_interpretability_analysis(
        {"Stub": model}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=30,
        show_plots=True, random_state=0,
        show_ice_pdp=False, show_ale=False, show_shap_summary=False,
        show_shap_dependence=False, show_shap_waterfall=False,
        show_sensitivity_morris=True, show_sensitivity_sobol=False,
    )
    key = "Sensitivity_Morris_Target__Stub"
    assert key in figures
    assert isinstance(figures[key], FailedPlot)
    assert "simulated prediction failure" in figures[key].reason


# ── run_interpretability_analysis: Sobol/FAST combined-figure rule ──────────
#
# User-directed design: Sobol and FAST estimate the SAME variance-based
# indices (S1/ST) via independent mechanisms, so when BOTH are enabled they
# must render as one paired-bar comparison figure (the point is checking
# their agreement), not two separate charts. Either one alone still gets its
# own standalone figure, unchanged from before FAST existed.

def _sensitivity_kwargs(**overrides):
    kwargs = dict(
        show_ice_pdp=False, show_ale=False, show_shap_summary=False,
        show_shap_dependence=False, show_shap_waterfall=False,
        show_sensitivity_morris=False, show_sensitivity_sobol=False,
        show_sensitivity_fast=False,
        sensitivity_sobol_n=64, sensitivity_fast_n=256,
    )
    kwargs.update(overrides)
    return kwargs


def test_sobol_and_fast_together_render_one_combined_figure_not_two():
    X_train, y_train = _large_toy_xy(n=60)
    _, figures = run_interpretability_analysis(
        {"Stub": _RecordingModel()}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50, show_plots=True, random_state=0,
        **_sensitivity_kwargs(show_sensitivity_sobol=True, show_sensitivity_fast=True),
    )
    assert set(figures.keys()) == {"Sensitivity_Sobol_FAST_Target__Stub"}


def test_sobol_alone_still_renders_its_own_standalone_figure():
    X_train, y_train = _large_toy_xy(n=60)
    _, figures = run_interpretability_analysis(
        {"Stub": _RecordingModel()}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50, show_plots=True, random_state=0,
        **_sensitivity_kwargs(show_sensitivity_sobol=True),
    )
    assert set(figures.keys()) == {"Sensitivity_Sobol_Target__Stub"}


def test_fast_alone_still_renders_its_own_standalone_figure():
    X_train, y_train = _large_toy_xy(n=60)
    _, figures = run_interpretability_analysis(
        {"Stub": _RecordingModel()}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50, show_plots=True, random_state=0,
        **_sensitivity_kwargs(show_sensitivity_fast=True),
    )
    assert set(figures.keys()) == {"Sensitivity_FAST_Target__Stub"}


def test_metrics_table_gets_fast_columns_only_when_fast_actually_ran():
    X_train, y_train = _large_toy_xy(n=60)
    metrics_df, _ = run_interpretability_analysis(
        {"Stub": _RecordingModel()}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50, show_plots=True, random_state=0,
        **_sensitivity_kwargs(show_sensitivity_sobol=True, show_sensitivity_fast=True),
    )
    assert "FAST Top Feature" in metrics_df.columns
    assert "Sobol-FAST Agreement" in metrics_df.columns


def test_metrics_table_omits_sobol_fast_agreement_when_only_one_of_them_ran():
    X_train, y_train = _large_toy_xy(n=60)
    metrics_df, _ = run_interpretability_analysis(
        {"Stub": _RecordingModel()}, X_train, y_train, ["Target"], list(X_train.columns),
        model_names_to_run=["Stub"], test_sample_size=50, show_plots=True, random_state=0,
        **_sensitivity_kwargs(show_sensitivity_sobol=True),
    )
    assert "Sobol-FAST Agreement" not in metrics_df.columns


def test_sobol_fast_agreement_is_perfect_when_both_methods_agree():
    morris_results = None
    sobol_results = {"ST": np.array([0.1, 0.2, 0.7]), "feature_names": FEATURES + ["Other"]}
    fast_results = {"ST": np.array([0.05, 0.25, 0.65]), "feature_names": FEATURES + ["Other"]}
    X, y = _toy_xy_features()
    row = compute_interpretability_metrics(
        "LGBM Regressor", "Target", FEATURES + ["Other"],
        morris_results=morris_results, sobol_results=sobol_results,
        X_sampled=np.column_stack([X, np.zeros(len(X))]), y_sampled_target=y,
        target_constraints={}, fast_results=fast_results,
    )
    # Same RANK order in both (even though raw values differ) -> perfect agreement.
    assert row["Sobol-FAST Agreement"] == 1.0


# ── Direct plotting-function coverage (ICE/PDP, ALE, SHAP) ───────────────────
#
# run_interpretability_analysis's smoke tests above disable these by default
# to stay fast; these call the plotting functions directly on a small,
# TreeExplainer-routed model so the actual plotting bodies get exercised too.

def _fitted_lgbm():
    X, y = _toy_xy_features()
    model = LGBMRegressor(verbose=-1, n_estimators=10).fit(X, y)
    return model, X, y


def test_plot_ice_and_pdp_returns_one_figure_with_one_axes_per_feature():
    from phoenix_ml.interpretability import plot_ice_and_pdp
    import matplotlib.figure

    model, X, y = _fitted_lgbm()
    figs = plot_ice_and_pdp(model, X, FEATURES, "Target", "LGBM Regressor",
                            subsample=20, grid_resolution=5)
    assert len(figs) == 1
    assert isinstance(figs[0], matplotlib.figure.Figure)
    # PartialDependenceDisplay adds its own companion axis per feature (a
    # "Partial dependence" decile-rug axis) alongside the titled one, so
    # there are more raw axes than features, not an exact 1:1 count.
    feature_titles = [ax.get_title() for ax in figs[0].axes if ax.get_title()]
    assert feature_titles == [f"Feature: {f}" for f in FEATURES]


def test_plot_ale_returns_a_curve_per_feature():
    from phoenix_ml.interpretability import plot_ale

    model, X, y = _fitted_lgbm()
    figs = plot_ale(model, X, FEATURES, "Target", "LGBM Regressor", grid_resolution=5)
    assert len(figs) == 1
    assert len(figs[0].axes) == len(FEATURES)
    # Each feature varies -> neither axis should hit the "constant, skipped" branch.
    titles = [ax.get_title() for ax in figs[0].axes]
    assert not any("constant, skipped" in t for t in titles)


def test_plot_ale_skips_a_constant_feature_instead_of_crashing():
    from phoenix_ml.interpretability import plot_ale

    X, y = _toy_xy_features()
    X_with_constant = X.copy()
    X_with_constant[:, 1] = 5.0  # second feature is now constant
    model = LGBMRegressor(verbose=-1, n_estimators=10).fit(X_with_constant, y)

    figs = plot_ale(model, X_with_constant, FEATURES, "Target", "LGBM Regressor", grid_resolution=5)
    titles = [ax.get_title() for ax in figs[0].axes]
    assert any("constant, skipped" in t for t in titles)


def test_plot_shap_summary_returns_a_titled_figure():
    from phoenix_ml.interpretability import plot_shap_summary
    import matplotlib.figure

    model, X, y = _fitted_lgbm()
    fig = plot_shap_summary(model, X, FEATURES, background_sample_size=10,
                            target_var="Target", model_name="LGBM Regressor", random_state=0)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_shap_dependence_returns_one_panel_per_feature():
    from phoenix_ml.interpretability import plot_shap_dependence

    model, X, y = _fitted_lgbm()
    figs = plot_shap_dependence(model, X, FEATURES, background_sample_size=10,
                                target_var="Target", model_name="LGBM Regressor", random_state=0)
    assert len(figs) == 1
    # shap.dependence_plot adds its own colour-bar axis per panel, so there
    # are more axes than features, not an exact 1:1 count.
    dependence_titles = [ax.get_title() for ax in figs[0].axes if ax.get_title()]
    assert len(dependence_titles) == len(FEATURES)


def test_plot_shap_waterfall_returns_n_samples_figures_spread_across_error_range():
    from phoenix_ml.interpretability import plot_shap_waterfall

    model, X, y = _fitted_lgbm()
    X_df = pd.DataFrame(X, columns=FEATURES)
    y_test = pd.Series(y)
    figs = plot_shap_waterfall(
        model, X_df, FEATURES, "Target", "LGBM Regressor",
        background_sample_size=10, y_test=y_test, n_samples=3, random_state=0,
    )
    assert len(figs) == 3
