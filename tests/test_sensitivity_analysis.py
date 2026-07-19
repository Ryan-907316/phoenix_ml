"""Direct unit tests for sensitivity_analysis.py's own public API.

Morris and Sobol are otherwise only exercised indirectly, through
compute_interpretability_metrics()/run_interpretability_analysis() in
test_interpretability.py (pre-built result dicts, or end-to-end). FAST gets
direct coverage here because its wrapper carries two things worth verifying
in isolation:
  - SALib 1.4.8's fast_sampler `seed` kwarg is silently not honoured (verified
    empirically while building this) -- compute_fast_indices works around it
    by seeding/restoring numpy's global RandomState, which needs its own
    reproducibility test rather than trusting SALib's documented API.
  - SALib hard-requires N > 4*M^2 for FAST and raises its own (unlabelled)
    ValueError otherwise -- compute_fast_indices re-raises this through the
    shared require_int_at_least() so the message actually names the
    parameter, matching the input-robustness standard applied everywhere
    else (see test_input_robustness.py).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

from phoenix_ml.sensitivity_analysis import (
    compute_fast_indices,
    compute_sobol_indices,
    plot_sensitivity_indices,
    plot_sobol_fast_comparison,
)


def _fitted_model_and_bounds(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.normal(0, 1, n)})
    y = 3.0 * X["a"] + 0.1 * X["b"]  # "a" clearly dominates output variance
    model = KNeighborsRegressor().fit(X, y)
    return model, X


# ── compute_fast_indices ─────────────────────────────────────────────────────

def test_fast_indices_are_reproducible_with_a_fixed_seed():
    model, X = _fitted_model_and_bounds()
    r1 = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    r2 = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    assert np.array_equal(r1["S1"], r2["S1"])
    assert np.array_equal(r1["ST"], r2["ST"])


def test_fast_indices_differ_with_a_different_seed():
    model, X = _fitted_model_and_bounds()
    r1 = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    r2 = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=1)
    assert not np.array_equal(r1["S1"], r2["S1"])


def test_fast_indices_correctly_rank_the_dominant_feature():
    model, X = _fitted_model_and_bounds()
    result = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    assert result["ST"][0] > result["ST"][1]  # "a" dominates by construction


def test_fast_indices_none_resolves_to_seed_zero_like_every_other_stage():
    model, X = _fitted_model_and_bounds()
    with_none = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=None)
    with_zero = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    assert np.array_equal(with_none["S1"], with_zero["S1"])


def test_global_numpy_state_is_restored_after_the_seeded_workaround():
    """The seed/restore workaround must not leak: numpy's global random state
    after a call must be exactly what it would have been without the call."""
    model, X = _fitted_model_and_bounds()
    np.random.seed(12345)
    expected_next = np.random.uniform()  # what would come next, undisturbed

    np.random.seed(12345)
    compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=99)
    actual_next = np.random.uniform()

    assert expected_next == actual_next


@pytest.mark.parametrize("bad_n", [0, -5, 10, 64, 2.5, "many"])
def test_fast_n_samples_below_or_at_the_salib_floor_raises_named_error(bad_n):
    # SALib requires N > 4*M^2; M=4 by default here, so the floor is 64
    # inclusive -- 64 itself must still raise (strictly greater than).
    model, X = _fitted_model_and_bounds()
    with pytest.raises(ValueError, match="sensitivity_fast_n"):
        compute_fast_indices(model, X, ["a", "b"], n_samples=bad_n, random_state=0)


def test_fast_n_samples_just_above_the_salib_floor_works():
    model, X = _fitted_model_and_bounds()
    result = compute_fast_indices(model, X, ["a", "b"], n_samples=65, random_state=0)
    assert len(result["S1"]) == 2


# ── plot_sobol_fast_comparison ───────────────────────────────────────────────

def test_combined_plot_has_two_panels_matching_s1_and_st():
    model, X = _fitted_model_and_bounds()
    sobol = compute_sobol_indices(model, X, ["a", "b"], n_base=64, random_state=0)
    fast = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    fig = plot_sobol_fast_comparison(sobol, fast, "T", "KNeighbors Regressor")
    assert len(fig.axes) == 2


def test_combined_plot_panels_each_get_their_own_title():
    # Regression test for a real usability gap: the two panels were only
    # distinguishable by reading the x-axis label; each now gets its own title.
    model, X = _fitted_model_and_bounds()
    sobol = compute_sobol_indices(model, X, ["a", "b"], n_base=64, random_state=0)
    fast = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    fig = plot_sobol_fast_comparison(sobol, fast, "T", "KNeighbors Regressor")
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["S1 Index by Feature", "ST Index by Feature"]


def test_combined_plot_labels_every_bar_with_its_value():
    # Matches plot_sensitivity_indices' single-method labelling: a number at
    # the tip of every bar, not just Morris/Sobol/FAST's own standalone plots.
    model, X = _fitted_model_and_bounds()
    sobol = compute_sobol_indices(model, X, ["a", "b"], n_base=64, random_state=0)
    fast = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    fig = plot_sobol_fast_comparison(sobol, fast, "T", "KNeighbors Regressor")
    # 2 features x 2 methods (Sobol + FAST) = 4 value labels per panel.
    for ax in fig.axes:
        assert len(ax.texts) == 4


def test_negative_value_labels_anchor_at_zero_not_the_bars_own_tip():
    """Regression test for a real report-review bug: a slightly negative
    index (sampling noise on a near-zero-effect feature -- entirely normal
    for finite-sample Sobol/FAST estimates) put its label at `v + offset`,
    which for negative v could still land inside or left of the bar instead
    of to its right. Both the single-method plot and the combined Sobol/FAST
    plot must anchor negative labels at the zero baseline instead."""
    results = {"ST": np.array([0.5, -0.02, 0.1]), "feature_names": ["a", "b", "c"]}
    fig = plot_sensitivity_indices(results, "T", "Model", "Sobol")
    label_by_value = {t.get_text(): t.get_position()[0] for t in fig.axes[0].texts}
    assert label_by_value["-0.020"] > 0  # right of the zero baseline
    assert label_by_value["-0.020"] < label_by_value["0.500"]  # not out past the biggest bar

    sobol = {"S1": np.array([0.5]), "ST": np.array([0.6]), "feature_names": ["a"]}
    fast = {"S1": np.array([-0.03]), "ST": np.array([-0.01]), "feature_names": ["a"]}
    fig2 = plot_sobol_fast_comparison(sobol, fast, "T", "Model")
    for ax in fig2.axes:
        xs = sorted(t.get_position()[0] for t in ax.texts)
        assert xs[0] > 0  # the negative FAST label is still right of x=0


def test_combined_plot_sorts_by_sobol_total_order_descending():
    model, X = _fitted_model_and_bounds()
    sobol = compute_sobol_indices(model, X, ["a", "b"], n_base=64, random_state=0)
    fast = compute_fast_indices(model, X, ["a", "b"], n_samples=256, random_state=0)
    fig = plot_sobol_fast_comparison(sobol, fast, "T", "KNeighbors Regressor")
    # "a" has the higher Sobol ST by construction -> must be the first (top,
    # since the axis is inverted) y-tick on either panel.
    ytick_labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert ytick_labels[0] == "a"
