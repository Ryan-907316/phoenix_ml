"""Tests for pareto_analysis.py — the dominance rule itself, and
run_pareto_analysis's HPO-vs-default performance selection.

compute_pareto_front cases are all hand-drawable: a handful of
(performance, time) points small enough to check dominance by eye.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from phoenix_ml.pareto_analysis import (
    _auto_log,
    _is_higher_better,
    compute_pareto_front,
    run_pareto_analysis,
    save_pareto_plots,
)


# ── compute_pareto_front ──────────────────────────────────────────────────────

def test_strictly_dominated_point_is_excluded():
    # B (0.5, 2.0) is worse than A (0.9, 1.0) on both axes -> dominated.
    # C (0.95, 3.0) trades time for performance -> non-dominated.
    mask = compute_pareto_front([0.9, 0.5, 0.95], [1.0, 2.0, 3.0], higher_is_better=True)
    assert list(mask) == [True, False, True]


def test_single_point_is_always_non_dominated():
    mask = compute_pareto_front([0.5], [10.0], higher_is_better=True)
    assert list(mask) == [True]


def test_all_identical_points_are_all_non_dominated():
    # Equal on both axes: nothing is strictly better on either, so nothing
    # dominates anything — all points stay on the front.
    mask = compute_pareto_front([0.7, 0.7, 0.7], [1.0, 1.0, 1.0], higher_is_better=True)
    assert list(mask) == [True, True, True]


def test_tie_on_performance_is_broken_by_time():
    # Same performance, but the second point is slower -> dominated.
    mask = compute_pareto_front([0.8, 0.8], [1.0, 5.0], higher_is_better=True)
    assert list(mask) == [True, False]


def test_lower_is_better_metric_flips_the_performance_axis():
    # With MSE-style metrics, SMALLER performance is better: (0.1, 2.0)
    # dominates (0.9, 3.0), while (0.5, 1.0) survives on speed.
    mask = compute_pareto_front([0.1, 0.9, 0.5], [2.0, 3.0, 1.0], higher_is_better=False)
    assert list(mask) == [True, False, True]


def test_float_noise_tie_does_not_flip_dominance():
    """Regression test for a real risk: dominance used exact float comparisons
    with no tolerance — two solutions that are really tied but differ only by
    float noise (e.g. the same value accumulated in a different operation
    order) could be classified as one strictly dominating the other,
    non-deterministically across machines/runs."""
    tiny = 1e-14  # far smaller than a real difference, well within float64 noise
    mask = compute_pareto_front([0.8, 0.8 + tiny], [1.0, 1.0 - tiny], higher_is_better=True)
    assert list(mask) == [True, True]


def test_a_real_difference_still_dominates_despite_the_tolerance():
    # The tolerance must not swallow genuine differences — only float-noise-scale
    # ones. A clearly better point on both axes still dominates.
    mask = compute_pareto_front([0.8, 0.9], [1.0, 0.5], higher_is_better=True)
    assert list(mask) == [False, True]


def test_is_higher_better_classification():
    for metric in ("MSE", "MAE", "RMSE", "NRMSE", "MAPE"):
        assert _is_higher_better(metric) is False
    for metric in ("R^2", "Q^2", "KGE", "ADJUSTED R^2"):
        assert _is_higher_better(metric) is True


def test_is_higher_better_reuses_hpo_lower_is_better_not_a_hand_duplicate():
    """Regression test for a real risk: _is_higher_better hand-duplicated
    hyperparameter_optimisation._HPO_LOWER_IS_BETTER as a separate literal —
    a metric added to one but not the other would silently drift out of
    sync. Now derived from the same set, so this can't happen."""
    from phoenix_ml.hyperparameter_optimisation import _HPO_LOWER_IS_BETTER

    for metric in _HPO_LOWER_IS_BETTER:
        assert _is_higher_better(metric) is False


def test_auto_log_requires_all_positive_and_wide_range():
    # Truthiness, not `is True`: the function returns a numpy bool.
    assert _auto_log([0.001, 1.0])          # ratio 1000
    assert not _auto_log([1.0, 5.0])        # ratio 5 < 10
    assert not _auto_log([1.0])             # single value
    assert not _auto_log([-1.0, 100.0])     # negatives filtered, 1 left


# ── run_pareto_analysis ───────────────────────────────────────────────────────

def _session_metrics():
    """Model A: HPO improved on the default (0.9 > 0.6). Model B: no HPO data,
    must fall back to its default score. Model C: no default entry at all
    (never trained) -> excluded entirely."""
    return {
        "default": {
            "Model A": {"T": {"Q^2": 0.6, "elapsed_time": 1.0}},
            "Model B": {"T": {"Q^2": 0.7, "elapsed_time": 2.0}},
        },
        "random": {
            "Model A": {"T": {"Q^2": 0.9}},
        },
        "hyperopt": {},
        "skopt": {},
    }


def test_pareto_uses_best_hpo_score_with_default_fallback():
    figures = run_pareto_analysis(
        _session_metrics(), ["T"], "Q^2", ["Model A", "Model B", "Model C"])
    assert "T" in figures
    fig = figures["T"]
    # The plotted y-data must be the HPO-improved 0.9 for A and the default
    # 0.7 for B — pull the scatter data back out of the figure to check.
    all_offsets = np.vstack([
        coll.get_offsets() for ax in fig.axes for coll in ax.collections
        if len(coll.get_offsets())
    ])
    plotted_perfs = set(np.round(all_offsets[:, 1], 6))
    assert 0.9 in plotted_perfs      # Model A: HPO beat the default
    assert 0.7 in plotted_perfs      # Model B: default fallback
    assert 0.6 not in plotted_perfs  # Model A's default must NOT be used
    plt.close(fig)


def test_pareto_skips_target_with_fewer_than_two_models():
    metrics = {
        "default": {"Model A": {"T": {"Q^2": 0.6, "elapsed_time": 1.0}}},
        "random": {}, "hyperopt": {}, "skopt": {},
    }
    figures = run_pareto_analysis(metrics, ["T"], "Q^2", ["Model A", "Model B"])
    assert figures == {}


# ── save_pareto_plots: one figure failing must not lose the others ──────────

def _figure_that_fails_to_save():
    # A real matplotlib Figure (plt.close() must still work on it after the
    # failure) whose savefig() is monkeypatched to raise, simulating a
    # disk/backend failure without needing a fake class plt.close() can't handle.
    fig, ax = plt.subplots()

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated disk/backend failure")

    fig.savefig = _raise
    return fig


def test_save_pareto_plots_continues_past_a_single_save_failure(tmp_path, capsys):
    """Regression test for an untested but real risk: one figure's savefig()
    failing (a real, if rare, possibility — disk full, a backend error) must
    not lose the other targets' figures that saved fine."""
    good_fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    figures = {
        "Target A": good_fig,
        "Target B": _figure_that_fails_to_save(),
    }

    paths = save_pareto_plots(figures, str(tmp_path))

    assert list(paths.keys()) == ["Target A"]
    assert os.path.exists(paths["Target A"])
    assert "[Pareto]" in capsys.readouterr().out


def test_save_pareto_plots_sanitizes_target_names_into_filenames(tmp_path):
    fig, ax = plt.subplots()
    paths = save_pareto_plots({"Residual Motor Speed/Torque": fig}, str(tmp_path))
    assert paths["Residual Motor Speed/Torque"].endswith(
        "pareto_residual_motor_speed_torque.png")
    assert os.path.exists(paths["Residual Motor Speed/Torque"])
