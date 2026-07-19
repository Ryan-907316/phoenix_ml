"""Pure-function tests for model_training.py's constraint/seed helpers.

These cover exactly the kind of logic bug that's bitten this project before:
build_monotone_constraints_kwarg's job is to say "no" for every model type
that can't accept a monotonicity constraint, and it's easy for a caller
elsewhere to forget that and apply a constraint (or flag a mismatch) for a
model that was never actually constrained.
"""
import numpy as np
import pytest
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

import pandas as pd
from sklearn.linear_model import LinearRegression

from phoenix_ml.model_training import (
    apply_monotone_constraints_for_target,
    build_monotone_constraints_kwarg,
    derive_seed,
    metrics_dict,
    reset_model_to_defaults,
    run_models,
    train_and_evaluate_model,
    _warn_if_large_dataset_for_model,
)

FEATURES = ["Input Torque", "Residual Armature Current"]


def test_lgbm_receives_constraint_as_list():
    kwargs = build_monotone_constraints_kwarg(
        "LGBM Regressor", FEATURES, {"Residual Armature Current": 1}
    )
    assert kwargs == {"monotone_constraints": [0, 1]}


def test_xgboost_receives_constraint_as_tuple():
    # XGBoost's constraint kwarg must be a tuple, not a list (model_training.py
    # applies this distinction deliberately) — pin it down explicitly.
    kwargs = build_monotone_constraints_kwarg(
        "XGBoost Regressor", FEATURES, {"Residual Armature Current": 1}
    )
    assert kwargs == {"monotone_constraints": (0, 1)}


def test_non_monotonic_capable_model_gets_nothing():
    # SVR, Random Forest, MLP, etc. cannot accept monotone_constraints at all —
    # this must return {} regardless of what constraints are requested, so
    # callers never mistakenly believe a constraint was applied.
    kwargs = build_monotone_constraints_kwarg(
        "SVR (RBF)", FEATURES, {"Residual Armature Current": 1}
    )
    assert kwargs == {}


def test_all_zero_constraints_returns_empty_even_for_capable_model():
    kwargs = build_monotone_constraints_kwarg(
        "LGBM Regressor", FEATURES, {"Residual Armature Current": 0}
    )
    assert kwargs == {}


def test_empty_constraints_dict_returns_empty():
    kwargs = build_monotone_constraints_kwarg("LGBM Regressor", FEATURES, {})
    assert kwargs == {}


def test_derive_seed_is_deterministic_and_offset_based():
    assert derive_seed(0, 3) == 3
    assert derive_seed(42, 3) == 45
    # Same base seed always reproduces the same derived seed.
    assert derive_seed(7, 1) == derive_seed(7, 1)
    # Different offsets from the same base must not collide.
    assert derive_seed(7, 1) != derive_seed(7, 2)


def test_warns_for_gpr_and_svr_above_the_row_threshold(capsys):
    """Regression test for a real risk: no size guard existed before the
    baseline .fit() call — GPR is O(n^2) memory / O(n^3) time and SVR scales
    similarly, so either can hang or OOM on a realistically large dataset
    with zero warning (only HPO subsamples internally)."""
    _warn_if_large_dataset_for_model("Gaussian Process Regressor", 5000)
    _warn_if_large_dataset_for_model("SVR (RBF)", 5000)
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Gaussian Process Regressor" in out
    assert "SVR (RBF)" in out


def test_no_warning_below_threshold_or_for_other_models(capsys):
    _warn_if_large_dataset_for_model("Gaussian Process Regressor", 100)
    _warn_if_large_dataset_for_model("Random Forest Regressor", 100000)
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "[WARN]" not in out


def test_reset_model_to_defaults_applies_random_state_when_supported():
    model = reset_model_to_defaults("Random Forest Regressor", random_state=123)
    assert model.get_params()["random_state"] == 123


def test_reset_model_to_defaults_resolves_none_to_seed_zero():
    """Package-wide convention (matching the HPO and UQ modules): a
    random_state of None resolves to seed 0 rather than leaving the model
    unseeded, so results are reproducible by default even when no seed is
    supplied. Regression test for the earlier inconsistency where model
    construction silently stayed nondeterministic under None."""
    model = reset_model_to_defaults("Random Forest Regressor", random_state=None)
    assert model.get_params()["random_state"] == 0


def test_reset_model_to_defaults_same_seed_gives_same_params():
    model_a = reset_model_to_defaults("MLP Regressor", random_state=5)
    model_b = reset_model_to_defaults("MLP Regressor", random_state=5)
    assert model_a.get_params()["random_state"] == model_b.get_params()["random_state"] == 5


def test_reset_model_to_defaults_forces_warm_start_off_even_if_requested(capsys):
    """Regression test for a real risk: override_params could pass
    warm_start=True through unfiltered — every caller refits ONE shared
    model instance across multiple targets in a loop, so target 2's fit
    would silently build on target 1's fitted state. reset_model_to_defaults
    promises a fresh instance and must never honour warm_start=True."""
    model = reset_model_to_defaults(
        "Random Forest Regressor", override_params={"warm_start": True})
    assert model.get_params()["warm_start"] is False
    assert "[WARN]" in capsys.readouterr().out


def test_reset_model_to_defaults_leaves_warm_start_false_silently():
    # The default (False, or simply unrequested) must not trigger a warning —
    # only an explicit attempt to turn it on is worth flagging.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = reset_model_to_defaults("Random Forest Regressor")
    assert model.get_params()["warm_start"] is False
    assert "[WARN]" not in buf.getvalue()


def test_reset_model_to_defaults_bakes_in_monotonic_constraint():
    model = reset_model_to_defaults(
        "LGBM Regressor",
        feature_names=FEATURES,
        monotonic_constraints={"Residual Armature Current": 1},
    )
    assert model.get_params()["monotone_constraints"] == [0, 1]


def test_reset_model_to_defaults_ignores_constraint_for_incapable_model():
    # Must not raise, and must not silently attach an unsupported kwarg.
    model = reset_model_to_defaults(
        "SVR (RBF)",
        feature_names=FEATURES,
        monotonic_constraints={"Residual Armature Current": 1},
    )
    assert "monotone_constraints" not in model.get_params()


# ── apply_monotone_constraints_for_target: cross-target leak regression ──────
#
# Real bug found via a systematic failure-mode sweep, present in FOUR separate
# call sites (model_training.run_models, uncertainty_quantification,
# interpretability, hyperparameter_optimisation) before this helper existed:
# each one shares ONE model instance across every target in a loop, refitting
# it in turn. The old code only called .set_params() when the CURRENT
# target's constraint was non-empty — so a target with no constraint of its
# own silently kept whatever a PREVIOUS target had set, since nothing ever
# reset it. This is the single most important finding of that sweep: it
# affects baseline training, before-HPO UQ, before-HPO interpretability, and
# HPO itself.

def test_constraint_does_not_leak_from_a_constrained_target_to_an_unconstrained_one():
    model = LGBMRegressor(verbose=-1)
    apply_monotone_constraints_for_target(
        model, "LGBM Regressor", FEATURES, {"Residual Armature Current": 1})
    assert model.get_params()["monotone_constraints"] == [0, 1]

    # Next target in the loop has no constraint at all — must reset to
    # neutral, not silently keep the previous target's [0, 1].
    apply_monotone_constraints_for_target(model, "LGBM Regressor", FEATURES, {})
    assert model.get_params()["monotone_constraints"] == [0, 0]


def test_constraint_leak_regression_xgboost_tuple_form():
    model = XGBRegressor()
    apply_monotone_constraints_for_target(
        model, "XGBoost Regressor", FEATURES, {"Input Torque": -1})
    assert model.get_params()["monotone_constraints"] == (-1, 0)

    apply_monotone_constraints_for_target(model, "XGBoost Regressor", FEATURES, {})
    assert model.get_params()["monotone_constraints"] == (0, 0)


def test_apply_constraints_is_a_noop_for_non_capable_model():
    # Must not raise, and must not add an attribute the model never had.
    # LightGBM/XGBoost only surface monotone_constraints in get_params() once
    # something has set it, unlike a formally-declared constructor parameter —
    # so this also guards against reflectively checking model.get_params()
    # instead of the model-name capability list (a real mistake made and
    # caught while writing this fix).
    model = SVR()
    apply_monotone_constraints_for_target(model, "SVR (RBF)", FEATURES, {"Input Torque": 1})
    assert "monotone_constraints" not in model.get_params()


def test_apply_constraints_noop_when_feature_names_missing():
    model = LGBMRegressor(verbose=-1)
    apply_monotone_constraints_for_target(model, "LGBM Regressor", None, {"Input Torque": 1})
    assert "monotone_constraints" not in model.get_params()


# ── Q^2 zero-variance guard ───────────────────────────────────────────────────

def test_q2_zero_variance_target_returns_nan_not_a_crash():
    """Regression test: Q^2's hand-rolled formula divides by the variance of
    y_true, unlike sklearn's r2_score (used for R^2, right next to it) which
    has its own internal zero-variance handling. A constant y_true — a
    genuinely constant column, or just an unlucky small test fold — divided
    by zero silently (producing inf/-inf, not an error) until this guard was
    added, matching the existing NRMSE/KGE pattern in the same dict."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    assert np.isnan(metrics_dict["Q^2"](y_true, y_pred))


def test_q2_normal_case_unaffected_by_the_guard():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert metrics_dict["Q^2"](y_true, y_pred) == 1.0


# ── n<=1 test split: every metric, not just Adjusted R^2 ─────────────────────

def test_single_row_test_split_does_not_crash_any_metric():
    """Investigated as a [RISK] item: 'n<=1 test splits only special-case
    Adjusted R^2', worded as if every other metric might be unsafe too.
    Traced through metrics_dict: every metric besides Adjusted R^2 already
    has its own independent zero-variance/zero-range guard (NRMSE, Q^2, KGE)
    or is inherently safe at n=1 (MSE/MAE/RMSE/MAPE), and sklearn's r2_score
    already returns NaN (with a warning) rather than raising for n=1. This
    locks in that n=1 is safe for the FULL metric set actually used by
    run_models, not just Adjusted R^2."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.DataFrame({"T": [1.0, 2.0, 3.0, 4.0]})
    X_test = pd.DataFrame({"a": [5.0]})
    y_test = pd.DataFrame({"T": [5.5]})
    model = LinearRegression().fit(X_train, y_train["T"])

    result = train_and_evaluate_model(
        model, X_train, X_test, y_train, y_test, "T", metrics_dict)

    assert np.isnan(result["ADJUSTED R^2"])
    for metric_name in metrics_dict:
        if metric_name == "ADJUSTED R^2":
            continue
        assert metric_name in result, f"{metric_name} missing from result"
        # Must be a real (possibly NaN) float, never a raised exception —
        # the assertion above already proves that by having reached here.


# ── run_models: cooperative pause/cancel checkpoint ──────────────────────────
#
# Regression tests for a real QA bug: checkpoint_fn was only ever threaded into
# HPO, so Stop/Pause during baseline Training did nothing until it ended on its
# own. run_models must call checkpoint_fn once per model (the same granularity
# HPO already used), so the UI's cooperative-cancel mechanism can interrupt it.

class _StopAfterN:
    """Raises on the Nth call — simulates the UI's checkpoint_fn once the
    user has pressed Stop partway through a multi-model loop."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls >= self.n:
            raise RuntimeError("simulated Stop")


def _tiny_training_data(n=20):
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y_train = pd.DataFrame({"T": rng.uniform(0, 1, n)})
    X_test = pd.DataFrame({"a": rng.uniform(0, 1, 5), "b": rng.uniform(0, 1, 5)})
    y_test = pd.DataFrame({"T": rng.uniform(0, 1, 5)})
    return X_train, X_test, y_train, y_test


def test_run_models_calls_checkpoint_fn_once_per_model():
    X_train, X_test, y_train, y_test = _tiny_training_data()
    checkpoint = _StopAfterN(n=999)  # never actually trips
    run_models(
        ["KNeighbors Regressor", "SVR (RBF)"],
        X_train, X_test, y_train, y_test, ["T"],
        {"R^2": metrics_dict["R^2"]},
        checkpoint_fn=checkpoint,
    )
    assert checkpoint.calls == 2  # once per model, not per target


def test_run_models_stops_partway_through_when_checkpoint_raises():
    X_train, X_test, y_train, y_test = _tiny_training_data()
    checkpoint = _StopAfterN(n=2)  # trips on the second model
    with pytest.raises(RuntimeError, match="simulated Stop"):
        run_models(
            ["KNeighbors Regressor", "SVR (RBF)", "Random Forest Regressor"],
            X_train, X_test, y_train, y_test, ["T"],
            {"R^2": metrics_dict["R^2"]},
            checkpoint_fn=checkpoint,
        )
    # Stopped before the third model was ever reached.
    assert checkpoint.calls == 2


def test_run_models_without_checkpoint_fn_runs_normally():
    # checkpoint_fn=None (the default) must be a complete no-op, not a crash —
    # every non-UI caller (workflow.py, tests) doesn't pass one.
    X_train, X_test, y_train, y_test = _tiny_training_data()
    results_df, *_ = run_models(
        ["KNeighbors Regressor"], X_train, X_test, y_train, y_test, ["T"],
        {"R^2": metrics_dict["R^2"]},
    )
    assert len(results_df) == 1
