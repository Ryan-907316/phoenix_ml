"""Standardised input-robustness sweep for user-settable knobs.

THE STANDARD (applies to every knob a user can change via the UI or the
run_workflow/WorkflowSession API): driven through the shared case grid --
  default value / zero / negative / very large / float-where-int-expected /
  wrong-type string / unknown name --
each knob must land in exactly one of two acceptable outcomes:

  1. WORKS: the value is valid, or is safely clamped with documented
     behaviour (e.g. oversized sample sizes clamp to the data available;
     any integer seed is normalised into numpy's valid range), or
  2. RAISES a ValueError/TypeError that names the parameter, immediately at
     the public entry point -- never a silent wrong answer, and never a
     cryptic crash deep inside numpy/sklearn/hyperopt far from the setting
     that caused it.

Every test here locks in one of those two outcomes. The empirical sweep that
seeded this file found (and fixed) four silent-nonsense behaviours and five
cryptic deep crashes -- see the 2026-07-18 input-robustness entry in
ISSUES.md. The UI's own string-to-number conversion layer (_sync_session's
messagebox validation) is display-bound and manually QA'd, so the layer
tested here is the module entry points that both the UI and scripting paths
flow through.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

from phoenix_ml.data_preprocessing import compute_feature_selection_recommendations
from phoenix_ml.model_training import build_monotone_constraints_kwarg, derive_seed
from phoenix_ml.models import param_spaces
from phoenix_ml.hyperparameter_optimisation import (
    run_hyperopt_optimisation,
    run_random_search,
)
from phoenix_ml.postprocessing import (
    _build_cv,
    apply_transformation,
    evaluate_transformations,
)
from phoenix_ml.uncertainty_quantification import run_uncertainty_quantification
from phoenix_ml.validation import (
    require_choice,
    require_in_range,
    require_int_at_least,
)


# ── Shared tiny dataset ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 40), "b": rng.normal(0, 1, 40)})
    y = pd.DataFrame({"T": 3.0 * X["a"] + rng.normal(0, 0.5, 40)})
    return X.iloc[:30], X.iloc[30:], y.iloc[:30], y.iloc[30:]


# ── The validators themselves ────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [0, -3, 2.5, 2.0, "five", True, None])
def test_require_int_at_least_rejects_the_standard_bad_cases(bad):
    with pytest.raises(ValueError, match="n_things"):
        require_int_at_least("n_things", bad, minimum=1)


def test_require_int_at_least_accepts_ints_and_numpy_ints():
    assert require_int_at_least("n", 3) == 3
    assert require_int_at_least("n", np.int64(7)) == 7


@pytest.mark.parametrize("bad", [-0.1, 1.5, "half", None, True])
def test_require_in_range_rejects_out_of_range_and_wrong_types(bad):
    with pytest.raises(ValueError, match="frac"):
        require_in_range("frac", bad, 0, 1)


def test_require_in_range_open_bounds_exclude_the_endpoints():
    with pytest.raises(ValueError):
        require_in_range("frac", 0.0, 0, 1, inclusive_low=False)
    with pytest.raises(ValueError):
        require_in_range("frac", 1.0, 0, 1, inclusive_high=False)
    assert require_in_range("frac", 0.5, 0, 1, inclusive_low=False,
                            inclusive_high=False) == 0.5


def test_require_choice_names_the_parameter_and_lists_the_options():
    with pytest.raises(ValueError, match="cv_method.*K-Fold"):
        require_choice("cv_method", "Bogus", {"K-Fold", "LOO"})


# ── random_seed (derive_seed) ────────────────────────────────────────────────
# WORKS for any integer: normalised into numpy's [0, 2**32) seed range, so a
# negative or huge base seed no longer crashes later, at model-fit time, with
# an error nowhere near the setting that caused it.

@pytest.mark.parametrize("base", [0, 1, -5, -(2**33), 2**40, 10**18])
def test_any_integer_seed_is_usable_by_numpy_randomstate(base):
    seed = derive_seed(base, 3)
    assert 0 <= seed < 2**32
    np.random.RandomState(seed)  # must not raise


def test_same_base_seed_still_maps_to_the_same_derived_seed():
    assert derive_seed(-5, 1) == derive_seed(-5, 1)
    assert derive_seed(-5, 1) != derive_seed(-5, 2)


def test_non_numeric_seed_raises_immediately():
    with pytest.raises(ValueError):
        derive_seed("abc", 1)


# ── UQ knobs (validated at run_uncertainty_quantification entry) ─────────────

def _uq(tiny, **overrides):
    # X as plain arrays, matching the real pipeline (X_train_scaled/X_test_scaled
    # are ndarrays from the scaler); y stays a DataFrame keyed by target name.
    Xtr, Xte, ytr, yte = tiny
    kwargs = dict(
        models_dict={"KNeighbors Regressor": KNeighborsRegressor()},
        X_train=np.asarray(Xtr), X_test=np.asarray(Xte),
        y_train=ytr.reset_index(drop=True), y_test=yte.reset_index(drop=True),
        target_columns=["T"], model_names_to_run=["KNeighbors Regressor"],
        uq_method="Conformal", n_bootstrap=2, confidence_interval=95,
        calibration_frac=0.2, subsample_test_size=8, n_jobs=1,
        show_plots=False, calibration_enabled=False, random_state=0,
    )
    kwargs.update(overrides)
    return run_uncertainty_quantification(**kwargs)


def test_uq_defaults_shaped_config_works(tiny):
    df, figs = _uq(tiny)
    assert not df.empty


@pytest.mark.parametrize("bad", [0, -2, 2.5, "ten"])
def test_uq_n_bootstrap_bad_values_raise_named_error(tiny, bad):
    # Used to be a cryptic IndexError inside np.percentile (0/-2) or a
    # TypeError from range() (2.5), only after the bootstrap loop had started.
    with pytest.raises(ValueError, match="n_bootstrap"):
        _uq(tiny, uq_method="Bootstrapping", n_bootstrap=bad)


@pytest.mark.parametrize("bad", [0, 100, 150, -10, "high"])
def test_uq_confidence_interval_bad_values_raise_named_error(tiny, bad):
    # ci=150/0/-10 used to SILENTLY produce nonsense conformal intervals (the
    # quantile just clamped); bootstrap crashed deep in numpy percentile.
    with pytest.raises(ValueError, match="confidence_interval"):
        _uq(tiny, confidence_interval=bad)


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2])
def test_uq_calibration_frac_bad_values_raise_named_error(tiny, bad):
    # 0/1.5 used to surface as sklearn's InvalidParameterError about
    # "test_size" -- a parameter name that exists nowhere in this app's UI.
    with pytest.raises(ValueError, match="calibration_frac"):
        _uq(tiny, calibration_frac=bad)


@pytest.mark.parametrize("bad", [0, -5, 7.5])
def test_uq_subsample_test_size_bad_values_raise_named_error(tiny, bad):
    with pytest.raises(ValueError, match="subsample_test_size"):
        _uq(tiny, subsample_test_size=bad)


def test_uq_oversized_subsample_clamps_and_works(tiny):
    # WORKS-with-clamp case: asking for more test rows than exist just uses
    # every row.
    df, _ = _uq(tiny, subsample_test_size=10**9)
    assert not df.empty


def test_uq_unknown_method_raises_named_error(tiny):
    with pytest.raises(ValueError, match="uq_method"):
        _uq(tiny, uq_method="Bogus Method")


# ── HPO knobs ────────────────────────────────────────────────────────────────

def _random_search(tiny, n_iter=4, sample_size=1000, sampling="Random"):
    Xtr, Xte, ytr, yte = tiny
    return run_random_search(
        KNeighborsRegressor(), param_spaces["KNeighbors Regressor"],
        Xtr, Xte, ytr, yte, sample_size, n_iter, 1, "T", "Q^2", sampling,
    )


def test_random_search_default_shaped_config_works(tiny):
    df, best_params, best_score, _, _ = _random_search(tiny)
    assert np.isfinite(best_score)


@pytest.mark.parametrize("bad", [0, -5, 2.5, "many"])
def test_random_search_n_iter_bad_values_raise_named_error(tiny, bad):
    # n_iter=0 used to crash with a bare KeyError('Q^2') on an empty tracking
    # DataFrame; -5 with "negative dimensions are not allowed" in the sampler.
    with pytest.raises(ValueError, match="n_iter"):
        _random_search(tiny, n_iter=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_random_search_sample_size_bad_values_raise_named_error(tiny, bad):
    with pytest.raises(ValueError, match="sample_size"):
        _random_search(tiny, sample_size=bad)


def test_random_search_unknown_sampling_method_raises_named_error(tiny):
    with pytest.raises(ValueError, match="[Ss]ampling"):
        _random_search(tiny, sampling="Bogus")


@pytest.mark.parametrize("bad", [0, -1, 3.5])
def test_hyperopt_evals_bad_values_raise_named_error(tiny, bad):
    # evals=0 used to crash inside hyperopt: "There are no evaluation tasks,
    # cannot return argmin of task losses."
    Xtr, Xte, ytr, yte = tiny
    with pytest.raises(ValueError, match="evals"):
        run_hyperopt_optimisation(
            "KNeighbors Regressor", KNeighborsRegressor(),
            param_spaces["KNeighbors Regressor"], bad,
            Xtr, Xte, ytr, yte, "T", "Q^2", 1000,
        )


# ── CV knobs (_build_cv) ─────────────────────────────────────────────────────

def test_cv_unknown_method_raises_named_error_not_bare_keyerror():
    with pytest.raises(ValueError, match="cv_method"):
        _build_cv("Bogus Method", {})


@pytest.mark.parametrize("bad_splits", [0, 1, -3, 2.5])
def test_cv_kfold_invalid_n_splits_raises_clearly_at_construction(bad_splits):
    # sklearn's own construction-time errors here are already clear and
    # immediate -- locked in as the acceptable outcome.
    with pytest.raises((ValueError, TypeError)):
        _build_cv("K-Fold", {"n_splits": bad_splits})


def test_cv_huge_n_splits_constructs_then_fails_clearly_at_split_time():
    cv = _build_cv("K-Fold", {"n_splits": 10**6})
    X = np.arange(40).reshape(20, 2)
    with pytest.raises(ValueError, match="[Cc]annot have number of splits"):
        list(cv.split(X))


# ── Residual transformations ─────────────────────────────────────────────────

def test_stale_transform_names_are_warned_and_skipped_not_mislabelled(tiny, capsys):
    """Regression test for a real silent-wrong-answer bug: names that were
    valid before the Yeo-Johnson consolidation ("Log", "Sqrt") -- or any typo
    -- produced rows whose values were the IDENTITY transform, labelled as if
    the requested transform had been applied (metrics identical to "None"
    with nothing saying why)."""
    Xtr, Xte, ytr, yte = tiny
    best_models = {"T": pd.Series({"model_name": "KNeighbors Regressor",
                                   "hyperparameters": {}})}
    df = evaluate_transformations(
        best_models, Xtr, Xte, ytr, yte,
        transforms=["Log", "Sqrt", "Arcsinh"], random_state=0)

    assert set(df["Transformation"]) == {"None", "Arcsinh"}
    out = capsys.readouterr().out
    assert "[WARN]" in out and "Log" in out and "Sqrt" in out


def test_apply_transformation_unknown_name_raises_named_error():
    with pytest.raises(ValueError, match="Unknown transformation"):
        apply_transformation(pd.Series([1.0, -2.0]), "Box-Cox")


# ── Monotonic constraints ────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [2, -2, "up", 0.5])
def test_monotone_constraint_values_outside_minus1_0_1_raise_named_error(bad):
    # A 2 used to pass straight through to LGBM's monotone_constraints list
    # (an undefined direction, silently accepted); "up" crashed with a bare
    # int() ValueError that never mentioned constraints.
    with pytest.raises(ValueError, match="[Mm]onotonic constraint"):
        build_monotone_constraints_kwarg("LGBM Regressor", ["a", "b"], {"a": bad})


@pytest.mark.parametrize("ok", [-1, 0, 1])
def test_monotone_constraint_valid_directions_still_work(ok):
    kw = build_monotone_constraints_kwarg("LGBM Regressor", ["a", "b"], {"a": ok})
    if ok == 0:
        assert kw == {}  # all-zero constraints mean "nothing to apply"
    else:
        assert kw == {"monotone_constraints": [ok, 0]}


# ── Feature-selection threshold ──────────────────────────────────────────────

@pytest.mark.parametrize("bad", [-1, 5, 1.001, "high"])
def test_redundancy_threshold_outside_0_1_raises_named_error(tiny, bad):
    # dcor lives in [0, 1]: negative used to flag EVERY pair redundant, > 1
    # silently disabled the check.
    Xtr, _, ytr, _ = tiny
    dist_corr_df = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ValueError, match="redundancy_threshold"):
        compute_feature_selection_recommendations(
            dist_corr_df, Xtr, ytr, ["a", "b"], ["T"], redundancy_threshold=bad)
