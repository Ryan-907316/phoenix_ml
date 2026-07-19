"""Tests for hyperparameter_optimisation.py — metric computation, random-search
early stopping, and the per-(model, target) tuned-instance lookup.

get_all_models_tuned_per_target is the successor to a real bug (its
predecessor picked one HPO method per model averaged across targets, then
applied a single target's hyperparameters to every target) — the tests here
lock in the property that motivated the rewrite: each (model, target) pair
gets its own winning method's hyperparameters.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVR

import phoenix_ml.hyperparameter_optimisation as hpo_module
from phoenix_ml.hyperparameter_optimisation import (
    _compute_metric,
    get_all_models_tuned_per_target,
    process_hyperparameters,
    run_all_models_optimisation,
    run_hyperopt_optimisation,
    run_random_search,
    run_skopt_optimisation,
)


# ── _compute_metric ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("metric,perfect_value", [
    ("MSE", 0.0), ("RMSE", 0.0), ("MAE", 0.0), ("NRMSE", 0.0), ("MAPE", 0.0),
    ("R^2", 1.0), ("Q^2", 1.0), ("KGE", 1.0),
])
def test_perfect_prediction_gives_each_metrics_ideal_value(metric, perfect_value):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _compute_metric(y, y.copy(), metric) == pytest.approx(perfect_value)


def test_error_metrics_match_hand_computed_values():
    # errors = [0, 0, 0, 4] -> hand-computable for every distance metric.
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 8.0])
    assert _compute_metric(y_true, y_pred, "MSE") == pytest.approx(4.0)
    assert _compute_metric(y_true, y_pred, "RMSE") == pytest.approx(2.0)
    assert _compute_metric(y_true, y_pred, "MAE") == pytest.approx(1.0)
    # NRMSE = RMSE / (max - min) = 2 / 3
    assert _compute_metric(y_true, y_pred, "NRMSE") == pytest.approx(2.0 / 3.0)
    # MAPE = mean(0, 0, 0, 4/4) * 100 = 25%
    assert _compute_metric(y_true, y_pred, "MAPE") == pytest.approx(25.0)
    # R^2 = Q^2 here: 1 - 16/5
    assert _compute_metric(y_true, y_pred, "R^2") == pytest.approx(-2.2)
    assert _compute_metric(y_true, y_pred, "Q^2") == pytest.approx(-2.2)


def test_adjusted_r2_requires_n_and_p():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="requires n and p"):
        _compute_metric(y, y, "ADJUSTED R^2")
    # With n/p supplied: 1 - (1 - r2)(n-1)/(n-p-1); r2 = 1 -> stays 1.
    assert _compute_metric(y, y, "ADJUSTED R^2", n=4, p=1) == pytest.approx(1.0)


def test_zero_variance_targets_give_nan_not_a_crash():
    # NRMSE divides by the target range and KGE by its std/mean — a constant
    # target must produce NaN, never a ZeroDivisionError or a bogus score an
    # optimiser could "improve".
    const = np.array([5.0, 5.0, 5.0])
    pred = np.array([4.0, 5.0, 6.0])
    assert np.isnan(_compute_metric(const, pred, "NRMSE"))
    assert np.isnan(_compute_metric(const, pred, "KGE"))
    # KGE also guards mean == 0 (its mean-ratio term would divide by zero).
    zero_mean = np.array([-1.0, 0.0, 1.0])
    assert np.isnan(_compute_metric(zero_mean, pred, "KGE"))
    # Q^2 (the pipeline's default metric) previously had no such guard at all,
    # unlike NRMSE/KGE right next to it in this same function — found via a
    # systematic failure-mode sweep.
    assert np.isnan(_compute_metric(const, pred, "Q^2"))


def test_unknown_metric_raises():
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported metric"):
        _compute_metric(y, y, "F1")


# ── run_random_search early stopping ─────────────────────────────────────────

class _FlatModel:
    """Minimal estimator whose hyperparameter never affects predictions —
    every random-search iteration therefore scores identically, making
    early-stopping behaviour exactly hand-traceable."""

    def __init__(self):
        self.k = 1.0
        self._mean = 0.0

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def get_params(self, deep=True):
        return {"k": self.k}

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


def _search_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(0, 1, (n, 2))
    X_test = rng.uniform(0, 1, (8, 2))
    y_train = pd.DataFrame({"T": rng.uniform(0, 1, n)})
    y_test = pd.DataFrame({"T": rng.uniform(0, 1, 8)})
    return X_train, X_test, y_train, y_test

_PARAM_SPACE = {"k": {"type": "uniform", "bounds": (0.0, 1.0)}}


def test_random_search_without_patience_runs_every_iteration():
    X_train, X_test, y_train, y_test = _search_data()
    results_df, best_params, best_value, elapsed, es_info = run_random_search(
        _FlatModel(), _PARAM_SPACE, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=10, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=None, random_state=0,
    )
    assert es_info["stopped_early"] is False
    assert es_info["actual_iters"] == 10
    assert len(results_df) == 10


def test_random_search_patience_one_stops_after_first_plateau_step():
    # _FlatModel scores identically every iteration: iteration 1 "improves"
    # from -inf/inf, iteration 2 does not -> patience=1 must stop there.
    X_train, X_test, y_train, y_test = _search_data()
    results_df, best_params, best_value, elapsed, es_info = run_random_search(
        _FlatModel(), _PARAM_SPACE, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=10, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=1, random_state=0,
    )
    assert es_info["stopped_early"] is True
    assert es_info["actual_iters"] == 2
    assert len(results_df) == 2


def test_random_search_patience_zero_stops_at_first_non_improving_step():
    """patience=0 means 'no patience': stop the moment an evaluation fails to
    improve — but never because of an evaluation that DID improve. Regression
    test for a bug where the patience check ran unconditionally, so patience=0
    stopped after the very first evaluation even though it had just improved
    on the -inf starting score."""
    X_train, X_test, y_train, y_test = _search_data()
    results_df, best_params, best_value, elapsed, es_info = run_random_search(
        _FlatModel(), _PARAM_SPACE, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=10, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=0, random_state=0,
    )
    # Iteration 1 improves (from the -inf/inf sentinel) and must NOT stop the
    # search; iteration 2 is the first non-improving step and must.
    assert es_info["stopped_early"] is True
    assert es_info["actual_iters"] == 2


class _FlatModelWithNJobs(_FlatModel):
    """Same flat-scoring stub as _FlatModel, plus an n_jobs param so the
    parallel-branch oversubscription guard has something to force."""

    def __init__(self):
        super().__init__()
        self.n_jobs = -1

    def get_params(self, deep=True):
        return {"k": self.k, "n_jobs": self.n_jobs}


def test_random_search_forces_single_threaded_model_in_parallel_branch():
    """Regression test for a real risk: Random Search's parallel branch
    (joblib.Parallel over trials) combined with a model that does its own
    internal multi-threading (e.g. XGBoost/LGBM) oversubscribes CPU cores,
    making a search take far longer wall-clock than its n_jobs/n_iter budget
    implies. The model's own n_jobs must be forced to 1 when trials are
    dispatched in parallel (patience=None)."""
    X_train, X_test, y_train, y_test = _search_data()
    model = _FlatModelWithNJobs()
    run_random_search(
        model, _PARAM_SPACE, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=5, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=None, random_state=0,
    )
    assert model.get_params()["n_jobs"] == 1


def test_random_search_leaves_model_n_jobs_alone_in_sequential_branch():
    # With early stopping (patience set), trials run sequentially in-process
    # -> no outer parallelism to conflict with, so the model's own n_jobs
    # must be left as the caller configured it.
    X_train, X_test, y_train, y_test = _search_data()
    model = _FlatModelWithNJobs()
    run_random_search(
        model, _PARAM_SPACE, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=5, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=5, random_state=0,
    )
    assert model.get_params()["n_jobs"] == -1


def test_random_search_same_seed_reproduces_identical_trials():
    X_train, X_test, y_train, y_test = _search_data()

    def run():
        results_df, *_ = run_random_search(
            SVR(kernel="rbf"),
            {"C": {"type": "loguniform", "bounds": (0.1, 100.0)}},
            X_train, X_test, y_train, y_test,
            sample_size=1000, n_iter=8, n_jobs=1, target_var="T",
            metric="MSE", sampling_method="Sobol", patience=None, random_state=7,
        )
        return results_df

    assert run().equals(run())


def test_random_search_applies_rf_constraint_fix_like_the_other_backends():
    """Regression test for a real risk: evaluate() skipped the RF
    min_samples_split > min_samples_leaf fix that hyperopt_objective and
    skopt_objective both apply, silently searching a slightly different RF
    space than the other two backends for the same nominal bounds."""
    class _RecordingParamsModel:
        def __init__(self):
            self.received_params = []

        def get_params(self, deep=True):
            return {}

        def set_params(self, **kwargs):
            self.received_params.append(dict(kwargs))
            return self

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X_train, X_test, y_train, y_test = _search_data()
    param_space = {
        "min_samples_split": {"type": "int", "bounds": (2, 4)},
        "min_samples_leaf": {"type": "int", "bounds": (2, 4)},
    }
    model = _RecordingParamsModel()
    run_random_search(
        model, param_space, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=15, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=None, random_state=0,
        model_name="Random Forest Regressor",
    )
    assert model.received_params  # sanity: trials actually ran
    for params in model.received_params:
        assert params["min_samples_split"] > params["min_samples_leaf"]


# ── hyperopt / skopt early stopping ──────────────────────────────────────────
#
# Same contract as random search (and same regression risk): patience=0 means
# "stop at the first NON-improving evaluation", never "stop right after an
# improving one". _FlatModel scores identically every evaluation, so with
# patience=0 each backend must run exactly 2 evaluations: #1 improves on the
# empty history (must not stop), #2 is the first non-improving one (must stop).
# The stub borrows the "Gaussian Process Regressor" identity because the skopt
# objective looks that name up in the global param_spaces registry, and GPR's
# space is a single 'alpha' parameter.

_ALPHA_SPACE = {"alpha": {"type": "loguniform", "bounds": (1e-3, 1e-1)}}


def test_hyperopt_patience_zero_stops_at_first_non_improving_eval():
    X_train, X_test, y_train, y_test = _search_data()
    best_params, tracking_lists, elapsed, es_info = run_hyperopt_optimisation(
        "Gaussian Process Regressor", _FlatModel(), _ALPHA_SPACE, 15,
        X_train, X_test, y_train, y_test, "T", "MSE", 1000,
        patience=0, random_state=0,
    )
    assert es_info["stopped_early"] is True
    assert es_info["actual_iters"] == 2


def test_hyperopt_without_patience_runs_every_evaluation():
    X_train, X_test, y_train, y_test = _search_data()
    best_params, tracking_lists, elapsed, es_info = run_hyperopt_optimisation(
        "Gaussian Process Regressor", _FlatModel(), _ALPHA_SPACE, 5,
        X_train, X_test, y_train, y_test, "T", "MSE", 1000,
        patience=None, random_state=0,
    )
    assert es_info["stopped_early"] is False
    assert es_info["actual_iters"] == 5


def test_skopt_patience_zero_stops_at_first_non_improving_call():
    X_train, X_test, y_train, y_test = _search_data()
    best_params, tracking_lists, best_value, elapsed, es_info = run_skopt_optimisation(
        "Gaussian Process Regressor", _FlatModel(), _ALPHA_SPACE, 12,
        X_train, X_test, y_train, y_test, "T", "MSE", 1000, n_jobs=1,
        patience=0, random_state=0,
    )
    assert es_info["stopped_early"] is True
    assert es_info["actual_iters"] == 2


# ── per-trial failure isolation ──────────────────────────────────────────────
#
# Regression tests for a real risk: KNN's n_neighbors bound (1, 50) ignores
# dataset size — a sampled/suggested n_neighbors exceeding the training set
# size crashes model.fit() with no per-trial exception handling anywhere,
# aborting the whole search instead of just that one bad trial. Simulated
# here with a stub whose fit() always raises, borrowing GPR's single-'alpha'
# param space identity (same reason as the early-stopping stubs above).

class _AlwaysFailsModel:
    def get_params(self, deep=True):
        return {"alpha": 0.01}

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def fit(self, X, y):
        raise ValueError("simulated fit failure (e.g. n_neighbors > n_samples)")

    def predict(self, X):
        raise AssertionError("predict should never be reached")


def test_random_search_survives_every_trial_failing(capsys):
    X_train, X_test, y_train, y_test = _search_data()
    results_df, best_params, best_value, elapsed, es_info = run_random_search(
        _AlwaysFailsModel(), _ALPHA_SPACE, X_train, X_test, y_train, y_test,
        sample_size=1000, n_iter=5, n_jobs=1, target_var="T",
        metric="MSE", sampling_method="Sobol", patience=None, random_state=0,
        model_name="Gaussian Process Regressor",
    )
    assert len(results_df) == 5           # every trial still recorded, none crashed the search
    assert best_value == np.inf            # MSE is lower-is-better -> worst is +inf
    assert "[WARN]" in capsys.readouterr().out


def test_hyperopt_survives_every_trial_failing(capsys):
    X_train, X_test, y_train, y_test = _search_data()
    best_params, tracking_lists, elapsed, es_info = run_hyperopt_optimisation(
        "Gaussian Process Regressor", _AlwaysFailsModel(), _ALPHA_SPACE, 5,
        X_train, X_test, y_train, y_test, "T", "MSE", 1000,
        patience=None, random_state=0,
    )
    # Every trial had STATUS_FAIL -> nothing usable to track or report as best.
    assert tracking_lists == ([], [])
    assert best_params == {}
    assert "[WARN]" in capsys.readouterr().out


def test_skopt_survives_every_trial_failing(capsys):
    X_train, X_test, y_train, y_test = _search_data()
    best_params, tracking_lists, best_value, elapsed, es_info = run_skopt_optimisation(
        "Gaussian Process Regressor", _AlwaysFailsModel(), _ALPHA_SPACE, 12,
        X_train, X_test, y_train, y_test, "T", "MSE", 1000, n_jobs=1,
        patience=None, random_state=0,
    )
    assert best_value == hpo_module._TRIAL_FAILURE_SENTINEL
    assert "[WARN]" in capsys.readouterr().out


# ── process_hyperparameters ──────────────────────────────────────────────────
#
# This function repairs hyperparameters that lost their types on the way
# through CSV/Excel/JSON round trips. Every branch is lenient by design: a
# value it can't confidently coerce passes through unchanged (and fails
# loudly later at set_params), rather than being guessed at.

def test_none_and_tuple_values_pass_through_untouched():
    processed = process_hyperparameters(
        {"max_depth": None, "hidden_layer_sizes": (64, 32)}, "MLP Regressor")
    assert processed["max_depth"] is None
    assert processed["hidden_layer_sizes"] == (64, 32)


def test_wellknown_integer_names_are_coerced_from_float_for_any_model():
    # n_estimators etc. are coerced by NAME (a hardcoded list), independent of
    # the model's registered param space — HPO samplers emit them as floats.
    processed = process_hyperparameters(
        {"n_estimators": 150.0, "min_samples_leaf": 2.0}, "Random Forest Regressor")
    assert processed["n_estimators"] == 150 and isinstance(processed["n_estimators"], int)
    assert processed["min_samples_leaf"] == 2 and isinstance(processed["min_samples_leaf"], int)


def test_param_space_int_is_parsed_from_string():
    # max_iter is an int in MLP's registered space but NOT in the hardcoded
    # name list — the space-driven branch must parse it, including via float
    # notation ("1500.0" -> 1500).
    processed = process_hyperparameters({"max_iter": "1500.0"}, "MLP Regressor")
    assert processed["max_iter"] == 1500 and isinstance(processed["max_iter"], int)


def test_uniform_param_parses_numeric_strings_but_keeps_named_options():
    # max_features is uniform-typed in RF's space; "0.5" is a stringified
    # number, "sqrt" is a legitimate sklearn named option that float() cannot
    # (and must not) touch.
    processed = process_hyperparameters(
        {"max_features": "0.5"}, "Random Forest Regressor")
    assert processed["max_features"] == pytest.approx(0.5)

    processed = process_hyperparameters(
        {"max_features": "sqrt"}, "Random Forest Regressor")
    assert processed["max_features"] == "sqrt"


def test_garbage_string_for_int_param_passes_through_unchanged():
    # Deliberately lenient: an unparseable value is passed through so the
    # failure surfaces at set_params() with the real offending value visible,
    # instead of being silently replaced here.
    processed = process_hyperparameters({"max_iter": "not a number"}, "MLP Regressor")
    assert processed["max_iter"] == "not a number"


def test_unknown_string_param_passes_through():
    processed = process_hyperparameters({"kernel": "rbf"}, "SVR (RBF)")
    assert processed["kernel"] == "rbf"


# ── get_all_models_tuned_per_target ──────────────────────────────────────────

def _hpo_results():
    """Hand-built HPO output where each target has a DIFFERENT winning method
    for the same model: random wins Target A (0.9 > 0.5), skopt wins
    Target B (0.8 > 0.4)."""
    metrics = {
        "random": {"SVR (RBF)": {
            "Target A": {"Q^2": 0.9}, "Target B": {"Q^2": 0.4}}},
        "hyperopt": {},
        "skopt": {"SVR (RBF)": {
            "Target A": {"Q^2": 0.5}, "Target B": {"Q^2": 0.8}}},
    }
    params = {
        "random": {"SVR (RBF)": {
            "Target A": {"C": 1.0}, "Target B": {"C": 2.0}}},
        "hyperopt": {},
        "skopt": {"SVR (RBF)": {
            "Target A": {"C": 50.0}, "Target B": {"C": 99.0}}},
    }
    return metrics, params


def test_each_target_gets_its_own_winning_methods_hyperparameters():
    metrics, params = _hpo_results()
    tuned = get_all_models_tuned_per_target(
        ["SVR (RBF)"], ["Target A", "Target B"], metrics, params,
        "Q^2", {"SVR (RBF)": SVR(kernel="rbf")},
    )
    # Target A's winner is random (C=1.0); Target B's is skopt (C=99.0).
    assert tuned["SVR (RBF)"]["Target A"].get_params()["C"] == 1.0
    assert tuned["SVR (RBF)"]["Target B"].get_params()["C"] == 99.0
    # And the two targets hold distinct instances, not one shared object.
    assert tuned["SVR (RBF)"]["Target A"] is not tuned["SVR (RBF)"]["Target B"]


def test_target_with_no_tuned_data_is_skipped_without_crashing():
    metrics, params = _hpo_results()
    # Ask for a target that no method ever produced results for.
    tuned = get_all_models_tuned_per_target(
        ["SVR (RBF)"], ["Target A", "Target C"], metrics, params,
        "Q^2", {"SVR (RBF)": SVR(kernel="rbf")},
    )
    assert "Target A" in tuned["SVR (RBF)"]
    assert "Target C" not in tuned["SVR (RBF)"]


def test_model_with_no_tuned_data_at_all_is_omitted():
    tuned = get_all_models_tuned_per_target(
        ["SVR (RBF)"], ["Target A"],
        {"random": {}, "hyperopt": {}, "skopt": {}},
        {"random": {}, "hyperopt": {}, "skopt": {}},
        "Q^2", {"SVR (RBF)": SVR(kernel="rbf")},
    )
    assert tuned == {}


def test_stringified_hyperparameters_are_parsed_before_set_params():
    # Hyperparameters read back from CSV/Excel exports arrive as strings.
    metrics = {"random": {"SVR (RBF)": {"Target A": {"Q^2": 0.9}}},
               "hyperopt": {}, "skopt": {}}
    params = {"random": {"SVR (RBF)": {"Target A": "{'C': 3.5}"}},
              "hyperopt": {}, "skopt": {}}
    tuned = get_all_models_tuned_per_target(
        ["SVR (RBF)"], ["Target A"], metrics, params,
        "Q^2", {"SVR (RBF)": SVR(kernel="rbf")},
    )
    assert tuned["SVR (RBF)"]["Target A"].get_params()["C"] == 3.5


# ── run_all_models_optimisation: per-model failure isolation ────────────────

def test_one_models_failure_does_not_abort_the_multi_model_sweep(monkeypatch):
    """Regression test for a real bug: one model failing (e.g. a hyperopt
    import failure) aborted the whole multi-model sweep, losing every
    already-computed result for the other models."""
    def _fake_workflow(model_name, **kwargs):
        if model_name == "Broken Model":
            raise RuntimeError("simulated hyperopt import failure")
        return {
            "random": {
                "Target A": {
                    "elapsed_time": 0.01,
                    "tracking": pd.DataFrame({"C": [1.0], "MSE": [0.5]}),
                    "es_info": {},
                }
            }
        }

    monkeypatch.setattr(hpo_module, "run_hyperparameter_optimisation_workflow", _fake_workflow)
    models_dict = {"Broken Model": SVR(kernel="rbf"), "SVR (RBF)": SVR(kernel="rbf")}
    all_metrics, all_params, all_times, all_plots = run_all_models_optimisation(
        models_dict, X_train=None, X_test=None, y_train=None, y_test=None,
        target_columns=["Target A"], methods_to_run=["random"], metric="MSE",
    )
    # The failing model must be present but empty, not missing or crashed-out.
    assert all_metrics["random"]["Broken Model"] == {}
    assert all_params["random"]["Broken Model"] == {}
    # The other model's results must still have been computed normally.
    assert "Target A" in all_metrics["random"]["SVR (RBF)"]
    assert all_metrics["random"]["SVR (RBF)"]["Target A"]["MSE"] == pytest.approx(0.5)


# ── run_hyperparameter_optimisation_workflow: per-METHOD failure isolation ──

def _fake_random_search(model, param_space, X_train, X_test, y_train, y_test,
                        sample_size, n_iter, n_jobs, target_var, metric,
                        sampling_method, **kwargs):
    return (pd.DataFrame({"C": [1.0], "MSE": [0.5]}), {"C": 1.0}, 0.5, 0.01, {})


def test_one_backends_failure_keeps_the_other_backends_results(monkeypatch, capsys):
    """Regression test for a real bug found in a live run: Scikit-Optimize
    raising (gp_minimize requires n_calls >= 10; the run used calls=5)
    discarded the ALREADY-COMPLETED Random Search and Hyperopt results for
    that model — the whole-model try/except in run_all_models_optimisation
    caught the exception after the successful methods' results had been
    computed but before they were returned. Downstream, every (model, target)
    pair then warned "No optimised hyperparameters found", UQ-After and
    Interpretability-After ran on zero models, and the report silently fell
    back to default hyperparameters. Each backend must now be contained
    individually: a failed method is absent from the result, the others'
    results survive."""
    monkeypatch.setattr(hpo_module, "run_random_search", _fake_random_search)

    def _failing_skopt(*args, **kwargs):
        raise ValueError("Expected `n_calls` >= 10, got 5")
    monkeypatch.setattr(hpo_module, "run_skopt_optimisation", _failing_skopt)

    results = hpo_module.run_hyperparameter_optimisation_workflow(
        model_name="SVR (RBF)", model=SVR(kernel="rbf"),
        X_train=None, X_test=None, y_train=None, y_test=None,
        target_columns=["Target A"], methods_to_run=["random", "skopt"],
        metric="MSE", plot=False,
    )

    # Random Search's completed results survive; skopt is simply absent.
    assert "random" in results
    assert "Target A" in results["random"]
    assert "skopt" not in results

    out = capsys.readouterr().out
    assert "[WARN] Scikit-Optimize failed" in out
    assert "results from the other HPO methods are kept" in out


def test_backend_failure_containment_flows_through_the_multi_model_sweep(monkeypatch):
    """Same scenario one level up: after run_all_models_optimisation, the
    surviving backend's metrics/params must be populated for the model (the
    exact dicts get_all_models_tuned_per_target reads), so UQ-After and
    Interpretability-After still have tuned instances to work with."""
    monkeypatch.setattr(hpo_module, "run_random_search", _fake_random_search)

    def _failing_skopt(*args, **kwargs):
        raise ValueError("Expected `n_calls` >= 10, got 5")
    monkeypatch.setattr(hpo_module, "run_skopt_optimisation", _failing_skopt)

    all_metrics, all_params, all_times, all_plots = run_all_models_optimisation(
        {"SVR (RBF)": SVR(kernel="rbf")},
        X_train=None, X_test=None, y_train=None, y_test=None,
        target_columns=["Target A"], methods_to_run=["random", "skopt"],
        metric="MSE", plot=False,
    )
    assert all_metrics["random"]["SVR (RBF)"]["Target A"]["MSE"] == pytest.approx(0.5)
    assert all_params["random"]["SVR (RBF)"]["Target A"] == {"C": 1.0}
    # skopt contributed nothing for this model - absent, not crashed-out.
    assert "SVR (RBF)" not in all_metrics["skopt"]
