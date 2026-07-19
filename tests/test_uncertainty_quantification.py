"""Tests for uncertainty_quantification.py — conformal intervals, calibration
metrics, and bootstrap reproducibility.

The conformal test pins the finite-sample-corrected quantile down by its
observable consequence: at 95% confidence with 10 calibration points,
ceil((10+1)*0.95)/10 = 1.1 clips to 1.0, so the interval half-width must be
the LARGEST calibration residual — a naive uncorrected quantile(0.95) would
produce a narrower (under-covering) interval.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from phoenix_ml.uncertainty_quantification import (
    bootstrap_uncertainty,
    calculate_coverage,
    compute_calibration_metrics,
    conformal_predictions,
    gp_posterior_uncertainty,
    run_uncertainty_quantification,
)


class _CountingConstantModel:
    """Predicts a constant; counts fit() calls on THIS instance so the
    deep-copy contract is observable."""

    def __init__(self, value=0.0):
        self.value = value
        self.fit_calls = 0

    def get_params(self, deep=True):
        return {"value": self.value}

    def set_params(self, **kw):
        self.__dict__.update(kw)
        return self

    def fit(self, X, y):
        self.fit_calls += 1
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def _uq_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(0, 1, (n, 2))
    y_train = pd.DataFrame({"T": rng.uniform(0, 10, n)})
    X_test = rng.uniform(0, 1, (15, 2))
    return X_train, y_train, X_test


# ── conformal_predictions ─────────────────────────────────────────────────────

def test_conformal_interval_is_symmetric_with_constant_width():
    X_train, y_train, X_test = _uq_data()
    preds, lower, upper, residuals = conformal_predictions(
        _CountingConstantModel(5.0), X_train, y_train, X_test, "T",
        calibration_frac=0.1, confidence_interval=95, random_state=0)
    widths_up = upper - preds
    widths_down = preds - lower
    assert np.allclose(widths_up, widths_down)          # symmetric
    assert np.allclose(widths_up, widths_up[0])         # same width everywhere
    assert len(residuals) == 10                         # 10% of 100


def test_conformal_small_calibration_set_uses_max_residual():
    # See module docstring: n_cal=10 at 95% -> corrected level clips to 1.0,
    # so the half-width must equal the largest calibration residual.
    X_train, y_train, X_test = _uq_data()
    preds, lower, upper, residuals = conformal_predictions(
        _CountingConstantModel(5.0), X_train, y_train, X_test, "T",
        calibration_frac=0.1, confidence_interval=95, random_state=0)
    half_width = float((upper - preds)[0])
    assert half_width == pytest.approx(float(np.max(residuals)))


def test_conformal_does_not_refit_the_callers_model_instance():
    # Documented contract: the caller's instance is shared across UQ methods
    # and later steps — conformal must fit a deep copy, never the original.
    X_train, y_train, X_test = _uq_data()
    model = _CountingConstantModel(5.0)
    conformal_predictions(model, X_train, y_train, X_test, "T",
                          calibration_frac=0.1, random_state=0)
    assert model.fit_calls == 0


# ── run_uncertainty_quantification: cooperative pause/cancel checkpoint ──────
#
# Regression test for a real QA bug: checkpoint_fn was only ever threaded into
# HPO, so Stop/Pause during UQ did nothing until it ended on its own.

class _StopAfterN:
    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls >= self.n:
            raise RuntimeError("simulated Stop")


def test_run_uncertainty_quantification_calls_checkpoint_fn_once_per_model():
    X_train, y_train, X_test = _uq_data(n=30)
    y_test = pd.DataFrame({"T": np.random.default_rng(1).uniform(0, 10, 15)})
    models = {"A": _CountingConstantModel(1.0), "B": _CountingConstantModel(2.0)}
    checkpoint = _StopAfterN(n=999)
    run_uncertainty_quantification(
        models, X_train, X_test, y_train, y_test, ["T"],
        model_names_to_run=["A", "B"], uq_method="Bootstrapping", n_bootstrap=2,
        show_plots=False, checkpoint_fn=checkpoint,
    )
    assert checkpoint.calls == 2


def test_run_uncertainty_quantification_stops_partway_when_checkpoint_raises():
    X_train, y_train, X_test = _uq_data(n=30)
    y_test = pd.DataFrame({"T": np.random.default_rng(1).uniform(0, 10, 15)})
    models = {"A": _CountingConstantModel(1.0), "B": _CountingConstantModel(2.0)}
    checkpoint = _StopAfterN(n=2)
    with pytest.raises(RuntimeError, match="simulated Stop"):
        run_uncertainty_quantification(
            models, X_train, X_test, y_train, y_test, ["T"],
            model_names_to_run=["A", "B"], uq_method="Bootstrapping", n_bootstrap=2,
            show_plots=False, checkpoint_fn=checkpoint,
        )
    assert checkpoint.calls == 2


# ── compute_calibration_metrics ───────────────────────────────────────────────

def test_bootstrap_branch_perfect_ensemble_gives_full_coverage_and_zero_crps():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # Every bootstrap replicate predicts y exactly: intervals are zero-width
    # AT the truth -> coverage 1.0 at every level, CRPS exactly 0.
    raw_preds = np.tile(y, (5, 1))
    out = compute_calibration_metrics({"raw_preds": raw_preds}, y, "Bootstrapping")
    assert out["CRPS"] == pytest.approx(0.0)
    assert all(actual == 1.0 for _, actual in out["reliability_curve"])
    assert len(out["reliability_curve"]) == 19


def test_conformal_branch_has_no_crps_by_design():
    y = np.array([1.0, 2.0, 3.0])
    out = compute_calibration_metrics(
        {"calibration_residuals": np.array([0.5, 1.0, 2.0]),
         "mean": np.array([1.1, 2.1, 2.5])},
        y, "Conformal")
    # Conformal intervals are set-valued, not distributional — CRPS must be
    # None, not 0 (0 would claim a perfect probabilistic forecast).
    assert out["CRPS"] is None
    assert len(out["reliability_curve"]) == 19
    assert out["RMSCE"] >= 0.0


def test_conformal_reliability_curve_uses_finite_sample_correction_not_a_plain_quantile():
    """Regression test for a real risk: the reliability curve used a plain
    quantile(residuals, conf) while the actually-reported conformal interval
    (conformal_predictions, see its docstring) uses a finite-sample-corrected
    quantile — the two visibly disagreed at small calibration sizes for the
    "same" nominal confidence level.

    n_cal=10, residuals=[1..10]: at nominal 95%, the corrected level is
    min(1, ceil(11*0.95)/10) = 1.0 -> quantile = max(residuals) = 10.
    A plain quantile(residuals, 0.95) would instead be 9.55 (linear
    interpolation). A point at distance 9.8 from the mean is covered under
    the corrected quantile but NOT under the plain one — a deterministic
    way to tell which formula is actually in use."""
    residuals = np.arange(1, 11, dtype=float)
    y_pred = np.array([0.0])
    y = np.array([9.8])

    out = compute_calibration_metrics(
        {"calibration_residuals": residuals, "mean": y_pred}, y, "Conformal")
    nominal_95_actual = dict(out["reliability_curve"])[0.95]
    assert nominal_95_actual == 1.0


def test_gp_branch_perfect_mean_with_tiny_sigma_gives_near_zero_crps():
    y = np.array([1.0, 2.0, 3.0])
    out = compute_calibration_metrics(
        {"mean": y.copy(), "y_std": np.zeros_like(y)},  # sigma clipped to 1e-9
        y, "GP Posterior")
    assert out["CRPS"] == pytest.approx(0.0, abs=1e-6)
    assert all(actual == 1.0 for _, actual in out["reliability_curve"])


@pytest.mark.parametrize("method,result", [
    ("Bootstrapping", {}),                                   # raw_preds missing
    ("Bootstrapping", {"raw_preds": np.empty((0, 0))}),      # raw_preds empty
    ("Conformal", {"mean": np.array([1.0])}),                # residuals missing
    ("Conformal", {"calibration_residuals": np.array([])}),  # residuals empty
    ("GP Posterior", {"mean": np.array([1.0])}),             # y_std missing
    ("Not A Method", {"mean": np.array([1.0])}),             # unknown method
])
def test_missing_or_invalid_inputs_return_none_not_a_crash(method, result):
    out = compute_calibration_metrics(result, np.array([1.0]), method)
    assert out is None


# ── bootstrap_uncertainty ─────────────────────────────────────────────────────

def test_bootstrap_same_seed_reproduces_identical_intervals():
    X_train, y_train, X_test = _uq_data(n=30)

    def run(seed):
        return bootstrap_uncertainty(
            LinearRegression(), X_train, y_train, X_test, "T",
            n_bootstrap=5, confidence_interval=95, n_jobs=1, random_state=seed)

    mean_a, lo_a, hi_a, preds_a = run(42)
    mean_b, lo_b, hi_b, preds_b = run(42)
    assert np.array_equal(preds_a, preds_b)
    assert np.array_equal(lo_a, lo_b) and np.array_equal(hi_a, hi_b)

    # And a different seed resamples differently.
    _, _, _, preds_c = run(43)
    assert not np.array_equal(preds_a, preds_c)


# ── gp_posterior_uncertainty ──────────────────────────────────────────────────

def test_gp_posterior_returns_none_for_any_non_gpr_model():
    # The native posterior interval only exists for GaussianProcessRegressor;
    # every other model type must get None (the caller skips the method),
    # never a bogus interval from some unrelated attribute.
    X_train, y_train, X_test = _uq_data(n=30)
    out = gp_posterior_uncertainty(
        LinearRegression(), X_train, y_train, X_test, "T", confidence_interval=95)
    assert out is None


def test_gp_posterior_interval_is_mean_plus_minus_z_sigma():
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor

    X_train, y_train, X_test = _uq_data(n=30)
    model = GaussianProcessRegressor(alpha=1e-2)
    y_pred, lower, upper, y_std = gp_posterior_uncertainty(
        model, X_train, y_train, X_test, "T", confidence_interval=95)

    z = norm.ppf(0.5 + 95 / 200.0)   # two-sided 95% -> z ≈ 1.96
    assert np.allclose(upper - y_pred, z * y_std)
    assert np.allclose(y_pred - lower, z * y_std)
    # And the caller's instance must not have been fitted in place.
    assert not hasattr(model, "X_train_")


def test_gp_posterior_returns_none_instead_of_crashing_on_a_fit_failure(capsys):
    """Regression test for a real risk: unlike every other risky call in this
    module, gp_posterior_uncertainty had no try/except around fit/predict — a
    singular-kernel LinAlgError (plausible with duplicate rows) would abort UQ
    for every remaining model/target instead of just skipping this one method
    (the same way a non-GPR model already skips via a None return)."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    class _BrokenGPR(GaussianProcessRegressor):
        def fit(self, X, y):
            raise np.linalg.LinAlgError("singular matrix")

    X_train, y_train, X_test = _uq_data(n=30)
    out = gp_posterior_uncertainty(
        _BrokenGPR(), X_train, y_train, X_test, "T", confidence_interval=95)
    assert out is None
    assert "singular matrix" in capsys.readouterr().out


# ── calculate_coverage ────────────────────────────────────────────────────────

def test_coverage_counts_inclusive_bounds():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([1.0, 0.0, 3.5, 0.0])   # y[0] exactly ON the bound
    upper = np.array([1.0, 3.0, 4.0, 3.0])   # y[3] above its upper bound
    # In-interval: y[0] (inclusive), y[1]; out: y[2] (below lower), y[3].
    assert calculate_coverage(y, lower, upper) == pytest.approx(50.0)
