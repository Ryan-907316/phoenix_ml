"""Tests for physics_model.py (Script Mode PERL engine) and the script-mode
half of PhoenixPredictor.

Script Mode is the package's headline feature — the user writes a physics
script exporting governing_function/constants/input_vars/output_vars, and the
engine runs it to produce '<var>_physics' estimate columns. Until this file
existed it had zero automated coverage despite being the mode the bundled DC
motor example (and the real report) uses.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from phoenix_ml.persistence import PhoenixPredictor
from phoenix_ml.physics_model import (
    compute_residuals,
    generate_residual_dataset,
    generate_simple_dataset,
    import_physics_script,
    round_and_clean_floats,
    run_physics_model,
)


def _governing_function(inputs, constants, time):
    # Hand-checkable physics: Speed = k * a  (k = 2)
    return pd.DataFrame({"Speed": constants["k"] * inputs["a"]})


_DATA = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [10.0, 20.0, 30.0],
    "Speed": [2.5, 3.5, 7.0],   # measured values, deliberately off the physics
})


def test_run_physics_model_renames_outputs_with_physics_suffix():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    assert list(phys["Speed_physics"]) == [2.0, 4.0, 6.0]


def test_run_physics_model_name_mapping_maps_internal_to_display_names():
    # A script may compute internal variable names ("Speed") while the dataset
    # uses display names ("Motor Speed") — name_mapping bridges the two.
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Motor Speed"],
        name_mapping={"Motor Speed": "Speed"},
    )
    assert list(phys["Motor Speed_physics"]) == [2.0, 4.0, 6.0]


def test_run_physics_model_raises_loudly_on_missing_input_column():
    """Regression test for a real risk: a config reused on a renamed/different
    dataset indexed straight into a missing column, raising a raw KeyError
    instead of naming what's missing."""
    with pytest.raises(ValueError, match="c"):
        run_physics_model(
            _DATA, time_col=None, governing_function=_governing_function,
            constants={"k": 2.0}, input_vars=["a", "c"], output_vars=["Speed"],
        )


def test_run_physics_model_raises_loudly_on_missing_time_column():
    with pytest.raises(ValueError, match="t"):
        run_physics_model(
            _DATA, time_col="t", governing_function=_governing_function,
            constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
        )


# ── round_and_clean_floats ────────────────────────────────────────────────────

def test_round_and_clean_floats_snaps_near_integers_and_rounds_others():
    df = pd.DataFrame({"x": [1.0000001, 2.123456789, 3.0]})
    out = round_and_clean_floats(df, decimal_places=6)
    assert list(out["x"]) == [1.0, 2.123457, 3.0]


def test_round_and_clean_floats_survives_nan_instead_of_crashing():
    """Regression test for a real bug found while vectorising this function:
    the old per-cell round(x) (no ndigits) raises ValueError on NaN — a live
    input here since inf/-inf are replaced with NaN two lines earlier in the
    same function, so any physics-modelled dataset producing +-inf crashed."""
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, np.nan]})
    out = round_and_clean_floats(df)
    assert out["x"].iloc[0] == 1.0
    assert out["x"].iloc[1:].isna().all()


def test_round_and_clean_floats_leaves_non_float_columns_untouched():
    df = pd.DataFrame({"x": [1.23456789], "label": ["A"], "n": [5]})
    out = round_and_clean_floats(df, decimal_places=3)
    assert out["x"].iloc[0] == 1.235
    assert out["label"].iloc[0] == "A"
    assert out["n"].iloc[0] == 5


def test_compute_residuals_is_measured_minus_simulated():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    residuals = compute_residuals(_DATA, phys, ["Speed"])
    # measured [2.5, 3.5, 7.0] - physics [2, 4, 6]
    assert np.allclose(residuals["Residual Speed"], [0.5, -0.5, 1.0])


def test_generate_residual_dataset_keeps_inputs_and_residual_targets():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    out = generate_residual_dataset(_DATA, phys, input_vars=["a", "b"], output_vars=["Speed"])
    assert list(out.columns) == ["a", "b", "Residual Speed"]
    assert np.allclose(out["Residual Speed"], [0.5, -0.5, 1.0])


def test_compute_residuals_raises_loudly_on_missing_measured_column():
    """Regression test for a real bug: a missing measured or physics column
    used to be silently skipped (the output var was just dropped from the
    result with no error), rather than raising like the third sibling
    function (generate_simple_dataset) already did via a raw KeyError."""
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    with pytest.raises(ValueError, match="Torque"):
        compute_residuals(_DATA, phys, ["Speed", "Torque"])


def test_compute_residuals_raises_loudly_on_missing_physics_column():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    with pytest.raises(ValueError, match="Torque_physics"):
        compute_residuals(_DATA.assign(Torque=1.0), phys, ["Speed", "Torque"])


def test_generate_residual_dataset_raises_loudly_on_missing_measured_column():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    with pytest.raises(ValueError, match="Torque"):
        generate_residual_dataset(_DATA, phys, input_vars=["a"], output_vars=["Speed", "Torque"])


def test_generate_residual_dataset_raises_loudly_on_missing_input_column():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    with pytest.raises(ValueError, match="c"):
        generate_residual_dataset(_DATA, phys, input_vars=["a", "c"], output_vars=["Speed"])


def test_generate_simple_dataset_raises_loudly_instead_of_a_raw_keyerror():
    """Regression test for a real bug: this function had no validation at
    all and raised an opaque pandas KeyError on a missing physics/input
    column, unlike its two siblings which at least did something (even if
    the wrong thing — silently skipping)."""
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    with pytest.raises(ValueError, match="Torque_physics"):
        generate_simple_dataset(_DATA, phys, input_vars=["a"], output_vars=["Speed", "Torque"])
    with pytest.raises(ValueError, match="c"):
        generate_simple_dataset(_DATA, phys, input_vars=["a", "c"], output_vars=["Speed"])


def test_generate_simple_dataset_combines_inputs_and_physics_outputs():
    phys = run_physics_model(
        _DATA, time_col=None, governing_function=_governing_function,
        constants={"k": 2.0}, input_vars=["a"], output_vars=["Speed"],
    )
    out = generate_simple_dataset(_DATA, phys, input_vars=["a", "b"], output_vars=["Speed"])
    assert list(out.columns) == ["a", "b", "Speed"]
    assert np.allclose(out["Speed"], [2.0, 4.0, 6.0])


def test_import_physics_script_loads_required_exports(tmp_path):
    script_path = tmp_path / "physics.py"
    script_path.write_text(
        "import pandas as pd\n"
        "constants = {'k': 2.0}\n"
        "input_vars = ['a']\n"
        "output_vars = ['Speed']\n"
        "def governing_function(inputs, constants, time):\n"
        "    return pd.DataFrame({'Speed': constants['k'] * inputs['a']})\n",
        encoding="utf-8",
    )
    module = import_physics_script(str(script_path))
    assert module.constants == {"k": 2.0}
    assert module.input_vars == ["a"]
    out = module.governing_function(_DATA[["a"]], module.constants, np.zeros(3))
    assert list(out["Speed"]) == [2.0, 4.0, 6.0]


# ── PhoenixPredictor script mode ─────────────────────────────────────────────

_SCRIPT_SOURCE = """
import pandas as pd
constants = {"k": 2.0}
input_vars = ["a"]
output_vars = ["Speed"]
def governing_function(inputs, constants, time):
    return pd.DataFrame({"Speed": constants["k"] * inputs["a"]})
"""

_SCRIPT_CONFIG = {
    "mode": "script",
    "reconstruction_map": {"Residual Speed": "Speed_physics"},
    "constants": {"k": 2.0},
    "input_vars": ["a"],
    "output_vars": ["Speed"],
    "time_col": None,
}


def _script_mode_predictor():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.1, 0.2, 0.3]})
    pipe = Pipeline([("model", LinearRegression())]).fit(X, X["a"] * 0.5)
    predictor = PhoenixPredictor(
        {"Residual Speed": pipe},
        physics_config=_SCRIPT_CONFIG,
        physics_script_source=_SCRIPT_SOURCE,
    )
    return X, predictor


def test_script_mode_prediction_adds_physics_estimate_to_ml_residual():
    X, predictor = _script_mode_predictor()
    out = predictor.predict(X)
    # ML residual model predicts 0.5*a; physics adds 2*a -> total 2.5*a.
    assert np.allclose(out["Residual Speed"], 2.5 * X["a"])


def test_script_mode_predictor_survives_save_after_predict(tmp_path):
    """Regression test for a real bug: calling predict() populates the
    exec'd physics namespace cache, which holds module objects that cannot
    pickle — so 'test the predictor, then save it' crashed with
    "cannot pickle 'module' object". The cache must be dropped at pickle time
    and rebuilt from the stored script source after load."""
    X, predictor = _script_mode_predictor()
    before = predictor.predict(X)          # populates the exec namespace cache

    path = tmp_path / "predictor.pkl"
    predictor.save(str(path))              # must not raise
    loaded = PhoenixPredictor.load(str(path))
    after = loaded.predict(X)              # must rebuild the namespace via exec

    assert np.allclose(before.values, after.values)
