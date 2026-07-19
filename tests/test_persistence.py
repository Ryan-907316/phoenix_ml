"""Tests for persistence.py — the deployable-artifact layer.

PhoenixPredictor is the single .pkl end users actually deploy (the report
tells them to load it and call .predict()), so the save -> load -> predict
round trip here is the most load-bearing contract in the package: if it
drifts, users find out at deployment time, not during a run.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phoenix_ml.persistence import (
    PhoenixPredictor,
    build_and_fit_best_models,
    build_uq_calibrators,
    load_bundle,
    load_pipeline,
    save_models_and_artifacts,
    save_predictor,
)


def _toy_pipelines(n=30, seed=0):
    """Two tiny fitted scaler+LinearRegression pipelines over shared features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-5, 5, n)})
    targets = {
        "Target One": 2.0 * X["a"] + 1.0,
        "Target Two": -3.0 * X["b"] + 4.0,
    }
    pipes = {}
    for name, y in targets.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipe.fit(X, y)
        pipes[name] = pipe
    return X, pipes


def test_predictor_save_load_predict_round_trip(tmp_path):
    X, pipes = _toy_pipelines()
    predictor = PhoenixPredictor(pipes, metadata={"timestamp": "test"})
    before = predictor.predict(X)

    path = tmp_path / "predictor.pkl"
    predictor.save(str(path))
    loaded = PhoenixPredictor.load(str(path))
    after = loaded.predict(X)

    assert list(after.columns) == ["Target One", "Target Two"]
    assert after.index.equals(X.index)
    assert np.allclose(before.values, after.values)


def test_predict_interval_only_offered_for_calibrated_targets():
    X, pipes = _toy_pipelines()
    predictor = PhoenixPredictor(
        pipes,
        uq_calibrators={"Target One": {"method": "Conformal", "half_width": 1.5}},
    )
    intervals = predictor.predict_interval(X)

    # Documented contract: targets without a stored calibrator are silently
    # absent from the result, not an error and not a zero-width interval.
    assert set(intervals) == {"Target One"}

    frame = intervals["Target One"]
    preds = predictor.predict(X)["Target One"]
    assert np.allclose(frame["lower"], preds - 1.5)
    assert np.allclose(frame["upper"], preds + 1.5)


def test_predict_adds_physics_reconstruction_in_expression_mode(tmp_path):
    # PERL deployment contract: for a target in the reconstruction map, the
    # final prediction is ML residual + physics estimate — and that must
    # survive a save/load cycle, since the physics config travels inside the
    # pickle.
    X, pipes = _toy_pipelines()
    physics_config = {
        "mode": "expression",
        "reconstruction_map": {"Target One": "phys_est"},
        "expressions": ["phys_est = `a` * 2"],
    }
    plain = PhoenixPredictor({k: v for k, v in pipes.items()})
    perl = PhoenixPredictor(dict(pipes), physics_config=physics_config)

    path = tmp_path / "perl.pkl"
    perl.save(str(path))
    loaded = PhoenixPredictor.load(str(path))
    out = loaded.predict(X)

    ml_only = plain.predict(X)
    assert np.allclose(out["Target One"], ml_only["Target One"] + 2.0 * X["a"])
    # Targets outside the reconstruction map are pure ML.
    assert np.allclose(out["Target Two"], ml_only["Target Two"])


def test_save_predictor_raises_clearly_when_script_mode_source_is_missing(tmp_path):
    """Regression test for a real bug: if a Script Mode PERL config's
    script_path has moved or been deleted between running PERL and saving the
    predictor, save_predictor used to silently set physics_script_source=None
    and save anyway. The resulting predictor would only fail later, at
    deploy-time predict(), with a confusing "'NoneType' object is not
    subscriptable" — instead of failing now, while the user can still fix the
    path. Found via a systematic failure-mode sweep."""
    X, pipes = _toy_pipelines()
    physics_config = {
        "mode": "script",
        "script_path": str(tmp_path / "does_not_exist.py"),
        "reconstruction_map": {"Target One": "phys_est"},
    }
    with pytest.raises(FileNotFoundError, match="does_not_exist.py"):
        save_predictor(
            output_dir=str(tmp_path), pipelines_by_target=pipes,
            physics_config=physics_config,
        )


def test_build_uq_calibrators_prefers_conformal_over_bootstrapping():
    # Bootstrapping row listed FIRST to prove selection follows the documented
    # preference order (Conformal > Bootstrapping > GP Posterior), not row order.
    uq_df = pd.DataFrame({
        "Model": ["M", "M"],
        "Target Variable": ["T", "T"],
        "UQ Method": ["Bootstrapping", "Conformal"],
        "Mean Range": [4.0, 2.0],
    })
    calibrators = build_uq_calibrators(uq_df, {"T": {"model_name": "M"}})
    assert calibrators == {"T": {"method": "Conformal", "half_width": 1.0}}


def test_build_uq_calibrators_falls_back_when_preferred_method_absent():
    uq_df = pd.DataFrame({
        "Model": ["M"],
        "Target Variable": ["T"],
        "UQ Method": ["Bootstrapping"],
        "Mean Range": [4.0],
    })
    calibrators = build_uq_calibrators(uq_df, {"T": {"model_name": "M"}})
    assert calibrators == {"T": {"method": "Bootstrapping", "half_width": 2.0}}


def test_build_uq_calibrators_omits_unmatched_targets_and_handles_empty_input():
    uq_df = pd.DataFrame({
        "Model": ["M"],
        "Target Variable": ["T"],
        "UQ Method": ["Conformal"],
        "Mean Range": [2.0],
    })
    # Target "U" has no UQ rows at all -> omitted, no error.
    calibrators = build_uq_calibrators(uq_df, {"U": {"model_name": "M"}})
    assert calibrators == {}
    assert build_uq_calibrators(None, {"T": {"model_name": "M"}}) == {}
    assert build_uq_calibrators(pd.DataFrame(), {"T": {"model_name": "M"}}) == {}


def test_build_uq_calibrators_skips_nan_mean_range():
    # A NaN Mean Range must never produce a calibrator (a NaN half-width would
    # poison every interval downstream).
    uq_df = pd.DataFrame({
        "Model": ["M"],
        "Target Variable": ["T"],
        "UQ Method": ["Conformal"],
        "Mean Range": [float("nan")],
    })
    calibrators = build_uq_calibrators(uq_df, {"T": {"model_name": "M"}})
    assert "T" not in calibrators


def test_build_uq_calibrators_nan_preferred_method_falls_back_to_next():
    """When the preferred method's Mean Range is NaN, the target must fall
    back to the next method in the preference order rather than being left
    with no interval at all. Regression test for the original behaviour, which
    broke out of the preference loop on the first method present regardless of
    whether it was usable."""
    uq_df = pd.DataFrame({
        "Model": ["M", "M"],
        "Target Variable": ["T", "T"],
        "UQ Method": ["Conformal", "Bootstrapping"],
        "Mean Range": [float("nan"), 4.0],
    })
    calibrators = build_uq_calibrators(uq_df, {"T": {"model_name": "M"}})
    assert calibrators == {"T": {"method": "Bootstrapping", "half_width": 2.0}}


def test_save_models_and_artifacts_round_trips_contents_not_just_files(tmp_path):
    """The saved artifacts must round-trip their CONTENTS: reloaded pipelines
    predict identically to the originals, the metadata JSON preserves the
    reproducibility settings it claims to record, and the bundle carries both.
    (Existence-only checks would pass even if joblib silently pickled a stale
    or empty object.)"""
    import json

    X, pipes = _toy_pipelines()
    hpo_settings = {"methods": ["random"], "n_iter": 50, "metric": "Q^2"}

    save_paths = save_models_and_artifacts(
        output_dir=str(tmp_path),
        pipelines_by_target=pipes,
        feature_names=["a", "b"],
        targets=list(pipes),
        metric_name="Q^2",
        dataset_path="C:/data/motors.csv",
        split_info={"method": "last", "test_size": 0.2},
        extra_meta={"selected_models": ["LinearRegression"]},
        hpo_settings=hpo_settings,
        make_bundle=True,
        prefix="t",
    )

    # Per-target pipelines: reload and compare predictions to the originals.
    for target, path in save_paths["by_target"].items():
        reloaded = load_pipeline(path)
        assert np.allclose(reloaded.predict(X), pipes[target].predict(X))
    # Fixed "{prefix} Pipeline_{target}.pkl" naming (no timestamp — artifacts
    # overwrite per run, matching the report/Excel naming convention).
    import os as _os
    assert _os.path.basename(save_paths["by_target"]["Target One"]) == "t Pipeline_Target One.pkl"
    assert _os.path.basename(save_paths["metadata"]) == "t Metadata.json"
    assert _os.path.basename(save_paths["bundle"]) == "t Bundle.pkl"

    # Metadata JSON: the reproducibility record must survive the round trip.
    with open(save_paths["metadata"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset"]["path"] == "C:/data/motors.csv"
    assert meta["dataset"]["targets"] == ["Target One", "Target Two"]
    assert meta["dataset"]["feature_names"] == ["a", "b"]
    assert meta["dataset"]["split_info"] == {"method": "last", "test_size": 0.2}
    assert meta["settings"]["metric"] == "Q^2"
    assert meta["settings"]["selected_models"] == ["LinearRegression"]
    assert meta["settings"]["hpo"] == hpo_settings

    # Bundle: one file carrying both the pipelines and the same metadata.
    bundle = load_bundle(save_paths["bundle"])
    assert bundle["metadata"] == meta
    for target in pipes:
        assert np.allclose(
            bundle["pipelines_by_target"][target].predict(X), pipes[target].predict(X))


def test_metadata_records_no_perl_when_physics_config_is_absent(tmp_path):
    import json

    X, pipes = _toy_pipelines()
    save_paths = save_models_and_artifacts(
        output_dir=str(tmp_path), pipelines_by_target=pipes,
        feature_names=["a", "b"], targets=list(pipes), metric_name="Q^2",
        dataset_path="x.csv", prefix="t",
    )
    with open(save_paths["metadata"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["physics"]["perl_enabled"] is False
    assert meta["physics"]["note"] is None


def test_metadata_flags_perl_residual_targets_so_pkl_is_not_misread(tmp_path):
    """Regression test for a real bug: metadata.json recorded no physics/PERL
    field at all — for a PERL run, nothing said the saved per-target .pkl
    pipelines predict residuals (measured - physics), not physical values,
    so a .pkl loaded without the report/UI context would be silently
    misread as predicting the physical quantity directly."""
    import json

    X, pipes = _toy_pipelines()
    physics_config = {
        "mode": "script",
        "reconstruction_map": {"Target One": "Target One_physics"},
    }
    save_paths = save_models_and_artifacts(
        output_dir=str(tmp_path), pipelines_by_target=pipes,
        feature_names=["a", "b"], targets=list(pipes), metric_name="Q^2",
        dataset_path="x.csv", prefix="t", physics_config=physics_config,
    )
    with open(save_paths["metadata"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["physics"]["perl_enabled"] is True
    assert meta["physics"]["mode"] == "script"
    assert meta["physics"]["reconstruction_map"] == {"Target One": "Target One_physics"}
    assert "residual" in meta["physics"]["note"].lower()


def test_deployment_toggles_skip_exactly_the_deselected_files(tmp_path):
    # Model Deployment checkboxes map to per-artifact flags: disabling
    # pipelines and metadata must leave only the bundle on disk.
    import os as _os
    X, pipes = _toy_pipelines()
    save_paths = save_models_and_artifacts(
        output_dir=str(tmp_path), pipelines_by_target=pipes,
        feature_names=["a", "b"], targets=list(pipes), metric_name="Q^2",
        dataset_path="x.csv", make_bundle=True, prefix="t",
        save_pipelines=False, save_metadata=False,
    )
    assert save_paths["by_target"] == {}
    assert "metadata" not in save_paths
    files = sorted(_os.listdir(str(tmp_path)))
    assert files == ["t Bundle.pkl"]


def _fit_args():
    rng = np.random.default_rng(1)
    X_train = rng.uniform(0, 1, (20, 2))
    y_train = pd.DataFrame({"T": rng.uniform(0, 1, 20)})
    from phoenix_ml.model_training import reset_model_to_defaults
    from phoenix_ml.hyperparameter_optimisation import process_hyperparameters
    return X_train, y_train, reset_model_to_defaults, process_hyperparameters


def test_build_and_fit_best_models_parses_stringified_dict_hyperparams():
    # Hyperparameters arrive stringified when read back from CSV/JSON exports —
    # the literal_eval path must recover the real dict and apply it.
    X_train, y_train, reset_fn, process_fn = _fit_args()
    fitted = build_and_fit_best_models(
        {"T": {"model_name": "Random Forest Regressor",
               "hyperparameters": "{'n_estimators': 5, 'max_depth': 3}"}},
        X_train, y_train, reset_fn, process_fn,
    )
    assert fitted["T"].get_params()["n_estimators"] == 5
    assert fitted["T"].get_params()["max_depth"] == 3


def test_build_and_fit_best_models_fails_loudly_on_malformed_hyperparameter_string():
    """A hyperparameter string that can't be parsed must raise, not silently
    train the model with default hyperparameters while the caller believes the
    tuned ones were applied — that would corrupt the deployed artifact in a way
    nothing downstream could detect. (The literal_eval failure itself is
    swallowed by design; the loud failure happens at the next step, which is
    what this locks in.)"""
    X_train, y_train, reset_fn, process_fn = _fit_args()
    with pytest.raises(Exception):
        build_and_fit_best_models(
            {"T": {"model_name": "Random Forest Regressor",
                   "hyperparameters": "{'n_estimators': 5"}},  # unclosed brace
            X_train, y_train, reset_fn, process_fn,
        )
