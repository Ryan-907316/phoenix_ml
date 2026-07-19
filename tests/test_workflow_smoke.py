"""End-to-end smoke test: runs a tiny synthetic dataset through the standalone
run_workflow() entry point and checks it completes and produces real output.

This is deliberately the slowest test in the suite, but it's the one most
likely to catch the kind of bug this project has actually hit — both the
monotonicity-mismatch bug and the interpretability visual-restriction bug
were wiring issues between modules, not broken logic in any single function,
and only a test that runs the real pipeline end-to-end would catch that class
of bug.

HPO and UQ are switched off and SHAP/waterfall/sensitivity plots are trimmed
down to keep this fast (a few seconds), while still exercising preprocessing,
baseline training, ICE/PDP + ALE interpretability, and report/Excel export
together.
"""
import os

import pandas as pd

from phoenix_ml.workflow import run_workflow


def test_tiny_workflow_runs_end_to_end(tmp_path, synthetic_dataset_csv):
    output_dir = tmp_path / "output"

    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["Random Forest Regressor"],
        targets=["Target"],
        perform_hpo=False,
        perform_uq=False,
        perform_cv=False,
        perform_interpretability=True,
        interpretability_settings=dict(
            test_sample_size=30, background_sample_size=5,
            subsample=10, grid_resolution=5,
            show_shap_summary=False, show_shap_dependence=False,
            show_shap_waterfall=False, show_sensitivity_sobol=False,
        ),
        random_seed=0,
    )

    assert os.path.exists(result["pdf"])
    assert os.path.getsize(result["pdf"]) > 0

    assert result["xlsx"] is not None
    assert os.path.exists(result["xlsx"])

    assert result["models"]
    for path in result["models"].values():
        if isinstance(path, str):
            assert os.path.exists(path)


def test_uq_after_hpo_is_skipped_entirely_when_hpo_did_not_run(tmp_path, synthetic_dataset_csv):
    """Regression test for a real bug found via a systematic failure-mode sweep:
    with perform_hpo=False and perform_uq=True, get_all_models_tuned_per_target()
    returns {} (nothing in metrics["random"/"hyperopt"/"skopt"] to look up), so
    run_uncertainty_quantification's per-model loop ran zero times and returned an
    empty-but-non-None DataFrame. The report/Excel gates only checked "is not
    None", so a full "Uncertainty Quantification (After HPO)" section rendered
    with an empty table — reading as "we quantified uncertainty and found
    nothing" rather than "this never ran". Fixed by gating the whole UQ-after
    computation on perform_hpo too, with an .empty check at every render/export
    site as defense in depth."""
    output_dir = tmp_path / "output"
    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["KNeighbors Regressor"],
        targets=["Target"],
        perform_hpo=False,
        perform_uq=True,
        perform_cv=False,
        perform_interpretability=False,
        uq_settings=dict(
            uq_method="Conformal", n_bootstrap=2, confidence_interval=95,
            calibration_frac=0.1, subsample_test_size=16, n_jobs=1,
            include_gp_posterior=False, calibration_enabled=False,
        ),
        random_seed=0,
    )

    assert os.path.exists(result["xlsx"])
    xlsx = pd.ExcelFile(result["xlsx"])
    # UQ Before HPO doesn't depend on HPO having run, and must have real data.
    assert "UQ Before HPO" in xlsx.sheet_names
    before_df = pd.read_excel(result["xlsx"], sheet_name="UQ Before HPO")
    assert not before_df.empty
    # UQ After HPO must be absent entirely -- not present-but-empty.
    assert "UQ After HPO" not in xlsx.sheet_names


def test_metadata_does_not_record_uq_settings_when_uq_was_skipped(tmp_path, synthetic_dataset_csv):
    """Regression test for a real risk: explicit uq_settings passed alongside
    perform_uq=False still got recorded in metadata.json, misleadingly
    claiming UQ was configured/used for a run where it never actually ran."""
    import json

    output_dir = tmp_path / "output"
    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["Random Forest Regressor"],
        targets=["Target"],
        perform_hpo=False,
        perform_uq=False,
        perform_cv=False,
        perform_interpretability=False,
        uq_settings=dict(
            uq_method="Conformal", n_bootstrap=2, confidence_interval=95,
            calibration_frac=0.1, subsample_test_size=16, n_jobs=1,
            include_gp_posterior=False, calibration_enabled=False,
        ),
        random_seed=0,
    )

    with open(result["models"]["metadata"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["settings"]["uq"] is None


def test_workflow_is_reproducible_with_same_seed(tmp_path, synthetic_dataset_csv):
    # Same seed, same config, twice -> the deployable predictor's baseline
    # training metrics should match exactly. This is the behaviour the random
    # seed unification work (derive_seed / reset_model_to_defaults) exists to
    # guarantee.
    from phoenix_ml.data_preprocessing import run_preprocessing_workflow
    from phoenix_ml.model_training import (
        run_model_training_workflow, derive_seed, SEED_OFFSET_MODEL_CONSTRUCTION,
        SEED_OFFSET_TRAIN_TEST_SPLIT, metrics_dict,
    )
    import copy
    from phoenix_ml.models import models_dict as ALL_MODELS

    def train_once():
        # split_method="random" deliberately exercises the seed-unification
        # guarantee end to end: workflow.py only derives a split seed (rather
        # than passing None) when the split method is "random" — this must be
        # done the same way here, or the split itself is unseeded and nothing
        # downstream can be reproducible regardless of model random_state.
        results = run_preprocessing_workflow(
            file_path=str(synthetic_dataset_csv),
            target_columns=["Target"],
            split_method="random",
            random_state=derive_seed(0, SEED_OFFSET_TRAIN_TEST_SPLIT),
        )
        selected = {"Random Forest Regressor": copy.deepcopy(ALL_MODELS["Random Forest Regressor"])}
        model_seed = derive_seed(0, SEED_OFFSET_MODEL_CONSTRUCTION)
        for model in selected.values():
            if "random_state" in model.get_params():
                model.set_params(random_state=model_seed)

        results_df, _, _, _ = run_model_training_workflow(
            X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
            y_train=results["y_train"], y_test=results["y_test"],
            target_columns=results["target_columns"],
            selected_model_names=["Random Forest Regressor"],
            selected_metrics={"Q^2": metrics_dict["Q^2"]},
            feature_names=results["feature_names"],
            random_state=model_seed,
        )
        return results_df

    df_a = train_once()
    df_b = train_once()
    # "Time Elapsed (s)" is wall-clock and never reproducible; every other
    # column (predictions, hence Q^2) must match exactly.
    cols = [c for c in df_a.columns if c != "Time Elapsed (s)"]
    assert df_a[cols].equals(df_b[cols])


# ── Minimal/degenerate configuration combinations ────────────────────────────
#
# workflow.py's standalone run_workflow() entry point has no way to enable
# PERL/physics reconstruction at all (save_predictor always passes
# physics_config=None) — every run through it is a "no-physics run" by
# construction. These four tests each isolate one minimal configuration
# dimension that no existing test covers explicitly.

def test_no_physics_run_records_perl_disabled_in_metadata(tmp_path, synthetic_dataset_csv):
    """workflow.py's standalone entry point never supports PERL — metadata.json
    must consistently record that (perl_enabled: False, no reconstruction_map)
    rather than omitting the physics block or leaving it ambiguous."""
    import json

    output_dir = tmp_path / "output"
    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["Random Forest Regressor"],
        targets=["Target"],
        perform_hpo=False,
        perform_uq=False,
        perform_cv=False,
        perform_interpretability=False,
        random_seed=0,
    )
    with open(result["models"]["metadata"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["physics"]["perl_enabled"] is False
    assert meta["physics"]["note"] is None


def test_single_target_run_skips_target_vs_target_plot_without_crashing(tmp_path, synthetic_dataset_csv):
    """plot_target_vs_target requires >= 2 targets and returns None otherwise
    (see data_preprocessing.py) — with the single-target dataset every other
    smoke test also uses, that figure must be cleanly absent from the report
    images, not attempted and not a crash."""
    output_dir = tmp_path / "output"
    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["Random Forest Regressor"],
        targets=["Target"],
        perform_hpo=False,
        perform_uq=False,
        perform_cv=False,
        perform_interpretability=False,
        random_seed=0,
    )
    assert os.path.exists(result["pdf"])
    assert not os.path.exists(
        os.path.join(result["images_dir"], "preprocessing_target_vs_target.png"))


def test_single_model_run_omits_pareto_section_without_crashing(tmp_path, synthetic_dataset_csv):
    """run_pareto_analysis is only called when len(selected_models) >= 2 (see
    workflow.py) — every existing end-to-end test happens to use exactly one
    model, so this guard itself was never explicitly exercised. With one
    model, no pareto_*.png should exist and the report must still complete."""
    output_dir = tmp_path / "output"
    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["Random Forest Regressor"],
        targets=["Target"],
        perform_hpo=True,
        perform_uq=False,
        perform_cv=False,
        perform_interpretability=False,
        methods_to_run=["random"],
        sampling_method="Random",
        n_iter=3,
        n_jobs=1,
        random_seed=0,
    )
    assert os.path.exists(result["pdf"])
    pareto_files = [f for f in os.listdir(result["images_dir"]) if f.startswith("pareto_")]
    assert pareto_files == []


def test_every_optional_step_off_still_produces_a_valid_minimal_report(tmp_path, synthetic_dataset_csv):
    """CV, UQ, and Interpretability all switched off together (HPO left on) —
    no existing test tries every optional analysis step off at once. The
    report/Excel must still generate, with those sections cleanly absent
    rather than rendering empty or crashing."""
    output_dir = tmp_path / "output"
    result = run_workflow(
        dataset_path=str(synthetic_dataset_csv),
        output_dir=str(output_dir),
        selected_models=["Random Forest Regressor"],
        targets=["Target"],
        perform_hpo=True,
        perform_uq=False,
        perform_cv=False,
        perform_interpretability=False,
        methods_to_run=["random"],
        sampling_method="Random",
        n_iter=3,
        n_jobs=1,
        random_seed=0,
    )
    assert os.path.exists(result["pdf"])
    assert os.path.getsize(result["pdf"]) > 0
    assert os.path.exists(result["xlsx"])

    xlsx = pd.ExcelFile(result["xlsx"])
    assert "Model Training Results" in xlsx.sheet_names
    assert "HPO Results" in xlsx.sheet_names
    for absent in ("UQ Before HPO", "UQ After HPO",
                   "Interpretability Before HPO", "Residual Transformations"):
        assert absent not in xlsx.sheet_names
