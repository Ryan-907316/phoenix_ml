"""End-to-end test of the WorkflowSession/step-function layer — the path the
GUI actually drives (workflow.py's standalone entry point is a separate,
simpler path already covered by test_workflow_smoke.py).

This layer is where several real bugs have lived (the interpretability
visual-restriction wiring, per-target monotonicity gating), because it's the
glue between every module — and until this file existed it had zero automated
coverage despite being the daily-use path.

Config is trimmed for speed (one fast model, random-search HPO only with 4
iterations, conformal-only UQ, metrics-only interpretability) but the step
sequence is the real one: preprocessing -> training -> interpretability
(before) -> HPO -> UQ (after) -> report, with the report step also persisting
the deployable predictor exactly as a UI run would.
"""
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from openpyxl import load_workbook

from phoenix_ml.persistence import PhoenixPredictor
from phoenix_ml.workflow_steps import (
    WorkflowSession,
    run_step_hpo,
    run_step_interpretability_before,
    run_step_perl,
    run_step_preprocessing,
    run_step_report,
    run_step_training,
    run_step_uq_after,
)


def test_reset_results_clears_every_result_but_keeps_settings():
    """Regression test for a real bug: running dataset A with PERL enabled, then
    dataset B without re-running every step, carried dataset A's stale PERL
    section (and other results) into dataset B's report. reset_results() must
    clear every per-run result while leaving user-configured settings alone."""
    session = WorkflowSession(
        dataset_path="a.csv", output_dir="out", targets=["T"],
        selected_models=["KNeighbors Regressor"], random_seed=7,
        monotonic_constraints={"T": {"f": 1}},
    )
    # Populate every result field as if a previous run had completed.
    session.preprocessing_results = {"stub": True}
    session.training_results = {"stub": True}
    session.uq_before = ({"stub": True}, {})
    session.uq_after = ({"stub": True}, {})
    session.interpretability_before = ({"stub": True}, {})
    session.interpretability_after = ({"stub": True}, {})
    session.hpo_results = {"stub": True}
    session.cv_results = {"stub": True}
    session.perl_results = {"stub": True}
    session.perl_config = {"stub": True}
    session.perl_output_df = pd.DataFrame({"a": [1]})
    session.cleaning_summary = {"export_path": "x"}
    session.metrics["random"] = {"stub": True}
    session.params["random"] = {"stub": True}
    session.total_elapsed = 123.4
    session.step_timings = [("Preprocessing", 1.0)]
    session.images_dir = "/old/images"
    session.report_dir = "/old/report"
    session.xlsx_path = "/old/x.xlsx"
    session.pdf_path = "/old/r.pdf"
    session.models_dir = "/old/models"

    session.reset_results()

    for field in ["preprocessing_results", "training_results", "uq_before", "uq_after",
                  "interpretability_before", "interpretability_after", "hpo_results",
                  "cv_results", "perl_results", "perl_config", "perl_output_df",
                  "cleaning_summary", "images_dir", "report_dir", "xlsx_path",
                  "pdf_path", "models_dir"]:
        assert getattr(session, field) is None, f"{field} was not cleared"
    assert session.metrics == {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}}
    assert session.params == {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}}
    assert session.total_elapsed == 0.0
    assert session.step_timings == []

    # Settings must survive untouched.
    assert session.dataset_path == "a.csv"
    assert session.output_dir == "out"
    assert session.targets == ["T"]
    assert session.selected_models == ["KNeighbors Regressor"]
    assert session.random_seed == 7
    assert session.monotonic_constraints == {"T": {"f": 1}}


def test_rerunning_preprocessing_clears_stale_results_from_a_previous_dataset(
    tmp_path, synthetic_dataset_csv,
):
    """End-to-end version of the same bug: a session that already has PERL/HPO
    results from a first dataset must have them gone the moment preprocessing
    runs again for a second (different) dataset — without the user needing to
    click anything else. This is the automatic, can't-forget-it fix; the manual
    Reset button in the UI calls the same reset_results() method directly."""
    session = WorkflowSession(
        dataset_path=str(synthetic_dataset_csv), output_dir=str(tmp_path),
        targets=["Target"], selected_models=["KNeighbors Regressor"], random_seed=0,
        show_target_vs_target=False, show_features_vs_targets=False,
        show_boxplots=False, show_distance_corr=False,
        show_multicollinearity=False, plot_pca_enabled=False, feat_sel_enabled=False,
    )
    # Simulate a completed prior run on a DIFFERENT dataset.
    session.perl_results = {"stale": "from a previous dataset"}
    session.perl_config = {"stale": True}
    session.hpo_results = {"stale": True}
    session.total_elapsed = 999.0
    session.step_timings = [("Hyperparameter Optimisation", 999.0)]
    old_images_dir = str(tmp_path / "stale_images")
    session.images_dir = old_images_dir

    run_step_preprocessing(session)

    assert session.perl_results is None
    assert session.perl_config is None
    assert session.hpo_results is None
    assert session.total_elapsed == 0.0
    assert session.step_timings == []
    # The stale path must have been recomputed, not silently kept.
    assert session.images_dir != old_images_dir
    assert session.preprocessing_results is not None


def test_selected_warns_once_for_a_model_name_absent_from_all_models(capsys):
    """Regression test for a real risk: session.selected silently dropped any
    selected_models entry not present in ALL_MODELS (e.g. a stale saved
    config referencing a model since renamed/removed) — the pipeline just
    ran with fewer models than configured, with nothing telling the user."""
    session = WorkflowSession(
        dataset_path="a.csv", output_dir="out", targets=["T"],
        selected_models=["KNeighbors Regressor", "Not A Real Model"],
    )
    models = session.selected
    assert list(models.keys()) == ["KNeighbors Regressor"]
    out = capsys.readouterr().out
    assert "[WARN]" in out and "Not A Real Model" in out

    # Repeated access must not reprint the same warning.
    _ = session.selected
    _ = session.selected
    assert capsys.readouterr().out == ""


def test_perl_warns_when_every_target_is_skipped(tmp_path, synthetic_dataset_csv, capsys):
    """Regression test for a real risk: a PERL run where every target hits a
    per-target skip (e.g. a reconstruction_map typo pointing at a physics
    column that was never produced) left perl_results = {}, indistinguishable
    from "PERL was never attempted" — the report section silently vanishes
    (gated on `if session.perl_results:`, falsy for {}) with nothing telling
    the user PERL ran and found nothing usable."""
    from phoenix_ml.physics_expressions import save_physics_config

    session = _tiny_session(synthetic_dataset_csv, tmp_path / "out")
    run_step_preprocessing(session)
    run_step_training(session)

    config_path = str(tmp_path / "physics_config.json")
    save_physics_config(
        config_path, expressions=[], output_cols_text="",
        reconstruction_map={"Target": "nonexistent_physics_col"},
    )
    session.perl_config_path = config_path

    run_step_perl(session)

    assert session.perl_results == {}
    assert "every target was skipped" in capsys.readouterr().out


def _tiny_session(dataset_csv, output_dir):
    return WorkflowSession(
        dataset_path=str(dataset_csv),
        output_dir=str(output_dir),
        targets=["Target"],
        selected_models=["KNeighbors Regressor"],
        random_seed=0,
        # Keep preprocessing plot generation off — the report must cope with
        # every optional figure absent (the UI exposes these as checkboxes).
        show_target_vs_target=False, show_features_vs_targets=False,
        show_boxplots=False, show_distance_corr=False,
        show_multicollinearity=False, plot_pca_enabled=False,
        feat_sel_enabled=False,
        # Metrics-only interpretability: no ICE/PDP/ALE/SHAP visuals, just a
        # tiny Morris pass feeding the comparable-metrics table.
        interpretability_settings=dict(
            test_sample_size=40, background_sample_size=5,
            subsample=10, grid_resolution=5,
            show_ice_pdp=False, show_ale=False, show_shap_summary=False,
            show_shap_dependence=False, show_shap_waterfall=False,
            show_sensitivity_morris=True, sensitivity_morris_trajectories=4,
            sensitivity_morris_levels=4, show_sensitivity_sobol=False,
        ),
        methods_to_run=["random"],
        sampling_method="Random",
        n_iter=4,
        n_jobs=1,
        early_stopping=None,
        uq_settings=dict(
            uq_method="Conformal", n_bootstrap=2, confidence_interval=95,
            calibration_frac=0.1, subsample_test_size=16, n_jobs=1,
            include_gp_posterior=False, calibration_enabled=True,
        ),
    )


def test_session_pipeline_end_to_end(tmp_path, synthetic_dataset_csv):
    session = _tiny_session(synthetic_dataset_csv, tmp_path / "out")

    # Prerequisite gates must open in order, exactly as the UI's step
    # checkboxes rely on them doing.
    assert session.can_run_preprocessing()
    assert not session.can_run_training()

    run_step_preprocessing(session)
    assert session.preprocessing_results is not None
    assert session.can_run_training()
    assert not session.can_run_hpo()

    run_step_training(session)
    assert session.training_results is not None
    assert not session.training_results["results_df"].empty
    assert session.can_run_hpo()
    assert not session.can_run_uq_after()          # needs HPO first

    run_step_interpretability_before(session)
    metrics_df, figures = session.interpretability_before
    # One comparable-metrics row per (model, target).
    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["Model"] == "KNeighbors Regressor"

    run_step_hpo(session)
    assert session.hpo_results is not None
    best = session.hpo_results["best_models_per_target"]
    assert "Target" in best
    assert session.can_run_uq_after()

    run_step_uq_after(session)
    uq_df, uq_figs = session.uq_after
    assert not uq_df.empty
    assert (uq_df["UQ Method"] == "Conformal").all()

    assert session.can_generate_report()
    run_step_report(session)

    # The two artifacts every UI run hands the user: the PDF report and the
    # deployable predictor.
    assert os.path.isfile(session.pdf_path)
    assert os.path.getsize(session.pdf_path) > 10_000   # a real multi-page PDF

    # Regression test: sheets used to be written with openpyxl's unset default
    # column width, clipping headers/values when the workbook was opened.
    # Every column must now be sized to fit its widest cell.
    assert session.xlsx_path and os.path.isfile(session.xlsx_path)
    wb = load_workbook(session.xlsx_path)
    assert wb.sheetnames, "workbook must contain at least one sheet"
    for sheet in wb.worksheets:
        for col_cells in sheet.columns:
            letter = col_cells[0].column_letter
            width = sheet.column_dimensions[letter].width
            longest = max((len(str(c.value)) for c in col_cells if c.value is not None), default=0)
            assert width is not None, f"{sheet.title}!{letter} has no explicit column width"
            if longest:
                assert width >= min(longest, 60), (
                    f"{sheet.title}!{letter} width {width} too narrow for its longest value ({longest} chars)"
                )

    predictor_files = [f for f in os.listdir(session.models_dir)
                       if f == "phoenix_ml Predictor.pkl"]
    assert len(predictor_files) == 1

    # And that predictor must actually work on fresh raw-feature data.
    predictor = PhoenixPredictor.load(os.path.join(session.models_dir, predictor_files[0]))
    raw = pd.read_csv(synthetic_dataset_csv).drop(columns=["Target"])
    preds = predictor.predict(raw)
    assert list(preds.columns) == ["Target"]
    assert len(preds) == len(raw)
    assert preds["Target"].notna().all()
