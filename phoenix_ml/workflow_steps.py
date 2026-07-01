# workflow_steps.py
# Stateful session object and per-step functions for the UI's step-by-step execution.
# The existing run_workflow() in workflow.py is unchanged.

from __future__ import annotations
import copy
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from phoenix_ml.models import models_dict as ALL_MODELS
from phoenix_ml.model_training import run_model_training_workflow, metrics_dict, reset_model_to_defaults
from phoenix_ml.data_preprocessing import run_preprocessing_workflow
from phoenix_ml.uncertainty_quantification import run_uncertainty_quantification
from phoenix_ml.interpretability import run_interpretability_analysis
from phoenix_ml.hyperparameter_optimisation import (
    run_all_models_optimisation,
    collect_results_as_dataframe,
    find_best_model_and_hyperparams,
    get_best_models_by_method_across_targets,
    process_hyperparameters,
)
from phoenix_ml.postprocessing import run_postprocessing_analysis
from phoenix_ml.persistence import build_and_fit_best_models, build_pipelines, save_models_and_artifacts
from phoenix_ml.pareto_analysis import run_pareto_analysis, save_pareto_plots
from phoenix_ml.report_generation import *


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


@dataclass
class WorkflowSession:
    # ── User config ──────────────────────────────────────────────────────────
    dataset_path: str = ""
    output_dir: str = ""
    targets: list[str] = field(default_factory=list)
    selected_models: list[str] = field(default_factory=list)

    # Preprocessing
    test_size: float = 0.2
    split_method: str = "last"
    split_random_state: int | None = None
    scaler_type: str = "Standard"
    show_preproc_plots: bool = True
    dist_corr_dummy: bool = True
    dist_corr_mp: bool = False

    # UQ
    uq_settings: dict = field(default_factory=lambda: dict(
        uq_method="Both", n_bootstrap=5, confidence_interval=95,
        calibration_frac=0.05, subsample_test_size=50, n_jobs=1,
    ))

    # Interpretability
    interpretability_settings: dict = field(default_factory=lambda: dict(
        preferred_model_name="XGBoost Regressor",
        test_sample_size=1000, background_sample_size=10,
        subsample=250, grid_resolution=10,
    ))

    # HPO
    hpo_metric: str = "Q^2"
    methods_to_run: list[str] = field(default_factory=lambda: ["random", "hyperopt", "skopt"])
    sampling_method: str = "Sobol"
    n_iter: int = 100
    sample_size: int = 1000
    evals: int = 50
    calls: int = 50
    n_jobs: int = -1
    early_stopping: dict | None = field(default_factory=lambda: {
        "random_search": {"patience": 10,  "min_delta": 1e-4},
        "hyperopt":      {"patience": 10,  "min_delta": 1e-4},
        "skopt":         {"patience": 10,  "min_delta": 1e-4},
    })

    # CV
    cv_method: str = "Shuffle Split"
    cv_args: dict = field(default_factory=lambda: {"n_splits": 10, "test_size": 0.2, "random_state": 0})
    scoring_metric: str = "R^2"

    # PERL Reconstruction
    perl_config_path: str = ""
    perl_results: Any = None
    perl_config: dict | None = None
    perl_output_df: Any = None

    # Report
    report_source: str = "ui"

    # ── Step outputs ─────────────────────────────────────────────────────────
    preprocessing_results: dict | None = None
    training_results: dict | None = None
    uq_before: tuple | None = None
    interpretability_figures: dict | None = None
    hpo_results: dict | None = None
    cv_results: Any = None
    uq_after: tuple | None = None

    # Accumulated metrics/params (training sets defaults; HPO updates them)
    metrics: dict = field(default_factory=lambda: {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}})
    params:  dict = field(default_factory=lambda: {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}})

    # Total elapsed time of all completed steps (set by UI before report generation)
    total_elapsed: float = 0.0

    # ── Paths ────────────────────────────────────────────────────────────────
    images_dir:  str | None = None
    report_dir:  str | None = None
    xlsx_path:   str | None = None
    pdf_path:    str | None = None
    models_dir:  str | None = None

    # ── Helpers ──────────────────────────────────────────────────────────────
    @property
    def selected(self) -> dict:
        return {n: copy.deepcopy(ALL_MODELS[n]) for n in self.selected_models if n in ALL_MODELS}

    def ensure_dirs(self) -> None:
        if self.images_dir is not None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H%M%S")
        rdir = _ensure_dir(os.path.join(self.output_dir, "Report"))
        self.report_dir  = rdir
        self.images_dir  = _ensure_dir(os.path.join(rdir, "Images"))
        self.models_dir  = _ensure_dir(os.path.join(self.output_dir, "Models"))
        self.pdf_path    = os.path.join(rdir, "Phoenix_ML Report.pdf")

    # ── Prerequisites ────────────────────────────────────────────────────────
    def can_run_preprocessing(self)   -> bool: return bool(self.dataset_path and self.targets and self.selected_models and self.output_dir)
    def can_run_training(self)        -> bool: return self.preprocessing_results is not None
    def can_run_uq_before(self)       -> bool: return self.training_results is not None
    def can_run_interpretability(self)-> bool: return self.training_results is not None
    def can_run_hpo(self)             -> bool: return self.training_results is not None
    def can_run_cv(self)              -> bool: return self.training_results is not None
    def can_run_uq_after(self)        -> bool: return self.training_results is not None and self.hpo_results is not None
    def can_run_perl(self)            -> bool: return self.training_results is not None and bool(self.perl_config_path) and os.path.isfile(self.perl_config_path)
    def can_generate_report(self)     -> bool: return self.preprocessing_results is not None and self.training_results is not None


# ── Individual step functions ─────────────────────────────────────────────────

def run_step_preprocessing(session: WorkflowSession) -> None:
    session.ensure_dirs()
    session.preprocessing_results = run_preprocessing_workflow(
        file_path=session.dataset_path,
        test_size=session.test_size,
        split_method=session.split_method,
        target_columns=session.targets,
        plot_target_vs_target_enabled=session.show_preproc_plots,
        plot_features_vs_targets_enabled=session.show_preproc_plots,
        plot_boxplots_enabled=session.show_preproc_plots,
        plot_distance_corr_enabled=session.show_preproc_plots,
        dist_corr_dummy=session.dist_corr_dummy,
        dist_corr_mp=session.dist_corr_mp,
        scaler_type=session.scaler_type,
        random_state=session.split_random_state,
    )


def run_step_training(session: WorkflowSession) -> None:
    r = session.preprocessing_results
    sel_metrics = {k: metrics_dict[k] for k in ["MSE", "R^2", "ADJUSTED R^2", "Q^2"]}
    results_df, def_metrics, def_params, _ = run_model_training_workflow(
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        target_columns=r["target_columns"],
        selected_model_names=session.selected_models,
        selected_metrics=sel_metrics,
    )
    session.training_results = {"results_df": results_df, "default_metrics": def_metrics, "default_params": def_params}
    session.metrics = {"default": def_metrics, "random": {}, "hyperopt": {}, "skopt": {}}
    session.params  = {"default": def_params,  "random": {}, "hyperopt": {}, "skopt": {}}


def run_step_uq_before(session: WorkflowSession) -> None:
    r = session.preprocessing_results
    df, figs = run_uncertainty_quantification(
        models_dict=session.selected,
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        target_columns=session.targets,
        model_names_to_run=session.selected_models,
        stage_label="Before HPO", show_plots=True,
        **session.uq_settings,
    )
    session.uq_before = (df, figs)


def run_step_interpretability(session: WorkflowSession) -> None:
    r = session.preprocessing_results
    session.interpretability_figures = run_interpretability_analysis(
        models_dict=session.selected,
        X_train=r["X_train_scaled"], y_train=r["y_train"],
        target_columns=r["target_columns"], feature_names=r["feature_names"],
        **session.interpretability_settings, show_plots=True,
    )


def run_step_hpo(session: WorkflowSession) -> None:
    session.ensure_dirs()
    r = session.preprocessing_results
    hpo_m, hpo_p, hpo_t, hpo_pl = run_all_models_optimisation(
        models_dict=session.selected,
        selected_model_names=session.selected_models,
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        target_columns=session.targets,
        methods_to_run=list(session.methods_to_run),
        metric=session.hpo_metric,
        sampling_method=session.sampling_method,
        sample_size=session.sample_size,
        n_iter=session.n_iter,
        evals=session.evals,
        calls=session.calls,
        n_jobs=session.n_jobs,
        plot=True,
        output_dir=session.images_dir,
        early_stopping=session.early_stopping,
    )
    session.metrics.update(hpo_m)
    session.params.update(hpo_p)

    collected = collect_results_as_dataframe(
        models_dict=session.selected, target_columns=session.targets,
        default_metrics=session.metrics["default"], default_params=session.params["default"],
        random_metrics=session.metrics["random"],   random_params=session.params["random"],
        random_sampling_method=session.sampling_method,
        hyperopt_metrics=session.metrics["hyperopt"], hyperopt_params=session.params["hyperopt"],
        skopt_metrics=session.metrics["skopt"],       skopt_params=session.params["skopt"],
        metric_name=session.hpo_metric,
    )
    session.hpo_results = {
        "hpo_metrics": hpo_m, "hpo_params": hpo_p,
        "hpo_times": hpo_t,   "hpo_plots": hpo_pl,
        "collected_df": collected,
        "best_models_per_target": find_best_model_and_hyperparams(collected, metric=session.hpo_metric),
    }


def run_step_cv(session: WorkflowSession) -> None:
    session.ensure_dirs()
    r = session.preprocessing_results
    if session.hpo_results:
        best = session.hpo_results["best_models_per_target"]
    else:
        collected = collect_results_as_dataframe(
            models_dict=session.selected, target_columns=session.targets,
            default_metrics=session.metrics["default"], default_params=session.params["default"],
            random_metrics={}, random_params={},
            random_sampling_method=session.sampling_method,
            hyperopt_metrics={}, hyperopt_params={},
            skopt_metrics={}, skopt_params={},
            metric_name=session.hpo_metric,
        )
        best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric)

    session.cv_results = run_postprocessing_analysis(
        best_models=best,
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        cv_method=session.cv_method, cv_args=session.cv_args,
        scoring_metric=session.scoring_metric,
        show_cv_summary=True, show_cooks_distance=True,
        show_residuals=True, show_transformation_plots=True,
        image_output_dir=session.images_dir,
    )


def run_step_uq_after(session: WorkflowSession) -> None:
    r = session.preprocessing_results
    best_instances = get_best_models_by_method_across_targets(
        session.selected_models, session.targets,
        session.metrics, session.params,
        session.hpo_metric, session.selected,
    )
    # Only run UQ for models that HPO successfully produced instances for
    available = [m for m in session.selected_models if m in best_instances]
    df, figs = run_uncertainty_quantification(
        models_dict=best_instances,
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        target_columns=session.targets,
        model_names_to_run=available,
        stage_label="After HPO", show_plots=True,
        **session.uq_settings,
    )
    session.uq_after = (df, figs)


def run_step_report(session: WorkflowSession) -> None:
    session.ensure_dirs()
    r  = session.preprocessing_results
    tr = session.training_results
    hr = session.hpo_results or {}
    is_settings = session.interpretability_settings or {}

    # Persist best models (mirrors what workflow.py does before building the PDF)
    save_paths = {}
    try:
        if hr:
            best = hr["best_models_per_target"]
        else:
            collected = collect_results_as_dataframe(
                models_dict=session.selected, target_columns=session.targets,
                default_metrics=session.metrics["default"], default_params=session.params["default"],
                random_metrics={}, random_params={},
                random_sampling_method=session.sampling_method,
                hyperopt_metrics={}, hyperopt_params={},
                skopt_metrics={}, skopt_params={},
                metric_name=session.hpo_metric,
            )
            best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric)

        fitted   = build_and_fit_best_models(best, r["X_train_scaled"], r["y_train"],
                                             reset_model_to_defaults, process_hyperparameters)
        pipelines = build_pipelines(fitted_models_dict=fitted, fitted_scaler=r["scaler"])
        save_paths = save_models_and_artifacts(
            output_dir=session.models_dir,
            pipelines_by_target=pipelines,
            feature_names=r["feature_names"], targets=r["target_columns"],
            metric_name=session.hpo_metric, dataset_path=session.dataset_path,
            split_info={"method": session.split_method, "test_size": session.test_size,
                        "train_count": len(r["X_train"]), "test_count": len(r["X_test"])},
            extra_meta={"selected_models": session.selected_models},
            hpo_settings={"methods": list(session.methods_to_run),
                          "sampling_method": session.sampling_method,
                          "n_iter": session.n_iter, "evals": session.evals,
                          "calls": session.calls, "sample_size": session.sample_size,
                          "n_jobs": session.n_jobs},
            uq_settings=session.uq_settings,
            interpretability_settings=is_settings,
            cv_settings={"method": session.cv_method, "args": session.cv_args,
                         "scoring_metric": session.scoring_metric},
            make_bundle=True, prefix="phoenix",
        )
    except Exception as e:
        print(f"  Warning: model persistence failed ({e}), continuing with report.")

    doc, elements, styles, _ = init_pdf_report(
        filename=os.path.basename(session.pdf_path),
        output_dir=session.report_dir,
        title="phoenix_ml: Summary Report", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )
    add_system_info_to_pdf(elements, styles)
    plot_paths = save_preprocessing_plots(r, output_dir=session.images_dir)
    add_preprocessing_section(elements, r, plot_paths, session.dataset_path, styles,
                              dist_corr_dummy=session.dist_corr_dummy,
                              dist_corr_mp=session.dist_corr_mp)
    add_model_selection_section(
        elements, styles,
        selected_model_names=session.selected_models,
        preferred_model_name=is_settings.get("preferred_model_name", "N/A"),
    )
    add_model_training_table_to_report(elements, tr["results_df"], styles)

    if session.uq_before is not None:
        uq_df, uq_figs = session.uq_before
        handle_uq_reporting_section(uq_df, uq_figs, "Before HPO", elements, styles,
                                    session.images_dir,
                                    uq_settings=session.uq_settings)

    if session.interpretability_figures is not None:
        add_interpretability_section(elements, session.interpretability_figures,
                                     styles, session.images_dir, is_settings)

    if hr:
        add_hpo_summary_section(
            elements, styles,
            hr["hpo_metrics"], hr["hpo_params"], hr["hpo_times"], hr["hpo_plots"],
            list(session.methods_to_run), session.hpo_metric, session.sampling_method,
            session.sample_size, session.n_iter, session.evals, session.calls, session.n_jobs,
            hr["best_models_per_target"],
            output_dir=session.images_dir,
            early_stopping=session.early_stopping,
        )

    # Pareto front: available whenever training results exist and 2+ models were used
    if tr and len(session.selected_models) >= 2:
        pareto_figs = run_pareto_analysis(
            session_metrics=session.metrics,
            target_columns=session.targets,
            perf_metric=session.hpo_metric,
            selected_models=session.selected_models,
        )
        if pareto_figs:
            pareto_paths = save_pareto_plots(pareto_figs, output_dir=session.images_dir)
            add_pareto_section(elements, styles, pareto_paths, perf_metric=session.hpo_metric)

    if session.cv_results is not None:
        add_postprocessing_section(elements, styles,
                                   postprocessing_results=session.cv_results,
                                   image_output_dir=session.images_dir)

    if session.uq_after is not None:
        uq_df, uq_figs = session.uq_after
        handle_uq_reporting_section(uq_df, uq_figs, "After HPO", elements, styles,
                                    session.images_dir,
                                    uq_settings=session.uq_settings)

    if session.perl_results and session.perl_config:
        add_perl_section(elements, styles,
                         perl_results=session.perl_results,
                         perl_config=session.perl_config,
                         output_dir=session.images_dir)

    elements.append(Spacer(1, 24))
    if session.total_elapsed > 0:
        total = session.total_elapsed
        if total >= 86400:
            d, r2 = divmod(total, 86400); h, r2 = divmod(r2, 3600); m, s = divmod(r2, 60)
            time_str = f"{int(d)}d {int(h)}h {int(m)}m {s:.1f}s"
        elif total >= 3600:
            h, r2 = divmod(total, 3600); m, s = divmod(r2, 60)
            time_str = f"{int(h)}h {int(m)}m {s:.1f}s"
        elif total >= 60:
            m, s = divmod(total, 60)
            time_str = f"{int(m)}m {s:.1f}s"
        else:
            time_str = f"{total:.1f}s"
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            f"Total workflow computation time: <b>{time_str}</b>",
            styles["CustomBody"],
        ))
    if save_paths:
        add_artifacts_section(elements, styles, save_paths, session.models_dir)

    build_pdf(doc, elements)
    print(f"\nReport saved to: {session.pdf_path}")

    # ── Excel multi-sheet results export ──────────────────────────────────────
    try:
        xlsx_path = os.path.join(session.report_dir, "Phoenix_ML_Results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_rows = [
                {"Sheet": "HPO Results",          "Contents": "All HPO trial scores and parameters"},
                {"Sheet": "UQ Before HPO",         "Contents": "Uncertainty quantification metrics before HPO"},
                {"Sheet": "UQ After HPO",          "Contents": "Uncertainty quantification metrics after HPO"},
                {"Sheet": "PERL Reconstruction",   "Contents": "PERL reconstruction output with physics, ML, and combined predictions"},
            ]
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if hr and "collected_df" in hr and not hr["collected_df"].empty:
                hr["collected_df"].to_excel(writer, sheet_name="HPO Results", index=False)
            if session.uq_before is not None:
                uq_before_df = session.uq_before[0]
                if uq_before_df is not None and not uq_before_df.empty:
                    uq_before_df.to_excel(writer, sheet_name="UQ Before HPO", index=False)
            if session.uq_after is not None:
                uq_after_df = session.uq_after[0]
                if uq_after_df is not None and not uq_after_df.empty:
                    uq_after_df.to_excel(writer, sheet_name="UQ After HPO", index=False)
            if session.perl_output_df is not None:
                session.perl_output_df.to_excel(writer, sheet_name="PERL Reconstruction", index=False)
        session.xlsx_path = xlsx_path
        print(f"Excel results saved to: {xlsx_path}")
    except Exception as e:
        print(f"  Warning: Excel export failed ({e})")


def run_step_perl(session: WorkflowSession) -> None:
    """
    PERL Reconstruction: combine physics estimates with the ML residual predictions
    to produce full physical-unit predictions for each target.

    For each residual target column (e.g. "Residual_Deflection"):
        y_PERL = f_physics(inputs) + f_ML(inputs)

    Prints a metrics summary and stores the reconstruction DataFrame in session.perl_output_df
    for inclusion in the Excel results file. If the original (pre-residual) dataset path is
    stored in the config, also computes physics-only vs PERL accuracy improvement.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from phoenix_ml.physics_expressions import (
        apply_expressions, load_physics_config, ExpressionError,
    )

    session.ensure_dirs()
    config = load_physics_config(session.perl_config_path)
    config["_config_path"] = session.perl_config_path   # carried through to report section
    mode          = config.get("mode", "expression")
    expressions   = config.get("expressions", [])  # only present in expression mode
    recon_map     = config["reconstruction_map"]   # {residual_target: physics_est_col}
    measured_map  = config.get("measured_map", {}) # {residual_target: original_measured_col}
    orig_path     = config.get("original_dataset_path", "")

    r = session.preprocessing_results
    X_test_raw    = r["X_test"]          # DataFrame, residual dataset test rows (unscaled)
    X_test_scaled = r["X_test_scaled"]   # numpy array for ML predict()
    y_test        = r["y_test"]          # DataFrame, residual target values (actual)

    # ── 1. Load original dataset to apply physics expressions ─────────────
    # Physics expressions reference columns from the ORIGINAL (pre-residual) dataset,
    # not the residual dataset that the ML model was trained on.
    orig_test_df  = None
    orig_train_df = None
    orig_full     = None
    phys_input_df = X_test_raw  # fallback: try residual features if no original available
    if orig_path and os.path.isfile(orig_path):
        try:
            orig_full = pd.read_csv(orig_path)
            X_train_raw = r.get("X_train")
            # Align test rows
            if all(i in orig_full.index for i in X_test_raw.index):
                orig_test_df = orig_full.loc[X_test_raw.index]
            else:
                print("[PERL] Warning: could not match test row indices to original dataset — aligning by position (last N rows). Results may be inaccurate if the dataset was shuffled or rows were dropped during cleaning.")
                orig_test_df = orig_full.iloc[-len(X_test_raw):].reset_index(drop=True)
            # Align train rows (used for ML-only baseline)
            if X_train_raw is not None:
                if all(i in orig_full.index for i in X_train_raw.index):
                    orig_train_df = orig_full.loc[X_train_raw.index]
                else:
                    orig_train_df = orig_full.iloc[:len(X_train_raw)].reset_index(drop=True)
            phys_input_df = orig_test_df
            print(f"[PERL] Loaded original dataset for physics evaluation: {orig_path}")
            print(f"[PERL] Original test rows: {len(phys_input_df)}, columns: {list(phys_input_df.columns)}")
        except Exception as e:
            print(f"[PERL] Warning: could not load original dataset ({e}); using residual features.")
    else:
        print("[PERL] No original dataset path in config; applying expressions to residual features.")

    # ── 2. Produce physics columns on original (or residual) test inputs ────
    if mode == "script":
        from phoenix_ml.physics_model import import_physics_script, run_physics_model
        script_path = config["script_path"]
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Physics script not found: {script_path}")
        module = import_physics_script(script_path)
        governing_function = module.governing_function
        constants = config.get("constants") or module.constants
        input_vars  = config["input_vars"]
        output_vars = config["output_vars"]
        name_mapping = config.get("name_mapping") or None
        time_col     = config.get("time_col")
        print("\n[PERL] Running physics script for test inputs...")
        physics_out = run_physics_model(
            phys_input_df, time_col, governing_function,
            constants, input_vars, output_vars, name_mapping,
        )
        phys_df = pd.concat(
            [phys_input_df.reset_index(drop=True), physics_out.reset_index(drop=True)],
            axis=1,
        )
        print(f"[PERL] Physics columns added: {list(physics_out.columns)}")
    else:
        print("\n[PERL] Applying physics expressions to test inputs...")
        phys_df, phys_log = apply_expressions(phys_input_df, expressions)
        for msg in phys_log:
            print(f"  {msg}")

    # ── 3. Rebuild best fitted models (same as report step) ────────────────
    if session.hpo_results:
        best = session.hpo_results["best_models_per_target"]
    else:
        collected = collect_results_as_dataframe(
            models_dict=session.selected, target_columns=session.targets,
            default_metrics=session.metrics["default"], default_params=session.params["default"],
            random_metrics={}, random_params={},
            random_sampling_method=session.sampling_method,
            hyperopt_metrics={}, hyperopt_params={},
            skopt_metrics={}, skopt_params={},
            metric_name=session.hpo_metric,
        )
        best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric)

    fitted = build_and_fit_best_models(
        best, r["X_train_scaled"], r["y_train"],
        reset_model_to_defaults, process_hyperparameters,
    )

    # ── 4. Reconstruct for each target ─────────────────────────────────────
    output_df = X_test_raw.copy()
    perl_results = {}

    for target in session.targets:
        phys_col = recon_map.get(target)
        if not phys_col or phys_col not in phys_df.columns:
            print(f"[PERL] '{target}': no physics estimate column '{phys_col}' found — skipping.")
            continue

        model = fitted.get(target)
        if model is None:
            print(f"[PERL] '{target}': no fitted model found — skipping.")
            continue

        y_physics = phys_df[phys_col].values
        y_ml      = model.predict(X_test_scaled).ravel()
        y_perl    = y_physics + y_ml

        # Metrics on residual (ML performance)
        if target in y_test.columns:
            y_actual_residual = y_test[target].values
            rmse_ml = float(np.sqrt(mean_squared_error(y_actual_residual, y_ml)))
        else:
            rmse_ml = float("nan")

        # Physics-only accuracy vs PERL accuracy (if original data available).
        # Resolution order:
        #   1. measured_map from config (explicit, works for any naming convention)
        #   2. Stem-strip fallback (Residual_X → X, try underscore and space forms)
        original_col = None
        if orig_test_df is not None:
            # 1. Prefer explicit measured_map entry
            candidate = measured_map.get(target)
            if candidate and candidate in orig_test_df.columns:
                original_col = candidate
            else:
                # 2. Stem-strip fallback for older configs or DC-motor-style naming
                for prefix in ("Residual_", "Residual "):
                    if target.startswith(prefix):
                        stem = target[len(prefix):]
                        for s in (stem, stem.replace("_", " "), stem.replace(" ", "_")):
                            if s in orig_test_df.columns:
                                original_col = s
                                break
                        break

        rmse_physics, rmse_perl = float("nan"), float("nan")
        y_actual_physical = None
        if orig_test_df is not None and original_col:
            y_actual_physical = orig_test_df[original_col].values[:len(y_physics)]
            rmse_physics = float(np.sqrt(mean_squared_error(y_actual_physical, y_physics)))
            rmse_perl    = float(np.sqrt(mean_squared_error(y_actual_physical, y_perl)))

        # ML-only baseline: same model architecture re-fitted on original target (no physics)
        y_ml_only     = None
        rmse_ml_only  = float("nan")
        if orig_train_df is not None and original_col and original_col in orig_train_df.columns:
            try:
                orig_train_y = orig_train_df[original_col].values[:len(r["X_train_scaled"])]
                ml_baseline  = copy.deepcopy(model)
                ml_baseline.fit(r["X_train_scaled"], orig_train_y)
                y_ml_only = ml_baseline.predict(r["X_test_scaled"]).ravel()
                if y_actual_physical is not None:
                    rmse_ml_only = float(np.sqrt(mean_squared_error(
                        y_actual_physical[:len(y_ml_only)], y_ml_only
                    )))
                print(f"  ML-only RMSE         : {rmse_ml_only:.4f}" if not np.isnan(rmse_ml_only) else "  ML-only RMSE         : n/a")
            except Exception as e:
                print(f"[PERL] ML-only baseline failed for {target}: {e}")

        perl_results[target] = {
            "physics_col":      phys_col,
            "y_physics":        y_physics,
            "y_ml":             y_ml,
            "y_perl":           y_perl,
            "y_ml_only":        y_ml_only,
            "rmse_ml_residual": rmse_ml,
            "rmse_physics":     rmse_physics,
            "rmse_ml_only":     rmse_ml_only,
            "rmse_perl":        rmse_perl,
            "y_actual":         y_actual_physical,
            "original_col":     original_col,
        }

        recon_col = original_col if original_col else re.sub(r"^Residual[_ ]", "PERL_", target)
        display_col = original_col if original_col else target
        output_df[phys_col]                         = y_physics
        output_df[f"ML Correction ({display_col})"] = y_ml
        output_df[f"PERL ({recon_col})"]            = y_perl

        print(f"\n[PERL] {target}")
        print(f"  Physics estimate col : {phys_col}")
        print(f"  ML residual RMSE     : {rmse_ml:.4f}" if not np.isnan(rmse_ml) else "  ML residual RMSE     : n/a")
        if not np.isnan(rmse_physics):
            improve = (rmse_physics - rmse_perl) / rmse_physics * 100
            print(f"  Physics-only RMSE    : {rmse_physics:.4f}")
            print(f"  PERL RMSE            : {rmse_perl:.4f}")
            print(f"  Improvement          : {improve:+.1f}%")

    session.perl_results    = perl_results
    session.perl_config     = config      # keep for report section
    session.perl_output_df  = output_df   # written to Excel by run_step_report
    print(f"\n[PERL] Columns in reconstruction output: {list(output_df.columns)}")
