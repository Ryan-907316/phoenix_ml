# workflow.py
# This module combines all the other modules together and make sure they all work together.

'''
This file ties together every modular component of the workflow:
  1. Loads and preprocesses the dataset.
  2. Trains baseline ML models using default hyperparameters.
  3. Quantifies uncertainty (bootstrapping + conformal) before optimisation.
  4. Generates interpretability plots (ICE, PDP, SHAP).
  5. Runs hyperparameter optimisation (Random, Hyperopt, Scikit-Optimize).
  6. Collects and compares optimisation results to find the best model per target.
  7. Performs post-processing (cross-validation, Cook's distance, residual analysis).
  8. Re-runs uncertainty quantification after optimisation.
  9. Saves fitted pipelines, metadata, and bundles for reproducibility.
 10. Automatically generates a full PDF report summarising all results.
'''
# Usage:
# from phoenix_ml.workflow import run_workflow
# This function is intended to be the *only* public entry point for users. All internal modules (preprocessing, HPO, UQ, etc.) are handled automatically behind the scenes.

from __future__ import annotations
import copy
import os, time
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from phoenix_ml.models import models_dict as ALL_MODELS
from phoenix_ml.model_training import (
    run_model_training_workflow, metrics_dict, reset_model_to_defaults,
    derive_seed,
    SEED_OFFSET_MODEL_CONSTRUCTION, SEED_OFFSET_TRAIN_TEST_SPLIT, SEED_OFFSET_CV,
    SEED_OFFSET_HPO, SEED_OFFSET_UQ_BOOTSTRAP, SEED_OFFSET_PERMUTATION_IMPORTANCE,
    SEED_OFFSET_SHAP_BACKGROUND,
)
from phoenix_ml.data_preprocessing import run_preprocessing_workflow
from phoenix_ml.uncertainty_quantification import run_uncertainty_quantification
from phoenix_ml.interpretability import run_interpretability_analysis
from phoenix_ml.hyperparameter_optimisation import (
    run_all_models_optimisation,
    collect_results_as_dataframe,
    find_best_model_and_hyperparams,
    get_all_models_tuned_per_target,
    process_hyperparameters,
)
from phoenix_ml.postprocessing import run_postprocessing_analysis
from phoenix_ml.pareto_analysis import run_pareto_analysis, save_pareto_plots
from phoenix_ml.system_info import SystemInfo
from phoenix_ml.persistence import (
    build_and_fit_best_models,
    build_pipelines,
    save_models_and_artifacts,
    save_predictor,
    build_uq_calibrators,
)
from phoenix_ml.report_generation import *

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def run_workflow(
    *,
    # Create directories for reports, images, and models; set up filenames
    dataset_path: str,
    output_dir: str,
    selected_models: list[str],
    targets: list[str],
    
    perform_hpo: bool = True,
    perform_interpretability: bool = True,
    perform_uq: bool = True,
    perform_cv: bool = True,

    # Random seed applied (via distinct per-stage derived offsets) everywhere randomness
    # appears: model construction, train/test split, CV, all three HPO backends,
    # bootstrap/conformal UQ, SHAP background sampling, Morris/Sobol sampling,
    # permutation importance. Same seed + same config reproduces identical results.
    random_seed: int = 0,

    # Preprocessing
    test_size: float = 0.2,
    split_method: str = "last",
    scaler_type: str = "Standard",
    show_target_vs_target: bool = True,
    show_features_vs_targets: bool = True,
    show_boxplots: bool = True,
    show_distance_corr: bool = True,
    dist_corr_dummy: bool = True,
    dist_corr_mp: bool = False,
    show_multicollinearity: bool = True,
    feat_sel_enabled: bool = True,
    feat_sel_redundancy_threshold: float = 0.90,
    plot_pca_enabled: bool = True,

    # Models
    monotonic_constraints: dict | None = None,

    # Interpretability
    interpretability_settings: dict | None = None,

    # HPO
    hpo_metric: str = "Q^2",
    methods_to_run: list[str] = ("random", "hyperopt", "skopt"),
    sampling_method: str = "Sobol",
    n_iter: int = 100,
    sample_size: int = 1000,
    evals: int = 50,
    calls: int = 50,
    n_jobs: int = -1,
    early_stopping: dict | None = None,

    # UQ
    uq_settings: dict | None = None,

    # CV / postprocessing
    cv_method: str = "Shuffle Split",
    cv_args: dict | None = None,
    scoring_metric: str = "R^2",
    show_lofo_importance: bool = False,
    transforms_to_run: list | None = None,
    # Whether the Best Transformation Normality Metrics table appears at all,
    # independent of which specific tests normality_tests below selects.
    show_normality_metrics: bool = True,
    # Which normality tests the report's Best Transformation Normality Metrics table
    # shows (all are always computed and always in the Excel export).
    normality_tests: list | None = None,

    # Model Deployment — which artifact files to write to the Models folder. With all
    # four off, the Models folder is never created and the report's Saved Models and
    # Artifacts section is omitted.
    save_pipelines: bool = True,
    save_metadata: bool = True,
    save_bundle: bool = True,
    save_predictor_file: bool = True,
) -> dict:
    # Setup paths
    report_dir = _ensure_dir(os.path.join(output_dir, "Report"))
    images_dir = _ensure_dir(os.path.join(report_dir, "Images"))
    # Path only — created lazily by the save functions when any deployment file is enabled
    models_dir = os.path.join(output_dir, "Models")

    pdf_path = os.path.join(report_dir, "phoenix_ml Report.pdf")

    # Build the ordered list of active steps for the progress header
    _steps = ["Preprocessing", "Baseline Training"]
    if perform_uq:               _steps.append("UQ (Before HPO)")
    if perform_interpretability: _steps.append("Interpretability")
    if perform_hpo:              _steps.append("Hyperparameter Optimisation")
    if perform_cv:               _steps.append("Postprocessing & CV")
    if perform_uq and perform_hpo: _steps.append("UQ (After HPO)")
    _steps.append("Report Generation")
    _total = len(_steps)
    _counter = {"n": 0}

    # Per-step wall-clock time, recorded as (name, seconds) each time the next step is announced.
    step_timings: list = []
    _last_checkpoint_time = time.time()
    _last_step_name = None

    def _announce(name):
        nonlocal _last_checkpoint_time, _last_step_name
        now = time.time()
        if _last_step_name is not None:
            step_timings.append((_last_step_name, now - _last_checkpoint_time))
        _last_checkpoint_time = now
        _last_step_name = name
        _counter["n"] += 1
        tqdm.write(f"\n{'-' * 60}")
        tqdm.write(f"  [{_counter['n']}/{_total}] {name}")
        tqdm.write(f"{'-' * 60}")

    # System info
    start_time = time.time()
    sysinfo = SystemInfo(); sysinfo.gather(); sysinfo.display()

    # Preprocessing
    _announce("Preprocessing")
    results = run_preprocessing_workflow(
        file_path=dataset_path,
        test_size=test_size,
        split_method=split_method,
        random_state=(
            derive_seed(random_seed, SEED_OFFSET_TRAIN_TEST_SPLIT)
            if split_method.lower() == "random" else None
        ),
        scaler_type=scaler_type,
        target_columns=targets,
        plot_target_vs_target_enabled=show_target_vs_target,
        plot_features_vs_targets_enabled=show_features_vs_targets,
        plot_boxplots_enabled=show_boxplots,
        plot_distance_corr_enabled=show_distance_corr,
        dist_corr_dummy=dist_corr_dummy,
        dist_corr_mp=dist_corr_mp,
        show_multicollinearity=show_multicollinearity,
        feat_sel_enabled=feat_sel_enabled,
        feat_sel_redundancy_threshold=feat_sel_redundancy_threshold,
        plot_pca_enabled=plot_pca_enabled,
    )

    # Baseline training
    _announce("Baseline Training")
    selected = {name: copy.deepcopy(ALL_MODELS[name]) for name in selected_models}
    # Mirror WorkflowSession.selected: random_state is safe to bake in once, here (it
    # doesn't vary per target). monotonic_constraints is per-target
    # ({target: {feature: +-1}}), so it can't be applied at this single, per-model-name
    # construction point — every consumer below (UQ, HPO) re-applies it per target,
    # right before that target's fit, since these instances are shared/refit per target.
    model_seed = derive_seed(random_seed, SEED_OFFSET_MODEL_CONSTRUCTION)
    for model in selected.values():
        if "random_state" in model.get_params():
            model.set_params(random_state=model_seed)
    selected_metrics = {k: metrics_dict[k] for k in ["MSE", "NRMSE", "MAPE", "R^2", "ADJUSTED R^2", "Q^2"]}

    results_df, default_metrics, default_params, _ = run_model_training_workflow(
        X_train=results["X_train_scaled"],
        X_test=results["X_test_scaled"],
        y_train=results["y_train"],
        y_test=results["y_test"],
        target_columns=results["target_columns"],
        selected_model_names=selected_models,
        selected_metrics=selected_metrics,
        feature_names=results["feature_names"],
        monotonic_constraints=monotonic_constraints,
        random_state=model_seed,
    )
    metrics = {"default": default_metrics, "random": {}, "hyperopt": {}, "skopt": {}}
    params  = {"default": default_params,  "random": {}, "hyperopt": {}, "skopt": {}}

    # UQ before HPO
    uq_df_before, uq_figures_before = None, None
    if perform_uq:
        _announce("UQ (Before HPO)")
        uq_settings = uq_settings or dict(uq_method="Both", n_bootstrap=5, confidence_interval=95,
                                        calibration_frac=0.05, subsample_test_size=50, n_jobs=1)
        uq_df_before, uq_figures_before = run_uncertainty_quantification(
            models_dict=selected,
            X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
            y_train=results["y_train"], y_test=results["y_test"],
            target_columns=targets, model_names_to_run=selected_models,
            stage_label="Before HPO", show_plots=True,
            random_state=derive_seed(random_seed, SEED_OFFSET_UQ_BOOTSTRAP),
            feature_names=results["feature_names"], monotonic_constraints=monotonic_constraints,
            **uq_settings
        )

    # Interpretability — every selected model, profiled once before HPO (this
    # standalone entry point doesn't expose the UI's separate Before/After HPO toggle;
    # use WorkflowSession via the UI for that split).
    interpretability_metrics_df, interpretability_figures = None, None
    if perform_interpretability:
        _announce("Interpretability")
        interpretability_settings = interpretability_settings or dict(
            test_sample_size=1000, background_sample_size=10,
            subsample=250, grid_resolution=10
        )
        interpretability_metrics_df, interpretability_figures = run_interpretability_analysis(
            models_dict=selected,
            X_train=results["X_train_scaled"], y_train=results["y_train"],
            target_columns=results["target_columns"], feature_names=results["feature_names"],
            model_names_to_run=selected_models,
            random_state=derive_seed(random_seed, SEED_OFFSET_SHAP_BACKGROUND),
            monotonic_constraints=monotonic_constraints,
            **interpretability_settings, show_plots=True
        )

    # HPO
    hpo_metrics, hpo_params, hpo_times, hpo_plots = {}, {}, {}, {}
    if perform_hpo:
        _announce("Hyperparameter Optimisation")
        hpo_metrics, hpo_params, hpo_times, hpo_plots = run_all_models_optimisation(
            models_dict=selected,
            selected_model_names=selected_models,
            X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
            y_train=results["y_train"], y_test=results["y_test"],
            target_columns=targets,
            methods_to_run=list(methods_to_run),
            metric=hpo_metric,
            sampling_method=sampling_method,
            sample_size=sample_size,
            n_iter=n_iter,
            evals=evals,
            calls=calls,
            n_jobs=n_jobs,
            plot=True,
            output_dir=images_dir,
            early_stopping=early_stopping,
            random_state=derive_seed(random_seed, SEED_OFFSET_HPO),
            feature_names=results["feature_names"],
            monotonic_constraints=monotonic_constraints,
        )
        metrics.update(hpo_metrics)
        params.update(hpo_params)

    # Collect and choose best
    collected_results_df = collect_results_as_dataframe(
        models_dict=selected, target_columns=targets,
        default_metrics=metrics["default"], default_params=params["default"],
        random_metrics=metrics["random"],  random_params=params["random"],
        random_sampling_method=sampling_method,
        hyperopt_metrics=metrics["hyperopt"], hyperopt_params=params["hyperopt"],
        skopt_metrics=metrics["skopt"],       skopt_params=params["skopt"],
        metric_name=hpo_metric,
    )
    best_models_per_target = find_best_model_and_hyperparams(collected_results_df, metric=hpo_metric)

    # Postprocessing (CV, residuals, transforms)
    post_results = None
    if perform_cv:
        _announce("Postprocessing & CV")
        cv_args = cv_args if cv_args is not None else {"n_splits": 10, "test_size": 0.2, "random_state": 0}
        cv_args = dict(cv_args)
        if "random_state" in cv_args:
            cv_args["random_state"] = derive_seed(random_seed, SEED_OFFSET_CV)
        post_results = run_postprocessing_analysis(
            best_models=best_models_per_target,
            X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
            y_train=results["y_train"], y_test=results["y_test"],
            cv_method=cv_method, cv_args=cv_args, scoring_metric=scoring_metric,
            show_cv_summary=True, show_cooks_distance=True,
            show_extended_diagnostics=True, show_residuals=True,
            show_transformation_plots=True, show_permutation_importance=True,
            show_lofo_importance=show_lofo_importance,
            transforms_to_run=transforms_to_run,
            feature_names=results["feature_names"], image_output_dir=images_dir,
            monotonic_constraints=monotonic_constraints,
            random_state=derive_seed(random_seed, SEED_OFFSET_PERMUTATION_IMPORTANCE),
        )

    # UQ after HPO — requires HPO to have actually produced tuned instances;
    # get_all_models_tuned_per_target() returns {} when perform_hpo is False (nothing in
    # metrics["random"/"hyperopt"/"skopt"] to look up), which previously still produced
    # an empty-but-non-None UQ DataFrame — the report/Excel gates below only checked
    # "is not None", so a full "Uncertainty Quantification (After HPO)" section rendered
    # with an empty table, reading as "we quantified uncertainty and found nothing"
    # rather than "this never ran" (found via a systematic failure-mode sweep).
    uq_df_after, uq_figures_after = None, None
    if perform_uq and perform_hpo:
        _announce("UQ (After HPO)")
        # Per-(model, target) tuned instances — each target gets its own correctly-tuned
        # hyperparameters, unlike the old across-targets-averaged lookup this replaced.
        best_model_instances = get_all_models_tuned_per_target(
            selected_models, targets, metrics, params, hpo_metric, selected
        )
        uq_df_after, uq_figures_after = run_uncertainty_quantification(
            models_dict=best_model_instances,
            X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
            y_train=results["y_train"], y_test=results["y_test"],
            target_columns=targets, model_names_to_run=list(best_model_instances.keys()),
            stage_label="After HPO", show_plots=True,
            random_state=derive_seed(random_seed, SEED_OFFSET_UQ_BOOTSTRAP),
            **uq_settings
    )

    # Persist best models. Wrapped like the predictor-save block below (and like
    # workflow_steps.py's equivalent report step): a failure here — a malformed
    # hyperparameter string, an unrecognised model name from a stale config — must not
    # crash the whole run and lose the PDF/Excel that the already-completed upstream
    # analysis (preprocessing/training/UQ/HPO) earned (found via a systematic
    # failure-mode sweep: this block previously had no try/except, unlike every other
    # persistence step in this function).
    save_paths = {}
    _save_any = save_pipelines or save_metadata or save_bundle or save_predictor_file
    pipelines_by_target = None
    if _save_any:
        try:
            fitted_models = build_and_fit_best_models(
                best_models_per_target, results["X_train_scaled"], results["y_train"],
                reset_model_to_defaults, process_hyperparameters,
                feature_names=results["feature_names"], monotonic_constraints=monotonic_constraints,
                random_state=derive_seed(random_seed, SEED_OFFSET_MODEL_CONSTRUCTION),
            )
            pipelines_by_target = build_pipelines(fitted_models_dict=fitted_models, fitted_scaler=results["scaler"])
            if save_pipelines or save_metadata or save_bundle:
                save_paths = save_models_and_artifacts(
                    output_dir=models_dir,
                    pipelines_by_target=pipelines_by_target,
                    feature_names=results["feature_names"], targets=results["target_columns"],
                    metric_name=hpo_metric, dataset_path=dataset_path,
                    split_info={
                        "method": split_method, "test_size": test_size,
                        "train_count": len(results["X_train"]), "test_count": len(results["X_test"]),
                    },
                    extra_meta={"selected_models": selected_models},
                    hpo_settings={
                        "methods": list(methods_to_run),
                        "sampling_method": sampling_method, "n_iter": n_iter, "evals": evals,
                        "calls": calls, "sample_size": sample_size, "n_jobs": n_jobs,
                    },
                    # perform_uq=False means UQ never ran regardless of what settings the
                    # caller passed in — recording them anyway would misleadingly claim
                    # they were used, the one settings field here not already implicitly
                    # None-means-skipped like the others.
                    uq_settings=(uq_settings if perform_uq else None),
                    interpretability_settings=interpretability_settings,
                    cv_settings={"method": cv_method, "args": cv_args, "scoring_metric": scoring_metric},
                    make_bundle=save_bundle, prefix="phoenix_ml",
                    save_pipelines=save_pipelines, save_metadata=save_metadata,
                )
                if not save_paths.get("by_target"):
                    save_paths.pop("by_target", None)
        except Exception as e:
            print(f"[WARN] Model persistence failed ({e}), continuing with report.")

    # Single deployable artifact: UQ intervals (if computed) travel with the model
    # so .predict() reproduces the report's numbers with no extra setup.
    if save_predictor_file and pipelines_by_target is not None:
        try:
            uq_calibrators = build_uq_calibrators(uq_df_after, best_models_per_target)
            save_paths["predictor"] = save_predictor(
                output_dir=models_dir,
                pipelines_by_target=pipelines_by_target,
                physics_config=None,
                uq_calibrators=uq_calibrators,
                metadata={
                    "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                    "dataset_path": dataset_path,
                    "targets": results["target_columns"],
                    "metric": hpo_metric,
                },
                prefix="phoenix_ml",
            )
        except Exception as e:
            print(f"[WARN] Deployable predictor save failed ({e}), continuing with report.")

    # Report
    _announce("Report Generation")
    doc, elements, styles, filepath, summary_index = init_pdf_report(
        filename=os.path.basename(pdf_path), output_dir=report_dir,
        title="Phoenix_ML: Report", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )
    add_system_info_to_pdf(elements, styles)
    plot_paths = save_preprocessing_plots(results, output_dir=images_dir)
    add_preprocessing_section(elements, results, plot_paths, dataset_path, styles,
                              dist_corr_dummy=dist_corr_dummy, dist_corr_mp=dist_corr_mp,
                              random_seed=random_seed)
    interpretability_settings = interpretability_settings or {}
    add_model_selection_section(
        elements, styles,
        selected_model_names=selected_models,
        monotonic_constraints=monotonic_constraints,
    )
    add_model_training_table_to_report(elements, results_df, styles)
    if perform_uq and uq_df_before is not None and not uq_df_before.empty:
        handle_uq_reporting_section(uq_df_before, uq_figures_before, "Before HPO", elements, styles, images_dir, uq_settings=uq_settings)

    if perform_interpretability and interpretability_figures is not None:
        add_interpretability_section(elements, interpretability_figures, styles, images_dir, interpretability_settings,
                                     n_features=len(results["feature_names"]), stage_label="Before HPO",
                                     target_columns=results["target_columns"])
        add_interpretability_metrics_table(elements, styles, interpretability_metrics_df, stage_label="Before HPO")

    if perform_hpo:
        add_hpo_summary_section(
            elements, styles, hpo_metrics, hpo_params, hpo_times, hpo_plots,
            list(methods_to_run), hpo_metric, sampling_method, sample_size, n_iter, evals, calls, n_jobs,
            best_models_per_target, output_dir=images_dir,
            early_stopping=early_stopping,
        )

    if len(selected_models) >= 2:
        pareto_figs = run_pareto_analysis(
            session_metrics=metrics,  # contains default + all HPO method results
            target_columns=list(targets),
            perf_metric=hpo_metric,
            selected_models=selected_models,
        )
        if pareto_figs:
            pareto_paths = save_pareto_plots(pareto_figs, output_dir=images_dir)
            add_pareto_section(elements, styles, pareto_paths, perf_metric=hpo_metric)

    if perform_cv and post_results is not None:
        add_postprocessing_section(elements, styles, postprocessing_results=post_results,
                                   image_output_dir=images_dir, normality_tests=normality_tests,
                                   show_normality_metrics=show_normality_metrics)
    if perform_uq and uq_df_after is not None and not uq_df_after.empty:
        handle_uq_reporting_section(uq_df_after, uq_figures_after, "After HPO", elements, styles, images_dir, uq_settings=uq_settings)

    elapsed = time.time() - start_time
    now = time.time()
    if _last_step_name is not None:
        step_timings.append((_last_step_name, now - _last_checkpoint_time))
    elements.append(Spacer(1, 24))
    if step_timings:
        add_time_breakdown_section(elements, styles, step_timings, images_dir)
    if save_paths:
        add_artifacts_section(elements, styles, save_paths, models_dir)

    add_executive_summary_section(
        elements, styles, summary_index,
        preprocessing_results=results,
        dataset_path=dataset_path,
        selected_model_names=selected_models,
        results_df=results_df,
        hpo_metric=hpo_metric,
        best_models_per_target=best_models_per_target,
        uq_before=uq_df_before,
        uq_after=uq_df_after,
        uq_settings=uq_settings,
        postprocessing_results=post_results,
        step_timings=step_timings,
        random_seed=random_seed,
    )

    build_pdf(doc, elements)

    # Excel results export — the Summary sheet lists only the sheets actually written
    xlsx_path = os.path.join(report_dir, "phoenix_ml Results.xlsx")
    try:
        interp_df = interpretability_metrics_df \
            if interpretability_metrics_df is not None and not interpretability_metrics_df.empty else None
        transformation_df = (post_results or {}).get("transformation_df") \
            if isinstance(post_results, dict) else None
        if transformation_df is not None and transformation_df.empty:
            transformation_df = None

        summary_rows = []
        if results_df is not None and not results_df.empty:
            summary_rows.append({"Sheet": "Model Training Results", "Contents": "Baseline training metrics for all models and target variables"})
        if collected_results_df is not None and not collected_results_df.empty:
            summary_rows.append({"Sheet": "HPO Results", "Contents": "All HPO trial scores and parameters"})
        if uq_df_before is not None and not uq_df_before.empty:
            summary_rows.append({"Sheet": "UQ Before HPO", "Contents": "Uncertainty quantification metrics before HPO"})
        if uq_df_after is not None and not uq_df_after.empty:
            summary_rows.append({"Sheet": "UQ After HPO", "Contents": "Uncertainty quantification metrics after HPO"})
        if interp_df is not None:
            summary_rows.append({"Sheet": "Interpretability Before HPO", "Contents": "Interpretability metrics summary (top features, rank agreement) before HPO"})
        if transformation_df is not None:
            summary_rows.append({"Sheet": "Residual Transformations", "Contents": "Residual transformation summary with every computed normality metric"})

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if results_df is not None and not results_df.empty:
                results_df.to_excel(writer, sheet_name="Model Training Results", index=False)
            if collected_results_df is not None and not collected_results_df.empty:
                collected_results_df.to_excel(writer, sheet_name="HPO Results", index=False)
            if uq_df_before is not None and not uq_df_before.empty:
                uq_df_before.to_excel(writer, sheet_name="UQ Before HPO", index=False)
            if uq_df_after is not None and not uq_df_after.empty:
                uq_df_after.to_excel(writer, sheet_name="UQ After HPO", index=False)
            if interp_df is not None:
                interp_df.to_excel(writer, sheet_name="Interpretability Before HPO", index=False)
            if transformation_df is not None:
                transformation_df.to_excel(writer, sheet_name="Residual Transformations", index=False)
    except Exception as e:
        print(f"[WARN] Excel export failed ({e})")
        xlsx_path = None

    return {
        "pdf": pdf_path,
        "xlsx": xlsx_path,
        "models": save_paths,
        "elapsed_seconds": elapsed,
        "images_dir": images_dir,
        "report_dir": report_dir,
    }
