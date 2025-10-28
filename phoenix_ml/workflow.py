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
import os, time
from datetime import datetime

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
from phoenix_ml.system_info import SystemInfo
from phoenix_ml.persistence import (
    build_and_fit_best_models,
    build_pipelines,
    save_models_and_artifacts,
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

    # Preprocessing
    test_size: float = 0.2,
    split_method: str = "last",
    show_preproc_plots: bool = True,

    # Interpretability
    interpretability_settings: dict | None = None,

    # HPO
    hpo_metric: str = "Q^2",
    methods_to_run: list[str] = ("random", "hyperopt", "skopt"),
    sampling_method: str = "Sobol",
    n_iter: int = 100,
    sample_size: int = 1000,
    evals: int = 10,
    calls: int = 10,
    n_jobs: int = -1,

    # UQ
    uq_settings: dict | None = None,

    # CV / postprocessing
    cv_method: str = "Shuffle Split",
    cv_args: dict | None = None,
    scoring_metric: str = "R^2",
) -> dict:
    # Setup paths
    report_dir = _ensure_dir(os.path.join(output_dir, "Report"))
    images_dir = _ensure_dir(os.path.join(report_dir, "Images"))
    models_dir = _ensure_dir(os.path.join(output_dir, "Models"))

    timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")
    csv_path = os.path.join(report_dir, f"Hyperparameter Optimisation Results {timestamp}.csv")
    pdf_path = os.path.join(report_dir, "Phoenix_ML Report.pdf")

    # System info
    start_time = time.time()
    sysinfo = SystemInfo(); sysinfo.gather(); sysinfo.display()

    # Preprocessing
    results = run_preprocessing_workflow(
        file_path=dataset_path,
        test_size=test_size,
        split_method=split_method,
        target_columns=targets,
        plot_target_vs_target_enabled=show_preproc_plots,
        plot_features_vs_targets_enabled=show_preproc_plots,
        plot_boxplots_enabled=show_preproc_plots,
        plot_distance_corr_enabled=show_preproc_plots,
    )

    # Baseline training
    selected = {name: ALL_MODELS[name] for name in selected_models}
    selected_metrics = {k: metrics_dict[k] for k in ["MSE", "R^2", "ADJUSTED R^2", "Q^2"]}

    results_df, default_metrics, default_params, _ = run_model_training_workflow(
        X_train=results["X_train_scaled"],
        X_test=results["X_test_scaled"],
        y_train=results["y_train"],
        y_test=results["y_test"],
        target_columns=results["target_columns"],
        selected_model_names=selected_models,
        selected_metrics=selected_metrics,
    )
    metrics = {"default": default_metrics, "random": {}, "hyperopt": {}, "skopt": {}}
    params  = {"default": default_params,  "random": {}, "hyperopt": {}, "skopt": {}}

    # UQ before HPO
    uq_settings = uq_settings or dict(uq_method="Both", n_bootstrap=5, confidence_interval=95,
                                      calibration_frac=0.05, subsample_test_size=50)
    uq_df_before, uq_figures_before = run_uncertainty_quantification(
        models_dict=selected,
        X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
        y_train=results["y_train"], y_test=results["y_test"],
        target_columns=targets, model_names_to_run=selected_models,
        stage_label="Before HPO", show_plots=True, **uq_settings
    )

    # Interpretability
    interpretability_settings = interpretability_settings or dict(
        preferred_model_name="XGBoost Regressor",
        test_sample_size=1000, background_sample_size=10,
        subsample=250, grid_resolution=10
    )
    interpretability_figures = run_interpretability_analysis(
        models_dict=selected,
        X_train=results["X_train_scaled"], y_train=results["y_train"],
        target_columns=results["target_columns"], feature_names=results["feature_names"],
        **interpretability_settings, show_plots=True
    )

    # HPO
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
    collected_results_df.to_csv(csv_path, index=False)
    best_models_per_target = find_best_model_and_hyperparams(collected_results_df, metric=hpo_metric)

    # Postprocessing (CV, residuals, transforms) 
    cv_args = cv_args or {"n_splits": 10, "test_size": 0.2, "random_state": 0}
    post_results = run_postprocessing_analysis(
        best_models=best_models_per_target,
        X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
        y_train=results["y_train"], y_test=results["y_test"],
        cv_method=cv_method, cv_args=cv_args, scoring_metric=scoring_metric,
        show_cv_summary=True, show_cooks_distance=True, show_residuals=True,
        show_transformation_plots=True, image_output_dir=images_dir
    )

    # UQ after HPO
    best_model_instances = get_best_models_by_method_across_targets(
        selected_models, targets, metrics, params, hpo_metric, selected
    )
    uq_df_after, uq_figures_after = run_uncertainty_quantification(
        models_dict=best_model_instances,
        X_train=results["X_train_scaled"], X_test=results["X_test_scaled"],
        y_train=results["y_train"], y_test=results["y_test"],
        target_columns=targets, model_names_to_run=selected_models,
        stage_label="After HPO", show_plots=True, **uq_settings
    )

    # Persist best models
    fitted_models = build_and_fit_best_models(
        best_models_per_target, results["X_train_scaled"], results["y_train"],
        reset_model_to_defaults, process_hyperparameters
    )
    pipelines_by_target = build_pipelines(fitted_models_dict=fitted_models, fitted_scaler=results["scaler"])
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
        uq_settings=uq_settings,
        interpretability_settings=interpretability_settings,
        cv_settings={"method": cv_method, "args": cv_args, "scoring_metric": scoring_metric},
        make_bundle=True, prefix="phoenix",
    )

    # Report
    doc, elements, styles, filepath = init_pdf_report(
        filename=os.path.basename(pdf_path), output_dir=report_dir,
        title="Phoenix_ML: Report", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )
    add_system_info_to_pdf(elements, styles)
    plot_paths = save_preprocessing_plots(results, output_dir=images_dir)
    add_preprocessing_section(elements, results, plot_paths, dataset_path, styles)
    add_model_selection_section(
        elements, styles,
        selected_model_names=selected_models,
        preferred_model_name=interpretability_settings["preferred_model_name"]
    )
    add_model_training_table_to_report(elements, results_df, styles)
    handle_uq_reporting_section(uq_df_before, uq_figures_before, "Before HPO", elements, styles, images_dir, report_dir, uq_settings=uq_settings)
    add_interpretability_section(elements, interpretability_figures, styles, images_dir, interpretability_settings)
    add_hpo_summary_section(
        elements, styles, hpo_metrics, hpo_params, hpo_times, hpo_plots,
        list(methods_to_run), hpo_metric, sampling_method, sample_size, n_iter, evals, calls, n_jobs,
        csv_path, best_models_per_target, output_dir=images_dir
    )
    add_postprocessing_section(elements, styles, postprocessing_results=post_results, image_output_dir=images_dir)
    handle_uq_reporting_section(uq_df_after, uq_figures_after, "After HPO", elements, styles, images_dir, report_dir, uq_settings=uq_settings)

    elapsed = time.time() - start_time
    hhmmss = f"{elapsed//3600:02.0f}:{(elapsed%3600)//60:02.0f}:{elapsed%60:05.2f}"
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Execution Summary", styles["CustomHeading"]))
    elements.append(Paragraph(f"The total time elapsed for running the full workflow was: <b>{hhmmss}</b>.", styles["CustomBody"]))
    add_artifacts_section(elements, styles, save_paths, models_dir)
    doc.build(elements)

    return {
        "csv": csv_path,
        "pdf": pdf_path,
        "models": save_paths,
        "elapsed_seconds": elapsed,
        "images_dir": images_dir,
        "report_dir": report_dir,
    }
