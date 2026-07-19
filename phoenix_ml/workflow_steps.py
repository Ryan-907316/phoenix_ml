# workflow_steps.py
# Stateful session object and per-step functions for the UI's step-by-step execution.
# The existing run_workflow() in workflow.py is unchanged.

from __future__ import annotations
import copy
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from phoenix_ml.models import models_dict as ALL_MODELS
from phoenix_ml.model_training import (
    run_model_training_workflow, metrics_dict, reset_model_to_defaults,
    derive_seed,
    SEED_OFFSET_MODEL_CONSTRUCTION, SEED_OFFSET_TRAIN_TEST_SPLIT, SEED_OFFSET_CV,
    SEED_OFFSET_HPO, SEED_OFFSET_UQ_BOOTSTRAP, SEED_OFFSET_PERMUTATION_IMPORTANCE,
    SEED_OFFSET_SHAP_BACKGROUND,
)
from phoenix_ml.data_preprocessing import run_preprocessing_workflow
from phoenix_ml.progress import log_step, log_info, log_warn
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
from phoenix_ml.persistence import (
    build_and_fit_best_models, build_pipelines, save_models_and_artifacts,
    save_predictor, build_uq_calibrators,
)
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
    # Names from selected_models already warned about via the `selected` property
    # below (not in ALL_MODELS — e.g. a stale saved config referencing a model
    # since renamed/removed from models.py). Tracked so repeated property reads
    # within one run don't reprint the same warning ~10 times.
    _warned_stale_model_names: set = field(default_factory=set)
    # Per-target, per-feature monotonicity constraint: {target: {feature_name: -1/0/+1}}.
    # Only applied to XGBoost/LGBM; a no-op for every other model. Set via the Models
    # tab's "Monotonicity Constraints..." popup. Per-target (not global) because a
    # direction that's physically correct for one target can fight the true relationship
    # for another target sharing the same underlying feature.
    monotonic_constraints: dict[str, dict[str, int]] = field(default_factory=dict)
    # Single seed applied (via distinct per-stage derived offsets, not reused literally)
    # everywhere randomness appears: model construction, train/test split, CV, all three
    # HPO backends, bootstrap/conformal UQ, SHAP background sampling, Morris/Sobol
    # sampling, outlier detection. Two identical runs with the same seed and config now
    # reproduce identical results end-to-end.
    random_seed: int = 0

    # Preprocessing
    test_size: float = 0.2
    split_method: str = "last"
    scaler_type: str = "Standard"
    show_target_vs_target: bool = True
    show_features_vs_targets: bool = True
    show_boxplots: bool = True
    show_distance_corr: bool = True
    dist_corr_dummy: bool = True
    dist_corr_mp: bool = False
    show_multicollinearity: bool = True
    plot_pca_enabled: bool = True
    feat_sel_enabled: bool = True
    feat_sel_redundancy_threshold: float = 0.90

    # Postprocessing / analysis
    show_cv_summary: bool = True
    show_cooks_distance: bool = True
    show_extended_diagnostics: bool = True
    show_residuals: bool = True
    show_transformation_plots: bool = True
    transforms_to_run: list[str] = field(default_factory=lambda: ["Yeo-Johnson", "Arcsinh"])
    show_permutation_importance: bool = True
    # Off by default: retrains once per feature (N+1 fits per target), unlike every
    # other diagnostic here which fits once.
    show_lofo_importance: bool = False

    # UQ
    uq_settings: dict = field(default_factory=lambda: dict(
        uq_method="Both", n_bootstrap=5, confidence_interval=95,
        calibration_frac=0.05, subsample_test_size=50, n_jobs=1,
        include_gp_posterior=False, calibration_enabled=True,
    ))

    # Interpretability — every selected model is now profiled (no single "preferred"
    # model), both before and after HPO as independently-toggleable steps (see STEPS).
    interpretability_settings: dict = field(default_factory=lambda: dict(
        test_sample_size=1000, background_sample_size=10,
        subsample=250, grid_resolution=10,
        show_ice_pdp=True, show_ale=True, show_shap_summary=True,
        show_shap_dependence=True, show_shap_waterfall=True,
        n_waterfall_samples=3,
        waterfall_percentiles=[1.0, 0.5, 0.0],
        show_sensitivity_morris=True, show_sensitivity_sobol=False,
        show_sensitivity_fast=False,
        sensitivity_morris_trajectories=10, sensitivity_morris_levels=4,
        sensitivity_sobol_n=512, sensitivity_fast_n=512,
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

    # Whether the "Best Transformation Normality Metrics" table appears at all,
    # independent of which specific tests normality_tests below selects -- so
    # unchecking every individual test correctly means "show zero columns",
    # not "hide the whole table" (that's this flag's job instead).
    show_normality_metrics: bool = True
    # Which normality tests the report's "Best Transformation Normality Metrics" table
    # shows. Purely a display choice: every test is always computed and always included
    # in the Excel export. Valid names: Shapiro-Wilk, Lilliefors, Filiben, Jarque-Bera,
    # D'Agostino (Anderson-Darling is always shown — it selects the best transform).
    normality_tests: list[str] = field(default_factory=lambda: ["Shapiro-Wilk", "Lilliefors", "Filiben"])

    # Model Deployment — which artifact files the report step writes to the Models
    # folder. With all four off, the Models folder is never created and the report's
    # "Saved Models and Artifacts" section is omitted entirely.
    save_pipelines: bool = True
    save_metadata: bool = True
    save_bundle: bool = True
    save_predictor_file: bool = True

    # PERL Reconstruction
    perl_config_path: str = ""
    perl_results: Any = None
    perl_config: dict | None = None
    perl_output_df: Any = None

    # Dataset cleaning (set by the UI's Clean tab on export; only used in the report if the
    # dataset actually loaded for this run matches the file that was exported)
    cleaning_summary: dict | None = None

    # Optional cooperative pause/cancel hook, set by the UI before each step starts.
    # Long-running steps call this periodically (currently: once per model in HPO) so the
    # UI's Pause/Stop controls can take effect without needing to kill any threads.
    checkpoint_fn: Any = None

    # Report
    report_source: str = "ui"
    # Metrics to show in the PDF training table (default: RMSE, Adjusted R², Q²/NSE, Time Elapsed)
    report_metric_cols: list = field(default_factory=lambda: ["RMSE", "ADJUSTED R^2", "Q^2", "Time Elapsed (s)"])

    # ── Step outputs ─────────────────────────────────────────────────────────
    preprocessing_results: dict | None = None
    training_results: dict | None = None
    uq_before: tuple | None = None
    # Each a (metrics_df, figures) tuple — same shape as uq_before/uq_after.
    interpretability_before: tuple | None = None
    interpretability_after: tuple | None = None
    hpo_results: dict | None = None
    cv_results: Any = None
    uq_after: tuple | None = None

    # Accumulated metrics/params (training sets defaults; HPO updates them)
    metrics: dict = field(default_factory=lambda: {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}})
    params:  dict = field(default_factory=lambda: {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}})

    # Total elapsed time of all completed steps (set by UI before report generation)
    total_elapsed: float = 0.0
    # Per-step wall-clock time as (display name, seconds), in execution order (set by UI before report generation)
    step_timings: list = field(default_factory=list)

    # ── Paths ────────────────────────────────────────────────────────────────
    images_dir:  str | None = None
    report_dir:  str | None = None
    xlsx_path:   str | None = None
    pdf_path:    str | None = None
    models_dir:  str | None = None

    # ── Helpers ──────────────────────────────────────────────────────────────
    def reset_results(self) -> None:
        """Clear every per-run result so a new analysis can't silently inherit stale
        data from a previous one — found as a real bug: running dataset A with PERL
        enabled, then dataset B without re-running every step, could carry dataset
        A's PERL section (and other stale results) into dataset B's report. Called
        automatically at the start of run_step_preprocessing (the one step that must
        always (re-)run for a new dataset, making it the natural "this is a fresh
        analysis" signal) and by the UI's manual Reset button.

        Clears only computed results/paths/timings — every user-configured setting
        (dataset path, targets, model selection, HPO/UQ/interpretability/CV
        settings, etc.) is left untouched, since switching datasets or starting over
        shouldn't discard configuration work.
        """
        self.preprocessing_results = None
        self.training_results = None
        self.uq_before = None
        self.uq_after = None
        self.interpretability_before = None
        self.interpretability_after = None
        self.hpo_results = None
        self.cv_results = None
        self.perl_results = None
        self.perl_config = None
        self.perl_output_df = None
        self.cleaning_summary = None
        self.metrics = {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}}
        self.params  = {"default": {}, "random": {}, "hyperopt": {}, "skopt": {}}
        self.total_elapsed = 0.0
        self.step_timings = []
        # Paths reset too: ensure_dirs() only (re)computes them when None, so a
        # changed output_dir between runs must not keep writing to the old location.
        self.images_dir = None
        self.report_dir = None
        self.xlsx_path = None
        self.pdf_path = None
        self.models_dir = None

    @property
    def selected(self) -> dict:
        stale = [n for n in self.selected_models if n not in ALL_MODELS]
        newly_stale = [n for n in stale if n not in self._warned_stale_model_names]
        if newly_stale:
            self._warned_stale_model_names.update(newly_stale)
            print(f"[WARN] Model(s) not found and skipped (removed/renamed since this "
                  f"config was saved?): {newly_stale}")
        models = {n: copy.deepcopy(ALL_MODELS[n]) for n in self.selected_models if n in ALL_MODELS}
        # random_state is safe to apply once, here — it doesn't vary per target the way
        # monotonic constraints do. monotonic_constraints is now per-target
        # ({target: {feature: +-1}}), so it can NOT be baked in at this single,
        # per-model-name (not per-target) construction point any more — every consumer
        # (UQ before/after, Interpretability before/after, HPO) is responsible for
        # (re-)applying constraints.get(target, {}) itself, right before that target's
        # fit, since these instances are shared/refit across every target in turn.
        seed = derive_seed(self.random_seed, SEED_OFFSET_MODEL_CONSTRUCTION)
        for model in models.values():
            if "random_state" in model.get_params():
                model.set_params(random_state=seed)
        return models

    def ensure_dirs(self) -> None:
        if self.images_dir is not None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H%M%S")
        rdir = _ensure_dir(os.path.join(self.output_dir, "Report"))
        self.report_dir  = rdir
        self.images_dir  = _ensure_dir(os.path.join(rdir, "Images"))
        # Path only — the Models folder is created lazily by the save functions
        # themselves, so runs with every Model Deployment checkbox off never
        # create an empty Models directory.
        self.models_dir  = os.path.join(self.output_dir, "Models")
        self.pdf_path    = os.path.join(rdir, "phoenix_ml Report.pdf")

    # ── Prerequisites ────────────────────────────────────────────────────────
    def can_run_preprocessing(self)   -> bool: return bool(self.dataset_path and self.targets and self.selected_models and self.output_dir)
    def can_run_training(self)        -> bool: return self.preprocessing_results is not None
    def can_run_uq_before(self)       -> bool: return self.training_results is not None
    def can_run_interpretability_before(self) -> bool: return self.training_results is not None
    def can_run_interpretability_after(self)  -> bool: return self.training_results is not None and self.hpo_results is not None
    def can_run_hpo(self)             -> bool: return self.training_results is not None
    def can_run_cv(self)              -> bool: return self.training_results is not None
    def can_run_uq_after(self)        -> bool: return self.training_results is not None and self.hpo_results is not None
    def can_run_perl(self)            -> bool: return self.training_results is not None and bool(self.perl_config_path) and os.path.isfile(self.perl_config_path)
    def can_generate_report(self)     -> bool: return self.preprocessing_results is not None and self.training_results is not None


# ── Individual step functions ─────────────────────────────────────────────────

def run_step_preprocessing(session: WorkflowSession) -> None:
    log_step("Preprocessing")
    # Checked once, here, before any work starts — usually fast enough that finer
    # granularity inside it isn't worth the complexity, but a Stop queued while a
    # previous step was still finishing must still take effect before this one begins.
    if session.checkpoint_fn is not None:
        session.checkpoint_fn()
    # Preprocessing is the one step that must always (re-)run for a new dataset, so
    # it's the natural point to invalidate every downstream result automatically —
    # this is the root-cause fix for a real bug: re-running only some steps on a
    # newly loaded dataset (without an explicit reset) could otherwise carry a
    # previous dataset's PERL section, HPO results, or computation-time breakdown
    # silently into the new report.
    session.reset_results()
    session.ensure_dirs()
    session.preprocessing_results = run_preprocessing_workflow(
        file_path=session.dataset_path,
        test_size=session.test_size,
        split_method=session.split_method,
        target_columns=session.targets,
        plot_target_vs_target_enabled=session.show_target_vs_target,
        plot_features_vs_targets_enabled=session.show_features_vs_targets,
        plot_boxplots_enabled=session.show_boxplots,
        plot_distance_corr_enabled=session.show_distance_corr,
        dist_corr_dummy=session.dist_corr_dummy,
        dist_corr_mp=session.dist_corr_mp,
        show_multicollinearity=session.show_multicollinearity,
        plot_pca_enabled=session.plot_pca_enabled,
        feat_sel_enabled=session.feat_sel_enabled,
        feat_sel_redundancy_threshold=session.feat_sel_redundancy_threshold,
        scaler_type=session.scaler_type,
        random_state=(
            derive_seed(session.random_seed, SEED_OFFSET_TRAIN_TEST_SPLIT)
            if session.split_method.lower() == "random" else None
        ),
    )


def run_step_training(session: WorkflowSession) -> None:
    log_step("Model Training")
    r = session.preprocessing_results
    sel_metrics = {k: metrics_dict[k] for k in ["MSE", "RMSE", "NRMSE", "MAPE", "R^2", "ADJUSTED R^2", "Q^2", "KGE"]}
    results_df, def_metrics, def_params, _ = run_model_training_workflow(
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        target_columns=r["target_columns"],
        selected_model_names=session.selected_models,
        selected_metrics=sel_metrics,
        feature_names=r["feature_names"],
        monotonic_constraints=session.monotonic_constraints,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_MODEL_CONSTRUCTION),
        checkpoint_fn=session.checkpoint_fn,
    )
    session.training_results = {"results_df": results_df, "default_metrics": def_metrics, "default_params": def_params}
    session.metrics = {"default": def_metrics, "random": {}, "hyperopt": {}, "skopt": {}}
    session.params  = {"default": def_params,  "random": {}, "hyperopt": {}, "skopt": {}}


def run_step_uq_before(session: WorkflowSession) -> None:
    log_step("Uncertainty Quantification (Before HPO)")
    r = session.preprocessing_results
    df, figs = run_uncertainty_quantification(
        models_dict=session.selected,
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        target_columns=session.targets,
        model_names_to_run=session.selected_models,
        stage_label="Before HPO", show_plots=True,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_UQ_BOOTSTRAP),
        feature_names=r["feature_names"],
        monotonic_constraints=session.monotonic_constraints,
        checkpoint_fn=session.checkpoint_fn,
        **session.uq_settings,
    )
    session.uq_before = (df, figs)


def run_step_interpretability_before(session: WorkflowSession) -> None:
    log_step("Interpretability (Before HPO)")
    # Mirrors run_step_uq_before: every selected model gets a metrics row (Morris/Sobol
    # + monotonicity check, cheap), but only each target's best BASELINE model gets the
    # expensive ICE/PDP/ALE/SHAP visuals — same "best per target" fallback used by
    # run_step_cv/run_step_report when HPO hasn't run yet.
    r = session.preprocessing_results
    collected = collect_results_as_dataframe(
        models_dict=session.selected, target_columns=session.targets,
        default_metrics=session.metrics["default"], default_params=session.params["default"],
        random_metrics={}, random_params={},
        random_sampling_method=session.sampling_method,
        hyperopt_metrics={}, hyperopt_params={},
        skopt_metrics={}, skopt_params={},
        metric_name=session.hpo_metric,
    )
    best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric, verbose=False)
    visual_model_by_target = {target: row["model_name"] for target, row in best.items()}

    metrics_df, figs = run_interpretability_analysis(
        models_dict=session.selected,
        X_train=r["X_train_scaled"], y_train=r["y_train"],
        target_columns=r["target_columns"], feature_names=r["feature_names"],
        model_names_to_run=session.selected_models,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_SHAP_BACKGROUND),
        monotonic_constraints=session.monotonic_constraints,
        visual_model_by_target=visual_model_by_target,
        checkpoint_fn=session.checkpoint_fn,
        **session.interpretability_settings, show_plots=True,
    )
    session.interpretability_before = (metrics_df, figs)


def run_step_interpretability_after(session: WorkflowSession) -> None:
    log_step("Interpretability (After HPO)")
    # Mirrors run_step_uq_after: every selected model gets a metrics row, each target's
    # own correctly-tuned hyperparameters (get_all_models_tuned_per_target) — but only
    # each target's actual best (HPO-selected) model gets the visuals.
    r = session.preprocessing_results
    best_instances = get_all_models_tuned_per_target(
        session.selected_models, session.targets,
        session.metrics, session.params,
        session.hpo_metric, session.selected,
    )
    available = [m for m in session.selected_models if m in best_instances]
    visual_model_by_target = {
        target: row["model_name"]
        for target, row in (session.hpo_results or {}).get("best_models_per_target", {}).items()
    }

    metrics_df, figs = run_interpretability_analysis(
        models_dict=best_instances,
        X_train=r["X_train_scaled"], y_train=r["y_train"],
        target_columns=r["target_columns"], feature_names=r["feature_names"],
        model_names_to_run=available,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_SHAP_BACKGROUND),
        monotonic_constraints=session.monotonic_constraints,
        visual_model_by_target=visual_model_by_target,
        checkpoint_fn=session.checkpoint_fn,
        **session.interpretability_settings, show_plots=True,
    )
    session.interpretability_after = (metrics_df, figs)


def run_step_hpo(session: WorkflowSession) -> None:
    log_step("Hyperparameter Optimisation")
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
        checkpoint_fn=session.checkpoint_fn,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_HPO),
        feature_names=r["feature_names"],
        monotonic_constraints=session.monotonic_constraints,
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
    log_step("Cross-Validation & Postprocessing")
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
        best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric, verbose=False)

    # cv_args carries user-facing knobs (n_splits, test_size, ...); random_state within
    # it is always the derived seed, not separately user-set.
    cv_args = dict(session.cv_args)
    if "random_state" in cv_args:
        cv_args["random_state"] = derive_seed(session.random_seed, SEED_OFFSET_CV)

    session.cv_results = run_postprocessing_analysis(
        best_models=best,
        X_train=r["X_train_scaled"], X_test=r["X_test_scaled"],
        y_train=r["y_train"],        y_test=r["y_test"],
        cv_method=session.cv_method, cv_args=cv_args,
        scoring_metric=session.scoring_metric,
        show_cv_summary=session.show_cv_summary,
        show_cooks_distance=session.show_cooks_distance,
        show_extended_diagnostics=session.show_extended_diagnostics,
        show_residuals=session.show_residuals,
        show_transformation_plots=session.show_transformation_plots,
        show_permutation_importance=session.show_permutation_importance,
        show_lofo_importance=session.show_lofo_importance,
        feature_names=r["feature_names"],
        image_output_dir=session.images_dir,
        transforms_to_run=session.transforms_to_run,
        monotonic_constraints=session.monotonic_constraints,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_PERMUTATION_IMPORTANCE),
        checkpoint_fn=session.checkpoint_fn,
    )


def run_step_uq_after(session: WorkflowSession) -> None:
    log_step("Uncertainty Quantification (After HPO)")
    r = session.preprocessing_results
    # Per-(model, target) tuned instances — each target gets its own correctly-tuned
    # hyperparameters, unlike the old across-targets-averaged lookup this replaced.
    best_instances = get_all_models_tuned_per_target(
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
        random_state=derive_seed(session.random_seed, SEED_OFFSET_UQ_BOOTSTRAP),
        checkpoint_fn=session.checkpoint_fn,
        **session.uq_settings,
    )
    session.uq_after = (df, figs)


def run_step_report(session: WorkflowSession) -> None:
    log_step("Generating Report")
    # Checked once, here, before anything is written — deliberately NOT threaded
    # further into this function's body: everything past this point writes files
    # (pipelines, metadata, the PDF/Excel themselves), and interrupting a Stop
    # mid-write risks a truncated/corrupt artifact, which is worse than letting
    # report generation finish once it has started.
    if session.checkpoint_fn is not None:
        session.checkpoint_fn()
    _report_gen_start = time.monotonic()
    session.ensure_dirs()
    r  = session.preprocessing_results
    tr = session.training_results
    hr = session.hpo_results or {}
    is_settings = session.interpretability_settings or {}

    # Persist best models (mirrors what workflow.py does before building the PDF).
    # Which artifact files get written is governed by the Model Deployment checkboxes
    # (save_pipelines/save_metadata/save_bundle/save_predictor_file); the best-model
    # LOOKUP always runs regardless, because the Executive Summary needs it.
    save_paths = {}
    best_models_per_target = None
    _save_any = (session.save_pipelines or session.save_metadata
                 or session.save_bundle or session.save_predictor_file)
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
            best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric, verbose=False)
        best_models_per_target = best

        if _save_any:
            fitted   = build_and_fit_best_models(best, r["X_train_scaled"], r["y_train"],
                                                 reset_model_to_defaults, process_hyperparameters,
                                                 feature_names=r["feature_names"],
                                                 monotonic_constraints=session.monotonic_constraints,
                                                 random_state=derive_seed(session.random_seed, SEED_OFFSET_MODEL_CONSTRUCTION))
            pipelines = build_pipelines(fitted_models_dict=fitted, fitted_scaler=r["scaler"])
            if session.save_pipelines or session.save_metadata or session.save_bundle:
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
                    physics_config=session.perl_config,
                    make_bundle=session.save_bundle, prefix="phoenix_ml",
                    save_pipelines=session.save_pipelines,
                    save_metadata=session.save_metadata,
                )
                if not save_paths.get("by_target"):
                    # Drop the empty placeholder so `if save_paths:` correctly reads
                    # "nothing was saved" when pipelines were the only thing enabled-off
                    save_paths.pop("by_target", None)

            # Single deployable artifact: physics reconstruction (if PERL) + UQ intervals
            # (if computed) travel with the model so .predict() reproduces the report's numbers.
            if session.save_predictor_file:
                try:
                    uq_df_for_calib = session.uq_after[0] if session.uq_after is not None else None
                    uq_calibrators = build_uq_calibrators(uq_df_for_calib, best)
                    predictor_path = save_predictor(
                        output_dir=session.models_dir,
                        pipelines_by_target=pipelines,
                        physics_config=session.perl_config,
                        uq_calibrators=uq_calibrators,
                        metadata={
                            "timestamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                            "dataset_path": session.dataset_path,
                            "targets": r["target_columns"],
                            "metric": session.hpo_metric,
                        },
                        prefix="phoenix_ml",
                    )
                    save_paths["predictor"] = predictor_path
                except Exception as e:
                    log_warn(f"Deployable predictor save failed ({e}), continuing with report.")
    except Exception as e:
        log_warn(f"Model persistence failed ({e}), continuing with report.")

    doc, elements, styles, _, summary_index = init_pdf_report(
        filename=os.path.basename(session.pdf_path),
        output_dir=session.report_dir,
        title="phoenix_ml: Summary Report", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )
    add_system_info_to_pdf(elements, styles)
    plot_paths = save_preprocessing_plots(r, output_dir=session.images_dir)
    add_preprocessing_section(elements, r, plot_paths, session.dataset_path, styles,
                              dist_corr_dummy=session.dist_corr_dummy,
                              dist_corr_mp=session.dist_corr_mp,
                              random_seed=session.random_seed)
    add_model_selection_section(
        elements, styles,
        selected_model_names=session.selected_models,
        monotonic_constraints=session.monotonic_constraints,
    )
    add_model_training_table_to_report(elements, tr["results_df"], styles,
                                       report_metric_cols=session.report_metric_cols)

    if session.uq_before is not None:
        uq_df, uq_figs = session.uq_before
        handle_uq_reporting_section(uq_df, uq_figs, "Before HPO", elements, styles,
                                    session.images_dir,
                                    uq_settings=session.uq_settings)

    if session.interpretability_before is not None:
        metrics_df, figs = session.interpretability_before
        add_interpretability_section(elements, figs, styles, session.images_dir, is_settings,
                                     n_features=len(r["feature_names"]), stage_label="Before HPO",
                                     target_columns=r["target_columns"])
        add_interpretability_metrics_table(elements, styles, metrics_df, stage_label="Before HPO")

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
                                   image_output_dir=session.images_dir,
                                   normality_tests=session.normality_tests,
                                   show_normality_metrics=session.show_normality_metrics)

    if session.uq_after is not None:
        uq_df, uq_figs = session.uq_after
        handle_uq_reporting_section(uq_df, uq_figs, "After HPO", elements, styles,
                                    session.images_dir,
                                    uq_settings=session.uq_settings)

    if session.interpretability_after is not None:
        metrics_df, figs = session.interpretability_after
        add_interpretability_section(elements, figs, styles, session.images_dir, is_settings,
                                     n_features=len(r["feature_names"]), stage_label="After HPO",
                                     target_columns=r["target_columns"])
        add_interpretability_metrics_table(elements, styles, metrics_df, stage_label="After HPO")

    if session.perl_results and session.perl_config:
        add_perl_section(elements, styles,
                         perl_results=session.perl_results,
                         perl_config=session.perl_config,
                         output_dir=session.images_dir)

    elements.append(Spacer(1, 24))
    report_gen_elapsed = time.monotonic() - _report_gen_start
    step_timings = list(session.step_timings) + [("Report Generation", report_gen_elapsed)]
    if any(t > 0 for _, t in step_timings):
        add_time_breakdown_section(elements, styles, step_timings, session.images_dir)
    if save_paths:
        add_artifacts_section(elements, styles, save_paths, session.models_dir)

    # Only attribute cleaning stats to this run if the dataset actually loaded here is the
    # file the Clean tab exported — otherwise a stale cleaning summary from an unrelated
    # dataset must not be shown.
    cleaning_summary = None
    cs = session.cleaning_summary
    if cs and cs.get("export_path") and os.path.abspath(session.dataset_path) == cs["export_path"]:
        cleaning_summary = cs

    uq_before_df = session.uq_before[0] if session.uq_before is not None else None
    uq_after_df  = session.uq_after[0] if session.uq_after is not None else None
    add_executive_summary_section(
        elements, styles, summary_index,
        preprocessing_results=r,
        dataset_path=session.dataset_path,
        selected_model_names=session.selected_models,
        results_df=tr["results_df"] if tr else None,
        hpo_metric=session.hpo_metric,
        best_models_per_target=best_models_per_target,
        uq_before=uq_before_df,
        uq_after=uq_after_df,
        uq_settings=session.uq_settings,
        postprocessing_results=session.cv_results,
        perl_results=session.perl_results,
        perl_config=session.perl_config,
        step_timings=step_timings,
        cleaning_summary=cleaning_summary,
        random_seed=session.random_seed,
    )

    build_pdf(doc, elements)
    print()
    log_info(f"Report saved to: {session.pdf_path}")

    # ── Excel multi-sheet results export ──────────────────────────────────────
    def _autosize_excel_columns(writer, min_width=8, max_width=60, padding=2):
        """Widen every column of every written sheet to fit its longest value,
        so opening the workbook doesn't show clipped headers/values by default."""
        for sheet in writer.sheets.values():
            for col_cells in sheet.columns:
                longest = max(
                    (len(str(cell.value)) for cell in col_cells if cell.value is not None),
                    default=0,
                )
                width = min(max(longest + padding, min_width), max_width)
                sheet.column_dimensions[col_cells[0].column_letter].width = width

    try:
        xlsx_path = os.path.join(session.report_dir, "phoenix_ml Results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            _tr = session.training_results or {}
            _tr_df = _tr.get("results_df")
            summary_rows = []
            if _tr_df is not None and not _tr_df.empty:
                summary_rows.append({"Sheet": "Model Training Results", "Contents": "Baseline training metrics for all models and target variables"})
            if hr and "collected_df" in hr and not hr["collected_df"].empty:
                summary_rows.append({"Sheet": "HPO Results", "Contents": "All HPO trial scores and parameters"})
            if (session.uq_before is not None
                    and session.uq_before[0] is not None
                    and not session.uq_before[0].empty):
                summary_rows.append({"Sheet": "UQ Before HPO", "Contents": "Uncertainty quantification metrics before HPO"})
            if (session.uq_after is not None
                    and session.uq_after[0] is not None
                    and not session.uq_after[0].empty):
                summary_rows.append({"Sheet": "UQ After HPO", "Contents": "Uncertainty quantification metrics after HPO"})

            def _interp_df(stage_tuple):
                df = stage_tuple[0] if stage_tuple is not None else None
                return df if df is not None and not df.empty else None

            interp_before_df = _interp_df(session.interpretability_before)
            interp_after_df  = _interp_df(session.interpretability_after)
            if interp_before_df is not None:
                summary_rows.append({"Sheet": "Interpretability Before HPO", "Contents": "Interpretability metrics summary (top features, rank agreement) before HPO"})
            if interp_after_df is not None:
                summary_rows.append({"Sheet": "Interpretability After HPO", "Contents": "Interpretability metrics summary (top features, rank agreement) after HPO"})

            transformation_df = (session.cv_results or {}).get("transformation_df") \
                if isinstance(session.cv_results, dict) else None
            if transformation_df is not None and not transformation_df.empty:
                summary_rows.append({"Sheet": "Residual Transformations", "Contents": "Residual transformation summary with every computed normality metric"})

            if session.perl_output_df is not None:
                summary_rows.append({"Sheet": "PERL Reconstruction", "Contents": "PERL reconstruction output with physics, ML, and combined predictions"})
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if _tr_df is not None and not _tr_df.empty:
                _tr_df.to_excel(writer, sheet_name="Model Training Results", index=False)
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
            if interp_before_df is not None:
                interp_before_df.to_excel(writer, sheet_name="Interpretability Before HPO", index=False)
            if interp_after_df is not None:
                interp_after_df.to_excel(writer, sheet_name="Interpretability After HPO", index=False)
            if transformation_df is not None and not transformation_df.empty:
                transformation_df.to_excel(writer, sheet_name="Residual Transformations", index=False)
            if session.perl_output_df is not None:
                session.perl_output_df.to_excel(writer, sheet_name="PERL Reconstruction", index=False)
            _autosize_excel_columns(writer)
        session.xlsx_path = xlsx_path
        log_info(f"Excel results saved to: {xlsx_path}")
    except Exception as e:
        log_warn(f"Excel export failed ({e})")


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

    log_step("Physics-Enhanced Residual Learning (PERL)")
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
                log_warn("PERL: could not match test row indices to original dataset - "
                         "aligning by position (last N rows). Results may be inaccurate if "
                         "the dataset was shuffled or rows were dropped during cleaning.")
                orig_test_df = orig_full.iloc[-len(X_test_raw):].reset_index(drop=True)
            # Align train rows (used for ML-only baseline)
            if X_train_raw is not None:
                if all(i in orig_full.index for i in X_train_raw.index):
                    orig_train_df = orig_full.loc[X_train_raw.index]
                else:
                    orig_train_df = orig_full.iloc[:len(X_train_raw)].reset_index(drop=True)
            phys_input_df = orig_test_df
            log_info(f"PERL: loaded original dataset for physics evaluation: {orig_path}")
            log_info(f"PERL: original test rows: {len(phys_input_df)}, columns: {list(phys_input_df.columns)}")
        except Exception as e:
            log_warn(f"PERL: could not load original dataset ({e}); using residual features.")
    else:
        log_info("PERL: no original dataset path in config; applying expressions to residual features.")

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
        best = find_best_model_and_hyperparams(collected, metric=session.hpo_metric, verbose=False)

    fitted = build_and_fit_best_models(
        best, r["X_train_scaled"], r["y_train"],
        reset_model_to_defaults, process_hyperparameters,
        feature_names=r["feature_names"], monotonic_constraints=session.monotonic_constraints,
        random_state=derive_seed(session.random_seed, SEED_OFFSET_MODEL_CONSTRUCTION),
    )

    # ── 4. Reconstruct for each target ─────────────────────────────────────
    output_df = X_test_raw.copy()
    perl_results = {}

    for target in session.targets:
        # Optional cooperative pause/cancel hook (e.g. from the UI): checked once per
        # target, the same granularity every other step function uses.
        if session.checkpoint_fn is not None:
            session.checkpoint_fn()
        phys_col = recon_map.get(target)
        if not phys_col or phys_col not in phys_df.columns:
            log_warn(f"PERL: '{target}': no physics estimate column '{phys_col}' found - skipping.")
            continue

        model = fitted.get(target)
        if model is None:
            log_warn(f"PERL: '{target}': no fitted model found - skipping.")
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
            except Exception as e:
                log_warn(f"PERL: ML-only baseline failed for {target}: {e}")

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

        print(f"\n=== PERL: {target} ===")
        print(f"  Physics estimate col : {phys_col}")
        print(f"  ML residual RMSE     : {rmse_ml:.4f}" if not np.isnan(rmse_ml) else "  ML residual RMSE     : n/a")
        if not np.isnan(rmse_ml_only):
            print(f"  ML-only RMSE         : {rmse_ml_only:.4f}")
        if not np.isnan(rmse_physics):
            improve = (rmse_physics - rmse_perl) / rmse_physics * 100
            print(f"  Physics-only RMSE    : {rmse_physics:.4f}")
            print(f"  PERL RMSE            : {rmse_perl:.4f}")
            print(f"  Improvement          : {improve:+.1f}%")

    if not perl_results:
        # Every target hit a per-target skip above (reconstruction_map typo, no
        # matching physics column, no fitted model, ...) — perl_results = {} is
        # otherwise indistinguishable from "PERL was never attempted", and the
        # report section is gated on `if session.perl_results:` (falsy for {}),
        # so it would silently vanish with no indication anything was wrong.
        print(f"[WARN] PERL: every target was skipped (see reasons above) - "
              f"no reconstruction output was produced; the PERL report section "
              f"will not appear.")

    session.perl_results    = perl_results
    session.perl_config     = config      # keep for report section
    session.perl_output_df  = output_df   # written to Excel by run_step_report
    log_info(f"PERL: columns in reconstruction output: {list(output_df.columns)}")
