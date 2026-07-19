# persistence.py
# This module essentially saves .pkl models for each target variable and for the whole thing, and generates a .JSON file with all the settings for reproducibility

import os
import ast
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline

def _safe_filename(name: str) -> str:
    """Make a target name safe for use inside a filename. Windows forbids
    \\ / : * ? " < > | outright (a '/' silently becomes a directory separator
    and the save fails with FileNotFoundError). Column names like "dw/dt" are
    normal in engineering datasets, so this cannot be left to luck. Spaces are
    kept as-is — the artifact filenames themselves contain spaces by design
    ("phoenix_ml Pipeline_...", matching "phoenix_ml Report.pdf")."""
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


# Instantiate, configure, and fit one model per target
def build_and_fit_best_models(best_models_per_target, X_train, y_train,
                               reset_model_to_defaults, process_hyperparameters,
                               feature_names=None, monotonic_constraints=None,
                               random_state=None):
    fitted_models = {}
    for target, model_info in best_models_per_target.items():
        model_name = model_info["model_name"]
        best_params = model_info["hyperparameters"]

        # Accept stringified dicts from CSVs/JSON without executing code.
        if isinstance(best_params, str):
            try:
                best_params = ast.literal_eval(best_params)
            except Exception:
                pass

        # Reset + set tuned params. monotonic_constraints is applied here (not part of
        # `processed`, since HPO never tunes it) and survives the set_params() below.
        # monotonic_constraints is per-target ({target: {feature: +-1}}) — this loop is
        # already per-target, so index straight in.
        model = reset_model_to_defaults(model_name, feature_names=feature_names,
                                        monotonic_constraints=(monotonic_constraints or {}).get(target, {}),
                                        random_state=random_state)
        processed = process_hyperparameters(best_params, model_name)
        model.set_params(**processed)

        model.fit(X_train, y_train[target])
        fitted_models[target] = model

    return fitted_models


def build_pipelines(fitted_models_dict, fitted_scaler=None):
    # Wrap each fitted model in a Pipeline with scaler (if provided).
    pipelines = {}
    for target, model in fitted_models_dict.items():
        if fitted_scaler:
            pipe = Pipeline([("scaler", fitted_scaler), ("model", model)])
        else:
            pipe = Pipeline([("model", model)]) # Yes, the order matters
        pipelines[target] = pipe
    return pipelines

# Persist per-target pipelines, metadata JSON, and (optionally) a single bundle.
def save_models_and_artifacts(
    output_dir,
    pipelines_by_target,
    feature_names,
    targets,
    metric_name,
    dataset_path,
    split_info=None,
    extra_meta=None,
    hpo_settings=None,
    uq_settings=None,
    interpretability_settings=None,
    cv_settings=None,
    physics_config=None,
    make_bundle=True,
    prefix="phoenix_ml",
    save_pipelines=True,
    save_metadata=True,
):
    # Save each pipeline as .pkl, plus metadata JSON, and optionally a single bundle .pkl.
    # Metadata includes reproducibility settings (dataset, models, HPO, UQ, CV, etc.).
    # Filenames are fixed per artifact type ("phoenix_ml Bundle.pkl", "phoenix_ml
    # Metadata.json", "phoenix_ml Pipeline_<target>.pkl") so they match the Results
    # folder's "phoenix_ml Report.pdf"/"phoenix_ml Results.xlsx" naming and simply
    # overwrite on re-run, the same way the report itself does. The run timestamp
    # lives in the metadata content, not the filenames.
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") # ISO 8601 supremacy

    save_paths = {"by_target": {}}

    # Save each pipeline separately
    if save_pipelines:
        for target, pipeline in pipelines_by_target.items():
            fname = f"{prefix} Pipeline_{_safe_filename(target)}.pkl"
            fpath = os.path.join(output_dir, fname)
            joblib.dump(pipeline, fpath)
            save_paths["by_target"][target] = fpath

    # Build reproducibility metadata
    metadata = {
        "timestamp": timestamp,
        "dataset": {
            "path": dataset_path,
            "split_info": split_info,
            "targets": targets,
            "feature_names": feature_names,
        },
        "settings": {
            "metric": metric_name,
            "selected_models": extra_meta.get("selected_models") if extra_meta else None,
            "hpo": hpo_settings,
            "uq": uq_settings,
            "interpretability": interpretability_settings,
            "cross_validation": cv_settings,
        },
        # For a PERL run, nothing else here says the saved per-target pipelines
        # predict RESIDUALS (measured - physics), not physical values — a raw
        # .pkl loaded without this context would silently be misread.
        "physics": {
            "perl_enabled": physics_config is not None,
            "mode": (physics_config or {}).get("mode"),
            "reconstruction_map": (physics_config or {}).get("reconstruction_map"),
            "note": (
                "Saved per-target pipelines predict RESIDUALS (measured minus physics "
                "estimate), not physical values, for every target listed in "
                "reconstruction_map. Add the physics estimate back to reconstruct the "
                "physical prediction, or use the deployable predictor artifact (if "
                "saved), which already does this."
            ) if physics_config else None,
        },
    }

    # Save metadata JSON
    if save_metadata:
        meta_path = os.path.join(output_dir, f"{prefix} Metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        save_paths["metadata"] = meta_path

    # Save bundle (optional)
    if make_bundle:
        bundle_path = os.path.join(output_dir, f"{prefix} Bundle.pkl")
        bundle_obj = {
            "pipelines_by_target": pipelines_by_target,
            "metadata": metadata,
        }
        joblib.dump(bundle_obj, bundle_path)
        save_paths["bundle"] = bundle_path

    return save_paths

# Loading helpers
def load_pipeline(path):
    return joblib.load(path)

def load_bundle(path):
    return joblib.load(path)


# ── Deployable single-file predictor ──────────────────────────────────────────
# Wraps the fitted per-target pipelines together with everything needed to
# reproduce the report's actual numbers with one call: optional physics
# reconstruction (Physics-Enhanced Residual Learning) and optional lightweight
# uncertainty intervals. Save with .save(path) / load with PhoenixPredictor.load(path).

_UQ_METHOD_PREFERENCE = ["Conformal", "Bootstrapping", "GP Posterior"]


class PhoenixPredictor:
    def __init__(self, pipelines_by_target, physics_config=None,
                 physics_script_source=None, uq_calibrators=None, metadata=None):
        self.pipelines = pipelines_by_target
        self.physics_config = physics_config
        self.physics_script_source = physics_script_source  # source text, script mode only
        self.uq_calibrators = uq_calibrators or {}
        self.metadata = metadata or {}
        self._physics_ns_cache = None  # rebuilt lazily from source after unpickling

    def __getstate__(self):
        # The lazily-exec'd physics namespace holds module objects and
        # exec-defined functions, neither of which can pickle — a predictor
        # that had predict() called before save() would crash with
        # "cannot pickle 'module' object". Drop the cache at pickle time;
        # _physics_namespace() rebuilds it from source after load.
        state = self.__dict__.copy()
        state["_physics_ns_cache"] = None
        return state

    def _physics_namespace(self):
        if self.physics_script_source is None:
            return None
        if self._physics_ns_cache is None:
            ns: dict = {}
            exec(self.physics_script_source, ns)
            self._physics_ns_cache = ns
        return self._physics_ns_cache

    def _physics_estimate(self, X, target):
        recon_map = self.physics_config["reconstruction_map"]
        est_col = recon_map[target]
        mode = self.physics_config.get("mode", "expression")
        if mode == "script":
            from phoenix_ml.physics_model import run_physics_model
            ns = self._physics_namespace()
            phys_out = run_physics_model(
                X, self.physics_config.get("time_col"),
                ns["governing_function"], self.physics_config["constants"],
                self.physics_config["input_vars"], self.physics_config["output_vars"],
                self.physics_config.get("name_mapping"),
            )
            return phys_out[est_col].values
        else:
            from phoenix_ml.physics_expressions import apply_expressions
            phys_df, _ = apply_expressions(X, self.physics_config["expressions"])
            return phys_df[est_col].values

    def predict(self, X):
        """Predict all targets. X must contain the same raw (unscaled) feature columns
        used for training — scaling is already baked into each per-target pipeline.
        For Physics-Enhanced (PERL) runs, X must also contain the raw columns referenced
        by the physics reconstruction (typically the same input columns)."""
        out = {}
        recon_map = (self.physics_config or {}).get("reconstruction_map", {})
        for target, pipe in self.pipelines.items():
            ml_pred = pipe.predict(X)
            if self.physics_config and target in recon_map:
                ml_pred = ml_pred + self._physics_estimate(X, target)
            out[target] = ml_pred
        return pd.DataFrame(out, index=getattr(X, "index", None))

    def predict_interval(self, X):
        """Predict all targets with an approximate prediction interval, for targets
        that have a stored UQ calibrator. Returns {target: DataFrame(prediction, lower, upper)}.
        Intervals are a fixed half-width derived from the original run's UQ summary
        (not re-derived from new data), so treat them as approximate."""
        preds = self.predict(X)
        intervals = {}
        for target in self.pipelines:
            cal = self.uq_calibrators.get(target)
            if cal is None:
                continue
            hw = cal["half_width"]
            intervals[target] = pd.DataFrame({
                "prediction": preds[target],
                "lower":      preds[target] - hw,
                "upper":      preds[target] + hw,
            }, index=preds.index)
        return intervals

    def summary(self) -> str:
        lines = ["PhoenixPredictor", f"  Targets: {', '.join(self.pipelines.keys())}"]
        if self.physics_config:
            lines.append("  Physics-Enhanced (PERL): yes (predictions include physics reconstruction)")
        if self.uq_calibrators:
            methods = ", ".join(f"{t} ({c['method']})" for t, c in self.uq_calibrators.items())
            lines.append(f"  Uncertainty intervals available: {methods}")
        ts = self.metadata.get("timestamp")
        if ts:
            lines.append(f"  Trained: {ts}")
        return "\n".join(lines)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def build_uq_calibrators(uq_df, best_models_per_target):
    """Extract one lightweight interval calibrator per target from a UQ results
    DataFrame (columns: Model, Target Variable, UQ Method, Mean Range, ...), matched
    to the specific model chosen as "best" for that target. Prefers Conformal, then
    Bootstrapping, then GP Posterior when multiple methods were run. Returns
    {target: {"method": ..., "half_width": ...}} — targets with no matching UQ row
    are omitted (predict_interval() simply won't offer an interval for them).
    """
    if uq_df is None or uq_df.empty:
        return {}

    calibrators = {}
    for target, model_info in best_models_per_target.items():
        model_name = model_info["model_name"] if isinstance(model_info, dict) else model_info.get("model_name")
        rows = uq_df[(uq_df["Model"] == model_name) & (uq_df["Target Variable"] == target)]
        if rows.empty:
            continue
        for method in _UQ_METHOD_PREFERENCE:
            match = rows[rows["UQ Method"] == method]
            if not match.empty:
                mean_range = match.iloc[0]["Mean Range"]
                if mean_range == mean_range:  # not NaN
                    calibrators[target] = {"method": method, "half_width": float(mean_range) / 2.0}
                    break
                # NaN Mean Range: fall through to the next preferred method
                # rather than leaving the target with no interval at all.
    return calibrators


def save_predictor(output_dir, pipelines_by_target, physics_config=None,
                    uq_calibrators=None, metadata=None, prefix="phoenix_ml", timestamp=None):
    """Save one self-contained PhoenixPredictor .pkl — the single file a user needs
    for deployment. Handles reading physics script source (script-mode PERL) so the
    file has no external path dependency. Saved as "{prefix} Predictor.pkl" (fixed
    name, overwritten per run — matching the other Results/Models artifacts); the
    `timestamp` parameter only feeds metadata, never the filename."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")

    physics_script_source = None
    if physics_config and physics_config.get("mode") == "script":
        script_path = physics_config.get("script_path", "")
        if script_path and os.path.isfile(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                physics_script_source = f.read()
        else:
            # Script-mode PERL is unusable without the source it embeds — the
            # predictor's predict() would otherwise crash later with a confusing
            # "'NoneType' object is not subscriptable" instead of this being caught
            # while the user can still fix the path (found via a systematic
            # failure-mode sweep: this previously saved a predictor that silently
            # could never produce physics estimates for any of its targets).
            raise FileNotFoundError(
                f"Physics script not found at '{script_path}' — cannot save a "
                f"Script Mode deployable predictor without it. Check the physics "
                f"script path hasn't moved or been deleted since this run."
            )

    predictor = PhoenixPredictor(
        pipelines_by_target=pipelines_by_target,
        physics_config=physics_config,
        physics_script_source=physics_script_source,
        uq_calibrators=uq_calibrators,
        metadata=metadata,
    )
    path = os.path.join(output_dir, f"{prefix} Predictor.pkl")
    predictor.save(path)
    return path
