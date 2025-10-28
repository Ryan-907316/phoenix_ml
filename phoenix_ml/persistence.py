# persistence.py
# This module essentially saves .pkl models for each target variable and for the whole thing, and generates a .JSON file with all the settings for reproducibility

import os
import ast
import json
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Instantiate, configure, and fit one model per target
def build_and_fit_best_models(best_models_per_target, X_train, y_train,
                               reset_model_to_defaults, process_hyperparameters):
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

        # Reset + set tuned params
        model = reset_model_to_defaults(model_name)
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
    make_bundle=True,
    prefix="model"
):
    # Save each pipeline as .pkl, plus metadata JSON, and optionally a single bundle .pkl.
    # Metadata includes reproducibility settings (dataset, models, HPO, UQ, CV, etc.). 
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") # ISO 8601 supremacy

    save_paths = {"by_target": {}}

    # Save each pipeline separately
    for target, pipeline in pipelines_by_target.items():
        fname = f"{prefix}_pipeline_{target.replace(' ', '_')}_{timestamp}.pkl"
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
    }

    # Save metadata JSON
    meta_path = os.path.join(output_dir, f"{prefix}_metadata_{timestamp}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    save_paths["metadata"] = meta_path

    # Save bundle (optional)
    if make_bundle:
        bundle_path = os.path.join(output_dir, f"{prefix}_bundle_{timestamp}.pkl")
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
