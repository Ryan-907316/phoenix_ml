# _runner.py
# This module is the control panel of the workflow, essentially.
# Here you can change settings in the workflow to your choosing, and are free to add or remove sections as well.

# Additional information about the workflow:
# Name of developer: Ryan Cheung
# This workflow was designed as part of a 3rd year individual dissertation project, titled "Developing a Python workflow to aid in Physics-Enhanced Machine Learning applications in engineering."
# Since then, additional work has been done to make the workflow as easier to use and contains more features, such as better UQ visualisation and the addition of a report.
# For more information, refer to my GitHub page: https://github.com/Ryan-907316 as well as the instructions to use this workflow and documentation.
# University email address is cheungkh@lancaster.ac.uk for a request of my dissertation, as well as other resources produced as a result of this workflow/advice.


# Imports
from __future__ import annotations
from pathlib import Path
import argparse
from phoenix_ml.workflow import run_workflow

# User Settings
# IMPORTANT: Ensure your dataset is saved as a .csv file. Otherwise you will get a UnicodeDecodeError.
# I would also suggest copying these as paths and pasting the raw string.
BASE_OUTPUT_DIR = "Results"
DATASET_PATH = "examples/DC_Motor_Dataset.csv"

TARGETS = ["Motor Speed", "Armature Current"]

SELECTED_MODELS = [
    "SVR (RBF)",
    "Random Forest Regressor",
    "Gaussian Process Regressor",
    "XGBoost Regressor",
    "HistGradientBoosting Regressor",
    "LGBM Regressor",
    "Bagging Regressor",
    "MLP Regressor",
    "KNeighbors Regressor",
    "Extra Trees Regressor",
] # To remove certain models, simply comment them out or remove them entirely.

# Preprocessing
TEST_SIZE = 0.2
SPLIT_METHOD = "last"
SHOW_PREPROC_PLOTS = True

# Uncertainty Quantification
UQ_SETTINGS = dict(
    uq_method="Both",
    n_bootstrap=5,
    confidence_interval=95,
    calibration_frac=0.05,
    subsample_test_size=50,
)

# Interpretability 
INTERP_SETTINGS = {
    "preferred_model_name": "XGBoost Regressor",
    "test_sample_size": 1000,
    "background_sample_size": 10,
    "subsample": 250,
    "grid_resolution": 10,
}

# Hyperparameter Optimisation (HPO) 
HPO_METRIC = "Q^2"                 # Options: "MSE", "R^2", "ADJUSTED R^2", or "Q^2"
METHODS_TO_RUN = ["random", "hyperopt", "skopt"]
SAMPLING_METHOD = "Sobol"          # Options: "Random", "Sobol", "Halton", or "Latin Hypercube"
N_ITER = 100
SAMPLE_SIZE = 1000
EVALS = 10
CALLS = 10
N_JOBS = -1

# Cross-Validation / Postprocessing
CV_METHOD = "Shuffle Split"        # Options: "K-Fold", "Repeated K-Fold", "Group K-Fold", "LOO", "LpO", or "Shuffle Split"
CV_ARGS = {"n_splits": 10, "test_size": 0.2, "random_state": 0}
SCORING_METRIC = "R^2"             # Options: "MAE", "MSE", "Explained Variance", "R^2", "ADJUSTED R^2", "Q^2"


def parse_args():
    # Optional CLI overrides for convenience.
    p = argparse.ArgumentParser(description="Run Phoenix-ML workflow.")
    p.add_argument("--dataset", "-d", default=DATASET_PATH, help="Path to CSV dataset.")
    p.add_argument("--out", "-o", default=BASE_OUTPUT_DIR, help="Base output directory.")
    p.add_argument("--metric", "-m", default=HPO_METRIC, help='HPO metric (e.g., "Q^2" or "MSE").')
    p.add_argument("--cv-metric", default=SCORING_METRIC, help='CV scoring metric (e.g., "R^2", "MSE").')
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    output_dir = Path(args.out).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phoenix_ML: running full workflow ===")
    print(f"Dataset     : {dataset_path}")
    print(f"Output Dir  : {output_dir}")
    print(f"HPO Metric  : {args.metric}")
    print(f"CV Metric   : {args.cv_metric}")
    print("Selected models:")
    for m in SELECTED_MODELS:
        print(f"  • {m}")

    results = run_workflow(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        selected_models=SELECTED_MODELS,
        targets=TARGETS,

        # Preprocessing
        test_size=TEST_SIZE,
        split_method=SPLIT_METHOD,
        show_preproc_plots=SHOW_PREPROC_PLOTS,

        # Option to skip advanced modules. Change these to "False" anytime you want to skip, no need to change anything else.
        perform_interpretability=True,
        perform_uq=True,
        perform_hpo=True,
        perform_cv=True,

        # Interpretability
        interpretability_settings=INTERP_SETTINGS,
        

        # HPO
        hpo_metric=args.metric,
        methods_to_run=METHODS_TO_RUN,
        sampling_method=SAMPLING_METHOD,
        n_iter=N_ITER,
        sample_size=SAMPLE_SIZE,
        evals=EVALS,
        calls=CALLS,
        n_jobs=N_JOBS,

        # UQ
        uq_settings=UQ_SETTINGS,
        
        # CV / postprocessing
        cv_method=CV_METHOD,
        cv_args=CV_ARGS,
        scoring_metric=args.cv_metric,
    )

    print(f"Report PDF : {results['pdf']}")
    print(f"HPO CSV    : {results['csv']}")
    print(f"Models     : {results['models']}")
    print(f"Images dir : {results['images_dir']}")
    print(f"Elapsed (s): {results['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
