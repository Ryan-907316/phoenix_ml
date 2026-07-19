"""
phoenix_ml: A Physics and Hybrid Optimised ENgine for Interpretability
and eXplainability for Machine Learning.

Graphical interface
-------------------
Launch from the terminal after installation::

    phoenix-ml

Or from the cloned repository::

    python app.py

Programmatic API
----------------
Run the full workflow from Python::

    from phoenix_ml import run_workflow

    results = run_workflow(
        dataset_path="data.csv",
        output_dir="Results/",
        selected_models=["XGBoost Regressor"],
        targets=["Target"],
    )
"""

__version__ = "1.2.0"

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from phoenix_ml.workflow import run_workflow  # noqa: F401

__all__ = ["run_workflow", "__version__"]
