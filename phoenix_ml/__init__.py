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

__version__ = "1.2.1"

import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

if os.name == "nt":
    # numba's on-disk JIT cache (used by `dcor`, a direct dependency) embeds the
    # full dotted-qualified name of the cached function in its cache filename --
    # up to ~165 characters for dcor's longer internal functions -- inside a
    # hashed-but-fixed-length subdirectory (45 characters). That's 209
    # characters of overhead phoenix_ml has no control over. Combined with a
    # deep install path (a nested venv inside a descriptively-named project
    # folder is a completely ordinary setup), this silently exceeded the
    # classic Windows 260-character MAX_PATH and crashed `import phoenix_ml`
    # outright on a real fresh install (see tests/ISSUES.md). Redirecting to a
    # short, fixed, drive-root path removes phoenix_ml's own install-path
    # length from the equation entirely. `setdefault` leaves any cache
    # location the user has already configured untouched, and if this
    # directory somehow isn't writable either, numba falls back to its own
    # default cache location exactly as it did before this existed -- this
    # can only help, never break an otherwise-working install.
    os.environ.setdefault("NUMBA_CACHE_DIR", r"C:\phoenix_ml_numba_cache")

from phoenix_ml.workflow import run_workflow  # noqa: F401

__all__ = ["run_workflow", "__version__"]
