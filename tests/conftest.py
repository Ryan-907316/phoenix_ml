"""Shared fixtures for the phoenix_ml test suite, plus the scoped coverage
table printed at the end of every run.

The coverage table shows one row per phoenix_ml module — but only the modules
targeted by the test files that actually ran, so `pytest tests/test_persistence.py`
reports just persistence.py while a full `pytest` reports everything tested.
Coverage measurement itself comes from pytest-cov (wired in pyproject.toml's
addopts); run `pytest --no-cov` to skip it.
"""
import os

import numpy as np
import pandas as pd
import pytest

# Which phoenix_ml module(s) each test file exercises. Test files not listed
# fall back to "test_<name>.py -> <name>.py" if that module exists.
_TEST_TO_MODULES = {
    "test_workflow_smoke.py": ["workflow.py", "model_training.py", "data_preprocessing.py"],
    "test_workflow_steps.py": ["workflow_steps.py", "report_generation.py"],
    "test_physics_model.py": ["physics_model.py", "persistence.py"],
    "test_hostile_datasets.py": ["data_preprocessing.py", "model_training.py", "persistence.py"],
    "test_interpretability.py": ["interpretability.py", "sensitivity_analysis.py"],
    "test_report_generation.py": ["report_generation.py"],
}

_ran_test_files: set = set()


def pytest_collection_modifyitems(config, items):
    for item in items:
        _ran_test_files.add(os.path.basename(str(item.fspath)))


def _modules_for_ran_tests(pkg_dir):
    modules = set()
    for test_file in _ran_test_files:
        mapped = _TEST_TO_MODULES.get(test_file)
        if mapped is None:
            guess = test_file.removeprefix("test_")
            mapped = [guess] if os.path.isfile(os.path.join(pkg_dir, guess)) else []
        modules.update(mapped)
    return sorted(modules)


@pytest.hookimpl(trylast=True)  # after pytest-cov has finalised its data
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    cov_plugin = config.pluginmanager.get_plugin("_cov")
    cov = getattr(getattr(cov_plugin, "cov_controller", None), "cov", None)
    if cov is None or not _ran_test_files:
        return

    pkg_dir = os.path.join(str(config.rootpath), "phoenix_ml")
    modules = _modules_for_ran_tests(pkg_dir)
    if not modules:
        return

    rows = []
    for module in modules:
        path = os.path.join(pkg_dir, module)
        if not os.path.isfile(path):
            continue
        try:
            _, statements, _, missing, _ = cov.analysis2(path)
        except Exception:
            continue  # module never imported during this run
        n_stmts, n_miss = len(statements), len(missing)
        pct = 100.0 * (n_stmts - n_miss) / n_stmts if n_stmts else 100.0
        rows.append((module, n_stmts, n_miss, pct))
    if not rows:
        return

    tr = terminalreporter
    tr.write_sep("=", "phoenix_ml coverage (modules exercised by this run)")
    name_w = max(len(r[0]) for r in rows)
    tr.write_line(f"{'Module'.ljust(name_w)}  Stmts  Miss  Cover")
    for module, n_stmts, n_miss, pct in rows:
        tr.write_line(f"{module.ljust(name_w)}  {n_stmts:>5}  {n_miss:>4}  {pct:>4.0f}%")


@pytest.fixture
def synthetic_dataset_csv(tmp_path):
    """A tiny, deterministic synthetic engineering-style dataset written to a
    real CSV file (several workflow entry points read from a file path, not a
    DataFrame, so a fixture returning a path is more broadly reusable than one
    returning a DataFrame).

    Two features feed the target with a clear linear relationship (so models
    have something real to learn instead of pure noise); a third feature is
    unrelated noise, mirroring the kind of "is this feature actually useful"
    question the real tool is built to help answer.
    """
    rng = np.random.default_rng(0)
    n = 80
    feature_a = rng.uniform(5, 30, n)
    feature_b = rng.uniform(0, 5, n)
    noise_feature = rng.normal(0, 1, n)
    target = 3.0 * feature_a - 8.0 * feature_b + rng.normal(0, 1.0, n)

    df = pd.DataFrame({
        "Feature A": feature_a,
        "Feature B": feature_b,
        "Noise Feature": noise_feature,
        "Target": target,
    })
    path = tmp_path / "synthetic_dataset.csv"
    df.to_csv(path, index=False)
    return path
