# phoenix_ml test suite

## How to use these tests, step by step

The following instructions assume you have no prior knowledge of testing whatsoever.

### 1. One-time setup

Open a terminal in the project root (the folder containing `pyproject.toml`). In VS Code: **Terminal → New Terminal**.

Create a dedicated virtual environment first, so the test dependencies stay isolated from anything else already installed on your machine. Installing straight into your main Python environment can silently upgrade packages that other, unrelated projects depend on:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

(On macOS/Linux, activate with `source .venv/bin/activate` instead of the second line.)

Your prompt should now start with `(.venv)`. With the environment active, install phoenix_ml and its test dependencies:

```bash
pip install -e ".[dev]"
```

This installs phoenix_ml in "editable" mode plus the test runner (`pytest`). You only need to repeat this after changing `pyproject.toml`, or after creating a fresh `.venv`.

On every future visit, just activate the environment again (`.\.venv\Scripts\Activate.ps1`) before running `pytest`; no reinstalling required. When you are finished, `deactivate` returns you to your normal terminal.

### 2. Run the whole suite

```bash
pytest
```

That is all that is required. pytest automatically finds every `test_*.py` file in `tests/` and runs every function inside them whose name starts with `test_`. The whole suite should take a couple of minutes; the last lines tell you everything, and should look something like this:

```
========= phoenix_ml coverage (modules exercised by this run) =========
Module                 Stmts  Miss  Cover
persistence.py           158    25    84%
...
================ 423 passed, 288 warnings in 83.46s (0:01:23) ================
```

(The exact number of tests will keep growing as the suite does; 423 is only a snapshot.)

The coverage table lists only the phoenix_ml modules targeted by the tests that actually ran: the full suite shows every tested module, while `pytest tests/test_persistence.py` shows just persistence. "Cover" is the percentage of that module's lines the run executed (executed ≠ verified, but 0% always means untested). Measuring coverage adds roughly 20% to runtime; skip it with `pytest --no-cov` when iterating quickly.

**All green ("N passed")** → every checked behaviour still holds. Safe to commit/push.

**Any red ("M failed")** → something you (or a dependency update) changed broke a checked behaviour. See step 5.

### 3. Run just part of the suite

```bash
pytest tests/test_persistence.py            # one file
pytest tests/test_persistence.py -v         # -v lists each test name + result
pytest -k "monotonic"                       # only tests with "monotonic" in the name
pytest -x                                   # stop at the first failure
```

Use a single file while working on that module (fast feedback), and the full `pytest` before committing.

### 4. When to run

- Before every commit/push: the whole point is catching breakage *before* it ships.
- After changing any module that has a matching test file.
- After updating dependencies (sklearn, pandas, etc.): several tests deliberately pin behaviour that could shift under a library upgrade.

### 5. When a test fails

pytest prints the failing test's name, the exact `assert` line, and the two values that disagreed. Read that first, as it usually answers the question outright. Then decide which of the two cases you are in:

1. **The code is wrong** (you introduced a bug): fix the code, re-run pytest, confirm green. This is the suite doing its job.
2. **The code is right and the expectation changed** (you *deliberately* changed behaviour): update the test to describe the new intended behaviour. Never delete a failing test just to get green, as that removes the tripwire without fixing anything.

If you cannot tell which case you are in, that is the signal to stop and investigate before shipping: the test has found exactly the kind of ambiguity it exists to catch.

### 6. Adding a test

Whenever you fix a bug, add a function to the matching `tests/test_<module>.py`:

```python
def test_short_sentence_describing_the_expected_behaviour():
    """One docstring line on the bug this guards against (if any)."""
    result = the_function_you_fixed(hand_crafted_input)
    assert result == the_value_you_can_verify_by_hand
```

Name it `test_...`, keep the input small enough to reason about by hand, and run `pytest` to confirm it passes. If there is no matching file yet, create one: pytest picks up any `tests/test_*.py` automatically, no registration needed anywhere.

`test_workflow_smoke.py` is the slowest file (it runs a real tiny end-to-end workflow); everything else should take milliseconds.

### 7. Cleaning up (optional)

Running the suite leaves behind `.pytest_cache/`, `.coverage`, and `__pycache__/` folders. All of it is gitignored and regenerates automatically, so there is nothing to clean up before committing. If you would like a tidy working directory anyway:

```powershell
Get-ChildItem -Path phoenix_ml, tests -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .pytest_cache, .coverage -ErrorAction SilentlyContinue
```

Once again, if you are finished with testing, typing in `deactivate` returns you to your normal terminal and out of the virtual environment.

## Layout

- `conftest.py`: shared fixtures (currently just `synthetic_dataset_csv`, a tiny deterministic dataset written to a temp CSV).
- `test_model_training.py`: pure-function tests for monotonicity-constraint gating and random-seed derivation/application.
- `test_interpretability.py`: the comparable-metrics-table logic (`compute_interpretability_metrics`), including a direct regression test for the monotonicity-mismatch false-positives.
- `test_report_generation.py`: the interpretability section's target/model grouping and heading structure.
- `test_workflow_smoke.py`: runs a full tiny synthetic dataset through `run_workflow()` end to end, and checks the same seed reproduces the same training metrics.
- `test_persistence.py`: PhoenixPredictor save/load/predict round trip (including PERL expression-mode reconstruction), UQ calibrator selection, and the malformed-hyperparameter-string failure contract.
- `test_data_preprocessing.py`: input validation errors, split-method edge cases, and the scaler lookup (including the scaler-"None" regression test).
- `test_dataset_cleaning.py`: stuck-value dropna-space index remapping, constant-step column auto-classification, multivariate outlier guards.
- `test_physics_expressions.py`: the expression validator treated as a security control (sandbox-escape constructs rejected), expression chaining, PERL mapping extraction, and validator/evaluator/LaTeX whitelist sync.
- `test_hyperparameter_optimisation.py`: `_compute_metric` for all metrics (incl. zero-variance NaN guards), random-search early stopping and seeding, and the per-(model, target) tuned-instance lookup.
- `test_pareto_analysis.py`: Pareto dominance on hand-drawn cases, and best-HPO-vs-default performance selection.
- `test_postprocessing.py`: CV splitter construction (kwarg filtering, K-Fold shuffle/random_state coupling) and residual transformations.
- `test_uncertainty_quantification.py`: conformal interval symmetry and the finite-sample quantile correction, calibration-metric branches, bootstrap seed reproducibility.
- `test_workflow_steps.py`: end-to-end run through the WorkflowSession/step layer (the path the GUI drives): preprocessing → training → interpretability → HPO → UQ → report, ending in a loadable deployable predictor.
- `test_physics_model.py`: Script Mode PERL: the physics engine's output naming/residual math, script importing, and the predictor's script-mode reconstruction surviving a save-after-predict round trip.
- `test_hostile_datasets.py`: pathological-but-plausible real-world data: constant targets, 10-row single-feature sets, slashes and unicode in column/target names.

`tests/ISSUES.md` is the single living issue register: open bugs, suspected risks, and test-coverage gaps sorted by priority, the `ui.py` manual QA checklist (not automatable), and the crossed-out history of everything already fixed with its regression test. When you fix something, move its entry to the Resolved section there; when you find something, add it there first. CI runs the whole suite on every push/PR via `.github/workflows/tests.yml` (Linux + Windows).

## Philosophy

Two kinds of test live side by side here on purpose:

1. **Pure-function tests** (fast, milliseconds): catch broken logic in one function in isolation.
2. **End-to-end smoke tests** (slower, seconds): catch wiring/integration bugs between modules. Both real bugs this project has hit so far during report review (the monotonicity-mismatch false positive, and interpretability visuals not respecting the best-per-target restriction) were integration bugs, not broken logic in any single function: a pure-function suite alone would have missed both.

## Extending this suite

When you fix a bug, add a test that would have caught it *before* fixing the underlying code, confirm it fails against the old behaviour (or just trust the fix, if reverting is inconvenient), then confirm it passes after. That keeps the suite growing in exactly the places this codebase has actually broken, rather than in whatever is easiest to test.
