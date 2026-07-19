# Changelog

All notable user-facing changes to phoenix_ml. The full defect history with per-fix regression-test references lives in [tests/ISSUES.md](tests/ISSUES.md), but the main changes are listed here.

## [1.2.0] - 2026-07-19

The first stable release of phoenix_ml: a systematic bug-fixing, testing, and quality-of-life pass over the whole package, plus several new analysis features. This is the first release marked Production/Stable. See tests/ISSUES.md for more information.

Verified with a fresh virtual environment and a full test run on each of Python 3.11, 3.12, and 3.13. Test suite reached 428 tests on all versions with per-module coverage reporting, plus CI on Linux and Windows via GitHub Actions.

- Fixed `dataset_cleaning.py`'s mostly-numeric text-column recovery silently breaking under pandas 3.0's new default string dtype, since four call sites checked `dtype == object` specifically.
- Fixed Sobol and FAST sensitivity analysis breaking outright under numpy 2.x, which removed the `ndarray.ptp()` method SALib < 1.5 depended on; bumped the SALib floor to 1.5 and reworked FAST's seeding to use SALib's own (now correct) seed handling, instead of the previous global-state workaround.
- Fixed a Python 3.11 incompatibility: `physics_expressions.py`'s LaTeX name-escaping helper used an f-string with a backslash inside the expression part, valid only from Python 3.12 (PEP 701) onwards, which crashed `import phoenix_ml` outright on 3.11.
- Added upper bounds to the core numerical dependencies (numpy, pandas, scipy, scikit-learn, SALib) so a fresh install cannot silently pick up an untested future major version again.
- Added a `CITATION.cff` file, so GitHub's "Cite this repository" button generates ready-to-use APA and BibTeX citations. A Zenodo-minted DOI will be added once the first GitHub Release is archived.
- Fixed switching datasets mid-session carrying stale results (the PERL section, HPO results, timings) from the previous dataset into the new report; re-running Preprocessing now invalidates every downstream result and step badge automatically.
- Fixed Stop/Pause only being cooperative inside HPO; it now takes effect during every long-running step, and settings are locked (visible but not editable) while a run is active.
- Added full reproducibility: a single random seed (unset resolves to 0) now drives every stochastic stage (train/test split, model construction, all three HPO backends, bootstrap/conformal UQ, SHAP background sampling, Morris/Sobol/FAST sampling, permutation importance) via per-stage derived seeds.
- Added per-target monotonicity constraints: constraints are now set per target rather than globally, so a physically-motivated direction on one target no longer forces constraints on other variables.
- Expanded the interpretability option for every model, before and after HPO: every selected model now gets a comparable metrics row (top features per method, cross-method rank agreement, monotonicity sanity check), running as two independently toggleable steps, with the computationally and spatially expensive visuals still rendering for each target's best model in the report.
- Added per-(model, target) HPO selection: UQ (After HPO) and Interpretability (After HPO) now use each target's own best hyperparameters for each model, replacing an across-targets-averaged lookup.
- Added Yeo-Johnson residual transformation with optimised lambda parameter, replacing the old fixed menu of named transforms (Log/Sqrt/Box-Cox/Modulus/Cube-root, now removed) with a single continuous search that fits Yeo-Johnson's lambda per target by directly minimising the Anderson-Darling statistic. Arcsinh is kept as a genuinely distinct alternative. The report shows the lambda-optimisation curve per target with a reference table of the AD statistic at each classic named lambda; configs passing one of the removed names get a warning and the name is skipped.
- Added a parsimony rule for residual transformations: residuals that already pass a Shapiro-Wilk normality check are left untransformed rather than always applying the lowest-AD transform.
- Standardised input validation: every user-settable knob now either works (with documented clamping) or fails immediately with an error naming the parameter, closing some silent-nonsense behaviours and cryptic deep crashes that were previously unidentified; new shared `phoenix_ml/validation.py` helpers enforce this consistently.
- Fixed one HPO backend failing (e.g. scikit-optimize's minimum-calls requirement) discarding the completed results of the other backends for that model; the UI also validates the scikit-optimize minimum up front.
- Fixed stale or unknown residual-transformation names (from configs predating the Yeo-Johnson consolidation) silently producing identity transforms mislabelled as the requested one.
- Fixed the Conformal reliability curve using an uncorrected quantile while the reported interval used the finite-sample-corrected one.
- Fixed per-plot interpretability failures only being logged to the console instead of shown in the report; removed a dead SHAP GradientExplainer branch.
- Fixed a synthetic "Dummy" noise column silently overwriting a real feature that happened to share the name.
- Fixed `detect_column_type` sampling only the first 20 non-null values, which could misclassify a sorted or placeholder-headed column; the sample is now a random, seeded draw.
- Fixed `missing_action='None'` leaving coercion-introduced NaNs completely unlogged.
- Added a "Show Best Transformation Normality Metrics" table (Shapiro-Wilk, Lilliefors, Filiben, Jarque-Bera, D'Agostino, individually toggleable) with an explicit master toggle, so hiding the whole table is a deliberate choice rather than a side effect of unchecking every individual test.
- Added an `[INFO]`/`[WARN]` structured progress log, consistent step banners, and uniform result tables across every pipeline step, replacing a mix of formats.
- Added Excel export column auto-sizing.
- Fixed a cross-target monotonic-constraint leak in four shared-instance training loops: a target with no constraint could silently keep the previous target's constraint.
- Fixed Q² dividing by zero on zero-variance targets in all three independent copies of the metric; all three now return NaN like the metrics beside them.
- Fixed a full "UQ (After HPO)" report section and Excel sheet rendering with an empty table when HPO had not actually run.
- Fixed `apply_cleaning` mutating the caller's column-role dictionary in place, which could carry a stale "Exclude" across runs or datasets in the same session.
- Fixed `save_predictor` silently saving an unusable Script Mode predictor when the underlying physics script had moved or been deleted; now raises at save time instead of failing later at deploy-time prediction.
- Fixed the standalone `workflow.py` model-persistence path having no error handling, unlike its predictor path, so one bad hyperparameter string could lose the entire PDF and Excel output.
- Fixed Script Mode `PhoenixPredictor` crashing on save if `predict()` had already been called (the executed physics namespace is not picklable).
- Fixed a slash in a target name (e.g. "Residual dw/dt") crashing model saving, since filename sanitisation covered spaces but not slashes.
- Decided and applied a consistent early-stopping rule: `patience=0` stops at the first non-improving evaluation, never after an improving one, across all three HPO backends.
- Decided that UQ calibrator selection falls back to the next method in the preference order on NaN, instead of leaving a target with no interval.
- Fixed LGBMRegressor and HistGradientBoostingRegressor missing from SHAP's TreeExplainer routing, so every SHAP output for those two models had been silently vanishing from reports.
- Fixed MLPRegressor being routed to `shap.GradientExplainer` (TensorFlow/PyTorch only), which failed every run and imported TensorFlow just to fail; now routes directly to KernelExplainer.
- Fixed the UI's Feature Scaling "None" option silently applying StandardScaler anyway (the no-scaling branch was unreachable).
- Fixed `atan2(y, x)` crashing the LaTeX preview despite being a valid, validated expression.
- Fixed "Monotonicity Mismatches" flagging every model on a constrained target, including model types that cannot accept the constraint at all.
- Fixed Interpretability generating full visual sets for every selected model instead of only each target's best model; every model still keeps a metrics-table row.
- Added extended regression diagnostics: VIF, condition number, and matrix rank (multicollinearity, with a colour-coded report table and a "Show Multicollinearity" toggle), plus DFFITS, leverage, and studentised residuals (with a warning when the design matrix is rank-deficient), Breusch-Pagan and White heteroscedasticity tests, and Durbin-Watson and Ljung-Box autocorrelation tests.
- Expanded metrics: KGE (Kling-Gupta Efficiency) alongside MSE/RMSE/NRMSE/MAPE/R²/Adjusted R²/Q², plus PICP and MPIW for uncertainty interval quality.
- Multivariate outlier detection: Isolation Forest, Local Outlier Factor, and Minimum Covariance Determinant (Elliptic Envelope), alongside the existing per-column IQR/Z-score/percentile methods.
- Added SHAP waterfall plots, showing per-prediction feature contributions across a spread of representative samples.
- Added Regression calibration reporting: CRPS scoring and reliability diagrams for uncertainty intervals.
- Added ALE (Accumulated Local Effects) plots, alongside the existing PDP/ICE.
- Added LOFO (Leave-One-Feature-Out) importance, alongside permutation importance.
- Added global sensitivity analysis: Morris elementary-effects screening, Sobol variance-based indices, and FAST (Fourier Amplitude Sensitivity Testing) via SALib. When Sobol and FAST are both enabled they render as a single side-by-side agreement plot, since they estimate the same S1/ST indices via independent mechanisms, and the interpretability metrics table gains a Sobol-FAST rank-agreement score.
- Added Deployable single-file predictor: `PhoenixPredictor`, a self-contained `.pkl` bundling the fitted per-target pipeline, optional physics reconstruction, and an optional uncertainty-interval calibrator, with `.predict()`/`.predict_interval()`/`.summary()`.
- Added Executive Summary and Time Breakdown report sections, giving a top-of-report overview and a per-step timing table.


## [1.1.2] - 2026-07-01

Report overhaul and bug fixes, normality tests, diagnostics, and API fixes.

- Added Table of Contents heading to report via multiBuild, a Two-pass PDF build, with page numbers.
- Unified all table header colours.
- Changed R^2 -> R², Q^2 -> Q², ADJUSTED R^2 -> Adjusted R² throughout.
- Model Training Results: fixed column headers and changed heading to CustomHeading so it appears in TOC.
- PERL reconstruction metrics table: reduced last two column widths to prevent overflow.
- Pareto section: metric name in caption now rendered properly.
- Best Models per Target: metric column rename uses the proper name instead of manual string replacement.
- HPO summary section: removed stale CSV path parameter and reference; updated body to reference Excel file.
- UQ filename generation: fixed stage label duplication (e.g. After_HPO_After_HPO) in saved plot filenames.
- Replaced dead import save_uncertainty_results with save_uq_plots.
- Replaced Shapiro-Wilk and KS tests with Lilliefors test.
- Retained Anderson-Darling (transformation selection) and Filiben Q-Q correlation coefficient.
- Cook's Distance plot: threshold label now shows count and percentage of points exceeding threshold.
- Added scaler_type and split_random_state parameters (previously missing in the GUI).
- Changed evals and calls defaults from 10 to 50 (consistent with WorkflowSession).
- Added Excel results export (HPO Results, UQ Before/After HPO sheets) matching GUI output.
- Return dict now includes "xlsx" key; removed stale "csv" key.
- Removed HPO and UQ results CSV output (results now in one Excel file only).
- Fixed doc.build() -> build_pdf() so page numbers are correct.
- Removed csv_path field from WorkflowSession and ensure_dirs().
- Fixed stale docstring on run_step_perl.
- Removed report_dir argument from handle_uq_reporting_section calls.
- Removed None csv_path argument from add_hpo_summary_section call.
- Comprehensive Future Work section update with near/medium/long-term additions (Optuna, SALib, MAPIE, symbolic regression, NGBoost, physics-informed GP, conformal calibration, dimensional analysis, multi-fidelity modelling, and more).
- Fixed stale Features section reference to CSV outputs.

## [1.1.1] - 2026-06-19

Bug fixes related to installation of phoenix_ml v1.1.0.

- Added setuptools<80 dependency to fix pkg_resources ImportError with hyperopt
- Wrapped hyperopt and skopt top-level imports in try/except so a broken optional dependency no longer prevents the UI from starting (but this does mean that if there are issues, hyperopt and skopt won't work)
- Bundled phoenix_ml/examples/ (DC motor + gas turbine datasets, generation script) as package data so pip users get them without cloning the repository
- Added --get-examples CLI flag to copy bundled examples to the working directory
- Fix orange_theme.json missing from wheel (package-data entry)
- Fix PyPI license format rejection (reverted to { text = 'MIT' })
- Update README with step-by-step pip installation guide and Future Work roadmap

## [1.1.0] - 2026-06-18

Full graphical interface, addition of new modules, and bug fixes

The addition of the following files:
- `phoenix_ml/ui.py`: complete CustomTkinter graphical interface with tabbed settings for every workflow step, live progress log, and step-by-step/full-run execution
- `phoenix_ml/workflow_steps.py`: step-by-step workflow runner consumed by the UI
- `phoenix_ml/cli.py`: console-script entry point (phoenix-ml command) with system info splash screen
- `phoenix_ml/dataset_cleaning.py`: interactive dataset cleaning with column role assignment, automatic fault detection, outlier handling, and imputation, which is all handled by the UI
- `phoenix_ml/pareto_analysis.py`: Pareto front analysis for HPO results with performance vs training-time trade-off charts
- `phoenix_ml/physics_expressions.py`: safe first-principles expression parsing and evaluation for PEML residual learning
- `phoenix_ml/orange_theme.json`: custom UI colour theme, because I like the colour orange
- app.py: wrapper for the UI launcher
- `phoenix_ml.bat`: Windows double-click launcher as a batch file
- `examples/Original Datasets/gt_2015.csv`: real gas turbine emissions dataset (from the open source UC Irvine ML Repository)
- `examples/DC_Motors_Dataset_Generation.py`: moved and renamed from dataset_generation.py at repo root

PDF report changes:
- plot_ice_and_pdp and plot_shap_dependence now return list[Figure], each capped at _MAX_ROWS_PER_FIG=3 rows, so no subplot row ever spans two pages
- plot_features_vs_targets returns {target: list[Figure]} and plot_boxplots returns list[Figure] with _MAX_ROWS_PER_FIG=4 for the wider 3-col layout
- report_generation: add_interpretability_section and add_preprocessing_section updated to iterate list-of-figures; CondPageBreak inserted before every image

UI improvements:
- Cross-validation tab is now fully dynamic: fields shown adapt per method (K-Fold shows n_splits + random_state; Repeated K-Fold adds n_repeats; LOO shows no fields; Leave p Out shows p; Shuffle Split shows all three)
- HPO: validation error if all three methods are unchecked
- HPO: Sampling Method dropdown greys out when Random Search is unchecked
- Interpretability defaults synced to match WorkflowSession (1000/10/250/10)
- UQ subsample_test_size default synced to 50

Bug fixes:
- Way too many to list here. But in general, better handling of edge cases at every single level of the workflow. Images in the PDF report handle a lot of exceptional circumstances, path hardcoding has been removed for example datasets, runtime optimisations have been made and can be adjusted, empty dicts treated as no arguments instead of breaking at runtime, etc.

Package / configuration additions:

- `pyproject.toml`: SPDX license format, Pillow and customtkinter added to deps, [project.scripts] phoenix-ml entry point, proper classifiers and keywords, license-files = ["LICENSE"]
- `phoenix_ml/__init__.py`: version string, module docstring, run_workflow export
- `README.md`: PyPI link, updated install instructions, programmatic example, corrected dataset path
- `examples/README.md`: rewritten to cover both DC Motor and GT 2015 datasets with UCI source attribution
- `.gitignore`: added Results*/, *.pdf, .idea/, .DS_Store, Thumbs.db

## [1.0.1] - 2025-11-07

Bug fixes:

- Changed interpretability fallback to work when TreeExplainer fails, it defaults to KernelExplainer which is slower, but more robust.
- Changed the dependencies list by specifying the exact version of each dependency (a lot less robust, but stable and works).

## [1.0.0] - 2025-10-28

Initial release of phoenix_ml.

- Changed name from "The Python Workflow for PEML" to "phoenix_ml", marking the beginning of the transition from 3rd year university project to continued development.
- Modularised and split all files into respective parts so that they are in different and callable files that can be used in sections instead of one, multiple thousand line script that had to be changed all at once.
- Added a new feature for UQ to be done either before or after HPO, or both.
- Added report generation so all the information is in a single handy .pdf file.
- Built the physics modelling in-house instead of having to use external tools (such as MATLAB) to do this; though it is more limited at the moment.
- Added .csv generation for HPO and UQ, and also supported the generation of .pkl files for the model and each target variable.
- Added .json file of workflow settings for reproducibility.
- Various bug fixes.

## [0.1.0] - 2025-03-28

Initial release of "The Python Workflow for PEML" (Currently untitled at the time).

- No official changelog exists of this version, but to see the capability of the workflow at this time, refer to the dissertation, this describes all features in detail.