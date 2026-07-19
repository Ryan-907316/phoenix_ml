# phoenix_ml

![tests](https://github.com/Ryan-907316/phoenix_ml/actions/workflows/tests.yml/badge.svg)

A **P**hysics and **H**ybrid **O**ptimised **EN**gine for **I**nterpretability and e**X**plainability for **M**achine **L**earning. Intended to make the full machine learning workflow experience easier, from dataset to report.

# Overview

This package takes you through the entire machine learning workflow with all the tools you need in a single package. Dataset cleaning, preprocessing, model evaluation, interpretability, hyperparameter optimisation, postprocessing, uncertainty quantification, physics-based modelling, residual learning, and report generation are all included, with no additional imports required. The interface is designed to be easy to use and highly customisable at every step of the workflow.

The package is primarily aimed at regression problems in engineering and scientific applications, including scenarios with real-world data from instrumentation and sensors, with built-in support for dataset cleaning and Physics-Enhanced Machine Learning (PEML) methods.

# Features

This package contains the following:
- **Dataset Cleaning**: Interactive column role assignment with automatic type detection. Sensor fault detection covering stuck values, clipping, and burst dropout. Configurable outlier detection (per-column IQR/Z-score/percentile, or multivariate Isolation Forest/Elliptic Envelope) with multiple handling strategies, missing value imputation, and duplicate removal.
- **Physics modelling**: Physics-Enhanced Machine Learning (PEML) methods including residual learning, safe input of first-principles expressions, script-based physics models, and automatic generation of residual datasets. Supports both expression mode and script mode, with physics-only vs ML-only vs PERL accuracy comparison in the report.
- **Preprocessing**: Customisable test/train split options (random, first-N, last-N). Scatter plots of features against target variables, highly detailed boxplots, a distance correlation matrix with a toggleable dummy variable and optional Marchenko-Pastur denoising, multicollinearity diagnostics (VIF, condition number, matrix rank), advisory feature-selection flags for noise-level and redundant features, and PCA scree/biplot analysis.
- **Model training and evaluation**: Support for ten regression models with the ability to add, change, or remove them. Compatible with single and multi-target optimisation. Results summarised as tables with MSE, RMSE, NRMSE, MAPE, R², Adjusted R², Q², and KGE metrics.
- **Monotonicity constraints**: Per-target, per-feature monotonic constraints (increasing/decreasing) for XGBoost, LightGBM, and HistGradientBoosting, set through a UI picker, so physically-motivated relationships (e.g. drag increasing with velocity) can be enforced per target rather than globally.
- **Interpretability**: Partial Dependence Plots (PDPs) with Individual Conditional Expectations (ICE), Accumulated Local Effects (ALE), SHAP summary, dependence, and per-prediction waterfall plots. Every selected model gets a comparable metrics row (top features and cross-method rank agreement), both before and after HPO; with images for each target's best model.
- **Global sensitivity analysis**: Morris elementary-effects screening, Sobol variance-based indices, and FAST (Fourier Amplitude Sensitivity Testing), with a side-by-side Sobol-vs-FAST agreement plot and rank-agreement score when both are enabled.
- **Hyperparameter Optimisation (HPO)**: Random search (with Monte Carlo, Sobol, Halton, or Latin Hypercube sampling), Hyperopt (Tree-structured Parzen Estimator), and scikit-optimize (Gaussian Process Minimisation). Configurable iterations, CPU cores, and early stopping. Best model and hyperparameters are selected per (model, target) pair by a user-defined metric.
- **Pareto front analysis**: Performance vs. training-time trade-off charts with non-dominated solution filtering.
- **Postprocessing**: Multiple cross-validation methods (K-Fold, Repeated K-Fold, LOO, LpO, Shuffle Split) with full argument and scoring metric customisation. Extended regression diagnostics: Cook's Distance, DFFITS, leverage, studentised residuals, Breusch-Pagan/White heteroscedasticity tests, and Durbin-Watson/Ljung-Box autocorrelation tests. Permutation and Leave-One-Feature-Out (LOFO) importance. Residual normality transformations: Yeo-Johnson with its λ optimised per target against the Anderson-Darling statistic (with an λ-optimisation curve and named-transform reference table in the report) plus Arcsinh, with a five-test normality metrics table.
- **Uncertainty Quantification (UQ)**: Bootstrap and conformal prediction intervals (plus the native GP posterior for Gaussian Process models) with a user-configurable confidence level, CRPS and RMS calibration error scoring, and ground truth overlaid on interval plots. Run before and/or after HPO to compare the effect of tuning.
- **Reproducibility**: A single random seed drives every stochastic stage (splitting, model construction, all three HPO backends, bootstrap/conformal UQ, SHAP background sampling, Morris/Sobol/FAST sampling, permutation importance) via per-stage derived seeds: the same seed and settings reproduce identical results.
- **Report generation**: All findings compiled into a single PDF, opening with an Executive Summary and a per-step Time Breakdown, with full-resolution images also saved separately. Results tables exported to a multi-sheet Excel file, model pipelines saved as .pkl files, a single self-contained deployable predictor (.pkl) that bundles scaler, models, physics reconstruction, and approximate intervals, and a .json metadata file for full reproducibility.

## Installation

### From PyPI

Available on [PyPI](https://pypi.org/project/phoenix-ml-workflow/).

**Step 1: Install the package:**

```bash
pip install phoenix-ml-workflow
```

**Step 2: Get the example datasets** (optional, but recommended for a quick start):

```bash
phoenix-ml --get-examples
```

Run this from whichever folder you want to work in. It will create an `examples/` folder there containing two ready-to-use datasets:
- `examples/Original Datasets/DC_Motor_Dataset.csv`: a synthetic DC motor dataset
- `examples/Original Datasets/gt_2015.csv`: a real gas turbine emissions dataset (from the [UC Irvine ML Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set))

**Step 3: Launch the interface:**

```bash
phoenix-ml
```

The terminal will display your system information on startup. Press any key to open the graphical interface.

**Step 4: Load your dataset:**

In the interface, set the **Dataset Path** to one of the CSV files from Step 2, for example:

```
examples/Original Datasets/DC_Motor_Dataset.csv
```

Set the **Output Directory** to wherever you want results saved (e.g. `Results/`), fill in the **Target Variables** (e.g. `Motor Speed, Armature Current` for the DC motor dataset, or `CO, NOX` for the gas turbine dataset), and you are ready to run.

### From GitHub (to explore or modify the source)

```bash
git clone https://github.com/Ryan-907316/phoenix_ml.git
cd phoenix_ml
pip install .
```

Then run:

```bash
python app.py
```

(On Windows you can also double-click `phoenix_ml.bat`, which launches the same thing and keeps the window open if anything goes wrong.)

The example datasets are already included in the repository at `phoenix_ml/examples/Original Datasets/`, ready to use, no separate download or generation step needed. See [phoenix_ml/examples/README.md](phoenix_ml/examples/README.md) for details on each dataset.

### Use from Python code

The full workflow can also be called directly from Python, for example:

```python
from phoenix_ml import run_workflow

results = run_workflow(
    dataset_path="examples/Original Datasets/DC_Motor_Dataset.csv",
    output_dir="Results/",
    selected_models=["XGBoost Regressor", "Random Forest Regressor"],
    targets=["Motor Speed", "Armature Current"],
)
```

Make sure to run `phoenix-ml --get-examples` first (or clone the repository) so the example datasets are available locally.

## Future Work

Items from earlier versions of this list that have since shipped (ALE, LOFO importance, Morris/Sobol sensitivity analysis, per-method UI toggles, monotonicity constraints, extended regression diagnostics, SHAP waterfall plots, calibration scoring, multivariate outlier detection, expanded metrics, feature-selection flags) now appear under Features above: see the [CHANGELOG](CHANGELOG.md) for when each arrived.

### Short-term
- **Advanced conformal prediction**: Jackknife+, CV+, and conformalized quantile regression (CQR) via MAPIE, extending the current split-conformal implementation to provide adaptive interval widths for heteroscedastic data and stronger distribution-free coverage guarantees.
- **Optuna HPO backend**: TPE, CMA-ES, Gaussian Process Bayesian optimisation, and NSGA-II/III multi-objective search in a single well-maintained pure-Python framework, extending and partially replacing the current scikit-optimize backend.
- **Multi-objective HPO**: Optimising across multiple metrics simultaneously (predictive performance vs. model complexity vs. training time) using Pareto-optimal selection, turning the existing post-hoc Pareto analysis into an active multi-objective optimiser via Optuna's NSGA-II/III samplers.
- **Advanced model selection**: More regression models, and the ability to choose which hyperparameters to optimise and customise their search spaces directly from the UI.
- **Additional residual transformations**: Lambert W × Gaussian (Goerg 2011) and Tukey's g-and-h transforms alongside the current Yeo-Johnson/Arcsinh pair, as both correct skewness and excess kurtosis independently, but need hand-rolled implementations (no scipy/sklearn support).
- **PRESS statistic and predicted R²**: rounding out the extended regression diagnostics already in place.
- **User documentation**: The creation of detailed user documentation to add existing ease-of-use, and explains everything they need to know in detail, including how to use phoenix_ml, how to use every setting in the UI, instructions on how to interpret information from the report, and step-by-step examples and tutorials.
- **Model documentation**: Auto-generated model cards from the existing metadata JSON, structured to document intended use, data provenance, metrics, and known limitations.

### Medium-term
- **Symbolic regression / equation discovery**: Integration of gplearn (pure Python, sklearn-compatible) as an equation discovery mode allowing users to find closed-form governing equations directly from data, with the discovered expression optionally fed back into the existing PERL expression mode. PySR (higher performance, Julia backend) as an optional advanced alternative.
- **NGBoost probabilistic predictions**: Natural Gradient Boosting as a native probabilistic regression model outputting a full predictive distribution per point, bridging the model training and uncertainty quantification modules.
- **Physics-informed Gaussian Process regression**: GPs with physics-based mean functions and physics-structured kernels as a Bayesian complement to the existing PERL framework, starting with the sklearn GP implementation and progressing to custom kernel classes encoding known physical constraints such as symmetry, periodicity, or monotonicity.
- **Physics-enhanced regularisation (soft constraints)**: Embedding first-principles expressions as soft penalty terms in the model loss function, allowing physical knowledge to guide training, compared to just residual learning.
- **Classification support**: Extending the workflow to classification problems, including appropriate metrics (accuracy, F1, AUC-ROC etc.), confusion matrices, and report sections.
- **Warm-starting and persistent HPO studies**: Seeding optimisation with known-good configurations and persisting study state to disk (SQLite via Optuna) to support pause/resume across sessions in the desktop workflow.
- **Buckingham Pi / dimensional analysis**: Automatic generation of dimensionless Pi-groups from user-specified variable units using sympy, enabling physically-motivated feature construction and improved model generalisation across operating conditions.
- **Interpretable additive models**: Penalised-spline Generalised Additive Models with per-feature uncertainty intervals and monotonicity constraints, providing an interpretable, smooth model family between linear regression and black-box models.
- **TabPFN v2 (optional)**: The small-data tabular foundation model (Hollmann et al., 2025) as an optional install providing a strong no-tuning baseline on datasets up to approximately 50,000 rows.
- **Time series support**: Time-aware train/test splitting, temporal cross-validation strategies, and sequence-capable models for datasets where observations are ordered in time. This would also allow the physics `gradient()` helper to detect unevenly-spaced samples after row-removing cleaning steps.
- **Model ensembling**: Stacking, blending, or voting ensembles built from the models already trained in the workflow, as a postprocessing step to squeeze out additional predictive performance.
- **Interactive reports**: HTML or dashboard-based output as an alternative to the PDF, allowing zoomable plots and filterable results tables.

### Long-term
- **Multi-fidelity modelling**: Co-kriging and Kennedy-O'Hagan multi-fidelity GP frameworks for combining cheap low-fidelity data (simulation) with expensive high-fidelity experimental data.
- **Physics-informed deep kernel GP**: Deep kernel learning with physics-structured kernels and latent force models (GPyTorch) for advanced Bayesian physics-informed inference, as an optional heavy dependency.
- **Dynamical systems identification**: Sparse identification of nonlinear dynamics (PySINDy) for time-dependent engineering systems: applicable once time-series support is in place.
- **Hard physics constraints**: Enforcing physical constraints exactly at every prediction (e.g. via Lagrange multipliers or projection methods), going beyond soft regularisation to guarantee physically consistent outputs.
- **Extended PEML/physics-informed learning**: Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNOs), and other architectures capable of learning solution operators for PDEs and a broader class of physics-governed problems.

---

Ongoing bug fixes and quality-of-life improvements based on user feedback.

## License

This project is licensed under the MIT License: see the [LICENSE](https://github.com/Ryan-907316/phoenix_ml/blob/main/LICENSE) file for details.

## Citation

If you use phoenix_ml in your work, please cite it. A ready-to-use APA and BibTeX citation is available via GitHub's "Cite this repository" button (top right of the repo page), generated from [CITATION.cff](CITATION.cff). A DOI for citing a specific version will be added once a release is archived on Zenodo.

## Credits

Package created by Ryan Cheung.

## Contact

University email: cheungkh@lancaster.ac.uk
