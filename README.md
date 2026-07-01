# phoenix_ml
A **P**hysics and **H**ybrid **O**ptimised **EN**gine for **I**nterpretability and e**X**plainability for **M**achine **L**earning. Intended to make the full machine learning workflow experience easier, from dataset to report.

# Overview

This package takes you through the entire machine learning workflow with all the tools you need in a single package. Dataset cleaning, preprocessing, model evaluation, interpretability, hyperparameter optimisation, postprocessing, uncertainty quantification, physics-based modelling, residual learning, and report generation are all included, with no additional imports required. The interface is designed to be easy to use and highly customisable at every step of the workflow.

The package is primarily aimed at regression problems in engineering and scientific applications, with built-in support for Physics-Enhanced Machine Learning (PEML) methods.

# Features

This package contains the following:
- **Dataset Cleaning**: Interactive column role assignment with automatic type detection. Sensor fault detection covering stuck values, clipping, and burst dropout. Configurable outlier detection (IQR, Z-score, or percentile-based) with multiple handling strategies, missing value imputation, and duplicate removal.
- **Physics modelling**: Physics-Enhanced Machine Learning (PEML) methods including residual learning, safe input of first-principles expressions, script-based physics models, and automatic generation of residual datasets. Supports both expression mode and script mode.
- **Preprocessing**: Customisable test/train split options (random, first-N, last-N). Scatter plots of features against target variables, highly detailed boxplots, and a distance correlation matrix with a toggleable dummy variable and optional Marchenko-Pastur denoising.
- **Model training and evaluation**: Support for ten regression models with the ability to add, change, or remove them. Compatible with single and multi-target optimisation. Results summarised as tables with MSE, R², Adjusted R², and Q² metrics.
- **Interpretability**: Partial Dependence Plots (PDPs) with Individual Conditional Expectations (ICE), SHAP summary plots, and SHAP dependence plots. Automatic selection of the preferred model for interpretability based on training performance.
- **Hyperparameter Optimisation (HPO)**: Random search (with Monte Carlo, Sobol, Halton, or Latin Hypercube sampling), Hyperopt (Tree-structured Parzen Estimator), and scikit-optimize (Gaussian Process Minimisation). Configurable number of iterations, CPU cores, and early stopping. Best model per target variable is selected by a user-defined metric (MSE, R², Adjusted R², or Q²).
- **Pareto front analysis**: Performance vs. training-time trade-off charts with non-dominated solution filtering, automatic log-scaling, and a numbered model legend.
- **Postprocessing**: Multiple cross-validation methods (K-Fold, Repeated K-Fold, LOO, LpO, Shuffle Split) with full argument and scoring metric customisation. Cook's Distance for influential point identification. Residual analysis including scatter, histogram, Q-Q plots, and automatic best-transform selection via the Anderson-Darling statistic.
- **Uncertainty Quantification (UQ)**: Bootstrap and conformal prediction intervals with a user-configurable confidence level. Ground truth overlaid on interval plots for direct visual comparison. Run before and/or after HPO to compare the effect of tuning.
- **Report generation**: All findings compiled into a single PDF with full-resolution images also saved separately. Results tables exported to a multi-sheet Excel file, model pipelines saved as .pkl files, and a .json metadata file for full reproducibility.

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
- `examples/Original Datasets/gt_2015.csv`: a real gas turbine emissions dataset ([UC Irvine ML Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set))

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

The example datasets are already included in the `examples/Original Datasets/` folder. The DC motor dataset can also be regenerated from scratch with:

```bash
python examples/DC_Motors_Dataset_Generation.py
```

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

### Near-term
- **User documentation**: Proper guides and tutorials covering machine learning concepts (HPO, interpretability, UQ) and how to get the most out of each step of the workflow.
- **Advanced model selection**: More regression models, and the ability to choose which hyperparameters to optimise and customise their search spaces directly from the UI.
- **Classification support**: Extending the workflow to classification problems, including appropriate metrics (accuracy, F1, AUC-ROC etc), confusion matrices, and report sections.
- **Physics-enhanced regularisation (soft constraints)**: Embedding first-principles expressions as soft penalty terms in the model loss function, allowing physical knowledge to guide training (compared to just residual learning).

### Medium-term
- **Symbolic regression / equation discovery**: Integration of gplearn (pure Python, sklearn-compatible) as an equation discovery mode allowing users to find closed-form governing equations directly from data, with the discovered expression optionally fed back into the existing PERL expression mode. PySR (higher performance, Julia backend) as an optional advanced alternative.
- **NGBoost probabilistic predictions**: Natural Gradient Boosting as a native probabilistic regression model outputting a full predictive distribution per point, bridging the model training and uncertainty quantification modules.
- **Physics-informed Gaussian Process regression**: GPs with physics-based mean functions and physics-structured kernels as a Bayesian complement to the existing PERL framework, starting with the sklearn GP implementation and progressing to custom kernel classes encoding known physical constraints such as symmetry, periodicity, or monotonicity.
- **Regression calibration reporting**: Reliability diagrams, proper scoring rules (CRPS, interval score), and calibration metrics (sharpness, RMS calibration error) for all interval-producing methods: bootstrap, conformal, and GP posterior, to assess calibration quality as well as coverage.
- **Warm-starting and persistent HPO studies**: Seeding optimisation with known-good configurations and persisting study state to disk (SQLite via Optuna) to support pause/resume across sessions in the desktop workflow.
- **Buckingham Pi / dimensional analysis**: Automatic generation of dimensionless Pi-groups from user-specified variable units using sympy (already a dependency), enabling physically-motivated feature construction and improved model generalisation across operating conditions.
- **Interpretable additive models**: Penalised-spline Generalised Additive Models with per-feature uncertainty intervals and monotonicity constraints, providing an interpretable, smooth model family between linear regression and black-box models.
- **SHAP waterfall plots**: Per-prediction local SHAP explanations for individual design points, for cases where an engineer must justify a single prediction.
- **TabPFN v2 (optional)**: The small-data tabular foundation model (Hollmann et al., 2025) as an optional install providing a strong no-tuning baseline on datasets up to approximately 50,000 rows.
- **Time series support**: Time-aware train/test splitting, temporal cross-validation strategies, and sequence-capable models for datasets where observations are ordered in time.
- **Model ensembling**: Stacking, blending, or voting ensembles built from the models already trained in the workflow, as a postprocessing step to squeeze out additional predictive performance.
- **Automated feature selection**: Using the distance correlation matrix and mutual information to automatically flag and optionally remove low-information or redundant features before training.
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

## Credits

Package created by Ryan Cheung.

## Contact

University email: cheungkh@lancaster.ac.uk
