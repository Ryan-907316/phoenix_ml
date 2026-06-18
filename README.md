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
- **Report generation**: All findings compiled into a single PDF with full-resolution images also saved separately. Results tables, HPO CSV outputs, model pipelines saved as .pkl files, and a .json metadata file for full reproducibility.

## Installation

### From PyPI

Available on [PyPI](https://pypi.org/project/phoenix-ml-workflow/).

```bash
pip install phoenix-ml-workflow
```

Then launch the interface with:

```bash
phoenix-ml
```

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

The terminal will display your system information on startup. Press any key to open the graphical interface, where you can configure and run the full workflow without writing any code.

To generate the example DC motor dataset, run:

```bash
python examples/DC_Motors_Dataset_Generation.py
```

### Programmatic use

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

## Future Work

- **User documentation**: Proper guides covering machine learning concepts (HPO, interpretability, UQ) and how to get the most out of each step of the workflow.
- **Extended PEML**: Physics-Informed Neural Networks (PINNs) and other non-residual physics-enhanced methods.
- **Classification support**: Extending the workflow beyond regression problems.
- Ongoing bug fixes and quality-of-life improvements based on user feedback.

## License

This project is licensed under the MIT License: see the [LICENSE](https://github.com/Ryan-907316/phoenix_ml/blob/main/LICENSE) file for details.

## Credits

Package created by Ryan Cheung.

## Contact

University email: cheungkh@lancaster.ac.uk
