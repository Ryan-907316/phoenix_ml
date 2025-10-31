# phoenix_ml
A **P**hysics and **H**ybrid **O**ptimised **EN**gine for **I**nterpretability and e**X**plainability for **M**achine **L**earning. Intended to make the full machine learning workflow experience easier, from dataset to report.

# Overview

This package intends to take you through the entire machine learning workflow with all the tools that you need in a single package. Preprocessing, model evaluation, interpretability, hyperparameter optimisation, postprocessing, uncertainty quantification, modelling with first-principles equations, residual learning, and report generation are all included without the additional importation of other packages, and has been made easy to use and highly customisible at every step of the workflow.

This package is intended to be used with regression datasets that are clean, though future versions may extend this to classification problems as well as an in-house dataset cleaner.

# Features

This package contains the following:
- **Physics modelling**: Physics-Enhanced Machine Learning (PEML) based methods such as residual learning, easy input of first-principles equations, and automatic generation of residual datasets.
- **Preprocessing**: Customisable test/train options, scatter plots of features and target variables, highly detailed boxplots of features, and distance correlation matrix with toggleable dummy variable.
- **Model training and evaluation**: The ability to add, change, or remove regression models when training, compatibility with single and multivariable optimisation, evaluation summarised as neat tables with different evaluation metrics.
- **Interpretability**: Partial Dependence Plots (PDPs) with Individual Condition Expectations (ICE), with the inclusion of Shapley Additive eXplanation (SHAP) summaries and plots. Automatic evaluation of preferred model based on model training performance used in interpretability.
- **Hyperparameter Optimisation (HPO)**: Inclusion of random HPO (with the choice of random sampling included, choose from Monte Carlo, Sobol, Halton, or Latin Hypercube sampling), Hyperopt (Adaptive Tree-based Parzen Estimators) and scikit-optimize (Gaussian Process Minimisation). Customise which method(s) to use and compare, as well as the number of iterations for all, and the number of CPU cores for HPO. Displays the best performing model for each target variable according to the user-defined metric (MSE, $R^2$, Adjusted $R^2$, or $Q^2$) and the time elapsed for each.
- **Postprocessing**: Variety of cross-validation methods with full customisation of arguments and scoring metrics. Influential points determination using Cook's Distance, residual analysis and automatic determination of transformed residuals using Anderson-Darling normality test, and Q-Q plots.
- **Uncertainty Quantification (UQ)**: Use of bootstrapping and conformal predictions as well as a user-customisable confidence and prediction interval. Ability to perform UQ before and/or after HPO for comparison.
- **Report generation**: Summarise all findings in a single .pdf file with the above features, with high quality images included in the report and additionally in a separate images folder for further analysis. Useful information is summarised in neat tables and .csv files. Models are saved as .pkl files overall and for each target variable. .json files included for full reproducibility.

## Installation

You can install phoenix_ml directly from PyPI using the following link:
[https://pypi.org/project/phoenix-ml-workflow/](https://pypi.org/project/phoenix-ml-workflow/)

Or install it via pip:

```bash
pip install phoenix-ml-workflow
```

## Quick Start

Included in this repository is a DC motor dataset for demonstration.

Clone the repository and run the workflow:

```bash
git clone https://github.com/Ryan-907316/phoenix_ml.git
cd phoenix_ml
python _runner.py
```

## TODO and Future Work

The next version of phoenix_ml (likely v1.1) plans to include several of the following improvements and new features. Here's a preview of what I have in mind:

- User documentation and proper instructions on how to use this package effectively. It will cover machine learning concepts such as explaining how hyperparameter optimisation works, the need for interpretability tools and how to use them, but also how to maximise the usability of this workflow so that almost anyone with an interest can use this.
- Improved image formatting in the generated reports, including better handling of datasets that contain a large number of features and target variables (as right now, they may appear a bit squished/have overlapping text etc)
- Integration of ground truth overlaid in uncertainty quantification plots, giving a better visual indication of the coverage within confidence and/or prediction intervals.
- Addition of Pareto Front analysis for performance-efficency trade offs in model selection, which includes dominance filtering and identification of optimal trade-offs.
- An optional Marchenko-Pastur thresholding as an addition to the dummy variable for improved distance correlation estimation.
- Expansion of PEML capabilities to more than just residual learning- the plan is to possibly integrate neural network models to support the inclusion of Physics-Informed Neural Networks (PINNs) in the long term, but also other non-NN physics-enhanced methods too, such as hybrid physics-data modelling.
- Development of an in-house dataset cleaning utility script to deal with incomplete and weird data, ways to deal with NaNs, floating point oddities, outliers, spikes, time drifts, ect. Probably will include some filtering like simple/exponential weighted average filters, Savitzky-Golay filters etc. Plan is to make everything user-defined and customisable too.
-  Potential extension to classification problems as well as combining regression and classification methods together (but this I know will take a lot of work, especially getting it to work seamlessly)
-  Ongoing bug fixes and QoL improvements, and also any feedback given from v1.0.

## License

This project is licensed under the MIT License - see the [license](https://github.com/Ryan-907316/phoenix_ml/blob/main/LICENSE) file for details.

## Credits

Package created by Ryan Cheung, and extends the work done previously for an individual undergraduate project.

## Contact

University email is cheungkh@lancaster.ac.uk for queries, instructions, or more information.

