# Examples

## Getting the examples

**If you installed via PyPI** (`pip install phoenix-ml-workflow`), run this once from whichever folder you want to work in:

```bash
phoenix-ml --get-examples
```

This copies this folder to `examples/` in your current directory, containing both datasets below plus this file.

**If you cloned the repository from GitHub**, the same files are already present; no separate step is needed. You will find them at `phoenix_ml/examples/` instead.

Paths in the examples below use the `--get-examples` layout (`examples/...`). If you are working from a cloned repository instead, use `phoenix_ml/examples/...` in their place.

## Using a dataset

**Via the graphical interface**: launch with `phoenix-ml` (PyPI install) or `python app.py` / `phoenix_ml.bat` (cloned repository), then in the interface set:

- **Dataset Path** to one of the CSV files below, e.g. `examples/Original Datasets/DC_Motor_Dataset.csv`
- **Output Directory** to wherever you want results saved, e.g. `Results/`
- **Target Variables** to the dataset's target columns, e.g. `Motor Speed, Armature Current`

then run.

**Via Python code**: call `run_workflow()` directly, as shown under each dataset below.

## DC Motor Dataset (`DC_Motor_Dataset.csv`)

A synthetic DC motor dataset for demonstrating the phoenix_ml workflow. It contains inputs such as voltage and load torque, and outputs such as motor speed and armature current.

`DC_Motors_Dataset_Generation.py` (in this folder) is not a prerequisite for using this dataset. Point the Physics Modelling tab's "Physics Script" field at it to use it in Script Mode, where it supplies the governing DC motor equations used to compute physics estimates and residuals for PERL.

Example use:

```python
from phoenix_ml.workflow import run_workflow

run_workflow(
    dataset_path="examples/Original Datasets/DC_Motor_Dataset.csv",
    output_dir="Results/",
    selected_models=["Random Forest Regressor", "XGBoost Regressor"],
    targets=["Motor Speed", "Armature Current"],
)
```

---

## Gas Turbine Emissions Dataset (`gt_2015.csv`)

A real-world dataset recorded from a gas turbine over one month in 2015, sourced from the UCI Machine Learning Repository.

**Source**: [Gas Turbine CO and NOx Emission Data Set: UCI ML Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

The dataset contains ambient condition measurements (temperature, pressure, humidity, air filter difference pressure, gas turbine exhaust pressure, turbine inlet temperature, turbine after temperature, compressor discharge pressure, and turbine energy yield) as inputs, with CO and NOX emission concentrations as target outputs.

Example use:

```python
from phoenix_ml.workflow import run_workflow

run_workflow(
    dataset_path="examples/Original Datasets/gt_2015.csv",
    output_dir="Results/",
    selected_models=["XGBoost Regressor", "HistGradientBoosting Regressor"],
    targets=["CO", "NOX"],
)
```
