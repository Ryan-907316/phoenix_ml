# Examples

## DC Motor Dataset (`DC_Motor_Dataset.csv`)

A synthetic DC motor dataset generated for demonstrating the Phoenix-ML workflow.

It contains inputs such as voltage and load torque, and outputs such as motor speed and armature current.

Run the generation script to produce the CSV:

```bash
python examples/DC_Motors_Dataset_Generation.py
```

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

**Source**: [Gas Turbine CO and NOx Emission Data Set — UCI ML Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

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
