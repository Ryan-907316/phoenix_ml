# Example: DC Motor Dataset

This dataset represents a synthetic DC motor system used for demonstrating the Phoenix-ML workflow.

It contains inputs such as voltage and torque, and outputs such as motor speed, torque, and current.

You can use it to test preprocessing, training, uncertainty quantification, and report generation.

Example use:
```python
from phoenix_ml.workflow import run_workflow

run_workflow(
    dataset_path="examples/DC_Motor_Dataset.csv",
    output_dir="results/",
    selected_models=["Random Forest Regressor", "XGBoost Regressor"],
    targets=["Motor Speed", "Motor Torque"]
)