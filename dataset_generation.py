import pandas as pd
import numpy as np
import os
from typing import Dict
from phoenix_ml.physics_model import (
    run_physics_model,
    generate_simple_dataset,
    generate_residual_dataset,
    round_and_clean_floats
)

# === CONFIGURATION ===
# Insert the folder in which you want to have your simple and residual datasets.
dataset_folder = "examples"
nonlinear_input_path = os.path.join(dataset_folder, "Nonlinear Dataset.csv")
simple_output_path = os.path.join(dataset_folder, "Simple Dataset.csv")
residuals_output_path = os.path.join(dataset_folder, "Residuals Dataset.csv")

# === LOAD INPUT DATASET ===
df = pd.read_csv(nonlinear_input_path)
df.columns = df.columns.str.strip()

# Rename for model use
df = df.rename(columns={
    'Input Voltage': 'Va',
    'Input Torque': 'AL',
    'Motor Speed': 'omega',
    'Motor Torque': 'Tm',
    'Armature Current': 'Ia'
})

# === CONSTANTS AND VARIABLES ===
constants = {
    'Ra': 0.8,
    'La': 0.01,
    'J': 0.01,
    'B': 0.001,
    'Kt': 0.1,
    'Ke': 0.1
}

input_vars = ['Va', 'AL']
output_vars = [
    'Motor Speed', 'Motor Torque', 'dw/dt', 'dTm/dt',
    'Temperature', 'Armature Current', 'Friction Coefficient'
]

name_mapping = {
    'Motor Speed': 'omega',
    'Motor Torque': 'Tm',
    'dw/dt': 'dw/dt',
    'dTm/dt': 'dTm/dt',
    'Temperature': 'Temperature',
    'Armature Current': 'Ia',
    'Friction Coefficient': 'Friction Coefficient'
}

# === USER-DEFINED PHYSICS MODEL ===
def simulate_dc_motor_snapshot_all_columns(
    data: pd.DataFrame,
    constants: Dict[str, float],
    time: np.ndarray
) -> pd.DataFrame:
    Ra, Kt, Ke, B, J = constants['Ra'], constants['Kt'], constants['Ke'], constants['B'], constants['J']
    Va, TL = data['Va'].values, data['AL'].values

    Ia = TL / Kt
    Tm = Kt * Ia
    omega = (Va - Ia * Ra) / Ke
    dw_dt = (Tm - TL - B * omega) / J
    dTL_dt = np.gradient(TL) / 1e-4
    dTm_dt = Kt * dTL_dt
    temperature = 25
    friction = B + 0.0001 * np.sin(omega)

    return pd.DataFrame({
        'omega': omega,
        'Tm': Tm,
        'dw/dt': dw_dt,
        'dTm/dt': dTm_dt,
        'Temperature': temperature,
        'Ia': Ia,
        'Friction Coefficient': friction
    })


# === RUN PHYSICS MODEL ===
physics_df = run_physics_model(
    data=df,
    time_col=None,
    governing_function=simulate_dc_motor_snapshot_all_columns,
    constants=constants,
    input_vars=input_vars,
    output_vars=output_vars,
    name_mapping=name_mapping
)

# === GENERATE SIMPLE AND RESIDUAL DATASETS ===
simple_df = generate_simple_dataset(df, physics_df, input_vars, output_vars)
residuals_df = generate_residual_dataset(df, physics_df, input_vars, output_vars, name_mapping)

# === RENAME INPUT COLUMNS BACK TO DISPLAY NAMES ===
input_display_names = {'Va': 'Input Voltage', 'AL': 'Input Torque'}
simple_df.rename(columns=input_display_names, inplace=True)
residuals_df.rename(columns=input_display_names, inplace=True)

# === ROUNDING AND SAVE ===
round_and_clean_floats(simple_df).to_csv(simple_output_path, index=False)
round_and_clean_floats(residuals_df).to_csv(residuals_output_path, index=False)


# === PRINT DATASET SUMMARY ===
def print_dataset_summary(name: str, df: pd.DataFrame):
    print(f"\nDataset Generated: {name}")
    print(f" - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f" - Columns: {list(df.columns)}")


# === DISPLAY CONFIRMATION ===
print_dataset_summary("Nonlinear Dataset", df)
print_dataset_summary("Simple Dataset", simple_df)
print_dataset_summary("Residuals Dataset", residuals_df)


"""
HOW TO USE THIS SCRIPT FOR YOUR OWN SYSTEM:

1. Place your input dataset (e.g., CSV file) in a folder and update `dataset_folder`.

2. Define:
   - Your physical constants (`constants`)
   - Input variable names from your dataset (`input_vars`)
   - Desired physical outputs (`output_vars`)
   - Mapping between friendly names and internal model names (`name_mapping`)

3. Write your physics model function:
   def simulate_your_system(data, constants, time):
       # Use vectorized physics equations to return a DataFrame of output variables

4. Rename your dataset columns (if needed) to match what your model expects.

5. That's it! The script will:
   - Simulate physics outputs
   - Compute residuals between real and simulated
   - Save simple and residual datasets
   - Print summary stats

All logic is in `physics_model.py`, no need to modify that.
"""
