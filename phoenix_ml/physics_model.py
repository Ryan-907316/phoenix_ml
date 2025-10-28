# physics_model.py
# This is a generalised helper module for integrating physics-based equations that defines the dataset and then used to generate a residual dataset.
# This method is known as Physics-Enhanced Residual Learning (PERL) and is a subset of Physics-Enhanced Machine Learning and is a hybrid modelling method.
# For the script to insert your dataset and to define your equations, see the test_dataset_generation.py module, it provides more instructions on how to use this.
# You should not have to modify this module in order to use the PERL part of the workflow.

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional

def run_physics_model(
    data: pd.DataFrame,
    time_col: Optional[str],
    governing_function: Callable[[pd.DataFrame, Dict[str, float], np.ndarray], pd.DataFrame],
    constants: Dict[str, float],
    input_vars: List[str],
    output_vars: List[str],
    name_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    
    # Run a physics model simulation defined by `governing_function`.
    # Returns a DataFrame with columns named as '<Display Name>_physics'.
    if time_col is not None:
        time = data[time_col].values
    else:
        time = np.zeros(len(data))

    inputs = data[input_vars].copy()
    physics_df = governing_function(inputs, constants, time)

    if name_mapping:
        for display_name, internal_name in name_mapping.items():
            if internal_name in physics_df.columns:
                physics_df.rename(columns={internal_name: f"{display_name}_physics"}, inplace=True)
    else:
        for var in output_vars:
            if var in physics_df.columns:
                physics_df.rename(columns={var: f"{var}_physics"}, inplace=True)

    return physics_df

# Compute (measured − simulated) residuals for each output variable.
def compute_residuals(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    output_vars: List[str]
) -> pd.DataFrame:
    residuals = {}
    for var in output_vars:
        sim_col = f"{var}_physics"
        if var in data.columns and sim_col in physics_df.columns:
            residuals[f"Residual {var}"] = data[var].values - physics_df[sim_col].values
    return pd.DataFrame(residuals)


def round_and_clean_floats(df: pd.DataFrame, decimal_places: int = 6) -> pd.DataFrame:
# Utility: round floats and remove tiny trailing decimals for readability.
    def clean_value(x):
        if isinstance(x, float):
            if abs(x - round(x)) < 10**-decimal_places:
                return round(x)
            return round(x, decimal_places)
        return x

    df = df.replace([np.inf, -np.inf], np.nan)  # Gets rid of infs and NaNs
    return df.map(clean_value)  # Map handles element-wise mapping 

# Look at what they have to do to mimic a fraction

def generate_simple_dataset(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    input_vars: List[str],
    output_vars: List[str]
) -> pd.DataFrame:
    
    # Combine input features and pure physics-based outputs into a simple dataset. 
    return pd.concat([
        data[input_vars].reset_index(drop=True),
        physics_df[[f"{var}_physics" for var in output_vars]].reset_index(drop=True)
    ], axis=1).rename(columns=dict(zip([f"{var}_physics" for var in output_vars], output_vars)))


def generate_residual_dataset(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    input_vars: List[str],
    output_vars: List[str],
    name_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    # Generate a residual dataset (input features + residuals).  
    input_cols = [col for col in data.columns if col.startswith("Input") or col in input_vars]

    # Map display names back to internal names
    mapped_vars = []
    for var in output_vars:
        internal_name = name_mapping.get(var, var) if name_mapping else var
        if internal_name in data.columns and f"{var}_physics" in physics_df.columns:
            mapped_vars.append((var, internal_name))

    # Compute residuals 
    residuals = {}
    for display_name, internal_name in mapped_vars:
        physics_col = f"{display_name}_physics"
        residuals[f"Residual {display_name}"] = data[internal_name].values - physics_df[physics_col].values

    residual_df = pd.DataFrame(residuals)
    return pd.concat([data[input_cols].reset_index(drop=True), residual_df.reset_index(drop=True)], axis=1)

