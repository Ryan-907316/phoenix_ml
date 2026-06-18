# physics_model.py
# Helper module for running physics-based models and generating residual datasets.
# Script Mode physics scripts must export: governing_function, constants, input_vars, output_vars
# See examples/DC_Motors_Dataset_Generation.py for a worked example.

import importlib.util
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional


def import_physics_script(path: str):
    """Dynamically import a user-written physics script and return the module.

    The script must define:
        governing_function(inputs, constants, time) -> pd.DataFrame
        constants  = {...}
        input_vars  = [...]
        output_vars = [...]
    Optional:
        name_mapping = {...}  # if the script uses internal variable names
        time_col     = None   # or a column name string
    """
    spec = importlib.util.spec_from_file_location("_user_physics_script", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load physics script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_physics_model(
    data: pd.DataFrame,
    time_col: Optional[str],
    governing_function: Callable[[pd.DataFrame, Dict[str, float], np.ndarray], pd.DataFrame],
    constants: Dict[str, float],
    input_vars: List[str],
    output_vars: List[str],
    name_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Run a physics model and return a DataFrame with columns named '<var>_physics'."""
    time = data[time_col].values if time_col is not None else np.zeros(len(data))
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


def compute_residuals(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    output_vars: List[str]
) -> pd.DataFrame:
    """Compute (measured − simulated) residuals for each output variable."""
    residuals = {}
    for var in output_vars:
        sim_col = f"{var}_physics"
        if var in data.columns and sim_col in physics_df.columns:
            residuals[f"Residual {var}"] = data[var].values - physics_df[sim_col].values
    return pd.DataFrame(residuals)


def round_and_clean_floats(df: pd.DataFrame, decimal_places: int = 6) -> pd.DataFrame:
    """Round floats and remove tiny trailing decimals for readability."""
    def clean_value(x):
        if isinstance(x, float):
            if abs(x - round(x)) < 10**-decimal_places:
                return round(x)
            return round(x, decimal_places)
        return x
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.map(clean_value)


def generate_simple_dataset(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    input_vars: List[str],
    output_vars: List[str]
) -> pd.DataFrame:
    """Combine input features and pure physics-based outputs into a simple dataset."""
    return pd.concat([
        data[input_vars].reset_index(drop=True),
        physics_df[[f"{var}_physics" for var in output_vars]].reset_index(drop=True)
    ], axis=1).rename(columns={f"{var}_physics": var for var in output_vars})


def generate_residual_dataset(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    input_vars: List[str],
    output_vars: List[str],
    name_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Generate a residual dataset: input features + (measured − physics) for each output."""
    # Only keep columns listed in input_vars
    input_cols = [col for col in data.columns if col in input_vars]

    # Map display names to internal names for residual computation
    mapped_vars = []
    for var in output_vars:
        internal_name = name_mapping.get(var, var) if name_mapping else var
        if internal_name in data.columns and f"{var}_physics" in physics_df.columns:
            mapped_vars.append((var, internal_name))

    residuals = {}
    for display_name, internal_name in mapped_vars:
        physics_col = f"{display_name}_physics"
        residuals[f"Residual {display_name}"] = (
            data[internal_name].values - physics_df[physics_col].values
        )

    residual_df = pd.DataFrame(residuals)
    return pd.concat(
        [data[input_cols].reset_index(drop=True), residual_df.reset_index(drop=True)],
        axis=1
    )
