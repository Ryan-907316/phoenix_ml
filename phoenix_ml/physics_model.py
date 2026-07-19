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


def _missing_columns(columns: List[str], available) -> List[str]:
    return [c for c in columns if c not in available]


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
    # A config reused on a renamed/different dataset used to index straight into
    # missing columns, raising a raw KeyError instead of naming what's missing.
    missing_inputs = _missing_columns(input_vars, data.columns)
    if missing_inputs:
        raise ValueError(f"Missing input column(s): {missing_inputs}.")
    if time_col is not None and time_col not in data.columns:
        raise ValueError(f"Missing time column: '{time_col}'.")
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
    missing_measured = _missing_columns(output_vars, data.columns)
    if missing_measured:
        raise ValueError(
            f"Missing measured column(s) for residual computation: {missing_measured}. "
            f"These must exist in the dataset with the exact output variable name."
        )
    physics_cols = [f"{var}_physics" for var in output_vars]
    missing_physics = _missing_columns(physics_cols, physics_df.columns)
    if missing_physics:
        raise ValueError(
            f"Missing physics-model output column(s): {missing_physics}. "
            f"The governing function did not produce these — check output_vars/name_mapping."
        )
    residuals = {}
    for var in output_vars:
        residuals[f"Residual {var}"] = data[var].values - physics_df[f"{var}_physics"].values
    return pd.DataFrame(residuals)


def round_and_clean_floats(df: pd.DataFrame, decimal_places: int = 6) -> pd.DataFrame:
    """Round floats and remove tiny trailing decimals for readability.

    Vectorised per float column instead of a Python-level element-wise
    df.map() call — the previous version was an O(rows x cols) Python
    function-call loop, a slow cliff on high-sample-rate data. This also
    fixes a latent crash: the old per-cell round(x) (no ndigits) raises
    ValueError on NaN, which is a live input here since inf/-inf were just
    replaced with NaN two lines above — np.round handles NaN as a no-op.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    result = df.copy()
    eps = 10 ** -decimal_places
    for col in result.select_dtypes(include="float").columns:
        vals = result[col].to_numpy(dtype=float)
        nearest_int = np.round(vals)
        is_near_int = np.abs(vals - nearest_int) < eps
        result[col] = np.where(is_near_int, nearest_int, np.round(vals, decimal_places))
    return result


def generate_simple_dataset(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    input_vars: List[str],
    output_vars: List[str]
) -> pd.DataFrame:
    """Combine input features and pure physics-based outputs into a simple dataset."""
    missing_inputs = _missing_columns(input_vars, data.columns)
    if missing_inputs:
        raise ValueError(f"Missing input column(s): {missing_inputs}.")
    physics_cols = [f"{var}_physics" for var in output_vars]
    missing_physics = _missing_columns(physics_cols, physics_df.columns)
    if missing_physics:
        raise ValueError(
            f"Missing physics-model output column(s): {missing_physics}. "
            f"The governing function did not produce these — check output_vars/name_mapping."
        )
    return pd.concat([
        data[input_vars].reset_index(drop=True),
        physics_df[physics_cols].reset_index(drop=True)
    ], axis=1).rename(columns={f"{var}_physics": var for var in output_vars})


def generate_residual_dataset(
    data: pd.DataFrame,
    physics_df: pd.DataFrame,
    input_vars: List[str],
    output_vars: List[str],
    name_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Generate a residual dataset: input features + (measured − physics) for each output."""
    missing_inputs = _missing_columns(input_vars, data.columns)
    if missing_inputs:
        raise ValueError(f"Missing input column(s): {missing_inputs}.")

    # Map display names to internal names for residual computation
    mapped_vars = []
    missing_measured = []
    missing_physics = []
    for var in output_vars:
        internal_name = name_mapping.get(var, var) if name_mapping else var
        physics_col = f"{var}_physics"
        if internal_name not in data.columns:
            missing_measured.append(internal_name)
        elif physics_col not in physics_df.columns:
            missing_physics.append(physics_col)
        else:
            mapped_vars.append((var, internal_name))
    if missing_measured:
        raise ValueError(
            f"Missing measured column(s) for residual computation: {missing_measured}. "
            f"These must exist in the dataset (see name_mapping if the script uses "
            f"internal variable names)."
        )
    if missing_physics:
        raise ValueError(
            f"Missing physics-model output column(s): {missing_physics}. "
            f"The governing function did not produce these — check output_vars/name_mapping."
        )

    residuals = {}
    for display_name, internal_name in mapped_vars:
        physics_col = f"{display_name}_physics"
        residuals[f"Residual {display_name}"] = (
            data[internal_name].values - physics_df[physics_col].values
        )

    residual_df = pd.DataFrame(residuals)
    return pd.concat(
        [data[input_vars].reset_index(drop=True), residual_df.reset_index(drop=True)],
        axis=1
    )
