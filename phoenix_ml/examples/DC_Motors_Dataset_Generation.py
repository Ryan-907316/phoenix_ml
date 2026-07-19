"""
DC Motor Physics Dataset Generator
=====================================
Script Mode source file (Physics Modelling tab -> Script Mode): point the
"Physics Script" field at this file. The framework will call
governing_function using constants, input_vars, and output_vars to compute
physics estimates, build a residual dataset, and save the physics configuration.

Physics model: DC motor snapshot equations.
   Ia       = TL / Kt                             (armature current from load torque)
   Tm       = Kt * Ia                             (motor torque)
   omega    = (Va - Ia*Ra) / Ke                   (angular velocity)
   dw/dt    = (Tm - TL - B*omega) / J             (angular acceleration)
   dTm/dt   = Kt * d(TL)/dt                       (rate of change of motor torque)
   T_est    = 25 + 5*Ia                           (temperature, simplified thermal model)
   Fc       = B + 0.0001*sin(omega)               (friction coefficient)
"""

import pandas as pd
import numpy as np
from typing import Dict

# ── Script Mode API ────────────────────────────────────────────────────────────
# The four variables below are required when using this file in Script Mode.

# Columns from your dataset that are INPUTS to the physics equations
input_vars = ["Input Voltage", "Input Torque"]

# Columns from your dataset that are physical OUTPUTS (measured values)
# A residual is computed for each: Residual = Measured - Physics_Estimate
output_vars = [
    "Motor Speed",
    "Armature Current",
    "Motor Torque",
    "dw/dt",
    "dTm/dt",
    "Temperature",
    "Friction Coefficient",
]

# Physical constants for the DC motor
constants = {
    "Ra": 0.8,     # Armature resistance (Ohm)
    "La": 0.01,    # Armature inductance (H)
    "J":  0.01,    # Rotor moment of inertia (kg.m^2)
    "B":  0.001,   # Viscous friction coefficient (N.m.s)
    "Kt": 0.1,     # Torque constant (N.m/A)
    "Ke": 0.1,     # Back-EMF constant (V.s/rad)
}

# Set to a column name string if your dataset has a time column; None otherwise
time_col = None


def governing_function(
    inputs: pd.DataFrame,
    constants: Dict[str, float],
    time: np.ndarray,
) -> pd.DataFrame:
    """
    DC motor physics equations: computes estimates for all output variables.

    inputs    : DataFrame containing the columns listed in input_vars
                ("Input Voltage" and "Input Torque")
    constants : dict of physical constants
    time      : numpy array of time values (not used for snapshot equations)

    Returns a DataFrame with one column per entry in output_vars.
    Column names must match output_vars exactly.
    """
    Ra = constants["Ra"]
    J  = constants["J"]
    B  = constants["B"]
    Kt = constants["Kt"]
    Ke = constants["Ke"]

    Va = inputs["Input Voltage"].values
    TL = inputs["Input Torque"].values

    # Steady-state DC motor equations
    Ia    = TL / Kt
    Tm    = Kt * Ia
    omega = (Va - Ia * Ra) / Ke

    # Dynamic equations (snapshot-based)
    dw_dt  = (Tm - TL - B * omega) / J

    # Rate of change of load torque via finite difference; divide by timestep 1e-4 s
    dTL_dt = np.gradient(TL) / 1e-4
    dTm_dt = Kt * dTL_dt

    # Simplified thermal model and nonlinear friction
    temperature = 25.0 + 5.0 * Ia
    friction    = B + 0.0001 * np.sin(omega)

    return pd.DataFrame({
        "Motor Speed":          omega,
        "Armature Current":     Ia,
        "Motor Torque":         Tm,
        "dw/dt":                dw_dt,
        "dTm/dt":               dTm_dt,
        "Temperature":          temperature,
        "Friction Coefficient": friction,
    })
