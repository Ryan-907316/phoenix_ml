"""Pathological-but-plausible dataset shapes that real users will feed this
tool sooner or later. Each test represents a way a first run on somebody
else's data could crash or mislead — the goal is that phoenix_ml either
handles the input or fails with a clear, named error, never a deep traceback.

The slash-in-target-name test is a regression test for a real bug found while
building this battery: model saving replaced spaces but not slashes in
filenames, so a target named "Residual dw/dt" (the naming style of the
bundled DC-motor dataset itself) crashed with FileNotFoundError on Windows,
where '/' is a path separator.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from phoenix_ml.data_preprocessing import load_and_preprocess_data
from phoenix_ml.model_training import metrics_dict, run_model_training_workflow
from phoenix_ml.persistence import load_pipeline, save_models_and_artifacts


def _write_csv(tmp_path, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _train_on(df, tmp_path, targets, models=("KNeighbors Regressor",)):
    path = _write_csv(tmp_path, df)
    (df_out, X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler, target_cols, feature_names) = \
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=list(targets))
    sel_metrics = {k: metrics_dict[k] for k in ["MSE", "NRMSE", "Q^2", "KGE"]}
    results_df, *_ = run_model_training_workflow(
        X_train=X_train_scaled, X_test=X_test_scaled,
        y_train=y_train, y_test=y_test, target_columns=target_cols,
        selected_model_names=list(models), selected_metrics=sel_metrics,
        feature_names=feature_names, verbose=False, random_state=0,
    )
    return results_df


def test_constant_target_trains_and_reports_nan_metrics_without_crashing(tmp_path):
    # A stuck/constant target column: range-normalised and variance-based
    # metrics are undefined (NaN is the honest answer), but training itself
    # must complete rather than dying on a divide-by-zero.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "f1": rng.uniform(0, 1, 30),
        "f2": rng.uniform(0, 1, 30),
        "y": np.full(30, 7.0),
    })
    with np.errstate(all="ignore"):
        results_df = _train_on(df, tmp_path, targets=["y"])
    row = results_df.iloc[0]
    assert np.isnan(row["NRMSE"])
    assert np.isnan(row["KGE"])


def test_ten_row_single_feature_dataset_survives_preprocess_and_train(tmp_path):
    # The smallest dataset anyone could plausibly try: 10 rows, one feature.
    df = pd.DataFrame({
        "only_feature": np.arange(10, dtype=float),
        "y": np.arange(10, dtype=float) * 2.0 + 1.0,
    })
    results_df = _train_on(df, tmp_path, targets=["y"])
    assert len(results_df) == 1
    assert np.isfinite(results_df.iloc[0]["MSE"])


def test_slash_in_target_name_still_saves_loadable_pipelines(tmp_path):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pipe = Pipeline([("model", LinearRegression())]).fit(X, X["a"])

    save_paths = save_models_and_artifacts(
        output_dir=str(tmp_path), pipelines_by_target={"Residual dw/dt": pipe},
        feature_names=["a"], targets=["Residual dw/dt"], metric_name="Q^2",
        dataset_path="x.csv", make_bundle=True, prefix="t",
    )
    path = save_paths["by_target"]["Residual dw/dt"]
    # The filename part (after the last separator) must carry no slash — the
    # target's "/" is sanitized to "_" ("t Pipeline_Residual dw_dt.pkl").
    assert "/" not in os.path.basename(path) and "dw_dt" in os.path.basename(path)
    reloaded = load_pipeline(path)
    assert np.allclose(reloaded.predict(X), pipe.predict(X))


def test_non_ascii_and_symbol_column_names_survive_preprocess_and_train(tmp_path):
    # Unicode units and symbol-heavy names are everyday engineering column
    # headers, not an edge case.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Temperatur (°C)": rng.uniform(20, 90, 30),
        "θ / rad": rng.uniform(0, 3.14, 30),
        "Δp [bar]": rng.uniform(0, 1, 30),
        "y": rng.uniform(0, 10, 30),
    })
    results_df = _train_on(df, tmp_path, targets=["y"])
    assert len(results_df) == 1
    assert np.isfinite(results_df.iloc[0]["MSE"])
