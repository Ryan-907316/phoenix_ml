"""Tests for load_and_preprocess_data — input validation, split-method edge
cases, and the scaler lookup.

The scaler-"None" test is a regression test for a real bug found while writing
this file: the UI's Feature Scaling dropdown offers "None", but the scaler
lookup's .get(..., StandardScaler) default silently substituted StandardScaler
for it — a user opting out of scaling was scaled anyway, with the no-scaling
branch in the code unreachable.
"""
import numpy as np
import pandas as pd
import pytest

from phoenix_ml.data_preprocessing import load_and_preprocess_data, plot_features_vs_targets


def _write_csv(tmp_path, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _clean_df(n=10):
    return pd.DataFrame({
        "f1": np.arange(n, dtype=float),
        "f2": np.arange(n, dtype=float) * 2.0,
        "y":  np.arange(n, dtype=float) * 3.0,
    })


def test_scaler_none_returns_unscaled_features(tmp_path):
    path = _write_csv(tmp_path, _clean_df())
    df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, targets, feats = \
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["y"], scaler_type="None")
    assert scaler is None
    # Unscaled means the arrays are the raw feature values, not standardized.
    assert np.array_equal(X_train_scaled, X_train.values)
    assert np.array_equal(X_test_scaled, X_test.values)


def test_unknown_scaler_string_falls_back_to_standard(tmp_path):
    # Documents the current contract: an unrecognized scaler_type string (e.g.
    # a lowercase typo) silently falls back to StandardScaler rather than
    # raising. No UI path can produce one (dropdown-only), but API callers
    # should know this is lenient, not validating.
    path = _write_csv(tmp_path, _clean_df())
    *_, scaler, targets, feats = load_and_preprocess_data(
        path, test_size=0.2, split_method="last",
        target_columns=["y"], scaler_type="standard")  # wrong case
    assert type(scaler).__name__ == "StandardScaler"


def test_minmax_scaler_is_actually_used(tmp_path):
    path = _write_csv(tmp_path, _clean_df())
    *_, X_train_scaled, X_test_scaled, scaler, targets, feats = \
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["y"], scaler_type="MinMax")
    assert type(scaler).__name__ == "MinMaxScaler"
    assert X_train_scaled.min() >= 0.0 and X_train_scaled.max() <= 1.0


def test_test_size_zero_clamps_to_one_row_not_whole_frame(tmp_path):
    """Regression guard for the -0 slicing edge: with test_size=0 and
    split_method='last', ceil() gives 0 test rows and X.iloc[-0:] would return
    the ENTIRE frame as the test set (Python's -0 == 0), leaving an empty
    training set. The clamp must instead give a 1-row test set."""
    path = _write_csv(tmp_path, _clean_df(n=10))
    df, X_train, X_test, *_ = load_and_preprocess_data(
        path, test_size=0.0, split_method="last", target_columns=["y"])
    assert len(X_test) == 1
    assert len(X_train) == 9


def test_test_size_one_still_leaves_training_data(tmp_path):
    # test_size=1.0 must not consume every row as test data — the clamp keeps
    # at least one training row so downstream .fit() calls can't get an empty X.
    path = _write_csv(tmp_path, _clean_df(n=10))
    df, X_train, X_test, *_ = load_and_preprocess_data(
        path, test_size=1.0, split_method="first", target_columns=["y"])
    assert len(X_train) >= 1
    assert len(X_test) == 9


def test_first_and_last_split_take_opposite_ends(tmp_path):
    path = _write_csv(tmp_path, _clean_df(n=10))
    _, _, X_test_first, *_ = load_and_preprocess_data(
        path, test_size=0.2, split_method="first", target_columns=["y"])
    _, _, X_test_last, *_ = load_and_preprocess_data(
        path, test_size=0.2, split_method="last", target_columns=["y"])
    assert list(X_test_first["f1"]) == [0.0, 1.0]
    assert list(X_test_last["f1"]) == [8.0, 9.0]


def test_invalid_split_method_raises(tmp_path):
    path = _write_csv(tmp_path, _clean_df())
    with pytest.raises(ValueError, match="split_method"):
        load_and_preprocess_data(path, test_size=0.2, split_method="middle",
                                 target_columns=["y"])


def test_non_numeric_feature_column_raises_and_names_it(tmp_path):
    df = _clean_df()
    df["sensor_id"] = ["A"] * len(df)
    path = _write_csv(tmp_path, df)
    with pytest.raises(ValueError, match="sensor_id"):
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["y"])


def test_nan_column_raises_and_names_it(tmp_path):
    df = _clean_df()
    df.loc[3, "f2"] = np.nan
    path = _write_csv(tmp_path, df)
    with pytest.raises(ValueError, match="f2"):
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["y"])


def test_inf_column_raises_and_names_it(tmp_path):
    df = _clean_df()
    df.loc[5, "f1"] = np.inf
    path = _write_csv(tmp_path, df)
    with pytest.raises(ValueError, match="f1"):
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["y"])


def test_missing_target_column_raises_and_names_it(tmp_path):
    path = _write_csv(tmp_path, _clean_df())
    with pytest.raises(ValueError, match="Not A Column"):
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["Not A Column"])


def test_non_numeric_target_column_raises_and_names_it(tmp_path):
    """Regression test for a real bug: the non-numeric check only scanned
    feature columns, never targets — a text/categorical target sailed past
    the exact check meant to catch it, even though the NaN/Inf checks right
    next to it already correctly covered targets."""
    df = _clean_df()
    df["y"] = ["a"] * len(df)  # non-numeric target
    path = _write_csv(tmp_path, df)
    with pytest.raises(ValueError, match="y"):
        load_and_preprocess_data(path, test_size=0.2, split_method="last",
                                 target_columns=["y"])


def test_test_size_out_of_range_raises_for_first_and_last_split(tmp_path):
    """Regression test for a real bug: "random" split delegates range checking
    to sklearn (which rejects e.g. test_size=50), but "first"/"last" used to
    silently clamp ANY value into an almost-empty split with no error at all —
    a user typing 50 meaning "50%" got a 1-row training set, silently."""
    path = _write_csv(tmp_path, _clean_df())
    for method in ("first", "last"):
        with pytest.raises(ValueError, match="test_size"):
            load_and_preprocess_data(
                path, test_size=50, split_method=method, target_columns=["y"])
        with pytest.raises(ValueError, match="test_size"):
            load_and_preprocess_data(
                path, test_size=-1, split_method=method, target_columns=["y"])


def test_test_size_boundary_values_still_clamp_not_raise(tmp_path):
    # 0 and 1 are valid inputs (handled by the existing clamp, tested
    # elsewhere in this file) — the new range check must not reject them.
    path = _write_csv(tmp_path, _clean_df(n=10))
    df, X_train, X_test, *_ = load_and_preprocess_data(
        path, test_size=0.0, split_method="last", target_columns=["y"])
    assert len(X_test) == 1
    df, X_train, X_test, *_ = load_and_preprocess_data(
        path, test_size=1.0, split_method="first", target_columns=["y"])
    assert len(X_train) >= 1


def test_plot_features_vs_targets_returns_empty_instead_of_crashing_with_zero_features():
    """Regression test for a real bug: with zero feature columns (reachable by
    default — target_columns defaults to the last two columns, so any
    2-column dataset run with default settings hits this), num_cols became 0
    and range(0, 0, 0) raised ValueError deep inside the function."""
    X_train = pd.DataFrame(index=range(5))  # zero columns
    y_train = pd.DataFrame({"y": range(5)})
    figs = plot_features_vs_targets(X_train, y_train, ["y"])
    assert figs == {}


def test_plot_distance_correlation_matrix_does_not_overwrite_a_real_dummy_column(capsys):
    """Regression test for a real risk: the synthetic noise-baseline column
    was unconditionally named "Dummy" and assigned via a plain df["Dummy"] =
    ..., silently overwriting a real feature that happened to be named
    "Dummy" — corrupting the correlation matrix and quietly poisoning the
    noise-baseline flagging in compute_feature_selection_recommendations."""
    import dcor
    from phoenix_ml.data_preprocessing import plot_distance_correlation_matrix

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.uniform(0, 1, 30),
        "Dummy": rng.uniform(100, 200, 30),  # a REAL feature, not noise
    })
    original_dummy = df["Dummy"].copy()

    dist_corr_df, fig = plot_distance_correlation_matrix(df, dummy=True)

    assert "[WARN]" in capsys.readouterr().out
    # No synthetic column was appended — still just the original 2 columns.
    assert list(dist_corr_df.columns) == ["a", "Dummy"]
    # The correlation actually used the REAL "Dummy" values, not fresh noise.
    expected = dcor.distance_correlation(df["a"].values, original_dummy.values)
    assert dist_corr_df.loc["a", "Dummy"] == pytest.approx(expected)


def test_mp_denoise_all_noise_falls_back_to_one_component():
    """Edge case: when every eigenvalue falls below the Marchenko-Pastur noise
    threshold (a small n_samples relative to feature count pushes lambda_plus
    above any eigenvalue a valid correlation-like matrix can have),
    _mp_denoise must not return a zero-signal, all-blank matrix — it forces
    the single largest eigen-component to survive instead, so the resulting
    heatmap has something to show rather than being entirely blank."""
    from phoenix_ml.data_preprocessing import _mp_denoise

    matrix = np.array([
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ])
    denoised, n_signal, lambda_plus = _mp_denoise(matrix, n_samples=1)
    assert n_signal == 1
    assert not np.allclose(denoised, 0.0)
    assert denoised.shape == matrix.shape
    assert np.all(denoised >= 0.0) and np.all(denoised <= 1.0)


def test_mp_denoise_keeps_multiple_signal_components_when_present():
    # Contrast case: a large n_samples relative to feature count keeps
    # lambda_plus small, so genuine signal eigenvalues survive normally —
    # locks in that the all-noise fallback doesn't fire when it shouldn't.
    from phoenix_ml.data_preprocessing import _mp_denoise

    matrix = np.array([
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ])
    denoised, n_signal, lambda_plus = _mp_denoise(matrix, n_samples=10_000)
    assert n_signal >= 1


def test_run_preprocessing_workflow_clears_a_caller_supplied_figures_dict(tmp_path):
    """Regression test for a real risk: a caller-supplied figures dict was
    only ever added to, never cleared — stale figures from a previous,
    differently-configured run could leak into a new run's output."""
    from phoenix_ml.data_preprocessing import run_preprocessing_workflow

    path = _write_csv(tmp_path, _clean_df(n=10))
    stale_figures = {"stale_key": "stale_value"}
    result = run_preprocessing_workflow(
        path, test_size=0.2, split_method="last", target_columns=["y"],
        plot_target_vs_target_enabled=False, plot_features_vs_targets_enabled=False,
        plot_boxplots_enabled=False, plot_distance_corr_enabled=False,
        show_multicollinearity=False, plot_pca_enabled=False, feat_sel_enabled=False,
        figures=stale_figures,
    )
    assert "stale_key" not in result["figures"]
    assert stale_figures == {}  # the caller's dict object itself was cleared in place


# ── compute_feature_selection_recommendations ────────────────────────────────

def _redundancy_inputs(feature_order):
    """Two features forming a redundant pair (hand-written inter-feature dcor
    0.95 > threshold 0.90). 'strong' is an exact copy of the target (max
    relevance 1.0); 'weak' is uncorrelated noise (low relevance)."""
    from phoenix_ml.data_preprocessing import compute_feature_selection_recommendations

    rng = np.random.default_rng(0)
    n = 40
    target = rng.uniform(0, 10, n)
    cols = {"strong": target.copy(), "weak": rng.uniform(0, 10, n)}
    X_train = pd.DataFrame({f: cols[f] for f in feature_order})
    y_train = pd.DataFrame({"T": target})

    dist_corr_df = pd.DataFrame(
        [[1.0, 0.95], [0.95, 1.0]], index=feature_order, columns=feature_order)

    return compute_feature_selection_recommendations(
        dist_corr_df, X_train, y_train,
        feature_names=feature_order, target_columns=["T"],
    )


def test_redundant_pair_drops_the_lower_relevance_feature_regardless_of_order():
    # Whichever order the redundant pair is listed in, the drop recommendation
    # must land on the feature with lower target relevance — the rule is
    # relevance-based, not position-based.
    assert _redundancy_inputs(["strong", "weak"])["recommended_drop"] == ["weak"]
    assert _redundancy_inputs(["weak", "strong"])["recommended_drop"] == ["weak"]


def test_multicollinearity_reports_rank_and_detects_exact_dependence():
    """compute_multicollinearity returns the scaled feature matrix's rank
    alongside the condition number: rank < n_features is a binary,
    self-explanatory flag for EXACT linear dependence (e.g. a derived column
    that is precisely the sum of two others), which a continuous condition
    number only implies via thresholds."""
    from phoenix_ml.data_preprocessing import compute_multicollinearity

    rng = np.random.default_rng(0)
    n = 40
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    X_ok = pd.DataFrame({"a": a, "b": b})
    _, cond_ok, rank_ok = compute_multicollinearity(X_ok, X_ok.values, ["a", "b"])
    assert rank_ok == 2

    # c = a + b exactly -> rank-deficient by construction.
    X_bad = pd.DataFrame({"a": a, "b": b, "c": a + b})
    _, cond_bad, rank_bad = compute_multicollinearity(X_bad, X_bad.values, ["a", "b", "c"])
    assert rank_bad == 2   # 3 features, only 2 independent dimensions


def test_redundancy_tie_breaks_deterministically_to_keep_the_earlier_feature():
    """With EXACTLY equal max relevance (one feature is a bitwise copy of the
    other, so their dcor against every target is identical), either drop
    choice is defensible — the contract locked here is that the tie-break is
    deterministic and keeps the earlier-listed feature, so repeated runs of
    the same config never flip their recommendation."""
    from phoenix_ml.data_preprocessing import compute_feature_selection_recommendations

    rng = np.random.default_rng(1)
    n = 40
    base = rng.uniform(0, 10, n)
    X_train = pd.DataFrame({"first": base.copy(), "second": base.copy()})
    y_train = pd.DataFrame({"T": base + rng.normal(0, 1, n)})
    dist_corr_df = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], index=["first", "second"], columns=["first", "second"])

    result = compute_feature_selection_recommendations(
        dist_corr_df, X_train, y_train,
        feature_names=["first", "second"], target_columns=["T"],
    )
    assert result["recommended_drop"] == ["second"]
