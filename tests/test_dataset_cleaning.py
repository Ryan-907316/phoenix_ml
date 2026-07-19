"""Tests for dataset_cleaning.py — stuck-value index remapping, column
auto-classification branching, and multivariate outlier handling.

The stuck-value tests use hand-computed inputs small enough to verify by eye:
detect_stuck_values() reports run positions in dropna()-space, and
apply_cleaning() must map those back to original row labels via the non-NaN
index before touching anything — a positional (iloc-style) mapping would
remove/interpolate the wrong rows whenever NaNs sit before or inside a run.
"""
import numpy as np
import pandas as pd

from phoenix_ml.dataset_cleaning import (
    ACTION_CAP,
    ACTION_INTERP,
    ACTION_NONE,
    ACTION_REMOVE,
    OUTLIER_IQR,
    OUTLIER_ISOLATION_FOREST,
    OUTLIER_LOF,
    OUTLIER_NONE,
    ROLE_EXCLUDE,
    ROLE_INPUT,
    ROLE_TIMESTAMP,
    TYPE_DATETIME,
    apply_cleaning,
    auto_classify_columns,
    coerce_numeric,
    detect_column_type,
    detect_outliers_multivariate,
    detect_stuck_values,
)


def test_detect_stuck_values_reports_runs_in_dropna_space():
    # Original: [1, NaN, 5, 5, NaN, 5, 5, 5, 9] -> dropna: [1, 5, 5, 5, 5, 5, 9]
    # The five 5s sit at dropna positions 1..5 even though NaNs interrupt them
    # in the original series.
    series = pd.Series([1.0, np.nan, 5.0, 5.0, np.nan, 5.0, 5.0, 5.0, 9.0])
    stuck = detect_stuck_values(series, min_run=5)
    assert stuck["n_runs"] == 1
    assert stuck["runs"] == [(1, 5, 5.0)]
    assert stuck["n_affected_rows"] == 5


def test_stuck_removal_maps_dropna_positions_back_to_original_rows():
    """With NaNs interspersed inside a stuck run, removal must drop exactly the
    rows holding the stuck value — not the rows at the same *positional*
    offsets. A positional bug here would keep stuck rows and delete good ones."""
    df = pd.DataFrame({
        "v": [1.0, np.nan, 5.0, 5.0, np.nan, 5.0, 5.0, 5.0, 9.0, np.nan, 2.0],
        "row_id": np.arange(11.0),  # survives cleaning; tracks original rows
    })
    col_info = {
        "v": {"role": ROLE_INPUT, "reason": ""},
        "row_id": {"role": ROLE_INPUT, "reason": ""},
    }
    result, log = apply_cleaning(
        df, col_info,
        missing_action=ACTION_NONE,
        outlier_method=OUTLIER_NONE, outlier_action=ACTION_NONE,
        stuck_enabled=True, stuck_min_run=5, stuck_action=ACTION_REMOVE,
        drop_excluded=False,
    )
    # Rows 2, 3, 5, 6, 7 held the stuck 5.0s; everything else survives,
    # including the NaN rows (missing_action is None).
    assert list(result["row_id"]) == [0.0, 1.0, 4.0, 8.0, 9.0, 10.0]


def test_stuck_interpolation_keeps_first_run_value_as_anchor():
    # [1, 5, 5, 5, 5, 5, 9]: interpolation NaNs the run EXCEPT its first member
    # (the anchor), then draws a line from that anchor (position 1, value 5) to
    # the next real value (position 6, value 9): step (9-5)/5 = 0.8.
    df = pd.DataFrame({"v": [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0]})
    col_info = {"v": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info,
        missing_action=ACTION_NONE,
        outlier_method=OUTLIER_NONE, outlier_action=ACTION_NONE,
        stuck_enabled=True, stuck_min_run=5, stuck_action=ACTION_INTERP,
        drop_excluded=False,
    )
    assert np.allclose(result["v"], [1.0, 5.0, 5.8, 6.6, 7.4, 8.2, 9.0])


def test_auto_classify_constant_step_resolves_to_two_roles_plus_a_review_flag():
    """The same structural signal (strictly increasing, constant step) must
    resolve by context: a time-named column is a Timestamp; every other
    constant-step column — INCLUDING a 0/1-based step-1 counter — stays an
    Input with a review note, never auto-Exclude. A step-1-from-0/1 column is
    often a record ID, but can just as easily be a genuine feature that
    coincidentally starts there (e.g. "Stage Number") — auto-excluding it by
    default was silent data loss on a coincidence (drop_excluded=True is the
    default), not the "slam dunk" the rest of this function's philosophy
    requires. Regression test for that real bug: the counter case used to
    default to Exclude while every other constant-step case defaulted to
    Input+review, an unjustified asymmetry."""
    n = 12
    df = pd.DataFrame({
        "time_s": np.arange(n) * 60.0,          # 0, 60, 120, ... — time axis
        "record_id": np.arange(n, dtype=float),  # 0, 1, 2, ... — counter-shaped
        "sweep_level": 3.0 + 7.0 * np.arange(n),  # 3, 10, 17, ... — neither
    })
    info = auto_classify_columns(df)

    assert info["time_s"]["suggested_role"] == ROLE_TIMESTAMP
    assert info["record_id"]["suggested_role"] == ROLE_INPUT
    assert info["sweep_level"]["suggested_role"] == ROLE_INPUT
    # Both non-timestamp cases must carry a visible review prompt, not pass
    # silently — but with different wording, since the counter case is still
    # worth calling out as LIKELY (not certainly) a record ID.
    assert any("review role" in issue for issue in info["sweep_level"]["issues"])
    assert any("record ID" in issue and "review role" in issue
              for issue in info["record_id"]["issues"])


def test_auto_classify_flags_zero_variance_as_exclude():
    df = pd.DataFrame({"const": [7.0] * 12, "ok": np.random.default_rng(0).normal(size=12)})
    info = auto_classify_columns(df)
    assert info["const"]["suggested_role"] == ROLE_EXCLUDE
    assert info["ok"]["suggested_role"] == ROLE_INPUT


def _blob_with_outlier(n=19, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
    })
    df.loc[len(df)] = [100.0, 100.0]  # one gross outlier
    return df


def test_multivariate_outliers_flag_gross_outlier_and_never_flag_nan_rows():
    df = _blob_with_outlier()
    outlier_idx = df.index[-1]
    # Two rows with a missing value: excluded from fitting, must come back
    # unflagged (missing-value handling deals with them, not outlier removal).
    df.loc[3, "x"] = np.nan
    df.loc[7, "y"] = np.nan

    mask = detect_outliers_multivariate(
        df, OUTLIER_ISOLATION_FOREST, contamination=0.05, random_state=0)

    assert bool(mask.loc[outlier_idx]) is True
    assert bool(mask.loc[3]) is False
    assert bool(mask.loc[7]) is False


def test_multivariate_outliers_return_all_false_below_ten_complete_rows():
    # Documented guard: with fewer than 10 complete rows there isn't enough
    # data to fit an outlier model — must return an all-False mask, not raise.
    df = _blob_with_outlier(n=8)
    mask = detect_outliers_multivariate(
        df, OUTLIER_ISOLATION_FOREST, contamination=0.1, random_state=0)
    assert not mask.any()


def test_coerce_numeric_at_exact_90_percent_boundary_coerces():
    # 10 non-null values, exactly 1 unparseable -> parse fraction is exactly
    # 0.90, and the threshold comparison is inclusive (>=), so the column
    # must still be coerced (the stray entry becomes NaN and is counted).
    series = pd.Series([str(v) for v in range(9)] + ["error"], dtype=object)
    coerced, n_bad = coerce_numeric(series)
    assert coerced is not None
    assert n_bad == 1
    assert list(coerced[:9]) == [float(v) for v in range(9)]
    assert np.isnan(coerced.iloc[9])


def test_coerce_numeric_below_threshold_leaves_column_untouched():
    # 8 of 10 parseable (0.80 < 0.90): the column is NOT mostly numeric, so
    # coercion must decline entirely rather than NaN-ing 20% of the data.
    series = pd.Series([str(v) for v in range(8)] + ["error", "----"], dtype=object)
    coerced, n_bad = coerce_numeric(series)
    assert coerced is None
    assert n_bad == 0


def test_timestamp_column_converts_to_elapsed_seconds():
    df = pd.DataFrame({
        "stamp": ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00"],
        "v": [1.0, 2.0, 3.0],
    })
    col_info = {
        "stamp": {"role": ROLE_TIMESTAMP, "reason": ""},
        "v": {"role": ROLE_INPUT, "reason": ""},
    }
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_NONE, outlier_action=ACTION_NONE,
        stuck_enabled=False,
    )
    assert list(result["stamp"]) == [0.0, 60.0, 120.0]
    assert any(entry.startswith("[CONV] 'stamp'") for entry in log)


def test_failed_timestamp_conversion_warns_and_excludes_the_column():
    """A Timestamp-role column that can't be parsed must (a) leave a [WARN]
    log entry naming the column so the failure is visible to the user, (b) be
    excluded rather than passed through as un-parseable text, and (c) not
    abort the rest of the cleaning run — the other columns still come out."""
    df = pd.DataFrame({
        "stamp": ["not a date", "also not", "nope"],
        "v": [1.0, 2.0, 3.0],
    })
    col_info = {
        "stamp": {"role": ROLE_TIMESTAMP, "reason": ""},
        "v": {"role": ROLE_INPUT, "reason": ""},
    }
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_NONE, outlier_action=ACTION_NONE,
        stuck_enabled=False, drop_excluded=True,
    )
    assert any(entry.startswith("[WARN] 'stamp'") for entry in log)
    assert "stamp" not in result.columns
    assert list(result["v"]) == [1.0, 2.0, 3.0]


def test_apply_cleaning_does_not_mutate_the_callers_col_info():
    """Regression test for a real bug: apply_cleaning used to reassign a failed
    timestamp column's role to Exclude directly on the caller's col_info dict.
    The UI reuses the same col_info object across repeated Apply calls and
    across datasets in a session, so that mutation would silently carry a
    stale Exclude forward into a later, unrelated run. The exclusion must
    still happen (checked above); only the CALLER's own dict must be
    unaffected by it."""
    df = pd.DataFrame({
        "stamp": ["not a date", "also not", "nope"],
        "v": [1.0, 2.0, 3.0],
    })
    col_info = {
        "stamp": {"role": ROLE_TIMESTAMP, "reason": ""},
        "v": {"role": ROLE_INPUT, "reason": ""},
    }
    apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_NONE, outlier_action=ACTION_NONE,
        stuck_enabled=False, drop_excluded=True,
    )
    assert col_info["stamp"]["role"] == ROLE_TIMESTAMP


def test_multivariate_outliers_are_reproducible_even_without_a_seed():
    # random_state=None resolves to seed 0 (package-wide convention), so
    # IsolationForest flags the same rows on every run by default.
    df = _blob_with_outlier()
    mask_a = detect_outliers_multivariate(df, OUTLIER_ISOLATION_FOREST, contamination=0.1)
    mask_b = detect_outliers_multivariate(df, OUTLIER_ISOLATION_FOREST, contamination=0.1)
    assert mask_a.equals(mask_b)


def test_multivariate_outliers_clamps_out_of_range_contamination_instead_of_crashing():
    """Regression test for a real bug: detect_outliers_multivariate() passes
    contamination straight through to sklearn, which requires it in (0, 0.5].
    outlier_threshold is shared with IQR (~1.5) and Percentage (~95.0), so a
    value left over from switching methods used to crash with a raw sklearn
    InvalidParameterError instead of this codebase's usual friendly handling."""
    df = _blob_with_outlier()
    # 1.5 (a plausible IQR threshold) and 95.0 (a plausible Percentage
    # threshold) must both be silently clamped into sklearn's valid range,
    # not raise.
    mask_high = detect_outliers_multivariate(df, OUTLIER_ISOLATION_FOREST, contamination=1.5)
    mask_pct = detect_outliers_multivariate(df, OUTLIER_ISOLATION_FOREST, contamination=95.0)
    assert mask_high.dtype == bool and mask_pct.dtype == bool


def test_apply_cleaning_clamps_contamination_and_logs_a_warning():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 2)), columns=["a", "b"])
    col_info = {"a": {"role": ROLE_INPUT, "reason": ""}, "b": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_ISOLATION_FOREST, outlier_threshold=1.5,
        outlier_action=ACTION_REMOVE, stuck_enabled=False,
    )
    assert any("out of range" in entry and "1.5" in entry for entry in log)


def test_detect_column_type_is_not_fooled_by_a_leading_placeholder_run():
    """Regression test for a real risk: detect_column_type sampled only the
    first 20 non-null values — a sorted or placeholder-headed column (real
    dates preceded by a run of "N/A"-style placeholders) had its actual type
    invisible to a head-only sample."""
    # A head(20) sample would be entirely placeholders (pd.to_datetime has no
    # per-element tolerance, so even one bad value fails the whole sample) —
    # a random sample from all 1000 rows is overwhelmingly likely to avoid
    # the 5 sparse placeholders entirely and see the column's real type.
    placeholders = ["N/A"] * 5
    dates = pd.date_range("2020-01-01", periods=995).strftime("%Y-%m-%d").tolist()
    series = pd.Series(placeholders + dates, dtype=object)
    assert detect_column_type(series) == TYPE_DATETIME


def test_apply_cleaning_warns_when_missing_action_none_leaves_nans_behind():
    """Regression test for a real risk: text->numeric coercion (step 1b) can
    introduce brand-new NaNs, and with missing_action='None' nothing logged
    their existence — the user's first sign anything was wrong used to be a
    less specific rejection downstream (load_and_preprocess_data's NaN check)."""
    # coerce_numeric only converts when >= 90% of values parse — 19 good
    # values + 1 bad one clears that bar (95%) while still leaving exactly
    # one unparseable value to become NaN.
    df = pd.DataFrame({"a": [str(float(i)) for i in range(19)] + ["error"]})
    col_info = {"a": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_NONE, outlier_action=ACTION_NONE,
        stuck_enabled=False,
    )
    assert result["a"].isna().sum() == 1
    assert any("[WARN]" in entry and "Missing values" in entry for entry in log)


def test_lof_outlier_detection_is_deterministic():
    """LocalOutlierFactor takes no random_state because the algorithm is
    deterministic (pure kNN distances) — this locks that in, so if sklearn's
    implementation ever grows a stochastic component the suite notices the
    reproducibility contract breaking."""
    df = _blob_with_outlier()
    mask_a = detect_outliers_multivariate(df, OUTLIER_LOF, contamination=0.1)
    mask_b = detect_outliers_multivariate(df, OUTLIER_LOF, contamination=0.1)
    assert mask_a.equals(mask_b)


# ── apply_cleaning outlier-action combinatorics: cap/interp x per-column/multivariate ──
#
# Every existing apply_cleaning test above uses outlier_action=ACTION_NONE or
# ACTION_REMOVE — Cap and Interpolate were untested for both the per-column
# (IQR/Z-Score/Percentage) and multivariate (Isolation Forest/LOF/MCD) methods.

def _col_with_outlier():
    return pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})


def test_per_column_iqr_cap_clips_the_outlier_in_place_not_removing_the_row():
    df = _col_with_outlier()
    col_info = {"v": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_IQR, outlier_threshold=1.5,
        outlier_action=ACTION_CAP, stuck_enabled=False,
    )
    assert len(result) == len(df)          # no row removed
    assert result["v"].iloc[-1] < 100.0    # clipped down to the IQR upper bound
    assert any("[FIX ]" in entry for entry in log)


def test_per_column_iqr_interpolate_replaces_the_outlier_value():
    df = _col_with_outlier()
    col_info = {"v": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_IQR, outlier_threshold=1.5,
        outlier_action=ACTION_INTERP, stuck_enabled=False,
    )
    assert len(result) == len(df)
    assert result["v"].iloc[-1] != 100.0
    assert result["v"].notna().all()


def test_multivariate_interpolate_replaces_outlier_features_not_the_whole_row():
    df = _blob_with_outlier()
    col_info = {"x": {"role": ROLE_INPUT, "reason": ""}, "y": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_ISOLATION_FOREST, outlier_threshold=0.1,
        outlier_action=ACTION_INTERP, stuck_enabled=False,
    )
    outlier_idx = df.index[-1]
    assert len(result) == len(df)                    # row kept, not removed
    assert outlier_idx in result.index
    assert result.loc[outlier_idx, "x"] != 100.0      # replaced via interpolation
    assert result["x"].notna().all() and result["y"].notna().all()


def test_multivariate_cap_action_is_unsupported_and_logged_not_silently_ignored():
    """Cap has no row-level meaning for a joint-feature-space method (there's
    no single per-column bound to clip to) — apply_cleaning must say so in the
    log and leave the data untouched, not silently do nothing."""
    df = _blob_with_outlier()
    col_info = {"x": {"role": ROLE_INPUT, "reason": ""}, "y": {"role": ROLE_INPUT, "reason": ""}}
    result, log = apply_cleaning(
        df, col_info, missing_action=ACTION_NONE,
        outlier_method=OUTLIER_ISOLATION_FOREST, outlier_threshold=0.1,
        outlier_action=ACTION_CAP, stuck_enabled=False,
    )
    outlier_idx = df.index[-1]
    assert len(result) == len(df)
    assert result.loc[outlier_idx, "x"] == 100.0   # left untouched
    assert any("[WARN]" in entry and "not supported" in entry for entry in log)
