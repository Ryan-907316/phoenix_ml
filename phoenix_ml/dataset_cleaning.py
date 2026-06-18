# dataset_cleaning.py
# Backend logic for the Dataset Cleaning tab.
# Handles column type detection, sensor issue detection, and cleaning operations.

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Column types ──────────────────────────────────────────────────────────────

TYPE_NUMERIC  = "Numeric"
TYPE_DATETIME = "Datetime"
TYPE_BINARY   = "Binary"
TYPE_STRING   = "String"

# ── Roles ─────────────────────────────────────────────────────────────────────

ROLE_INPUT     = "Input"
ROLE_TARGET    = "Target"
ROLE_TIMESTAMP = "Timestamp"
ROLE_EXCLUDE   = "Exclude"

# ── Outlier methods ───────────────────────────────────────────────────────────

OUTLIER_NONE       = "None"
OUTLIER_IQR        = "Interquartile Range"
OUTLIER_ZSCORE     = "Z-Score"
OUTLIER_PERCENTILE = "Percentage"

# ── Actions ───────────────────────────────────────────────────────────────────

ACTION_NONE   = "None"
ACTION_REMOVE = "Remove Rows"
ACTION_CAP    = "Cap (Winsorise)"
ACTION_INTERP = "Interpolate"
ACTION_FFILL  = "Forward Fill"
ACTION_BFILL  = "Backward Fill"
ACTION_MEAN   = "Replace with Mean"
ACTION_MEDIAN = "Replace with Median"
ACTION_DROP   = "Drop Rows"


# ── Type detection ────────────────────────────────────────────────────────────

def detect_column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return TYPE_DATETIME
    if series.dtype == object:
        sample = series.dropna().head(20)
        try:
            pd.to_datetime(sample)
            return TYPE_DATETIME
        except Exception:
            return TYPE_STRING
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return TYPE_BINARY
        return TYPE_NUMERIC
    return TYPE_STRING


def _is_likely_id(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna()
    if len(s) == 0:
        return False
    if len(s) != series.nunique():
        return False
    diffs = s.diff().dropna()
    if len(diffs) == 0:
        return False
    return bool((diffs > 0).all() and diffs.std() < 1e-6)


def _is_zero_variance(series: pd.Series) -> bool:
    """True only when every non-NaN value is identical (range == 0)."""
    s = series.dropna()
    if len(s) == 0:
        return False
    return float(s.max() - s.min()) == 0.0


# ── Per-column statistics ─────────────────────────────────────────────────────

def compute_column_stats(series: pd.Series) -> dict:
    n = len(series)
    n_missing = int(series.isna().sum())
    stats: dict = {
        "n_rows": n,
        "n_missing": n_missing,
        "missing_pct": round(100.0 * n_missing / n, 1) if n > 0 else 0.0,
    }
    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        if len(s) > 0:
            stats.update({
                "min":    round(float(s.min()),    4),
                "max":    round(float(s.max()),    4),
                "mean":   round(float(s.mean()),   4),
                "std":    round(float(s.std()),    4),
                "median": round(float(s.median()), 4),
            })
        else:
            stats.update({"min": None, "max": None, "mean": None,
                          "std": None, "median": None})
    else:
        stats["n_unique"] = int(series.nunique())
    return stats


# ── Sensor issue detection ────────────────────────────────────────────────────

def detect_stuck_values(series: pd.Series, min_run: int = 5) -> dict:
    empty = {"n_runs": 0, "n_affected_rows": 0, "runs": []}
    if not pd.api.types.is_numeric_dtype(series):
        return empty
    s = series.dropna().reset_index(drop=True)
    if len(s) == 0:
        return empty
    runs = []
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s.iloc[j] == s.iloc[i]:
            j += 1
        if j - i >= min_run:
            runs.append((int(i), int(j - 1), float(s.iloc[i])))
        i = j
    return {
        "n_runs": len(runs),
        "n_affected_rows": sum(e - b + 1 for b, e, _ in runs),
        "runs": runs,
    }


def detect_outliers_iqr(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(False, index=series.index)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    mask = (series < lower) | (series > upper)
    return mask.fillna(False)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(False, index=series.index)
    mean = float(s.mean())
    std  = float(s.std())
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs().gt(threshold).fillna(False)


def percentile_bounds(series: pd.Series, pct: float = 95.0) -> tuple[float, float]:
    """Two-sided bounds that keep the middle `pct`% of values (e.g. 95 -> 2.5th/97.5th)."""
    s = series.dropna()
    tail  = (100.0 - pct) / 2.0 / 100.0
    lower = float(s.quantile(tail))
    upper = float(s.quantile(1.0 - tail))
    return lower, upper


def detect_outliers_percentile(series: pd.Series, pct: float = 95.0) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(False, index=series.index)
    lower, upper = percentile_bounds(series, pct)
    mask = (series < lower) | (series > upper)
    return mask.fillna(False)


def detect_clipping(series: pd.Series, min_pct: float = 0.02) -> dict:
    if not pd.api.types.is_numeric_dtype(series):
        return {"clipped_low": 0, "clipped_high": 0}
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"clipped_low": 0, "clipped_high": 0}
    threshold_count = max(2, int(n * min_pct))
    low_count  = int((s == s.min()).sum())
    high_count = int((s == s.max()).sum())
    return {
        "clipped_low":  low_count  if low_count  >= threshold_count else 0,
        "clipped_high": high_count if high_count >= threshold_count else 0,
    }


def detect_duplicates(df: pd.DataFrame) -> int:
    """Return count of exact duplicate rows (ignoring the first occurrence)."""
    return int(df.duplicated().sum())


def detect_burst_dropout(series: pd.Series, max_normal_gap: int = 3) -> dict:
    nan_mask = series.isna().reset_index(drop=True)
    bursts = []
    i = 0
    while i < len(nan_mask):
        if nan_mask.iloc[i]:
            j = i + 1
            while j < len(nan_mask) and nan_mask.iloc[j]:
                j += 1
            length = j - i
            if length > max_normal_gap:
                bursts.append((i, length))
            i = j
        else:
            i += 1
    return {"n_bursts": len(bursts), "bursts": bursts}


# ── Auto-classification ───────────────────────────────────────────────────────
#
# Philosophy: only auto-exclude slam-dunk cases (datetime, non-numeric string,
# monotonic IDs, truly constant columns).  Binary, low-variance, stuck values,
# and clipping are informational notes — the user decides what to do with them.
# IQR outlier detection is NOT run here; it belongs in the Apply step.

def auto_classify_columns(df: pd.DataFrame) -> dict:
    n = len(df)
    result = {}
    for col in df.columns:
        series  = df[col]
        coltype = detect_column_type(series)
        issues: list[str] = []
        reason = ""

        # ── Slam-dunk role assignments ────────────────────────────────────────
        if coltype == TYPE_DATETIME:
            suggested = ROLE_TIMESTAMP
            reason    = "Datetime column"

        elif coltype == TYPE_STRING:
            suggested = ROLE_EXCLUDE
            reason    = "Non-numeric string - incompatible with regression workflow"

        elif _is_likely_id(series):
            suggested = ROLE_EXCLUDE
            reason    = "Monotonic integer - likely a record ID"
            issues.append("Likely record ID")

        elif _is_zero_variance(series):
            suggested = ROLE_EXCLUDE
            reason    = "Constant column (every value is identical)"
            issues.append("Zero variance - no information content")

        else:
            # Everything else defaults to Input; user can promote to Target or
            # demote to Exclude after reviewing the column manager.
            suggested = ROLE_INPUT
            reason    = ""

            # ── Informational notes (no role change) ─────────────────────────
            if coltype == TYPE_BINARY:
                issues.append("Binary (0/1) - check if suitable as a regression feature")

            if coltype == TYPE_NUMERIC:
                # Stuck values: require a longer run (min_run=10) to reduce
                # false positives from legitimate stepped/setpoint signals.
                stuck = detect_stuck_values(series, min_run=10)
                if stuck["n_runs"] > 0:
                    k = stuck["n_runs"]
                    issues.append(
                        f"{stuck['n_affected_rows']} stuck values "
                        f"({k} run{'s' if k > 1 else ''})"
                    )

                # Clipping: require 5 % of rows at the exact min or max,
                # so stepped inputs don't falsely trigger this.
                clip = detect_clipping(series, min_pct=0.05)
                if clip["clipped_low"] or clip["clipped_high"]:
                    parts = []
                    if clip["clipped_low"]:  parts.append(f"low: {clip['clipped_low']}")
                    if clip["clipped_high"]: parts.append(f"high: {clip['clipped_high']}")
                    issues.append(f"Possible clipping ({', '.join(parts)})")

        # ── Missing values: always flag, regardless of role ───────────────────
        n_missing = int(series.isna().sum())
        if n_missing > 0:
            burst = detect_burst_dropout(series)
            if burst["n_bursts"] > 0:
                k = burst["n_bursts"]
                issues.append(
                    f"{n_missing} missing values "
                    f"({k} burst{'s' if k > 1 else ''})"
                )
            else:
                pct = round(100.0 * n_missing / n, 1) if n > 0 else 0.0
                issues.append(f"{n_missing} missing values ({pct}%)")

        result[col] = {
            "type":           coltype,
            "suggested_role": suggested,
            "role":           suggested,
            "reason":         reason,
            "issues":         issues,
            "stats":          compute_column_stats(series),
        }
    return result


# ── Row-level issue mask (for data preview colouring) ─────────────────────────

def build_issue_mask(df: pd.DataFrame, col_info: dict) -> dict:
    row_issues: dict[int, set] = {i: set() for i in range(len(df))}
    for col, info in col_info.items():
        if info.get("role") == ROLE_EXCLUDE or col not in df.columns:
            continue
        series = df[col].reset_index(drop=True)
        # NaN
        for i in series[series.isna()].index:
            row_issues[int(i)].add("nan")
        if pd.api.types.is_numeric_dtype(series):
            # Outliers
            outlier_mask = detect_outliers_iqr(series)
            for i in outlier_mask[outlier_mask].index:
                row_issues[int(i)].add("outlier")
            # Stuck values
            stuck = detect_stuck_values(series)
            for start, end, _ in stuck["runs"]:
                for i in range(start, end + 1):
                    if i in row_issues:
                        row_issues[i].add("stuck")
    return row_issues


# ── Cleaning orchestrator ─────────────────────────────────────────────────────

def apply_cleaning(
    df: pd.DataFrame,
    col_info: dict,
    missing_action:    str   = ACTION_FFILL,
    outlier_method:    str   = OUTLIER_IQR,
    outlier_threshold: float = 1.5,
    outlier_action:    str   = ACTION_REMOVE,
    stuck_enabled:     bool  = True,
    stuck_min_run:     int   = 10,
    stuck_action:      str   = ACTION_NONE,
    drop_excluded:     bool  = True,
    remove_duplicates: bool  = False,
) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    log: list[str] = []
    rows_before   = len(result)
    cols_excluded = 0

    # ── 0. Remove duplicate rows ──────────────────────────────────────────────
    if remove_duplicates:
        n_dupes = int(result.duplicated().sum())
        if n_dupes > 0:
            result = result.drop_duplicates()
            log.append(f"[FIX ] Duplicate rows removed: {n_dupes}")

    # ── 1. Convert timestamp columns ──────────────────────────────────────────
    for col, info in col_info.items():
        if info.get("role") != ROLE_TIMESTAMP or col not in result.columns:
            continue
        try:
            dt      = pd.to_datetime(result[col])
            elapsed = (dt - dt.min()).dt.total_seconds()
            result[col] = elapsed
            log.append(
                f"[CONV] '{col}': datetime -> elapsed seconds "
                f"(0.0 s -> {elapsed.max():.1f} s)"
            )
        except Exception as exc:
            log.append(f"[WARN] '{col}': could not convert datetime - {exc}")
            col_info[col]["role"] = ROLE_EXCLUDE

    # ── 2. Drop excluded columns ──────────────────────────────────────────────
    if drop_excluded:
        to_drop = [
            c for c, info in col_info.items()
            if info.get("role") == ROLE_EXCLUDE and c in result.columns
        ]
        for c in to_drop:
            result = result.drop(columns=[c])
            log.append(f"[EXCL] '{c}': removed - {col_info[c]['reason']}")
            cols_excluded += 1

    active_cols = list(result.columns)

    # ── 3. Handle stuck values ────────────────────────────────────────────────
    # Note: detect_stuck_values returns positions within the dropna() series,
    # so we map back to label-based index via non_nan_idx before modifying rows.
    if stuck_enabled and stuck_action != ACTION_NONE:
        for col in active_cols:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            stuck = detect_stuck_values(result[col], min_run=stuck_min_run)
            if stuck["n_runs"] == 0:
                continue
            n_affected = stuck["n_affected_rows"]
            non_nan_idx = result[col].dropna().index
            if stuck_action == ACTION_INTERP:
                col_data = result[col].copy()
                for start, end, _ in stuck["runs"]:
                    labels = non_nan_idx[start + 1: end + 1]
                    col_data.loc[labels] = np.nan
                result[col] = col_data.interpolate(
                    method="linear", limit_direction="both"
                )
                log.append(
                    f"[FIX ] '{col}': {n_affected} stuck values in "
                    f"{stuck['n_runs']} run(s) -> interpolated"
                )
            elif stuck_action == ACTION_REMOVE:
                mask = pd.Series(False, index=result.index)
                for start, end, _ in stuck["runs"]:
                    labels = non_nan_idx[start: end + 1]
                    mask.loc[labels] = True
                result = result[~mask]
                log.append(f"[FIX ] '{col}': {n_affected} stuck rows removed")

    # ── 4. Handle outliers ────────────────────────────────────────────────────
    if outlier_method != OUTLIER_NONE and outlier_action != ACTION_NONE:
        rows_to_drop   = pd.Series(False, index=result.index)
        outlier_counts: dict[str, int] = {}

        for col in result.columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            if outlier_method == OUTLIER_IQR:
                mask = detect_outliers_iqr(result[col], outlier_threshold)
            elif outlier_method == OUTLIER_PERCENTILE:
                mask = detect_outliers_percentile(result[col], outlier_threshold)
            else:
                mask = detect_outliers_zscore(result[col], outlier_threshold)
            n_out = int(mask.sum())
            if n_out == 0:
                continue
            outlier_counts[col] = n_out

            if outlier_action == ACTION_REMOVE:
                rows_to_drop = rows_to_drop | mask
            elif outlier_action == ACTION_CAP:
                s = result[col].dropna()
                if outlier_method == OUTLIER_ZSCORE:
                    mean_ = float(s.mean())
                    std_  = float(s.std())
                    lower = mean_ - outlier_threshold * std_
                    upper = mean_ + outlier_threshold * std_
                elif outlier_method == OUTLIER_PERCENTILE:
                    lower, upper = percentile_bounds(result[col], outlier_threshold)
                else:  # IQR
                    q1 = float(s.quantile(0.25))
                    q3 = float(s.quantile(0.75))
                    iqr   = q3 - q1
                    lower = q1 - outlier_threshold * iqr
                    upper = q3 + outlier_threshold * iqr
                result[col] = result[col].clip(lower=lower, upper=upper)
            elif outlier_action == ACTION_INTERP:
                col_data = result[col].copy()
                col_data[mask] = np.nan
                result[col] = col_data.interpolate(
                    method="linear", limit_direction="both"
                )

        if outlier_counts:
            detail = ", ".join(f"{c} ({n})" for c, n in outlier_counts.items())
            if outlier_action == ACTION_REMOVE:
                result      = result[~rows_to_drop]
                total_removed = int(rows_to_drop.sum())
                log.append(
                    f"[FIX ] {total_removed} outlier rows removed ({outlier_method}) - {detail}"
                )
            else:
                log.append(
                    f"[FIX ] Outliers {outlier_action} ({outlier_method}) - {detail}"
                )

    # ── 5. Handle missing values ──────────────────────────────────────────────
    n_nan = int(result.isna().sum().sum())
    if n_nan > 0:
        col_nan = {
            c: int(result[c].isna().sum())
            for c in result.columns if result[c].isna().any()
        }
        if missing_action == ACTION_FFILL:
            result = result.ffill().bfill()
        elif missing_action == ACTION_BFILL:
            result = result.bfill().ffill()
        elif missing_action == "Interpolate":
            result = result.interpolate(method="linear").bfill().ffill()
        elif missing_action == ACTION_MEAN:
            for col in result.columns:
                if result[col].isna().any() and pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())
        elif missing_action == ACTION_MEDIAN:
            for col in result.columns:
                if result[col].isna().any() and pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())
        elif missing_action == ACTION_DROP:
            result = result.dropna()

        if missing_action != ACTION_NONE:
            remaining = int(result.isna().sum().sum())
            detail    = ", ".join(f"'{c}' ({n})" for c, n in col_nan.items())
            suffix    = f" - {remaining} NaN remaining" if remaining > 0 else ""
            log.append(
                f"[FIX ] Missing values ({missing_action}) - {detail}{suffix}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    rows_after  = len(result)
    rows_removed = rows_before - rows_after
    log.append(
        f"[INFO] Export: {rows_after} rows x{len(result.columns)} columns"
        + (f" ({rows_removed} rows removed," if rows_removed else " (")
        + f" {cols_excluded} column(s) excluded)"
    )

    return result.reset_index(drop=True), log
