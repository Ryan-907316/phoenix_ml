# dataset_cleaning.py
# Backend logic for the Dataset Cleaning tab.
# Handles column type detection, sensor issue detection, and cleaning operations.

from __future__ import annotations
import copy
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

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

OUTLIER_NONE              = "None"
OUTLIER_IQR               = "Interquartile Range"
OUTLIER_ZSCORE            = "Z-Score"
OUTLIER_PERCENTILE        = "Percentage"
OUTLIER_ISOLATION_FOREST  = "Isolation Forest"
OUTLIER_LOF               = "Local Outlier Factor"
OUTLIER_MCD               = "Minimum Covariance Determinant"

# Multivariate methods work on the full feature matrix at once and return a
# row-level mask directly, unlike IQR/Z-Score/Percentage which run per-column.
MULTIVARIATE_OUTLIER_METHODS = {OUTLIER_ISOLATION_FOREST, OUTLIER_LOF, OUTLIER_MCD}

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

# An object column counts as numeric when at least this fraction of its non-null
# values parse as numbers. Real-world sensor CSVs often carry a handful of text
# entries ("error", "----", "NA") in an otherwise numeric channel; one stray string
# should not cost the user the whole column.
NUMERIC_COERCE_THRESHOLD = 0.90


def _is_stringlike_dtype(series: pd.Series) -> bool:
    """True for legacy object-dtype text columns and pandas >= 3.0's dedicated
    string dtype (PDEP-14 made the latter the default for `pd.read_csv` text
    columns, so checking `dtype == object` alone silently stops matching)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def coerce_numeric(series: pd.Series):
    """Try converting an object column to numeric. Returns (coerced, n_unparseable)
    when at least NUMERIC_COERCE_THRESHOLD of non-null values parse (unparseable
    entries become NaN), else (None, 0)."""
    if not _is_stringlike_dtype(series):
        return None, 0
    non_null = series.dropna()
    if len(non_null) == 0:
        return None, 0
    coerced = pd.to_numeric(series, errors="coerce")
    n_bad = int(coerced[series.notna()].isna().sum())
    if 1.0 - n_bad / len(non_null) >= NUMERIC_COERCE_THRESHOLD:
        return coerced, n_bad
    return None, 0


def detect_column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return TYPE_DATETIME
    if _is_stringlike_dtype(series):
        # Random, not head(20): a sorted column or one with a placeholder value
        # clustered at the start (e.g. blank/"Unknown" rows before real data
        # begins) would otherwise misclassify off a head-sample that isn't
        # representative of the column as a whole.
        non_null = series.dropna()
        sample = (non_null.sample(n=min(20, len(non_null)), random_state=0)
                  if len(non_null) > 0 else non_null)
        try:
            pd.to_datetime(sample)
            return TYPE_DATETIME
        except Exception:
            pass
        # Mostly-numeric text column (sensor channel with stray text entries)
        if coerce_numeric(series)[0] is not None:
            return TYPE_NUMERIC
        return TYPE_STRING
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return TYPE_BINARY
        return TYPE_NUMERIC
    return TYPE_STRING


def _constant_step_info(series: pd.Series):
    """Detect strictly-increasing, constant-step columns.

    Returns (is_constant_step, step, first_value). Both record IDs (0,1,2,...)
    and regularly-sampled time axes (0,60,120,... seconds) look like this, so the
    caller has to disambiguate — a time axis is a legitimate model input, an ID
    is not.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False, None, None
    s = series.dropna()
    if len(s) < 3 or len(s) != series.nunique():
        return False, None, None
    diffs = s.diff().dropna()
    if len(diffs) == 0 or not (diffs > 0).all() or diffs.std() >= 1e-6:
        return False, None, None
    return True, float(diffs.iloc[0]), float(s.iloc[0])


_TIME_NAME_TOKENS = {
    "t", "time", "timestamp", "stamp", "date", "datetime", "elapsed",
    "sec", "secs", "second", "seconds", "min", "mins", "minute", "minutes",
    "hour", "hours", "hr", "hrs", "day", "days",
}


def _looks_time_named(name: str) -> bool:
    tokens = re.split(r"[^a-zA-Z]+", str(name).lower())
    return any(tok in _TIME_NAME_TOKENS for tok in tokens if tok)


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


def detect_outliers_multivariate(
    df_numeric: pd.DataFrame, method: str, contamination: float = 0.1,
    random_state: int | None = None,
) -> pd.Series:
    """Row-level multivariate outlier mask.

    `df_numeric` should already contain only the numeric feature columns to
    score (targets/timestamps/excluded columns removed by the caller) — these
    methods look at the joint feature space, not one column at a time.
    Rows with any NaN in `df_numeric` are excluded from fitting and returned
    as not-flagged; missing-value handling deals with those separately.
    """
    mask = pd.Series(False, index=df_numeric.index)
    complete = df_numeric.dropna()
    if len(complete) < 10 or complete.shape[1] == 0:
        return mask

    # None resolves to seed 0 (package-wide convention) so IsolationForest/
    # EllipticEnvelope flag the same rows on every run by default.
    random_state = 0 if random_state is None else random_state

    # Defensive clamp: sklearn requires contamination in (0, 0.5]. The UI shares one
    # "outlier_threshold" field across IQR (~1.5), Percentage (~95.0), and these
    # multivariate methods, so a value left over from switching methods would
    # otherwise crash here with a raw sklearn InvalidParameterError instead of this
    # codebase's usual friendly handling. apply_cleaning() clamps (and logs a
    # [WARN] about it) before calling this function; this is a second, silent
    # safety net for any other caller.
    contamination = min(max(float(contamination), 1e-4), 0.5)

    if method == OUTLIER_ISOLATION_FOREST:
        model = IsolationForest(contamination=contamination, random_state=random_state)
    elif method == OUTLIER_LOF:
        model = LocalOutlierFactor(contamination=contamination, novelty=False)
    elif method == OUTLIER_MCD:
        model = EllipticEnvelope(contamination=contamination, random_state=random_state)
    else:
        return mask

    preds = model.fit_predict(complete.values)  # -1 = outlier, 1 = inlier
    mask.loc[complete.index] = preds == -1
    return mask


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

        # Object column that is really a numeric sensor channel with stray text:
        # classify/analyse it using the coerced values (text entries become NaN,
        # which the missing-value handling below then picks up and flags).
        n_coerce_bad = 0
        if coltype == TYPE_NUMERIC and _is_stringlike_dtype(series):
            coerced, n_coerce_bad = coerce_numeric(series)
            if coerced is not None:
                series = coerced
            if n_coerce_bad > 0:
                issues.append(
                    f"{n_coerce_bad} non-numeric value(s) - will become NaN on Apply"
                )

        # ── Slam-dunk role assignments ────────────────────────────────────────
        if coltype == TYPE_DATETIME:
            suggested = ROLE_TIMESTAMP
            reason    = "Datetime column"

        elif coltype == TYPE_STRING:
            suggested = ROLE_EXCLUDE
            reason    = "Non-numeric string - incompatible with regression workflow"

        elif (_cs := _constant_step_info(series))[0]:
            _, step, first = _cs
            if _looks_time_named(col):
                # A regularly-sampled time axis: valid input or timestamp, not an ID
                suggested = ROLE_TIMESTAMP
                reason    = f"Constant sample interval ({step:g}) - looks like a time axis"
            else:
                # Constant step (including the classic 0- or 1-based counter:
                # 0,1,2,... or 1,2,3,...) — defaults to Input with a review flag
                # rather than auto-Exclude, even for the counter-shaped case: a
                # step-1-from-0/1 column is OFTEN a record ID, but can just as
                # easily be a genuine feature that coincidentally starts there
                # (e.g. "Stage Number"), and drop_excluded=True by default meant
                # auto-Exclude here was silent data loss on a coincidence, not a
                # slam dunk. No naming heuristic (e.g. checking for "id" in the
                # column name) is used to disambiguate — a real "Batch Number" or
                # "Run Count" feature could just as easily false-positive on that,
                # so it isn't a more reliable signal than just asking the user.
                suggested = ROLE_INPUT
                if step == 1.0 and first in (0.0, 1.0):
                    issues.append(
                        f"Monotonic integer counter (step 1, starts at {first:g}) - "
                        f"likely a record ID, but could be a genuine feature; review role"
                    )
                else:
                    issues.append(
                        f"Monotonic constant-step column (step {step:g}) - "
                        f"could be a time axis or record ID; review role"
                    )

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
    random_state:      int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    # Deep-copied so a failed timestamp conversion below (which reassigns a column's
    # role to Exclude) can't mutate the CALLER's col_info in place — the UI reuses the
    # same col_info object across repeated Apply calls and across datasets in a
    # session, so without this a stale Exclude from one failed run would silently
    # carry forward into every later one (found via a systematic failure-mode sweep).
    col_info = copy.deepcopy(col_info)
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

    # ── 1b. Coerce mostly-numeric text columns ────────────────────────────────
    # Real sensor exports often have one stray text entry ("error", "----") that
    # makes pandas type the whole channel as object. Convert those columns to
    # numeric here (stray text -> NaN) so the numeric steps below can see them;
    # the missing-value handling in step 5 then deals with the NaNs.
    for col in result.columns:
        if not _is_stringlike_dtype(result[col]):
            continue
        if col_info.get(col, {}).get("role") == ROLE_EXCLUDE:
            continue
        if col_info.get(col, {}).get("role") == ROLE_TIMESTAMP:
            continue
        coerced, n_bad = coerce_numeric(result[col])
        if coerced is not None:
            result[col] = coerced
            log.append(
                f"[CONV] '{col}': text -> numeric"
                + (f" ({n_bad} unparseable value(s) -> NaN)" if n_bad else "")
            )

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
    if outlier_method in MULTIVARIATE_OUTLIER_METHODS and outlier_action != ACTION_NONE:
        # These methods score the joint feature space and return a row-level
        # mask directly, rather than running column-by-column.
        feature_cols = [
            c for c, info in col_info.items()
            if info.get("role") == ROLE_INPUT
            and c in result.columns
            and pd.api.types.is_numeric_dtype(result[c])
        ]
        if not feature_cols:
            log.append(
                f"[WARN] '{outlier_method}': no numeric Input-role columns available - skipped"
            )
        else:
            # outlier_threshold is shared with IQR (~1.5) and Percentage (~95.0), but
            # this method needs a fraction in (0, 0.5] (sklearn's contamination) — a
            # value left over from switching methods would otherwise crash with a raw
            # sklearn error instead of this codebase's usual friendly handling.
            mv_contamination = outlier_threshold
            if not (0 < mv_contamination <= 0.5):
                mv_contamination = min(max(mv_contamination, 1e-4), 0.5)
                log.append(
                    f"[WARN] '{outlier_method}': threshold {outlier_threshold:g} is out of "
                    f"range for this method (needs 0-0.5, it's a fraction of rows) - using "
                    f"{mv_contamination:g} instead"
                )
            mask = detect_outliers_multivariate(
                result[feature_cols], outlier_method, contamination=mv_contamination,
                random_state=random_state,
            )
            n_out = int(mask.sum())
            if n_out > 0:
                if outlier_action == ACTION_REMOVE:
                    result = result[~mask]
                    log.append(
                        f"[FIX ] {n_out} outlier rows removed ({outlier_method}, "
                        f"multivariate over {len(feature_cols)} feature column(s))"
                    )
                elif outlier_action == ACTION_INTERP:
                    for c in feature_cols:
                        result.loc[mask, c] = np.nan
                    result[feature_cols] = result[feature_cols].interpolate(
                        method="linear", limit_direction="both"
                    )
                    log.append(
                        f"[FIX ] {n_out} outlier rows interpolated ({outlier_method}, "
                        f"multivariate over {len(feature_cols)} feature column(s))"
                    )
                else:
                    log.append(
                        f"[WARN] '{outlier_method}' flagged {n_out} outlier rows but "
                        f"action '{outlier_action}' is not supported for row-level "
                        f"multivariate methods - no rows changed"
                    )

    elif outlier_method != OUTLIER_NONE and outlier_action != ACTION_NONE:
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
        else:
            # Missing-value handling explicitly disabled — these NaNs (some possibly
            # introduced moments ago by the text->numeric coercion in step 1b) are
            # left exactly as-is. Log them so a later, less specific rejection
            # downstream (e.g. load_and_preprocess_data's NaN check) isn't the
            # user's first sign anything needs attention.
            detail = ", ".join(f"'{c}' ({n})" for c, n in col_nan.items())
            log.append(f"[WARN] Missing values present but action is 'None' - {detail}")

    # ── Summary ───────────────────────────────────────────────────────────────
    rows_after  = len(result)
    rows_removed = rows_before - rows_after
    log.append(
        f"[INFO] Export: {rows_after} rows x{len(result.columns)} columns"
        + (f" ({rows_removed} rows removed," if rows_removed else " (")
        + f" {cols_excluded} column(s) excluded)"
    )

    return result.reset_index(drop=True), log
