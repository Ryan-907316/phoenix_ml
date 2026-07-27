# model_cards.py
# Generates a one-page, poster-style PDF "model card" per target: a compact,
# non-expert-facing summary of the final deployed model (what it predicts, its
# headline metrics, and a short list of known limitations), saved as a standalone
# artifact in the Models/Model Cards folder alongside the pipeline/metadata/bundle.
# Deliberately not a shrunk-down copy of the main PDF report -- that documents the
# whole run, this documents one deployed model, meant to travel with it. Every
# limitation flag below is a procedural rule over already-computed results, no
# free-text generation, and only fires if the step it depends on actually ran.

import ast
import os
import re
from datetime import datetime

import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_RIGHT

from phoenix_ml.report_generation import _fmt_metric_long, _fmt_metric
from phoenix_ml.model_training import _MONOTONIC_CAPABLE_MODELS
from phoenix_ml.postprocessing import select_best_transformation_indices, NORMALITY_P_THRESHOLD
from phoenix_ml.hyperparameter_optimisation import process_hyperparameters

# Static -- always the same regardless of model/target, so this isn't "free-text
# generation" in the sense the module docstring rules out (compare footer_note,
# also a fixed string passed down from generate_model_cards). Standard "Intended
# Use" content for a model card: states the boundary of validated use without
# claiming to know whether any specific training-range edge is a real physical
# limit or just a sampling artefact (that distinction needs domain/unit context
# this module doesn't have -- see the Known Limitations range flags below).
SCOPE_NOTE = (
    "Intended for prediction within the training data ranges shown below. Not "
    "validated for extrapolation beyond those ranges, or for safety-critical or "
    "real-time control use without further validation."
)

# ── Visual identity ──────────────────────────────────────────────────────────
# Kept local to model cards: the main report stays visually neutral, but a model
# card is meant to be presentable/poster-style on its own, so it uses the same
# orange accent the desktop app itself already uses (ui.py/code_conventions).
_ACCENT       = colors.HexColor("#E07818")
_ACCENT_DEEP  = colors.HexColor("#B85D0F")
_INK          = colors.HexColor("#2B2620")
_INK_SOFT     = colors.HexColor("#756B5A")
_RULE         = colors.HexColor("#DDD2BA")
_PAPER_RAISED = colors.HexColor("#FBF9F4")
_NEUTRAL_FLAG = colors.HexColor("#7B97C7")

# ── Rule thresholds (confirmed with user, 2026-07-21) ────────────────────────
PRESS_GAP_THRESHOLD = 0.2
UQ_UNDERCOVERAGE_THRESHOLD_POINTS = 5.0
CLEANING_REMOVED_THRESHOLD_FRAC = 0.10
SMALL_DATASET_MIN_ROWS_FLOOR = 50
SMALL_DATASET_ROWS_PER_FEATURE = 20
N_RANGE_FEATURES_SHOWN = 3


def _format_range_value(x):
    """3 significant figures, formatted for a general reader -- never scientific
    notation ("92,600" rather than "9.26e+04"), since this card is meant to be
    presentable to a non-expert, not a numerics-heavy technical audience."""
    rounded = float(f"{x:.3g}")
    if abs(rounded) >= 1000:
        return f"{rounded:,.0f}"
    return f"{rounded:.6f}".rstrip("0").rstrip(".")
_MAX_FLAGS_SHOWN = 6


def _format_hyperparams(hyperparams_raw, model_name):
    """hyperparams_raw: the row's "hyperparameters" value, a stringified dict
    (collect_results_as_dataframe stores it via str()) whether the winning method
    was a real HPO backend or "Default" -- the default-hyperparameters case still
    carries the small curated subset a model was actually constructed with (see
    model_training.py's default_hyperparameters), so this works either way.
    Returns a compact "key=value, key=value" string, or None if unavailable."""
    if isinstance(hyperparams_raw, str):
        try:
            hyperparams_raw = ast.literal_eval(hyperparams_raw)
        except Exception:
            return None
    if not isinstance(hyperparams_raw, dict) or not hyperparams_raw:
        return None
    processed = process_hyperparameters(hyperparams_raw, model_name)
    parts = []
    for k, v in processed.items():
        v_str = f"{v:.4g}" if isinstance(v, float) else str(v)
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


def _format_cv_params(method, params):
    """method/params: a target's "CV Method"/"CV Parameters" cell from
    cv_summary_df -- params is the real cv_args dict (n_splits, test_size, ...).
    random_state is dropped: it's a reproducibility detail, not something a
    non-expert reader reading this card needs to see."""
    if isinstance(params, str):
        try:
            params = ast.literal_eval(params)
        except Exception:
            params = {}
    if not isinstance(params, dict):
        params = {}
    parts = [f"{k}={v}" for k, v in params.items() if k != "random_state"]
    return f"{method} ({', '.join(parts)})" if parts else str(method)


def compute_model_card_flags(
    *,
    model_name,
    headline_metric_name=None,
    headline_metric_value=None,
    predicted_r2=None,
    uq_nominal_coverage=None,
    uq_actual_coverages=None,
    lofo_top_feature=None,
    permutation_top_feature=None,
    residual_transform_passes_normality=None,
    train_rows=None,
    n_features=None,
    rows_before_cleaning=None,
    rows_after_cleaning=None,
    monotonic_constraints_for_target=None,
    top_feature_ranges=None,
    max_flags=_MAX_FLAGS_SHOWN,
):
    """Every rule only fires if the data it needs was actually computed -- a step
    that didn't run is never silently assumed to have "passed". Returns a list of
    {"severity": "warn"|"neutral", "lead": str, "detail": str}, warn-tier first
    (stable within each tier), truncated to max_flags so a page with many
    applicable rules still fits on one page.
    """
    flags = []

    if headline_metric_value is not None and predicted_r2 is not None:
        gap = headline_metric_value - predicted_r2
        if gap >= PRESS_GAP_THRESHOLD:
            flags.append({
                "severity": "warn",
                "lead": "Cross-validated performance is lower than the headline figure.",
                "detail": (
                    f"The reported {_fmt_metric_long(headline_metric_name)} of "
                    f"{headline_metric_value:.3f} dropped to a Predicted R² of "
                    f"{predicted_r2:.3f} under cross-validation, so treat predictions on "
                    f"genuinely new data with extra caution."
                ),
            })

    if uq_nominal_coverage is not None and uq_actual_coverages:
        worst = min(uq_actual_coverages)
        if worst <= uq_nominal_coverage - UQ_UNDERCOVERAGE_THRESHOLD_POINTS:
            flags.append({
                "severity": "warn",
                "lead": "Uncertainty intervals are not perfectly calibrated.",
                "detail": (
                    f"The reported {uq_nominal_coverage:.0f}% intervals covered as little as "
                    f"{worst:.1f}% of test points, so treat interval widths as approximate "
                    f"rather than exact."
                ),
            })

    if lofo_top_feature and permutation_top_feature and lofo_top_feature != permutation_top_feature:
        flags.append({
            "severity": "warn",
            "lead": "Feature importance methods disagree on the top driver.",
            "detail": (
                f"LOFO importance ranks {lofo_top_feature} highest, while permutation "
                f"importance ranks {permutation_top_feature} highest, which can happen when "
                f"input features are correlated with each other."
            ),
        })

    if residual_transform_passes_normality is False:
        flags.append({
            "severity": "warn",
            "lead": "Residuals are not fully normally distributed.",
            "detail": (
                "Prediction intervals may be less reliable than for a model with normally "
                "distributed residuals."
            ),
        })

    if train_rows is not None and n_features is not None and n_features > 0:
        floor = max(SMALL_DATASET_MIN_ROWS_FLOOR, SMALL_DATASET_ROWS_PER_FEATURE * n_features)
        if train_rows < floor:
            flags.append({
                "severity": "warn",
                "lead": "Trained on relatively few rows for the number of features used.",
                "detail": (
                    f"{train_rows} training rows and {n_features} features were used. With "
                    f"limited data density like this, performance on new data may vary more "
                    f"than the reported metrics suggest."
                ),
            })

    if rows_before_cleaning is not None and rows_after_cleaning is not None and rows_before_cleaning > 0:
        removed = rows_before_cleaning - rows_after_cleaning
        frac = removed / rows_before_cleaning
        if frac >= CLEANING_REMOVED_THRESHOLD_FRAC:
            flags.append({
                "severity": "neutral",
                "lead": "A notable share of rows were removed during cleaning.",
                "detail": (
                    f"{removed} of {rows_before_cleaning} rows ({frac:.0%}) were removed "
                    f"before training; this model reflects the cleaned dataset only."
                ),
            })

    if model_name in _MONOTONIC_CAPABLE_MODELS and not (monotonic_constraints_for_target or {}):
        flags.append({
            "severity": "neutral",
            "lead": "No monotonicity constraints were applied.",
            "detail": (
                "The model can learn locally inconsistent trends where training data is "
                "sparse, even where the true relationship is known to only increase or "
                "decrease."
            ),
        })

    if top_feature_ranges:
        for name, lo, hi in top_feature_ranges[:N_RANGE_FEATURES_SHOWN]:
            flags.append({
                "severity": "neutral",
                "lead": f"Trained on a limited range of {name}.",
                "detail": (
                    f"Ranged from {_format_range_value(lo)} to {_format_range_value(hi)} "
                    f"in training data. Predictions outside "
                    f"this range are extrapolations."
                ),
            })

    flags.sort(key=lambda f: 0 if f["severity"] == "warn" else 1)
    return flags[:max_flags]


# ── PDF rendering ─────────────────────────────────────────────────────────────

def _styles():
    return {
        "eyebrow": ParagraphStyle(
            name="_MCEyebrow", fontName="Helvetica-Bold", fontSize=9,
            textColor=_ACCENT_DEEP, leading=11,
        ),
        "title": ParagraphStyle(
            name="_MCTitle", fontName="Helvetica-Bold", fontSize=22,
            textColor=_INK, leading=25,
        ),
        "subtitle": ParagraphStyle(
            name="_MCSubtitle", fontName="Helvetica", fontSize=10,
            textColor=_INK_SOFT, leading=13,
        ),
        "meta": ParagraphStyle(
            name="_MCMeta", fontName="Courier", fontSize=8, textColor=_INK_SOFT,
            leading=11, alignment=TA_RIGHT,
        ),
        "stat_label": ParagraphStyle(
            name="_MCStatLabel", fontName="Helvetica-Bold", fontSize=7.5,
            textColor=_INK_SOFT, leading=9,
        ),
        "stat_value": ParagraphStyle(
            name="_MCStatValue", fontName="Courier-Bold", fontSize=16,
            textColor=_INK, leading=19,
        ),
        "block_title": ParagraphStyle(
            name="_MCBlockTitle", fontName="Helvetica-Bold", fontSize=9.5,
            textColor=_INK, leading=12, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            name="_MCBody", fontName="Helvetica", fontSize=8.5, textColor=_INK,
            leading=11.5, spaceAfter=4,
        ),
        "fact_label": ParagraphStyle(
            name="_MCFactLabel", fontName="Helvetica", fontSize=8, textColor=_INK_SOFT, leading=11,
        ),
        "fact_value": ParagraphStyle(
            name="_MCFactValue", fontName="Courier-Bold", fontSize=8, textColor=_INK,
            leading=11, alignment=TA_RIGHT,
        ),
        "limit": ParagraphStyle(
            name="_MCLimit", fontName="Helvetica", fontSize=8, textColor=_INK, leading=11, spaceAfter=6,
        ),
        "footer": ParagraphStyle(
            name="_MCFooter", fontName="Courier", fontSize=7, textColor=_INK_SOFT, leading=9,
        ),
    }


def _header_block(styles, model_name, target, dataset_name, is_residual,
                   version, date_str, seed, hpo_label):
    left = [
        Paragraph("MODEL CARD", styles["eyebrow"]),
        Paragraph(model_name, styles["title"]),
        Paragraph(
            f"Predicts <b>{target}</b>. Trained on {dataset_name}."
            if not is_residual else
            f"Predicts a residual for <b>{target}</b>. Trained on {dataset_name}.",
            styles["subtitle"],
        ),
    ]
    right = Paragraph(
        f"phoenix_ml v{version}<br/>generated {date_str}<br/>seed {seed} &middot; {hpo_label}",
        styles["meta"],
    )
    header_table = Table([[left, right]], colWidths=[115 * mm, 55 * mm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LINEBELOW", (0, 0), (-1, -1), 1.5, _INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return header_table


def _stat_tiles(styles, stats):
    """stats: list of (label, value_str) pairs, whichever metrics were actually
    computed for this run -- not a fixed set."""
    if not stats:
        return Spacer(1, 0)
    cells = [[
        [Paragraph(label.upper(), styles["stat_label"]), Paragraph(value, styles["stat_value"])]
        for label, value in stats
    ]]
    col_w = 170 * mm / len(stats)
    tbl = Table(cells, colWidths=[col_w] * len(stats))
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PAPER_RAISED),
        ("GRID", (0, 0), (-1, -1), 0.5, _RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


_FEATURE_BAR_NAME_W  = 24 * mm
_FEATURE_BAR_TRACK_W = 32 * mm
_FEATURE_BAR_VALUE_W = 12 * mm
# Must fit within the left column's own width (see build_model_card_elements'
# body table, colWidths=[85mm, 85mm]) -- a nested Table wider than the cell it's
# placed in is not clipped or auto-shrunk by reportlab, it overflows straight
# into the neighbouring column instead. 24+32+12 = 68mm, safely under 85mm.


def _feature_bars(styles, top_features, method_label=None):
    """top_features: list of (name, normalised_share 0-1, display_value_str).
    method_label: which importance method these came from (e.g. "LOFO",
    "Permutation") -- shown in the heading so it's never ambiguous which one is
    plotted, particularly since the Known Limitations column right beside it may
    say the two methods disagree on the ranking."""
    if not top_features:
        return None
    heading = f"TOP DRIVERS ({method_label})" if method_label else "TOP DRIVERS"
    rows = [Paragraph(heading, styles["block_title"])]
    max_share = max(s for _, s, _ in top_features) or 1.0
    for name, share, value_str in top_features:
        frac = max(0.03, share / max_share)
        bar = Table([[""]], colWidths=[frac * _FEATURE_BAR_TRACK_W], rowHeights=[3.2 * mm])
        bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _ACCENT)]))
        row = Table(
            [[Paragraph(name, styles["fact_label"]), bar, Paragraph(value_str, styles["fact_value"])]],
            colWidths=[_FEATURE_BAR_NAME_W, _FEATURE_BAR_TRACK_W, _FEATURE_BAR_VALUE_W],
        )
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        rows.append(row)
    return rows


def _limitation_flowables(styles, flags):
    out = [Paragraph("KNOWN LIMITATIONS", styles["block_title"])]
    dot_colors = {"warn": _ACCENT, "neutral": _NEUTRAL_FLAG}
    for f in flags:
        dot_hex = dot_colors[f["severity"]].hexval().replace("0x", "#")
        out.append(Paragraph(
            f'<font color="{dot_hex}">&#9679;</font>&nbsp;'
            f'<b>{f["lead"]}</b> {f["detail"]}',
            styles["limit"],
        ))
    if not flags:
        out.append(Paragraph("No automatic caveats were flagged for this model.", styles["limit"]))
    return out


def build_model_card_elements(
    *, model_name, target, dataset_name, is_residual, version, date_str, seed, hpo_label,
    stats, prediction_note, training_facts, top_features, flags, footer_note,
    top_features_method=None, scope_note=None, feature_names_list=None,
    key_hyperparameters=None,
):
    """Pure builder: returns a flat list of reportlab flowables for one target's
    model card. Kept separate from save_model_card_pdf so it's testable without
    writing an actual PDF file (mirrors report_generation.py's own
    add_*_section() functions, which build elements lists the tests inspect
    directly rather than parsing rendered PDF output).
    """
    styles = _styles()
    elements = [
        _header_block(styles, model_name, target, dataset_name, is_residual,
                      version, date_str, seed, hpo_label),
        Spacer(1, 8),
        _stat_tiles(styles, stats),
        Spacer(1, 10),
    ]

    left_col = [Paragraph("WHAT THIS MODEL PREDICTS", styles["block_title"]),
                Paragraph(prediction_note, styles["body"])]
    if scope_note:
        left_col.append(Paragraph(scope_note, styles["body"]))
    left_col += [Spacer(1, 6), Paragraph("TRAINING DATA", styles["block_title"])]
    for label, value in training_facts:
        # 48/32 (was 55/25): the newer facts (CV method, Preprocessing) run longer
        # than the original short values ("8", "last") the narrower split was
        # sized for, and were wrapping awkwardly onto two lines.
        row = Table([[Paragraph(label, styles["fact_label"]), Paragraph(str(value), styles["fact_value"])]],
                     colWidths=[48 * mm, 32 * mm])
        row.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, _RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        left_col.append(row)
    if feature_names_list:
        left_col += [Spacer(1, 6), Paragraph("INPUT FEATURES", styles["block_title"]),
                     Paragraph(", ".join(feature_names_list), styles["body"])]
    if key_hyperparameters:
        left_col += [Spacer(1, 6), Paragraph("KEY HYPERPARAMETERS", styles["block_title"]),
                     Paragraph(key_hyperparameters, styles["body"])]
    bars = _feature_bars(styles, top_features, method_label=top_features_method)
    if bars:
        left_col.append(Spacer(1, 8))
        left_col.extend(bars)

    right_col = _limitation_flowables(styles, flags)

    body = Table([[left_col, right_col]], colWidths=[85 * mm, 85 * mm])
    body.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 10), ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    elements.append(body)
    elements.append(Spacer(1, 10))

    footer = Table([[Paragraph(footer_note, styles["footer"]),
                     Paragraph("phoenix_ml", ParagraphStyle(
                         name="_MCFooterBrand", parent=styles["footer"],
                         textColor=_ACCENT_DEEP, fontName="Courier-Bold", alignment=TA_RIGHT))]],
                    colWidths=[130 * mm, 40 * mm])
    footer.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), 1, _INK),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(footer)

    return elements


def save_model_card_pdf(output_dir, filename, elements):
    """Writes one target's model card elements to a standalone single-page PDF.
    Intentionally its own small SimpleDocTemplate, not init_pdf_report()'s title
    page + table of contents machinery -- a model card has neither."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=12 * mm,
        title=f"phoenix_ml Model Card", author="phoenix_ml",
    )
    doc.build(elements)
    return filepath


# ── Orchestrator ──────────────────────────────────────────────────────────────
# Explicit parameters rather than a session/results object, so this can be called
# identically from workflow_steps.py's WorkflowSession-based UI path and
# workflow.py's standalone flat-parameter run_workflow() -- both already persist
# models the same way (see save_predictor/save_models_and_artifacts), this
# mirrors that pattern rather than introducing a session dependency here.

def _row_get(row, key, default=None):
    """best_models_per_target rows are pandas Series in every real call site, but
    tests (and any future dict-shaped caller) pass plain dicts too -- Series.get
    and dict.get both exist, but a KeyError/TypeError path is kept for anything
    that supports neither."""
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _hpo_label_for_row(row):
    """This target's *actual* winning method (e.g. "Default", "Random Search
    (Sobol)", "Hyperopt (TPE)") -- read from the same find_best_model_and_hyperparams()
    row that already determines this model's hyperparameters. Deliberately NOT
    derived from session-level "which HPO methods are ticked right now" state:
    that can be stale (e.g. still the dataclass default of all three methods)
    when the HPO step didn't actually run this session at all, which is exactly
    the bug this function exists to avoid reintroducing.
    """
    return _row_get(row, "hpo_method", "default hyperparameters")


def _uq_coverages_for_target(target, model_name, hpo_label, uq_after_df, uq_before_df):
    """Prefer UQ (After HPO); fall back to UQ (Before HPO) only when this
    target's deployed model IS the default-hyperparameter one (hpo_label ==
    "Default"), i.e. before and after are the same model -- so the "before"
    numbers genuinely describe what's deployed, not a different, non-deployed
    model. Without this, a run with HPO off entirely (or only "UQ Before HPO"
    enabled) silently showed no UQ data at all even though real,
    directly-applicable coverage numbers existed. Returns a list of coverage
    percentages (possibly empty) or None if no usable UQ data was found.
    """
    df = uq_after_df
    if (df is None or df.empty) and hpo_label == "Default":
        df = uq_before_df
    if df is None or df.empty:
        return None
    tgt_uq = df[(df["Target Variable"] == target) & (df["Model"] == model_name)]
    if tgt_uq.empty:
        return None
    return [c for c in tgt_uq["Coverage (%)"].tolist() if pd.notna(c)]


def generate_model_cards(
    *,
    best_models_per_target,
    target_columns,
    X_train, X_test,
    feature_names,
    hpo_metric,
    cv_results,
    uq_after_df,
    uq_before_df=None,
    uq_settings=None,
    cleaning_summary,
    monotonic_constraints,
    perl_config,
    dataset_path,
    random_seed,
    split_method,
    models_dir,
    scaler_type=None,
    log_warn=print,
):
    """One standalone, one-page PDF per target summarising its final deployed
    model, saved to <models_dir>/Model Cards/. Every value fed to
    compute_model_card_flags is pulled from results that were actually computed
    this run -- a step that didn't run (no CV, no UQ, ...) simply leaves that
    rule unable to fire, rather than guessing. Each target is wrapped
    independently so one bad extraction doesn't take down the rest of the report.
    Returns {target: filepath} for whichever targets succeeded.
    """
    from phoenix_ml import __version__ as _pml_version

    cv_res = cv_results or {}
    cv_summary_df     = cv_res.get("cv_summary_df")
    cv_scoring_metric = cv_res.get("scoring_metric")
    lofo_results      = cv_res.get("lofo_importance") or {}
    perm_results      = cv_res.get("permutation_importance") or {}
    transformation_df = cv_res.get("transformation_df")

    uq_nominal = (uq_settings or {}).get("confidence_interval")
    rows_before = (cleaning_summary or {}).get("rows_before")
    rows_after  = (cleaning_summary or {}).get("rows_after")
    is_perl = perl_config is not None
    n_features = len(feature_names)

    out_dir = os.path.join(models_dir, "Model Cards")
    saved_paths = {}

    for target in target_columns:
        try:
            row = best_models_per_target[target]
            model_name = row["model_name"]
            hpo_label = _hpo_label_for_row(row)

            headline_value = None
            try:
                headline_value = float(row[hpo_metric])
            except Exception:
                headline_value = None

            # The full CV summary row for this target, not just PRED_R^2's special
            # case: whichever scoring metric was actually selected, its real
            # cross-validated Mean Score is shown on the card -- previously any
            # metric other than PRED_R^2 was silently never surfaced here at all.
            cv_mean_score = None
            cv_method_label = None
            cv_params_raw = None
            if cv_summary_df is not None:
                tgt_rows = cv_summary_df[cv_summary_df["Target Variable"] == target]
                if not tgt_rows.empty:
                    tgt_row = tgt_rows.iloc[0]
                    cv_mean_score = float(tgt_row["Mean Score"])
                    # CV Method/Parameters are always present on a real cv_summary_df
                    # (create_cv_summary_df always sets them); .get() only guards
                    # against a minimal/malformed frame, not an expected real case.
                    method_val = tgt_row.get("CV Method")
                    cv_method_label = str(method_val) if method_val is not None else None
                    cv_params_raw = tgt_row.get("CV Parameters")
            # PRESS-gap flag/label specifically need PRED_R^2's value -- the pooled
            # cross-validated R^2 property the gap check relies on doesn't carry
            # over to averaged per-fold metrics the same way.
            predicted_r2 = cv_mean_score if cv_scoring_metric == "PRED_R^2" else None

            uq_actual_coverages = _uq_coverages_for_target(
                target, model_name, hpo_label, uq_after_df, uq_before_df)

            def _top_feature(res_dict, key_values, key_names):
                res = res_dict.get(target)
                if not res:
                    return None
                vals, names = res.get(key_values), res.get(key_names)
                if vals is None or names is None or len(vals) == 0:
                    return None
                return names[max(range(len(vals)), key=lambda i: vals[i])]

            lofo_top = _top_feature(lofo_results, "importances", "feature_names")
            perm_top = _top_feature(perm_results, "importances_mean", "feature_names")

            passes_normality = None
            if transformation_df is not None and not transformation_df.empty:
                try:
                    best_idx = select_best_transformation_indices(transformation_df)
                    best_rows = transformation_df.loc[best_idx]
                    best_rows = best_rows[best_rows["Target Variable"] == target]
                    if not best_rows.empty:
                        p = best_rows.iloc[0].get("Shapiro-Wilk p")
                        if p is not None and pd.notna(p):
                            passes_normality = bool(p >= NORMALITY_P_THRESHOLD)
                except Exception:
                    pass

            train_rows_n = len(X_train) if X_train is not None else None

            # Feature ranges/importance for the "Top Drivers" bars and the range
            # caveat: LOFO preferred (same reasoning as the Executive Summary's
            # own top-feature choice), falling back to permutation importance.
            # Which one is tracked explicitly and shown in the card's heading --
            # left ambiguous, this reads as contradicting the Known Limitations
            # column right beside it whenever the two methods disagree.
            ranking = lofo_results.get(target)
            ranking_method = "LOFO"
            if not ranking:
                ranking = perm_results.get(target)
                ranking_method = "Permutation"
            top_features_for_bars = []
            top_feature_ranges = []
            if ranking:
                vals = ranking.get("importances", ranking.get("importances_mean"))
                names = ranking.get("feature_names")
                if vals is not None and names is not None and len(vals):
                    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
                    for i in order[:3]:
                        top_features_for_bars.append((names[i], float(vals[i]), f"{vals[i]:.2f}"))
                        if X_train is not None and names[i] in getattr(X_train, "columns", []):
                            col = X_train[names[i]]
                            top_feature_ranges.append((names[i], float(col.min()), float(col.max())))

            flags = compute_model_card_flags(
                model_name=model_name,
                headline_metric_name=hpo_metric,
                headline_metric_value=headline_value,
                predicted_r2=predicted_r2,
                uq_nominal_coverage=uq_nominal,
                uq_actual_coverages=uq_actual_coverages,
                lofo_top_feature=lofo_top,
                permutation_top_feature=perm_top,
                residual_transform_passes_normality=passes_normality,
                train_rows=train_rows_n,
                n_features=n_features,
                rows_before_cleaning=rows_before,
                rows_after_cleaning=rows_after,
                monotonic_constraints_for_target=(monotonic_constraints or {}).get(target, {}),
                top_feature_ranges=top_feature_ranges,
            )

            prediction_note = (
                f"This model predicts the difference between a physics-based estimate of "
                f"{target} and the measured value, not {target} itself. Add the physics "
                f"estimate back onto the prediction to get the physical value, or use the "
                f"saved deployable predictor, which does this automatically."
                if is_perl else
                f"This model estimates {target} directly from {n_features} measured input features."
            )

            # Which split each number comes from is stated explicitly rather than
            # left implicit -- the headline figure is a single held-out test-set
            # evaluation, genuinely distinct from the (cross-validated) CV score
            # even when they happen to use the same underlying metric.
            stats = []
            if headline_value is not None:
                stats.append((f"{_fmt_metric(hpo_metric)} (Test)", f"{headline_value:.3f}"))
            if cv_mean_score is not None:
                cv_label = (f"{_fmt_metric(cv_scoring_metric)} (PRESS)"
                            if cv_scoring_metric == "PRED_R^2" else
                            f"{_fmt_metric(cv_scoring_metric)} (CV)")
                stats.append((cv_label, f"{cv_mean_score:.3f}"))
            if uq_actual_coverages:
                stats.append((f"{uq_nominal:.0f}% Coverage" if uq_nominal else "Coverage",
                              f"{min(uq_actual_coverages):.1f}%"))

            training_facts = [
                ("Rows (train / test)",
                 f"{train_rows_n or '?'} / {len(X_test) if X_test is not None else '?'}"),
                ("Features used", n_features),
                ("Split method", split_method),
            ]
            if scaler_type:
                training_facts.append(("Preprocessing", f"{scaler_type} scaling"))
            if cv_method_label:
                training_facts.append(("CV method", _format_cv_params(cv_method_label, cv_params_raw)))

            key_hyperparameters = _format_hyperparams(_row_get(row, "hyperparameters"), model_name)

            elements = build_model_card_elements(
                model_name=model_name, target=target,
                dataset_name=os.path.basename(str(dataset_path)),
                is_residual=is_perl, version=_pml_version,
                date_str=datetime.now().strftime("%Y-%m-%d"),
                seed=random_seed, hpo_label=hpo_label,
                stats=stats, prediction_note=prediction_note,
                training_facts=training_facts, top_features=top_features_for_bars,
                top_features_method=(ranking_method if top_features_for_bars else None),
                flags=flags,
                footer_note=("Reproducibility metadata and full diagnostics: see the "
                             "paired .json file and the main PDF report."),
                scope_note=SCOPE_NOTE,
                feature_names_list=list(feature_names),
                key_hyperparameters=key_hyperparameters,
            )
            safe_target = re.sub(r'[<>:"/\\|?*]', "_", str(target))
            path = save_model_card_pdf(out_dir, f"phoenix_ml Model Card_{safe_target}.pdf", elements)
            saved_paths[target] = path
        except Exception as e:
            log_warn(f"[WARN] Model card generation failed for target '{target}' ({e}), continuing with report.")

    return saved_paths
