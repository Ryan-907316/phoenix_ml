"""Regression test for add_interpretability_section()'s target/model grouping.

Locks in the ordering and heading structure requested during report review:
sections are grouped by target variable (in target_columns order, matching
every other section of the report) with the model name as a lighter
sub-heading — not grouped by model with the target buried in the caption.
"""
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phoenix_ml.interpretability import FailedPlot
from phoenix_ml.report_generation import add_interpretability_section, init_pdf_report


def _heading_texts(elements, style_names):
    out = []
    for el in elements:
        text = getattr(el, "text", None)
        style_name = getattr(getattr(el, "style", None), "name", None)
        if text and style_name in style_names:
            out.append((style_name, text))
    return out


def test_sections_grouped_by_target_in_target_columns_order(tmp_path):
    doc, elements, styles, filepath, summary_index = init_pdf_report(
        filename="test.pdf", output_dir=str(tmp_path), title="Test", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )

    def fig():
        f, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        return f

    # Figure keys mirror the real "{PREFIX}_{target}__{model}" naming, with the
    # model iteration order (LGBM before MLP) deliberately opposite to the
    # target_columns order (Speed before Torque) — this is exactly the
    # mismatch that was reported: figures ended up ordered by whichever model
    # was selected first, not by target.
    figs = {
        "ICE_PDP_Residual Motor Torque__LGBM Regressor": fig(),
        "ICE_PDP_Residual Motor Speed__MLP Regressor": fig(),
    }
    settings = {
        "test_sample_size": 100, "background_sample_size": 10,
        "show_ice_pdp": True, "show_ale": False, "show_shap_summary": False,
        "show_shap_dependence": False, "show_shap_waterfall": False,
        "show_sensitivity_morris": False, "show_sensitivity_sobol": False,
    }

    add_interpretability_section(
        elements, figs, styles, str(tmp_path), settings, n_features=7,
        stage_label="After HPO",
        target_columns=["Residual Motor Speed", "Residual Motor Torque"],
    )

    headings = _heading_texts(elements, {"CustomSubheading"})
    heading_texts = [text for _, text in headings]

    assert heading_texts == [
        "Interpretability Analysis for Residual Motor Speed",
        "Interpretability Analysis for Residual Motor Torque",
    ]

    # The model name must appear as a plain sentence, not its own heading.
    model_sentences = [
        getattr(el, "text", "") for el in elements
        if "<b>Model:</b>" in getattr(el, "text", "")
    ]
    assert any("MLP Regressor" in t for t in model_sentences)
    assert any("LGBM Regressor" in t for t in model_sentences)


def test_a_failed_plot_renders_an_explicit_note_not_a_silently_missing_image(tmp_path):
    """Regression test for a real risk: a plot that failed to generate left
    its figures-dict slot simply absent, so the PDF gave no indication a
    plot was attempted and failed vs. never requested. A FailedPlot sentinel
    must render as an explicit note instead."""
    doc, elements, styles, filepath, summary_index = init_pdf_report(
        filename="test.pdf", output_dir=str(tmp_path), title="Test", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )
    figs = {
        "ICE_PDP_Residual Motor Torque__LGBM Regressor": FailedPlot("simulated failure"),
    }
    settings = {
        "test_sample_size": 100, "background_sample_size": 10,
        "show_ice_pdp": True, "show_ale": False, "show_shap_summary": False,
        "show_shap_dependence": False, "show_shap_waterfall": False,
        "show_sensitivity_morris": False, "show_sensitivity_sobol": False,
    }

    add_interpretability_section(
        elements, figs, styles, str(tmp_path), settings, n_features=7,
        target_columns=["Residual Motor Torque"],
    )

    note_texts = [
        getattr(el, "text", "") for el in elements
        if "failed to generate" in getattr(el, "text", "")
    ]
    assert any("simulated failure" in t for t in note_texts)


def test_unrecognized_figure_key_falls_back_to_its_raw_name_not_a_blank_heading(tmp_path, capsys):
    """Regression test for a real risk: a figure key matching none of the
    hardcoded _PREFIXES (naming drift upstream, or a typo) fell through to an
    empty target_name, silently bucketing the figure into a blank ""
    "Interpretability Analysis for " heading instead of somewhere identifiable."""
    doc, elements, styles, filepath, summary_index = init_pdf_report(
        filename="test.pdf", output_dir=str(tmp_path), title="Test", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )

    def fig():
        f, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        return f

    figs = {"Totally_Unknown_Prefix_Something": fig()}
    settings = {
        "test_sample_size": 100, "background_sample_size": 10,
        "show_ice_pdp": True, "show_ale": False, "show_shap_summary": False,
        "show_shap_dependence": False, "show_shap_waterfall": False,
        "show_sensitivity_morris": False, "show_sensitivity_sobol": False,
    }

    add_interpretability_section(elements, figs, styles, str(tmp_path), settings, n_features=7)

    headings = _heading_texts(elements, {"CustomSubheading"})
    heading_texts = [text for _, text in headings]
    assert heading_texts == ["Interpretability Analysis for Totally_Unknown_Prefix_Something"]
    assert "[WARN]" in capsys.readouterr().out


def test_uq_method_names_expand_both_plus_gp_posterior():
    # The report's settings block lists the actual method names selected,
    # never the stored "Both" shorthand.
    from phoenix_ml.report_generation import _uq_method_names

    assert _uq_method_names({"uq_method": "Both"}) == \
        ["Bootstrapping", "Conformal Predictions"]
    assert _uq_method_names({"uq_method": "Both", "include_gp_posterior": True}) == \
        ["Bootstrapping", "Conformal Predictions", "GP Posterior"]
    assert _uq_method_names({"uq_method": "Conformal"}) == ["Conformal Predictions"]
    assert _uq_method_names({"uq_method": "Bootstrapping"}) == ["Bootstrapping"]


def test_falls_back_to_first_seen_order_without_target_columns(tmp_path):
    doc, elements, styles, filepath, summary_index = init_pdf_report(
        filename="test.pdf", output_dir=str(tmp_path), title="Test", font_name="Helvetica",
        font_size=10, title_font_size=20, heading_font_size=14,
    )

    def fig():
        f, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        return f

    figs = {"ICE_PDP_Residual Motor Torque__LGBM Regressor": fig()}
    settings = {
        "test_sample_size": 100, "background_sample_size": 10,
        "show_ice_pdp": True, "show_ale": False, "show_shap_summary": False,
        "show_shap_dependence": False, "show_shap_waterfall": False,
        "show_sensitivity_morris": False, "show_sensitivity_sobol": False,
    }

    # No target_columns given (e.g. an older caller) — must not raise, and
    # must still render the one figure present.
    add_interpretability_section(
        elements, figs, styles, str(tmp_path), settings, n_features=7,
    )
    headings = _heading_texts(elements, {"CustomSubheading"})
    assert ("CustomSubheading", "Interpretability Analysis for Residual Motor Torque") in headings


# ── add_hpo_summary_section: per-model breakdown model-name union ───────────

def test_hpo_breakdown_includes_a_model_that_only_the_second_method_has_results_for():
    """Regression test for a real risk: the per-model breakdown listed models
    from only the FIRST method with results — a model failing under that
    method (e.g. a hyperopt import failure) but succeeding under another was
    absent from the whole section."""
    from phoenix_ml.report_generation import add_hpo_summary_section

    hpo_metrics = {
        "random": {"LGBM Regressor": {"T": {"Q^2": 0.9, "elapsed_time": 1.0}}},
        # "hyperopt" produced no data for LGBM at all (empty dict) -> not "active".
        "hyperopt": {},
        # skopt succeeded for a DIFFERENT model that random never even ran.
        "skopt": {"MLP Regressor": {"T": {"Q^2": 0.8, "elapsed_time": 2.0}}},
    }
    hpo_params = {
        "random": {"LGBM Regressor": {"T": {"n_estimators": 100}}},
        "hyperopt": {},
        "skopt": {"MLP Regressor": {"T": {"alpha": 0.01}}},
    }
    hpo_times = {"random": {}, "hyperopt": {}, "skopt": {}}

    elements, styles = [], _report_styles()
    add_hpo_summary_section(
        elements, styles, hpo_metrics, hpo_params, hpo_times, {},
        methods_used=["random", "hyperopt", "skopt"], metric_used="Q^2",
        sampling_method="Sobol", sample_size=1000, n_iter=10, evals=10, calls=10,
        n_jobs=1, best_models_per_target={},
    )

    subheadings = {text for _, text in _heading_texts(elements, {"CustomSubheading"})}
    assert "LGBM Regressor" in subheadings
    assert "MLP Regressor" in subheadings


# ── add_postprocessing_section: Best Transformation Normality Metrics ───────

def _normality_transformation_df():
    import pandas as pd
    return pd.DataFrame({
        "Target Variable": ["T"], "Model": ["LGBM"], "Transformation": ["None"],
        "Skewness": [0.1], "Excess Kurtosis": [0.1], "AD Statistic": [0.5],
        "Shapiro-Wilk p": [0.9], "Lilliefors p": [0.8], "Filiben": [0.95],
    })


def test_normality_tests_empty_list_shows_zero_columns_not_the_defaults(tmp_path):
    """Regression test for a real bug: `normality_tests or DEFAULT_NORMALITY_TESTS`
    treats an explicit "I unchecked every test" ([]) the same as "not specified"
    (None) because an empty list is falsy in Python -- so unchecking every
    checkbox in the UI silently brought back Shapiro-Wilk/Lilliefors/Filiben
    anyway. An empty list must now mean exactly zero columns (no table at all,
    since there'd be nothing to show), while None still falls back to the
    defaults."""
    from phoenix_ml.report_generation import add_postprocessing_section

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {"transformation_df": _normality_transformation_df()},
        image_output_dir=str(tmp_path),
        normality_tests=[],
    )
    headings = [t for _, t in _heading_texts(elements, {"CustomSubheading"})]
    assert "Best Transformation Normality Metrics" not in headings


def test_normality_tests_none_falls_back_to_defaults(tmp_path):
    from phoenix_ml.report_generation import add_postprocessing_section

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {"transformation_df": _normality_transformation_df()},
        image_output_dir=str(tmp_path),
        normality_tests=None,
    )
    headings = [t for _, t in _heading_texts(elements, {"CustomSubheading"})]
    assert "Best Transformation Normality Metrics" in headings


def test_show_normality_metrics_false_hides_the_table_even_with_tests_selected(tmp_path):
    """The master toggle must be able to hide the whole table independent of
    which individual tests are selected -- unlike relying on an empty
    normality_tests list (which is really "zero columns", a different thing
    from "no table")."""
    from phoenix_ml.report_generation import add_postprocessing_section

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {"transformation_df": _normality_transformation_df()},
        image_output_dir=str(tmp_path),
        normality_tests=["Shapiro-Wilk"],
        show_normality_metrics=False,
    )
    headings = [t for _, t in _heading_texts(elements, {"CustomSubheading"})]
    assert "Best Transformation Normality Metrics" not in headings


# ── add_postprocessing_section: intro reflects what actually ran ────────────

def test_postprocessing_intro_only_lists_sections_actually_present(tmp_path):
    """Regression test: the section's intro used to unconditionally name all
    four sub-analyses (cross-validation, influential points, residual
    statistical tests, residual transformations) regardless of which ones
    actually ran -- misleading a reader who only enabled a subset. It must
    list only what's actually present in postprocessing_results, gated the
    same way each subsection gates itself further down."""
    import pandas as pd
    from phoenix_ml.report_generation import add_postprocessing_section

    transformation_df = pd.DataFrame({
        "Target Variable": ["T"], "Model": ["LGBM"], "Transformation": ["None"],
        "Skewness": [0.1], "Excess Kurtosis": [0.1], "AD Statistic": [0.5],
        "Shapiro-Wilk p": [0.9],
    })

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {"transformation_df": transformation_df},
        image_output_dir=str(tmp_path),
    )

    paragraph_texts = [getattr(el, "text", "") for el in elements
                       if type(el).__name__ == "Paragraph"]
    included_line = next((t for t in paragraph_texts if t.startswith("Included below")), None)
    assert included_line is not None, "expected an 'Included below: ...' summary line"
    assert "residual transformations" in included_line
    assert "cross-validation" not in included_line
    assert "influential points" not in included_line
    assert "residual statistical tests" not in included_line


# ── add_postprocessing_section: lambda curve placement + reference table ────

def test_lambda_curve_appears_right_after_summary_table_with_arcsinh_row(tmp_path):
    """Report-review feedback: (1) the λ optimisation curve used to render
    after the Q-Q diagnostic plots; it belongs right after the Residual
    Transformation Summary table instead, since it explains that table's
    Yeo-Johnson row. (2) it must render as one plot beside its own reference
    table per target (not a combined table below every plot), with no
    "Target Variable" column (redundant once the table is target-specific)
    and no leftover "Only shown when..." caveat text. (3) only the row for
    whichever transform was actually selected as best overall is highlighted
    -- not the Yeo-Johnson optimum and Arcsinh unconditionally, regardless of
    which one is actually lower."""
    import pandas as pd
    from phoenix_ml.report_generation import add_postprocessing_section

    # Yeo-Johnson has the lowest AD Statistic and None fails normality, so
    # Yeo-Johnson must be the one selected as best -- Arcsinh (AD=0.25, worse
    # than Yeo-Johnson's 0.2 but better than None's 0.9) must NOT be
    # highlighted even though it's a real, actually-evaluated candidate.
    transformation_df = pd.DataFrame({
        "Target Variable": ["T", "T", "T"],
        "Model": ["LGBM", "LGBM", "LGBM"],
        "Transformation": ["None", "Yeo-Johnson", "Arcsinh"],
        "Lambda": [None, 1.1, None],
        "Skewness": [0.3, 0.1, 0.1],
        "Excess Kurtosis": [0.3, 0.1, 0.1],
        "AD Statistic": [0.9, 0.2, 0.25],
        "Shapiro-Wilk p": [0.01, 0.5, 0.4],
    })
    lambda_reference_df = pd.DataFrame([
        {"Target Variable": "T", "Transform": "Log", "Lambda": 0.0, "AD Statistic": 0.6},
        {"Target Variable": "T", "Transform": "Identity", "Lambda": 1.0, "AD Statistic": 0.4},
        {"Target Variable": "T", "Transform": "Yeo-Johnson (optimum)", "Lambda": 1.1, "AD Statistic": 0.2},
    ])

    def fig():
        f, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        return f

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {
            "transformation_df": transformation_df,
            "lambda_reference_df": lambda_reference_df,
            "lambda_curve_figs": {"T": fig()},
            "transformation_figs": {"residual": fig(), "qq": fig()},
        },
        image_output_dir=str(tmp_path),
    )

    headings = _heading_texts(elements, {"CustomSubheading"})
    heading_texts = [text for _, text in headings]
    lambda_idx = heading_texts.index("Yeo-Johnson λ Optimisation Curve")
    diagnostic_idx = heading_texts.index("Transformed Residual Diagnostic Plots")
    assert lambda_idx < diagnostic_idx, "λ curve must render before the diagnostic plots, not after"

    all_text = " ".join(getattr(el, "text", "") for el in elements)
    assert "Only shown when" not in all_text

    # The plot+table row is a 2-column outer Table whose second cell is
    # itself a nested Table (the per-target reference table).
    row_tables = [el for el in elements if type(el).__name__ == "Table"
                 and len(getattr(el, "_cellvalues", [])) == 1
                 and type(el._cellvalues[0][1]).__name__ == "Table"]
    assert len(row_tables) == 1, "expected exactly one image+table row (one target)"
    ref_table = row_tables[0]._cellvalues[0][1]

    cell_text = lambda cell: getattr(cell, "text", cell)
    header = [cell_text(c) for c in ref_table._cellvalues[0]]
    assert "Target Variable" not in header, "target column is redundant in a per-target table"
    transform_col = header.index("Transform")
    transform_values = [cell_text(row[transform_col]) for row in ref_table._cellvalues[1:]]
    assert "Log" in transform_values
    assert "Identity" in transform_values
    assert "Yeo-Johnson (optimum)" in transform_values

    # Exactly one row highlighted green (excluding the header's own dark
    # background), and it must be Yeo-Johnson's, not Arcsinh's (Arcsinh isn't
    # even in this reference table's rows, but the highlight logic must key
    # off the actual winner regardless).
    import reportlab.lib.colors as rl_colors
    highlighted_rows = {r for (_, (_, r), (_, r2), colour) in ref_table._bkgrndcmds
                        if r == r2 and colour == rl_colors.lightgreen}
    assert len(highlighted_rows) == 1
    highlighted_transform = cell_text(ref_table._cellvalues[list(highlighted_rows)[0]][transform_col])
    assert highlighted_transform == "Yeo-Johnson (optimum)"
    assert "Arcsinh" in transform_values


# ── add_postprocessing_section: all-NaN AD Statistic group ──────────────────

def test_transformation_table_survives_a_target_with_no_valid_ad_statistic(tmp_path):
    """Regression test for a real bug: groupby(...).idxmin() on AD Statistic
    assumed at least one non-NaN value per target — a degenerate residual
    target failing every normality fit made that target's group all-NaN,
    raising ValueError and taking down the whole table (every target, not
    just the degenerate one)."""
    import pandas as pd
    from phoenix_ml.report_generation import add_postprocessing_section

    transformation_df = pd.DataFrame({
        "Target Variable": ["Good Target", "Good Target", "Bad Target", "Bad Target"],
        "Model": ["LGBM", "LGBM", "LGBM", "LGBM"],
        "Transformation": ["None", "Log", "None", "Log"],
        "Skewness": [0.1, 0.05, float("nan"), float("nan")],
        "Excess Kurtosis": [0.2, 0.1, float("nan"), float("nan")],
        "AD Statistic": [0.5, 0.3, float("nan"), float("nan")],
        "Shapiro-Wilk p": [0.6, 0.7, float("nan"), float("nan")],
    })

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {"transformation_df": transformation_df},
        image_output_dir=str(tmp_path),
    )  # must not raise

    tables = [el for el in elements if type(el).__name__ == "Table"]
    assert tables, "transformation table must still render for the healthy target"


def test_transformation_selection_agrees_between_table_and_metrics_section(tmp_path):
    """Regression test: the main table's highlighted row and the "Best
    Transformation Normality Metrics" section below it used to each run their
    own idxmin() independently -- now both call the same
    select_best_transformation_indices(), so they can no longer disagree on
    which transformation is "best" for a given target. Also exercises the
    parsimony gate end-to-end: "None" has the higher AD statistic but already
    passes Shapiro-Wilk, so it must be the one selected, not "Log"."""
    import pandas as pd
    from phoenix_ml.report_generation import add_postprocessing_section

    transformation_df = pd.DataFrame({
        "Target Variable": ["T", "T"],
        "Model": ["LGBM", "LGBM"],
        "Transformation": ["None", "Log"],
        "Skewness": [0.1, 0.05],
        "Excess Kurtosis": [0.2, 0.1],
        "AD Statistic": [0.5, 0.1],
        "Shapiro-Wilk p": [0.9, 0.01],
    })

    elements, styles = [], _report_styles()
    add_postprocessing_section(
        elements, styles,
        {"transformation_df": transformation_df},
        image_output_dir=str(tmp_path),
    )

    tables = [el for el in elements if type(el).__name__ == "Table"]
    assert len(tables) == 2, "expected the main transformation table plus the metrics table"
    main_table, metrics_table = tables

    def _cell_text(cell):
        # Main table cells are plain strings; the metrics table wraps each
        # cell in a Paragraph.
        return getattr(cell, "text", cell)

    def _col(table, name):
        header = [_cell_text(c) for c in table._cellvalues[0]]
        return header.index(name)

    def _rows(table, col):
        return [_cell_text(row[col]) for row in table._cellvalues[1:]]

    # The main table lists every candidate transformation (unfiltered) — both
    # rows must still be present regardless of which one is "best".
    main_col = _col(main_table, "Transformation")
    assert set(_rows(main_table, main_col)) == {"None", "Log"}

    # The metrics table is filtered to the selected-best row per target — this
    # is where the parsimony gate's outcome is directly observable: "None"
    # already passes Shapiro-Wilk (p=0.9), so it must win over "Log" despite
    # "Log" having the lower AD statistic.
    metrics_col = _col(metrics_table, "Transformation")
    assert _rows(metrics_table, metrics_col) == ["None"]


def _report_styles():
    _, _, styles, _, _ = init_pdf_report(
        filename="styles_only.pdf", output_dir=tempfile.mkdtemp(), title="Test",
        font_name="Helvetica", font_size=10, title_font_size=20, heading_font_size=14,
    )
    return styles
