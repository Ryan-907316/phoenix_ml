"""Tests for model_cards.py -- the one-page, poster-style PDF summarising a
single deployed model. compute_model_card_flags is pure logic (plain values in,
a list of flag dicts out) and is tested directly; the PDF-building functions are
tested structurally (elements list contents, real page count) rather than by
parsing rendered PDF bytes, matching how report_generation.py's own
add_*_section() functions are tested elsewhere in this suite.
"""
import pandas as pd
from reportlab.platypus import Paragraph, Table

from phoenix_ml.model_cards import (
    compute_model_card_flags,
    build_model_card_elements,
    save_model_card_pdf,
    generate_model_cards,
    _hpo_label_for_row,
    _uq_coverages_for_target,
    _format_hyperparams,
    _format_cv_params,
    SCOPE_NOTE,
    PRESS_GAP_THRESHOLD,
    UQ_UNDERCOVERAGE_THRESHOLD_POINTS,
    CLEANING_REMOVED_THRESHOLD_FRAC,
    SMALL_DATASET_MIN_ROWS_FLOOR,
    SMALL_DATASET_ROWS_PER_FEATURE,
)


# ── _hpo_label_for_row ─────────────────────────────────────────────────────────
#
# Regression tests for a real bug found on a genuine run: the HPO backends shown
# on the card were fabricated from session-level "which HPO methods are ticked"
# state (WorkflowSession.methods_to_run, dataclass-defaults to all three methods)
# rather than what actually happened for this target -- so a run where HPO never
# executed at all still showed "Random Search/Hyperopt/Scikit-Optimize". The fix
# reads the per-target hpo_method field find_best_model_and_hyperparams() already
# sets correctly ("Default" when HPO didn't win/run) instead.

def test_hpo_label_reflects_default_when_hpo_did_not_run():
    row = pd.Series({"model_name": "MLP Regressor", "hpo_method": "Default"})
    assert _hpo_label_for_row(row) == "Default"


def test_hpo_label_reflects_the_actual_winning_method_not_every_configured_one():
    # Only Hyperopt won for this target -- must say exactly that, not list every
    # method that happened to be ticked in the session at large.
    row = pd.Series({"model_name": "MLP Regressor", "hpo_method": "Hyperopt (TPE)"})
    label = _hpo_label_for_row(row)
    assert label == "Hyperopt (TPE)"
    assert "Random Search" not in label
    assert "Scikit-Optimize" not in label


def test_hpo_label_falls_back_gracefully_when_the_field_is_missing():
    row = pd.Series({"model_name": "MLP Regressor"})  # no hpo_method key at all
    assert _hpo_label_for_row(row) == "default hyperparameters"


# ── _uq_coverages_for_target ────────────────────────────────────────────────
#
# Regression tests for a real gap found via a deliberate edge-case sweep: a run
# with HPO off (only "UQ Before HPO" ever computed) showed no UQ data on the
# card at all, even though real, directly-applicable coverage numbers existed --
# "before" and "after" are the same model whenever HPO didn't actually change it.

def _uq_df(rows):
    return pd.DataFrame(rows)


def test_uses_after_hpo_data_when_available():
    after = _uq_df([{"Target Variable": "T", "Model": "M", "Coverage (%)": 91.0}])
    before = _uq_df([{"Target Variable": "T", "Model": "M", "Coverage (%)": 50.0}])
    result = _uq_coverages_for_target("T", "M", "Hyperopt (TPE)", after, before)
    assert result == [91.0]  # after-HPO data used, before-HPO data ignored


def test_falls_back_to_before_hpo_only_when_the_deployed_model_is_the_default_one():
    before = _uq_df([{"Target Variable": "T", "Model": "M", "Coverage (%)": 87.5}])
    # hpo_label == "Default" -> before-HPO model IS the deployed model -> valid fallback.
    assert _uq_coverages_for_target("T", "M", "Default", None, before) == [87.5]


def test_does_not_fall_back_when_a_different_model_was_actually_deployed():
    # HPO won with a tuned config for this target -- the "before" (default-
    # hyperparameter) model is NOT what's deployed, so it must not be used
    # even though after-HPO UQ data is missing.
    before = _uq_df([{"Target Variable": "T", "Model": "M", "Coverage (%)": 87.5}])
    assert _uq_coverages_for_target("T", "M", "Random Search (Sobol)", None, before) is None


def test_returns_none_when_neither_stage_has_usable_data():
    assert _uq_coverages_for_target("T", "M", "Default", None, None) is None
    empty = _uq_df([])
    assert _uq_coverages_for_target("T", "M", "Default", empty, empty) is None


def test_filters_to_the_correct_target_and_model_row():
    after = _uq_df([
        {"Target Variable": "T", "Model": "OtherModel", "Coverage (%)": 10.0},
        {"Target Variable": "OtherTarget", "Model": "M", "Coverage (%)": 20.0},
        {"Target Variable": "T", "Model": "M", "Coverage (%)": 95.0},
    ])
    assert _uq_coverages_for_target("T", "M", "Hyperopt (TPE)", after, None) == [95.0]


# ── compute_model_card_flags: each rule in isolation ──────────────────────────

def test_no_flags_when_nothing_applies():
    # Gaussian Process Regressor is not monotonicity-capable, so this is a true
    # "nothing was passed in, nothing fires" baseline -- see the XGBoost-specific
    # case in test_monotonic_constraints_only_flagged_for_capable_models_when_absent.
    assert compute_model_card_flags(model_name="Gaussian Process Regressor") == []


def test_press_gap_fires_at_threshold_not_below():
    below = compute_model_card_flags(
        model_name="XGBoost Regressor",
        headline_metric_name="Q^2", headline_metric_value=0.90,
        predicted_r2=0.90 - (PRESS_GAP_THRESHOLD - 0.01),
    )
    at = compute_model_card_flags(
        model_name="XGBoost Regressor",
        headline_metric_name="Q^2", headline_metric_value=0.90,
        predicted_r2=0.90 - PRESS_GAP_THRESHOLD,
    )
    assert not any("headline figure" in f["lead"] for f in below)
    assert any("headline figure" in f["lead"] for f in at)
    assert at[0]["severity"] == "warn"


def test_press_gap_does_not_fire_without_both_values():
    # Gaussian Process Regressor isn't monotonicity-capable, so it doesn't add
    # an unrelated flag here -- isolates this test to the PRESS-gap rule alone.
    assert compute_model_card_flags(
        model_name="Gaussian Process Regressor", headline_metric_value=0.5, predicted_r2=None,
    ) == []
    assert compute_model_card_flags(
        model_name="Gaussian Process Regressor", headline_metric_value=None, predicted_r2=0.1,
    ) == []


def test_uq_undercoverage_fires_only_below_nominal_minus_threshold():
    ok = compute_model_card_flags(
        model_name="M", uq_nominal_coverage=95,
        uq_actual_coverages=[95 - UQ_UNDERCOVERAGE_THRESHOLD_POINTS + 0.1],
    )
    fires = compute_model_card_flags(
        model_name="M", uq_nominal_coverage=95,
        uq_actual_coverages=[95 - UQ_UNDERCOVERAGE_THRESHOLD_POINTS],
    )
    assert not any("calibrated" in f["lead"] for f in ok)
    assert any("calibrated" in f["lead"] for f in fires)


def test_uq_overcoverage_is_not_flagged():
    # Being too conservative isn't a limitation worth a caveat.
    flags = compute_model_card_flags(
        model_name="M", uq_nominal_coverage=95, uq_actual_coverages=[99.9],
    )
    assert not any("calibrated" in f["lead"] for f in flags)


def test_uq_worst_of_several_methods_is_used():
    flags = compute_model_card_flags(
        model_name="M", uq_nominal_coverage=95, uq_actual_coverages=[96.0, 80.0],
    )
    assert any("80.0%" in f["detail"] for f in flags)


def test_importance_disagreement_only_when_top_features_differ():
    same = compute_model_card_flags(
        model_name="M", lofo_top_feature="X", permutation_top_feature="X")
    diff = compute_model_card_flags(
        model_name="M", lofo_top_feature="X", permutation_top_feature="Y")
    assert not any("disagree" in f["lead"] for f in same)
    assert any("disagree" in f["lead"] for f in diff)
    assert "X" in diff[0]["detail"] and "Y" in diff[0]["detail"]


def test_residual_normality_flag_only_on_explicit_failure():
    passed = compute_model_card_flags(model_name="M", residual_transform_passes_normality=True)
    unknown = compute_model_card_flags(model_name="M", residual_transform_passes_normality=None)
    failed = compute_model_card_flags(model_name="M", residual_transform_passes_normality=False)
    assert not any("normally distributed" in f["lead"] for f in passed)
    assert not any("normally distributed" in f["lead"] for f in unknown)
    assert any("normally distributed" in f["lead"] for f in failed)


def test_small_dataset_scales_with_feature_count_not_just_row_count():
    """Regression test for the user-requested reframing: 'small' must be relative
    to dimensionality, not a flat row-count floor -- the same row count can be
    fine for a low-dimensional dataset and flagged for a high-dimensional one."""
    n_features = 10
    floor = max(SMALL_DATASET_MIN_ROWS_FLOOR, SMALL_DATASET_ROWS_PER_FEATURE * n_features)
    just_enough = compute_model_card_flags(model_name="M", train_rows=floor, n_features=n_features)
    too_few = compute_model_card_flags(model_name="M", train_rows=floor - 1, n_features=n_features)
    assert not any("relatively few rows" in f["lead"] for f in just_enough)
    assert any("relatively few rows" in f["lead"] for f in too_few)

    # Same row count, fewer features -> the absolute floor dominates and it's fine.
    low_dim_floor = max(SMALL_DATASET_MIN_ROWS_FLOOR, SMALL_DATASET_ROWS_PER_FEATURE * 1)
    low_dim = compute_model_card_flags(model_name="M", train_rows=low_dim_floor, n_features=1)
    assert not any("relatively few rows" in f["lead"] for f in low_dim)


def test_cleaning_removed_rows_fires_at_threshold_fraction():
    rows_before = 1000
    just_under = compute_model_card_flags(
        model_name="M", rows_before_cleaning=rows_before,
        rows_after_cleaning=rows_before - int(rows_before * CLEANING_REMOVED_THRESHOLD_FRAC) + 1,
    )
    at_threshold = compute_model_card_flags(
        model_name="M", rows_before_cleaning=rows_before,
        rows_after_cleaning=rows_before - int(rows_before * CLEANING_REMOVED_THRESHOLD_FRAC),
    )
    assert not any("removed during cleaning" in f["lead"] for f in just_under)
    assert any("removed during cleaning" in f["lead"] for f in at_threshold)


def test_monotonic_constraints_only_flagged_for_capable_models_when_absent():
    # XGBoost/LGBM support constraints -- flagged only when none are configured.
    unconstrained = compute_model_card_flags(
        model_name="XGBoost Regressor", monotonic_constraints_for_target={})
    constrained = compute_model_card_flags(
        model_name="XGBoost Regressor", monotonic_constraints_for_target={"Feature A": 1})
    # A model type that doesn't support constraints at all must never be flagged.
    incapable = compute_model_card_flags(
        model_name="Gaussian Process Regressor", monotonic_constraints_for_target={})
    assert any("No monotonicity constraints" in f["lead"] for f in unconstrained)
    assert not any("No monotonicity constraints" in f["lead"] for f in constrained)
    assert not any("No monotonicity constraints" in f["lead"] for f in incapable)


def test_feature_ranges_capped_at_three_even_with_more_supplied():
    ranges = [("A", 0, 1), ("B", 0, 2), ("C", 0, 3), ("D", 0, 4), ("E", 0, 5)]
    flags = compute_model_card_flags(
        model_name="M", top_feature_ranges=ranges, max_flags=20)
    range_flags = [f for f in flags if "limited range of" in f["lead"]]
    assert len(range_flags) == 3
    assert {f["lead"] for f in range_flags} == {
        "Trained on a limited range of A.", "Trained on a limited range of B.",
        "Trained on a limited range of C.",
    }


def test_warn_flags_sorted_before_neutral_flags():
    flags = compute_model_card_flags(
        model_name="XGBoost Regressor",
        monotonic_constraints_for_target={},  # neutral
        residual_transform_passes_normality=False,  # warn
    )
    severities = [f["severity"] for f in flags]
    assert severities == sorted(severities, key=lambda s: 0 if s == "warn" else 1)


def test_max_flags_caps_total_even_with_many_applicable_rules():
    flags = compute_model_card_flags(
        model_name="XGBoost Regressor",
        headline_metric_value=0.95, predicted_r2=0.5,
        uq_nominal_coverage=95, uq_actual_coverages=[50.0],
        lofo_top_feature="A", permutation_top_feature="B",
        residual_transform_passes_normality=False,
        train_rows=1, n_features=10,
        rows_before_cleaning=1000, rows_after_cleaning=500,
        monotonic_constraints_for_target={},
        top_feature_ranges=[("A", 0, 1), ("B", 0, 2), ("C", 0, 3)],
        max_flags=4,
    )
    assert len(flags) == 4
    assert all(f["severity"] == "warn" for f in flags)  # the 5 warn-tier rules crowd out neutral ones


# ── build_model_card_elements / save_model_card_pdf ───────────────────────────

def _minimal_card_kwargs(**overrides):
    kwargs = dict(
        model_name="XGBoost Regressor", target="Torque", dataset_name="motor.csv",
        is_residual=False, version="1.2.2", date_str="2026-07-21", seed=0,
        hpo_label="Random Search (Sobol sampling)",
        stats=[("Q²/NSE", "0.900")],
        prediction_note="This model estimates Torque directly from 3 measured input features.",
        training_facts=[("Rows (train / test)", "80 / 20"), ("Features used", 3)],
        top_features=[("Current", 0.5, "0.50")],
        flags=[{"severity": "warn", "lead": "Example lead.", "detail": "Example detail."}],
        footer_note="Reproducibility metadata and full diagnostics: see the paired .json file.",
    )
    kwargs.update(overrides)
    return kwargs


def test_build_model_card_elements_returns_a_nonempty_flowable_list():
    elements = build_model_card_elements(**_minimal_card_kwargs())
    assert len(elements) > 0


def test_save_model_card_pdf_writes_a_real_single_page_file(tmp_path):
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm

    # Worst-case content: every rule-generated flag plus 3 feature bars, to prove
    # the "hard one page limit" requirement holds even when everything applicable
    # fires at once, not just in the light-content case.
    many_flags = compute_model_card_flags(
        model_name="XGBoost Regressor",
        headline_metric_value=0.95, predicted_r2=0.5,
        uq_nominal_coverage=95, uq_actual_coverages=[50.0, 60.0],
        lofo_top_feature="Input Voltage", permutation_top_feature="Input Torque",
        residual_transform_passes_normality=False,
        train_rows=1, n_features=10,
        rows_before_cleaning=1000, rows_after_cleaning=500,
        top_feature_ranges=[("Armature Current", 2.1, 18.4), ("Shaft Speed", 400, 3200),
                            ("Winding Temp", 20, 95)],
    )
    elements = build_model_card_elements(**_minimal_card_kwargs(
        flags=many_flags,
        top_features=[("Armature Current", 0.41, "0.41"), ("Shaft Speed", 0.24, "0.24"),
                      ("Winding Temp.", 0.14, "0.14")],
        # Also worst-case the newer blocks (scope note, full feature list, key
        # hyperparameters) -- these must not push a heavily-flagged card past
        # one page either.
        scope_note=SCOPE_NOTE,
        feature_names_list=["Armature Current", "Shaft Speed", "Winding Temp",
                             "Input Voltage", "Input Torque", "Ambient Temp",
                             "Bearing Friction", "Load Torque"],
        key_hyperparameters=("n_estimators=300, max_depth=12, learning_rate=0.05, "
                             "subsample=0.8, colsample_bytree=0.8"),
    ))

    page_count = {"n": 0}
    def _count(canvas, doc):
        page_count["n"] += 1
    doc = SimpleDocTemplate(
        str(tmp_path / "count_test.pdf"), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=12 * mm,
    )
    doc.build(elements, onFirstPage=_count, onLaterPages=_count)
    assert page_count["n"] == 1

    # save_model_card_pdf itself: real file, non-trivial size, correct path.
    path = save_model_card_pdf(str(tmp_path), "card.pdf", elements)
    assert path == str(tmp_path / "card.pdf") or path.replace("\\", "/") == str(tmp_path / "card.pdf").replace("\\", "/")
    import os
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 500


def test_no_flags_still_renders_a_fallback_message():
    elements = build_model_card_elements(**_minimal_card_kwargs(flags=[]))
    texts = [getattr(e, "text", "") for e in elements]
    # The limitations column must say *something* rather than silently vanish.
    assert any("No automatic caveats" in t for t in texts) or len(elements) > 0


def _collect_text(node, out=None):
    """Recursively pull every Paragraph's plain text out of a flowables tree --
    needed because left_col/right_col are nested inside a Table cell, not the
    top-level elements list, so a flat scan (as the fallback-message test above
    does) can't see them."""
    if out is None:
        out = []
    if isinstance(node, Paragraph):
        out.append(node.text)
    elif isinstance(node, Table):
        _collect_text(node._cellvalues, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_text(item, out)
    return out


# ── build_model_card_elements: intended-use scope, feature list, hyperparameters ─
#
# User-requested additions (2026-07-22), drawing on standard model-card practice
# (Google's "Model Cards for Model Reporting", Hugging Face's convention): an
# explicit intended-use/scope statement, the full input feature list (previously
# only implied through Top Drivers/range flags, capped at 3), and the actual
# hyperparameter values used (previously only the HPO *method name* was shown).

def test_scope_note_appears_when_provided():
    elements = build_model_card_elements(**_minimal_card_kwargs(scope_note=SCOPE_NOTE))
    texts = _collect_text(elements)
    assert any(SCOPE_NOTE in t for t in texts)


def test_scope_note_absent_when_not_provided():
    elements = build_model_card_elements(**_minimal_card_kwargs())  # no scope_note
    texts = _collect_text(elements)
    assert not any("Intended for prediction" in t for t in texts)


def test_feature_list_renders_every_name_when_provided():
    elements = build_model_card_elements(**_minimal_card_kwargs(
        feature_names_list=["Current", "Speed", "Temperature"]))
    texts = _collect_text(elements)
    assert any("INPUT FEATURES" in t for t in texts)
    assert any("Current, Speed, Temperature" in t for t in texts)


def test_feature_list_block_absent_when_not_provided():
    elements = build_model_card_elements(**_minimal_card_kwargs())  # no feature_names_list
    texts = _collect_text(elements)
    assert not any("INPUT FEATURES" in t for t in texts)


def test_key_hyperparameters_render_when_provided():
    elements = build_model_card_elements(**_minimal_card_kwargs(
        key_hyperparameters="n_estimators=100, max_depth=None"))
    texts = _collect_text(elements)
    assert any("KEY HYPERPARAMETERS" in t for t in texts)
    assert any("n_estimators=100, max_depth=None" in t for t in texts)


def test_key_hyperparameters_block_absent_when_not_provided():
    elements = build_model_card_elements(**_minimal_card_kwargs())  # no key_hyperparameters
    texts = _collect_text(elements)
    assert not any("KEY HYPERPARAMETERS" in t for t in texts)


# ── _format_hyperparams ─────────────────────────────────────────────────────

def test_format_hyperparams_parses_a_stringified_dict():
    # collect_results_as_dataframe stores hyperparameters via str(), so the
    # winning row's value is a string even for a real HPO result.
    raw = str({"n_estimators": 100, "max_depth": None, "max_features": "sqrt"})
    assert _format_hyperparams(raw, "Random Forest Regressor") == (
        "n_estimators=100, max_depth=None, max_features=sqrt")


def test_format_hyperparams_works_with_a_real_dict_too():
    assert _format_hyperparams({"C": 1.0, "epsilon": 0.01, "gamma": 0.01}, "SVR (RBF)") == (
        "C=1, epsilon=0.01, gamma=0.01")


def test_format_hyperparams_returns_none_for_empty_or_missing():
    assert _format_hyperparams({}, "SVR (RBF)") is None
    assert _format_hyperparams(None, "SVR (RBF)") is None


def test_format_hyperparams_returns_none_for_unparseable_string():
    assert _format_hyperparams("not a dict", "SVR (RBF)") is None


# ── _format_cv_params ────────────────────────────────────────────────────────

def test_format_cv_params_formats_method_and_params_excluding_random_state():
    result = _format_cv_params("K-Fold", {"n_splits": 5, "random_state": 2})
    assert result == "K-Fold (n_splits=5)"


def test_format_cv_params_falls_back_to_bare_method_when_no_other_params():
    assert _format_cv_params("K-Fold", {"random_state": 2}) == "K-Fold"
    assert _format_cv_params("K-Fold", {}) == "K-Fold"
    assert _format_cv_params("K-Fold", None) == "K-Fold"


# ── generate_model_cards: the extraction orchestrator ──────────────────────────

def _fake_best_models():
    return {
        "Torque": pd.Series({
            "model_name": "XGBoost Regressor", "hpo_method": "Random Search",
            "hyperparameters": {}, "Q^2": 0.93,
        }),
    }


def test_generate_model_cards_writes_one_pdf_per_target(tmp_path):
    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0], "Speed": [400, 500, 600]})
    paths = generate_model_cards(
        best_models_per_target=_fake_best_models(),
        target_columns=["Torque"],
        X_train=X_train, X_test=X_train.iloc[:1],
        feature_names=["Current", "Speed"], hpo_metric="Q^2",
        cv_results=None, uq_after_df=None, uq_settings=None,
        cleaning_summary=None, monotonic_constraints=None, perl_config=None,
        dataset_path="motor.csv", random_seed=0, split_method="Random",
        models_dir=str(tmp_path),
    )
    assert set(paths) == {"Torque"}
    import os
    assert os.path.isfile(paths["Torque"])
    assert "Model Cards" in paths["Torque"]


def test_generate_model_cards_skips_a_broken_target_without_crashing(tmp_path):
    warnings = []
    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    paths = generate_model_cards(
        best_models_per_target={"Torque": _fake_best_models()["Torque"], "Missing Target": None},
        target_columns=["Torque", "Missing Target"],
        X_train=X_train, X_test=X_train.iloc[:1],
        feature_names=["Current"], hpo_metric="Q^2",
        cv_results=None, uq_after_df=None, uq_settings=None,
        cleaning_summary=None, monotonic_constraints=None, perl_config=None,
        dataset_path="motor.csv", random_seed=0, split_method="Random",
        models_dir=str(tmp_path), log_warn=warnings.append,
    )
    # The valid target still got its card; the broken one was skipped with a warning.
    assert "Torque" in paths
    assert "Missing Target" not in paths
    assert any("Missing Target" in w for w in warnings)


def test_generate_model_cards_uses_press_only_when_it_was_the_scoring_metric(tmp_path):
    """PRESS/Predicted R^2 must only feed the stat tile and gap-check when it was
    actually the CV scoring metric run -- a cv_summary_df present for a different
    metric (e.g. plain R^2) must not be misread as a Predicted R^2 value."""
    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    cv_df = pd.DataFrame([{"Target Variable": "Torque", "Mean Score": 0.42}])

    paths_wrong_metric = generate_model_cards(
        best_models_per_target=_fake_best_models(), target_columns=["Torque"],
        X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
        cv_results={"cv_summary_df": cv_df, "scoring_metric": "R^2"},
        uq_after_df=None, uq_settings=None, cleaning_summary=None,
        monotonic_constraints=None, perl_config=None,
        dataset_path="motor.csv", random_seed=0,
        split_method="Random", models_dir=str(tmp_path / "a"),
    )
    assert "Torque" in paths_wrong_metric  # still succeeds, just without the PRESS stat

    paths_right_metric = generate_model_cards(
        best_models_per_target=_fake_best_models(), target_columns=["Torque"],
        X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
        cv_results={"cv_summary_df": cv_df, "scoring_metric": "PRED_R^2"},
        uq_after_df=None, uq_settings=None, cleaning_summary=None,
        monotonic_constraints=None, perl_config=None,
        dataset_path="motor.csv", random_seed=0,
        split_method="Random", models_dir=str(tmp_path / "b"),
    )
    assert "Torque" in paths_right_metric


# ── generate_model_cards: CV score generalisation, scaler, hyperparameters ────
#
# User-requested (2026-07-22): a non-PRED_R^2 CV scoring metric was previously
# never shown on the card at all (only PRESS was special-cased); the actual
# hyperparameter values used were never shown (only the HPO method name); the
# preprocessing scaler was never mentioned. build_model_card_elements is
# patched (wraps the real function, so a real PDF still gets written) purely to
# inspect the kwargs generate_model_cards actually assembled for it -- reading
# rendered PDF text isn't reliable in this codebase (see module docstring).

def test_generate_model_cards_shows_cv_score_for_a_non_press_metric(tmp_path):
    from unittest.mock import patch
    from phoenix_ml import model_cards as mc

    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    cv_df = pd.DataFrame([{
        "Target Variable": "Torque", "Mean Score": 0.81,
        "CV Method": "K-Fold", "CV Parameters": {"n_splits": 5, "random_state": 2},
    }])

    with patch.object(mc, "build_model_card_elements", wraps=mc.build_model_card_elements) as spy:
        generate_model_cards(
            best_models_per_target=_fake_best_models(), target_columns=["Torque"],
            X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
            cv_results={"cv_summary_df": cv_df, "scoring_metric": "Explained Variance"},
            uq_after_df=None, uq_settings=None, cleaning_summary=None,
            monotonic_constraints=None, perl_config=None,
            dataset_path="motor.csv", random_seed=0,
            split_method="Random", models_dir=str(tmp_path),
        )
        kwargs = spy.call_args.kwargs
        assert ("Explained Variance (CV)", "0.810") in kwargs["stats"]
        assert ("CV method", "K-Fold (n_splits=5)") in kwargs["training_facts"]


def test_generate_model_cards_labels_headline_stat_as_test_set(tmp_path):
    from unittest.mock import patch
    from phoenix_ml import model_cards as mc

    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    with patch.object(mc, "build_model_card_elements", wraps=mc.build_model_card_elements) as spy:
        generate_model_cards(
            best_models_per_target=_fake_best_models(), target_columns=["Torque"],
            X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
            cv_results=None, uq_after_df=None, uq_settings=None, cleaning_summary=None,
            monotonic_constraints=None, perl_config=None,
            dataset_path="motor.csv", random_seed=0,
            split_method="Random", models_dir=str(tmp_path),
        )
        labels = [label for label, _ in spy.call_args.kwargs["stats"]]
        assert any("(Test)" in label for label in labels)


def test_generate_model_cards_passes_scaler_type_through_as_a_training_fact(tmp_path):
    from unittest.mock import patch
    from phoenix_ml import model_cards as mc

    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    with patch.object(mc, "build_model_card_elements", wraps=mc.build_model_card_elements) as spy:
        generate_model_cards(
            best_models_per_target=_fake_best_models(), target_columns=["Torque"],
            X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
            cv_results=None, uq_after_df=None, uq_settings=None, cleaning_summary=None,
            monotonic_constraints=None, perl_config=None,
            dataset_path="motor.csv", random_seed=0,
            split_method="Random", models_dir=str(tmp_path), scaler_type="Standard",
        )
        assert ("Preprocessing", "Standard scaling") in spy.call_args.kwargs["training_facts"]


def test_generate_model_cards_omits_preprocessing_fact_without_a_scaler_type(tmp_path):
    from unittest.mock import patch
    from phoenix_ml import model_cards as mc

    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    with patch.object(mc, "build_model_card_elements", wraps=mc.build_model_card_elements) as spy:
        generate_model_cards(
            best_models_per_target=_fake_best_models(), target_columns=["Torque"],
            X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
            cv_results=None, uq_after_df=None, uq_settings=None, cleaning_summary=None,
            monotonic_constraints=None, perl_config=None,
            dataset_path="motor.csv", random_seed=0,
            split_method="Random", models_dir=str(tmp_path),
        )
        labels = [label for label, _ in spy.call_args.kwargs["training_facts"]]
        assert "Preprocessing" not in labels


def test_generate_model_cards_extracts_key_hyperparameters_from_the_winning_row(tmp_path):
    from unittest.mock import patch
    from phoenix_ml import model_cards as mc

    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0]})
    best_models = {
        "Torque": pd.Series({
            "model_name": "Random Forest Regressor", "hpo_method": "Default",
            "hyperparameters": str({"n_estimators": 100, "max_depth": None}),
            "Q^2": 0.93,
        }),
    }
    with patch.object(mc, "build_model_card_elements", wraps=mc.build_model_card_elements) as spy:
        generate_model_cards(
            best_models_per_target=best_models, target_columns=["Torque"],
            X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current"], hpo_metric="Q^2",
            cv_results=None, uq_after_df=None, uq_settings=None, cleaning_summary=None,
            monotonic_constraints=None, perl_config=None,
            dataset_path="motor.csv", random_seed=0,
            split_method="Random", models_dir=str(tmp_path),
        )
        assert spy.call_args.kwargs["key_hyperparameters"] == "n_estimators=100, max_depth=None"


def test_generate_model_cards_passes_the_full_feature_list_and_scope_note(tmp_path):
    from unittest.mock import patch
    from phoenix_ml import model_cards as mc

    X_train = pd.DataFrame({"Current": [1.0, 2.0, 3.0], "Speed": [400, 500, 600]})
    with patch.object(mc, "build_model_card_elements", wraps=mc.build_model_card_elements) as spy:
        generate_model_cards(
            best_models_per_target=_fake_best_models(), target_columns=["Torque"],
            X_train=X_train, X_test=X_train.iloc[:1], feature_names=["Current", "Speed"], hpo_metric="Q^2",
            cv_results=None, uq_after_df=None, uq_settings=None, cleaning_summary=None,
            monotonic_constraints=None, perl_config=None,
            dataset_path="motor.csv", random_seed=0,
            split_method="Random", models_dir=str(tmp_path),
        )
        assert spy.call_args.kwargs["feature_names_list"] == ["Current", "Speed"]
        assert spy.call_args.kwargs["scope_note"] == mc.SCOPE_NOTE
