"""Tests for the Executive Summary's UQ description sentence
(_exec_modelling_group, via report_generation.py).

Regression tests for two real bugs found by the user reviewing a generated
report line by line: (1) the UQ description always said "before"/"after
hyperparameter optimisation" purely from whether uq_before/uq_after were
present, with no awareness of whether HPO actually ran -- so a run with HPO
off entirely still claimed uncertainty was "computed before hyperparameter
optimisation", implying an "after" that never happened. (2) the UQ method
list used a separate, stale static dict (_UQ_METHOD_NAMES, now removed) that
never accounted for the GP Posterior flag, silently omitting it from the
Executive Summary's own sentence even when actually enabled and shown
throughout the rest of the report.
"""
import pandas as pd

from phoenix_ml.report_generation import _exec_dataset_group, _exec_modelling_group, init_pdf_report


def _styles():
    _, _, styles, _, _ = init_pdf_report(filename="unused.pdf", output_dir="scratch_unused_styles")
    return styles


def _uq_sentence(elements):
    for el in elements:
        text = getattr(el, "text", "")
        if "Prediction uncertainty was quantified" in text:
            return text
    return None


def _run(styles, uq_before=None, uq_after=None, uq_settings=None, hpo_ran=False):
    return _exec_modelling_group(
        styles, ["Model A"], None, "Q^2", None, None,
        uq_before, uq_after, uq_settings, None, None, hpo_ran=hpo_ran,
    )


def test_uq_before_only_with_hpo_off_says_default_hyperparameters(tmp_path):
    styles = _styles()
    elements = _run(styles, uq_before=pd.DataFrame({"x": [1]}), uq_after=None,
                    uq_settings={"uq_method": "Both", "confidence_interval": 95}, hpo_ran=False)
    sentence = _uq_sentence(elements)
    assert sentence is not None
    assert "using default hyperparameters" in sentence
    assert "before hyperparameter optimisation" not in sentence
    assert "after hyperparameter optimisation" not in sentence


def test_uq_before_only_with_hpo_on_says_before_hyperparameter_optimisation():
    styles = _styles()
    elements = _run(styles, uq_before=pd.DataFrame({"x": [1]}), uq_after=None,
                    uq_settings={"uq_method": "Both", "confidence_interval": 95}, hpo_ran=True)
    sentence = _uq_sentence(elements)
    assert "before hyperparameter optimisation" in sentence


def test_uq_after_only_says_after_hyperparameter_optimisation():
    # uq_after can only exist if HPO produced results -- hpo_ran is irrelevant here,
    # but pass True since that's the only way uq_after is reachable in practice.
    styles = _styles()
    elements = _run(styles, uq_before=None, uq_after=pd.DataFrame({"x": [1]}),
                    uq_settings={"uq_method": "Both", "confidence_interval": 95}, hpo_ran=True)
    sentence = _uq_sentence(elements)
    assert "after hyperparameter optimisation" in sentence
    assert "before hyperparameter optimisation" not in sentence
    assert "before and after" not in sentence


def test_uq_before_and_after_says_both():
    styles = _styles()
    elements = _run(styles, uq_before=pd.DataFrame({"x": [1]}), uq_after=pd.DataFrame({"x": [1]}),
                    uq_settings={"uq_method": "Both", "confidence_interval": 95}, hpo_ran=True)
    sentence = _uq_sentence(elements)
    assert "before and after hyperparameter optimisation" in sentence


def test_uq_method_list_includes_gp_posterior_when_enabled():
    styles = _styles()
    elements = _run(
        styles, uq_before=pd.DataFrame({"x": [1]}), uq_after=None,
        uq_settings={"uq_method": "Both", "confidence_interval": 95, "include_gp_posterior": True},
        hpo_ran=False,
    )
    sentence = _uq_sentence(elements)
    assert "GP Posterior" in sentence
    assert "Bootstrapping" in sentence
    assert "Conformal" in sentence


def test_uq_method_list_omits_gp_posterior_when_disabled():
    styles = _styles()
    elements = _run(
        styles, uq_before=pd.DataFrame({"x": [1]}), uq_after=None,
        uq_settings={"uq_method": "Both", "confidence_interval": 95, "include_gp_posterior": False},
        hpo_ran=False,
    )
    sentence = _uq_sentence(elements)
    assert "GP Posterior" not in sentence


def test_no_uq_sentence_when_neither_stage_present():
    styles = _styles()
    elements = _run(styles, uq_before=None, uq_after=None, uq_settings=None, hpo_ran=False)
    assert _uq_sentence(elements) is None


def _dataset_sentence(elements):
    for el in elements:
        text = getattr(el, "text", "")
        if "Physics-Enhanced Residual Learning (PERL)" in text:
            return text
    return None


def _minimal_preprocessing_results():
    return {"meta": {
        "dataset_path": "data.csv", "n_rows": 10, "n_cols": 5, "n_features": 4,
        "targets": ["y"], "train_count": 8, "train_prop": 0.8,
        "test_count": 2, "test_prop": 0.2, "split_method": "random",
    }}


def test_perl_mode_label_defaults_to_expression_when_mode_key_missing():
    # Regression test: this line's own missing-key fallback used to disagree with
    # add_perl_section's (this fell through to "Script Mode", add_perl_section fell
    # through to "Expression Mode") -- both must now default to Expression Mode.
    styles = _styles()
    elements = _exec_dataset_group(
        styles, _minimal_preprocessing_results(), "data.csv",
        perl_config={"_config_path": "physics.json"}, cleaning_summary=None,
    )
    sentence = _dataset_sentence(elements)
    assert sentence is not None
    assert "Expression Mode" in sentence
    assert "Script Mode" not in sentence


def test_perl_mode_label_says_script_mode_when_mode_is_script():
    styles = _styles()
    elements = _exec_dataset_group(
        styles, _minimal_preprocessing_results(), "data.csv",
        perl_config={"mode": "script"}, cleaning_summary=None,
    )
    sentence = _dataset_sentence(elements)
    assert "Script Mode" in sentence
    assert "Expression Mode" not in sentence


def test_perl_mode_label_says_expression_mode_when_mode_is_expression():
    styles = _styles()
    elements = _exec_dataset_group(
        styles, _minimal_preprocessing_results(), "data.csv",
        perl_config={"mode": "expression"}, cleaning_summary=None,
    )
    sentence = _dataset_sentence(elements)
    assert "Expression Mode" in sentence
    assert "Script Mode" not in sentence
