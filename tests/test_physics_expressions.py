"""Tests for physics_expressions.py — the safe expression DSL users author
physics equations in.

The validator tests treat the AST whitelist as a security control, not just a
correctness check: parse_expression() is the only barrier between a text field
in the GUI and evaluated code, so every rejected construct here is an escape
route being held shut.
"""
import numpy as np
import pandas as pd
import pytest

from phoenix_ml.physics_expressions import (
    ExpressionError,
    apply_expressions,
    expression_to_latex,
    extract_measured_mapping,
    extract_reconstruction_mapping,
    parse_expression,
    select_output_columns,
)


# ── Validator as a security control ───────────────────────────────────────────

@pytest.mark.parametrize("expression", [
    # Attribute access — the classic sandbox escape route (obj.__class__...).
    "out = x.__class__",
    "out = ().__class__",
    # Subscripting.
    "out = x[0]",
    # Calling something that isn't a whitelisted function name.
    "out = __import__('os')",
    "out = eval('1')",
    "out = exec('1')",
    "out = open('secrets.txt')",
    # Attribute-call form (func is not a bare Name).
    "out = np.loadtxt('f')",
    # Lambdas and comprehensions.
    "out = (lambda: 1)()",
    "out = [i for i in x]",
    # String constants (blocks string-smuggling into any future function).
    "out = 'text'",
    # f-strings.
    "out = f'{x}'",
    # Unsupported operators.
    "out = x % 2",       # Mod
    "out = x // 2",      # FloorDiv
    "out = x @ y",       # MatMult
    "out = x | y",       # BitOr
    "out = x if y else 0",  # ternary
    "out = x == y",      # comparison
])
def test_validator_rejects_dangerous_or_unsupported_constructs(expression):
    with pytest.raises(ExpressionError):
        parse_expression(expression)


def test_statements_cannot_reach_the_validator():
    # mode="eval" parsing makes statements a syntax error long before
    # evaluation — and that syntax error must surface as ExpressionError.
    with pytest.raises(ExpressionError):
        parse_expression("out = import os")
    with pytest.raises(ExpressionError):
        parse_expression("out = x; y")


def test_whitelisted_functions_and_operators_are_accepted():
    # The flip side of the security tests: the whitelist must not be so tight
    # that legitimate physics expressions fail.
    for expression in [
        "out = sqrt(x) + exp(y) * 2",
        "out = sin(x) ** 2 + cos(x) ** 2",
        "out = atan2(y, x)",
        "out = -x + abs(y) / pi",
        "out = gradient(`Motor Speed`)",
    ]:
        lhs, tree, name_map = parse_expression(expression)
        assert lhs == "out"


def test_function_arity_is_enforced():
    with pytest.raises(ExpressionError):
        parse_expression("out = sqrt(x, y)")     # 1-arg func given 2
    with pytest.raises(ExpressionError):
        parse_expression("out = atan2(x)")       # 2-arg func given 1


def test_huge_literal_exponent_is_rejected():
    """Regression test for a real risk: `2 ** 999999999` between bare
    constants has no magnitude bound — int**int is arbitrary precision, so
    this computes an unbounded-size Python bignum (not just slow — can
    exhaust memory) rather than erroring quickly."""
    with pytest.raises(ExpressionError, match="Exponent"):
        parse_expression("out = 2 ** 999999999")


def test_huge_negative_literal_exponent_is_not_blocked():
    # A negative int exponent falls back to float division (base**exp
    # computed as a normal float op, no bignum growth) — not the risk this
    # guard targets, and blocking it would reject a legitimate tiny-value
    # expression like `x ** -1000`.
    parse_expression("out = 2 ** -999999999")


def test_moderate_and_variable_exponents_are_still_accepted():
    # The bound only applies to literal constant exponents far outside any
    # real physics use — small literals and variable exponents (whose
    # runtime magnitude can't be checked statically) must pass through.
    parse_expression("out = x ** 2")
    parse_expression("out = 2 ** x")
    parse_expression("out = x ** y")


# ── apply_expressions ─────────────────────────────────────────────────────────

def test_apply_expressions_chains_columns_created_by_earlier_expressions():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result, log = apply_expressions(df, [
        "b = a * 2",
        "c = b + a",   # references the column created one step earlier
    ])
    assert list(result["b"]) == [2.0, 4.0, 6.0]
    assert list(result["c"]) == [3.0, 6.0, 9.0]
    assert len(log) == 2


def test_apply_expressions_unknown_variable_names_the_offender_and_position():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ExpressionError, match=r"Expression 2 .*not_a_column"):
        apply_expressions(df, ["b = a * 2", "c = not_a_column + 1"])


def test_apply_expressions_handles_backticked_column_names():
    df = pd.DataFrame({"Motor Speed": [10.0, 20.0], "dw/dt": [1.0, 2.0]})
    result, _ = apply_expressions(df, ["out = `Motor Speed` - 2 * `dw/dt`"])
    assert list(result["out"]) == [8.0, 16.0]


def test_apply_expressions_overwrites_existing_column_in_place():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    result, _ = apply_expressions(df, ["a = a * 10"])
    assert list(result["a"]) == [10.0, 20.0]
    assert list(result.columns) == ["a"]


# ── PERL mapping extraction ───────────────────────────────────────────────────

def test_reconstruction_and_measured_mappings_split_residual_definitions():
    # "Residual = `Measured Col` - physics_est" is the PERL residual pattern:
    # reconstruction map points at the physics estimate (bare identifier),
    # measured map points at the dataset column (backticked or bare).
    exprs = ["`Residual Speed` = `Motor Speed` - speed_est"]
    assert extract_reconstruction_mapping(exprs) == {"Residual Speed": "speed_est"}
    assert extract_measured_mapping(exprs) == {"Residual Speed": "Motor Speed"}


def test_backticked_right_operand_is_not_treated_as_physics_estimate():
    # A backticked name on the right is a dataset column, not an intermediate
    # physics estimate — no reconstruction mapping should be extracted.
    exprs = ["`Residual Speed` = `Motor Speed` - `Other Column`"]
    assert extract_reconstruction_mapping(exprs) == {}


def test_mapping_extraction_raises_instead_of_silently_dropping_a_bad_expression():
    """Regression test for a real bug: extract_reconstruction_mapping and
    extract_measured_mapping silently `continue` past any expression that
    fails to parse — callers that never separately validate the same texts
    (the Save Configuration path in the UI does not) got an incomplete
    mapping with no indication anything was wrong. A genuine parse error must
    now name the broken expression instead of vanishing."""
    exprs = ["`Residual Speed` = `Motor Speed` - speed_est", "bad = ("]
    with pytest.raises(ExpressionError, match=r"Expression 2 .*bad = \("):
        extract_reconstruction_mapping(exprs)
    with pytest.raises(ExpressionError, match=r"Expression 2"):
        extract_measured_mapping(exprs)


def test_mapping_extraction_still_skips_non_residual_expressions_silently():
    # The structural skip (expression isn't of the "a - b" residual shape) is
    # intentional selectivity, not an error — must remain silent.
    exprs = ["other = a + b"]
    assert extract_reconstruction_mapping(exprs) == {}
    assert extract_measured_mapping(exprs) == {}


# ── LaTeX preview ─────────────────────────────────────────────────────────────

def test_backticked_names_render_their_real_column_name_in_latex():
    # The parser swaps `Motor Speed` for a placeholder like __bt0__ before
    # AST parsing; the LaTeX renderer must translate the placeholder BACK to
    # the real column name — a leaked __bt0__ in the preview would be
    # meaningless to the user.
    lhs, tree, name_map = parse_expression("out = `Motor Speed` - 2 * `dw/dt`")
    latex = expression_to_latex(lhs, tree, name_map)
    assert "__bt" not in latex
    # Spaces are escaped for LaTeX math mode; the slash passes through.
    assert r"Motor\ Speed" in latex
    assert "dw/dt" in latex


def test_backticked_lhs_and_underscores_are_latex_escaped():
    # Underscores are LaTeX subscript operators — an unescaped one would
    # silently typeset "speed_est" as "speed" with subscript "est".
    lhs, tree, name_map = parse_expression("`Residual Speed` = speed_est + 1")
    latex = expression_to_latex(lhs, tree, name_map)
    assert r"Residual\ Speed" in latex
    assert r"speed\_est" in latex


def test_every_validator_accepted_function_renders_to_latex():
    """Anything parse_expression() accepts must also render in the LaTeX
    preview — the validator, evaluator, and renderer are three parallel
    whitelists that must stay in sync. Regression test for a real bug found
    while writing this file: atan2 validated and evaluated fine but crashed
    to_latex with a KeyError, because the two-argument LaTeX table existed but
    was never consulted."""
    from phoenix_ml.physics_expressions import _FUNCS, _FUNCS_2ARG

    for fname in _FUNCS:
        lhs, tree, name_map = parse_expression(f"out = {fname}(x)")
        assert expression_to_latex(lhs, tree, name_map)  # must not raise
    for fname in _FUNCS_2ARG:
        lhs, tree, name_map = parse_expression(f"out = {fname}(y, x)")
        assert expression_to_latex(lhs, tree, name_map)  # must not raise


# ── select_output_columns ─────────────────────────────────────────────────────

def test_select_output_columns_rejects_a_duplicated_name():
    """Regression test for a real risk: a duplicated name in the comma-
    separated list ("a, a, b") silently produced a repeated column in the
    output instead of being rejected."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ExpressionError, match="a"):
        select_output_columns(df, "a, a, b")


def test_select_output_columns_reorders_without_duplicates():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    out = select_output_columns(df, "c, a")
    assert list(out.columns) == ["c", "a"]


# ── physics config JSON round-trip ────────────────────────────────────────────

def test_save_and_load_physics_config_round_trips_special_characters(tmp_path):
    """Expression-mode config: unicode, quotes, backslashes, and spaces in
    expressions/paths/column names must survive a save -> load round trip
    byte-for-byte, since these are exactly the characters real Windows paths
    and backtick-quoted column names introduce."""
    from phoenix_ml.physics_expressions import save_physics_config, load_physics_config

    path = str(tmp_path / "config.json")
    expressions = [
        "`Motor Speed (rpm)` = `Raw Speed` - speed_est",
        "`Résidu Vitesse` = `Vitesse Mesurée \"brute\"` - speed_est",
    ]
    reconstruction_map = {"Motor Speed (rpm)": "speed_est", "Résidu Vitesse": "speed_est"}
    measured_map = {"Motor Speed (rpm)": "Raw Speed", "Résidu Vitesse": 'Vitesse Mesurée "brute"'}
    original_dataset_path = "C:\\Users\\Test's PC\\data\\motor_data.csv"

    save_physics_config(
        path, expressions, output_cols_text="`Motor Speed (rpm)`, `Résidu Vitesse`",
        reconstruction_map=reconstruction_map,
        original_dataset_path=original_dataset_path,
        measured_map=measured_map,
    )
    loaded = load_physics_config(path)

    assert loaded["expressions"] == expressions
    assert loaded["reconstruction_map"] == reconstruction_map
    assert loaded["measured_map"] == measured_map
    assert loaded["original_dataset_path"] == original_dataset_path
    assert loaded["output_columns_text"] == "`Motor Speed (rpm)`, `Résidu Vitesse`"


def test_save_and_load_script_physics_config_round_trips_special_characters(tmp_path):
    """Script-mode config: same round-trip guarantee, plus the constants dict
    (numeric values) and name_mapping (another special-character-bearing
    dict) that only this mode carries."""
    from phoenix_ml.physics_expressions import save_script_physics_config, load_physics_config

    path = str(tmp_path / "script_config.json")
    script_path = "C:\\Users\\Test's PC\\scripts\\motor_physics.py"
    input_vars = ["Input Torque", "Armature Current"]
    output_vars = ["Motor Speed (rpm)"]
    constants = {"k_τ": 0.05, "R (Ω)": 1.2}  # unicode keys, float values
    reconstruction_map = {"Residual Motor Speed": "Motor Speed (rpm)_physics"}
    measured_map = {"Residual Motor Speed": "Motor Speed (rpm)"}
    name_mapping = {"Motor Speed (rpm)": "speed_internal"}

    save_script_physics_config(
        path, script_path=script_path, input_vars=input_vars, output_vars=output_vars,
        constants=constants, reconstruction_map=reconstruction_map,
        measured_map=measured_map, name_mapping=name_mapping, time_col="t (s)",
        original_dataset_path=script_path,
    )
    loaded = load_physics_config(path)

    assert loaded["script_path"] == script_path
    assert loaded["input_vars"] == input_vars
    assert loaded["output_vars"] == output_vars
    assert loaded["constants"] == constants
    assert loaded["reconstruction_map"] == reconstruction_map
    assert loaded["measured_map"] == measured_map
    assert loaded["name_mapping"] == name_mapping
    assert loaded["time_col"] == "t (s)"
    assert loaded["mode"] == "script"
