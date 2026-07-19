# physics_expressions.py
# Safe, beginner-friendly math-expression engine for the Physics Modelling "Expression Mode".
# Each expression has the form  NewColumn = expression over existing dataset columns,
# parsed with Python's ast module against a strict whitelist (no eval() of arbitrary code)
# and evaluated vectorised (numpy/pandas) over a DataFrame. Expressions are applied in
# list order, so later expressions may reference columns created by earlier ones.
#
# Column names containing spaces or symbols (e.g. "Motor Speed", "dw/dt") are not valid
# Python identifiers, so they must be wrapped in backticks, e.g. `Motor Speed` — the same
# convention pandas uses in DataFrame.query()/eval().

from __future__ import annotations
import ast
import operator
import re
import numpy as np
import pandas as pd
from scipy.special import erf as _erf
from typing import Dict, List, Tuple

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCS = {
    # Roots / exponentials
    "sqrt":     np.sqrt,
    "exp":      np.exp,
    # Logarithms
    "ln":       np.log,        # natural log (alias for log)
    "log":      np.log,        # natural log
    "log2":     np.log2,
    "log10":    np.log10,
    # Trigonometric
    "sin":      np.sin,
    "cos":      np.cos,
    "tan":      np.tan,
    # Inverse trigonometric
    "asin":     np.arcsin,
    "acos":     np.arccos,
    "atan":     np.arctan,
    # Hyperbolic
    "sinh":     np.sinh,
    "cosh":     np.cosh,
    "tanh":     np.tanh,
    # Misc
    "abs":      np.abs,
    "floor":    np.floor,
    "ceil":     np.ceil,
    "gradient": np.gradient,   # finite-difference d(x)/d(row index)
    # Special functions
    "erf":      _erf,          # Gauss error function — useful for cumulative normal distributions
}

# Two-argument functions (atan2 only for now — separate from _FUNCS to keep validator clean)
_FUNCS_2ARG = {
    "atan2": np.arctan2,       # atan2(y, x) — full-circle arctangent, avoids quadrant ambiguity
}

_LATEX_FUNCS = {
    "sqrt":     lambda a: rf"\sqrt{{{a}}}",
    "exp":      lambda a: rf"e^{{{a}}}",
    "ln":       lambda a: rf"\ln\!\left({a}\right)",
    "log":      lambda a: rf"\ln\!\left({a}\right)",
    "log2":     lambda a: rf"\log_{{2}}\!\left({a}\right)",
    "log10":    lambda a: rf"\log_{{10}}\!\left({a}\right)",
    "sin":      lambda a: rf"\sin\!\left({a}\right)",
    "cos":      lambda a: rf"\cos\!\left({a}\right)",
    "tan":      lambda a: rf"\tan\!\left({a}\right)",
    "asin":     lambda a: rf"\arcsin\!\left({a}\right)",
    "acos":     lambda a: rf"\arccos\!\left({a}\right)",
    "atan":     lambda a: rf"\arctan\!\left({a}\right)",
    "sinh":     lambda a: rf"\sinh\!\left({a}\right)",
    "cosh":     lambda a: rf"\cosh\!\left({a}\right)",
    "tanh":     lambda a: rf"\tanh\!\left({a}\right)",
    "abs":      lambda a: rf"\left|{a}\right|",
    "floor":    lambda a: rf"\lfloor {a} \rfloor",
    "ceil":     lambda a: rf"\lceil {a} \rceil",
    "gradient": lambda a: rf"\frac{{d}}{{dt}}\!\left({a}\right)",
    "erf":      lambda a: rf"\mathrm{{erf}}\!\left({a}\right)",
}

_LATEX_FUNCS_2ARG = {
    "atan2": lambda a, b: rf"\mathrm{{atan2}}\!\left({a},\,{b}\right)",
}

# Recognised symbolic constants (appear as Name nodes, not function calls)
_CONSTANTS = {"pi": np.pi}

_BACKTICK_RE = re.compile(r"`([^`]+)`")

# Generous for any real physics equation; blocks a typo like `2 ** 999999999`
# from computing an unbounded-magnitude Python bignum (int**int is arbitrary
# precision, so this isn't just slow — it can exhaust memory before Python's
# int-to-str guard ever gets a chance to kick in).
_MAX_POW_EXPONENT = 1000


class ExpressionError(ValueError):
    pass


def _extract_backticked(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace each `Column Name` with a safe placeholder identifier; return (text, map)."""
    name_map: Dict[str, str] = {}
    counter = [0]

    def _sub(m):
        placeholder = f"__bt{counter[0]}__"
        counter[0] += 1
        name_map[placeholder] = m.group(1)
        return placeholder

    return _BACKTICK_RE.sub(_sub, text), name_map


def _parse_lhs(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("`") and raw.endswith("`") and len(raw) >= 2:
        return raw[1:-1].strip()
    if not raw.isidentifier():
        raise ExpressionError(
            f"'{raw}' is not a valid column name — wrap names with spaces or symbols in "
            f"backticks, e.g. `Motor Speed`")
    return raw


def parse_expression(text: str) -> Tuple[str, ast.Expression, Dict[str, str]]:
    """Split 'LHS = RHS' and parse RHS into a validated AST. Raises ExpressionError."""
    if "=" not in text:
        raise ExpressionError("Expected the form:  NewColumn = expression")
    lhs_raw, _, rhs_raw = text.partition("=")
    lhs = _parse_lhs(lhs_raw)
    rhs_raw = rhs_raw.strip()
    if not rhs_raw:
        raise ExpressionError("Right-hand side is empty")

    rhs_sub, name_map = _extract_backticked(rhs_raw)
    try:
        tree = ast.parse(rhs_sub, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Syntax error: {e.msg}")
    _validate(tree.body)
    return lhs, tree, name_map


def _validate(node: ast.AST) -> None:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ExpressionError(f"Operator '{type(node.op).__name__}' is not supported")
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) \
                and isinstance(node.right.value, (int, float)) \
                and abs(node.right.value) > _MAX_POW_EXPONENT:
            raise ExpressionError(
                f"Exponent {node.right.value:g} exceeds the maximum allowed "
                f"magnitude ({_MAX_POW_EXPONENT}) — likely a typo"
            )
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ExpressionError(f"Operator '{type(node.op).__name__}' is not supported")
        _validate(node.operand)
    elif isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else None
        if fname in _FUNCS_2ARG:
            if len(node.args) != 2 or node.keywords:
                raise ExpressionError(f"'{fname}' takes exactly two arguments: {fname}(y, x)")
            _validate(node.args[0]); _validate(node.args[1])
        elif fname in _FUNCS:
            if len(node.args) != 1 or node.keywords:
                raise ExpressionError(f"'{fname}' takes exactly one argument")
            _validate(node.args[0])
        else:
            names = ", ".join(sorted(list(_FUNCS) + list(_FUNCS_2ARG)))
            raise ExpressionError(f"Unknown function — supported: {names}")
    elif isinstance(node, ast.Name):
        return
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ExpressionError("Only numeric constants are supported")
    else:
        raise ExpressionError(f"'{type(node).__name__}' is not supported in expressions")


def referenced_names(node: ast.AST, name_map: Dict[str, str]) -> set:
    """Variable names referenced by an expression (excludes function names and constants like pi)."""
    if isinstance(node, ast.BinOp):
        return referenced_names(node.left, name_map) | referenced_names(node.right, name_map)
    if isinstance(node, ast.UnaryOp):
        return referenced_names(node.operand, name_map)
    if isinstance(node, ast.Call):
        refs = referenced_names(node.args[0], name_map)
        if len(node.args) > 1:
            refs |= referenced_names(node.args[1], name_map)
        return refs
    if isinstance(node, ast.Name):
        actual = name_map.get(node.id, node.id)
        if actual in _CONSTANTS:
            return set()
        return {actual}
    return set()


def evaluate(node: ast.AST, namespace: Dict[str, pd.Series], name_map: Dict[str, str]):
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](
            evaluate(node.left, namespace, name_map), evaluate(node.right, namespace, name_map))
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](evaluate(node.operand, namespace, name_map))
    if isinstance(node, ast.Call):
        fname = node.func.id
        if fname in _FUNCS_2ARG:
            return _FUNCS_2ARG[fname](
                evaluate(node.args[0], namespace, name_map),
                evaluate(node.args[1], namespace, name_map),
            )
        return _FUNCS[fname](evaluate(node.args[0], namespace, name_map))
    if isinstance(node, ast.Name):
        actual = name_map.get(node.id, node.id)
        if actual in _CONSTANTS:
            return _CONSTANTS[actual]
        if actual not in namespace:
            raise ExpressionError(f"Unknown variable '{actual}' — check it matches a dataset column")
        return namespace[actual]
    if isinstance(node, ast.Constant):
        return node.value
    raise ExpressionError(f"'{type(node).__name__}' is not supported in expressions")


def _mathrm(name: str) -> str:
    # Escaping done outside the f-string expression part: Python 3.11 and
    # earlier reject a backslash anywhere inside `{...}`, even nested inside
    # a string literal argument (relaxed only by PEP 701 in 3.12).
    escaped = name.replace('_', r'\_').replace(' ', r'\ ')
    return rf"\mathrm{{{escaped}}}"


def to_latex(node: ast.AST, name_map: Dict[str, str], parent_prec: int = 0) -> str:
    """Render a validated expression AST as a LaTeX math string (preview only)."""
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op is ast.Add:
            s, prec = f"{to_latex(node.left, name_map, 1)} + {to_latex(node.right, name_map, 1)}", 1
        elif op is ast.Sub:
            s, prec = f"{to_latex(node.left, name_map, 1)} - {to_latex(node.right, name_map, 2)}", 1
        elif op is ast.Mult:
            s, prec = rf"{to_latex(node.left, name_map, 2)} \cdot {to_latex(node.right, name_map, 2)}", 2
        elif op is ast.Div:
            return rf"\frac{{{to_latex(node.left, name_map, 0)}}}{{{to_latex(node.right, name_map, 0)}}}"
        elif op is ast.Pow:
            s, prec = f"{to_latex(node.left, name_map, 3)}^{{{to_latex(node.right, name_map, 0)}}}", 3
        return rf"\left({s}\right)" if prec < parent_prec else s
    if isinstance(node, ast.UnaryOp):
        inner = to_latex(node.operand, name_map, 2)
        s = f"-{inner}" if isinstance(node.op, ast.USub) else inner
        return rf"\left({s}\right)" if parent_prec > 2 else s
    if isinstance(node, ast.Call):
        if node.func.id in _LATEX_FUNCS_2ARG:
            return _LATEX_FUNCS_2ARG[node.func.id](
                to_latex(node.args[0], name_map, 0), to_latex(node.args[1], name_map, 0))
        return _LATEX_FUNCS[node.func.id](to_latex(node.args[0], name_map, 0))
    if isinstance(node, ast.Name):
        actual = name_map.get(node.id, node.id)
        if actual in _CONSTANTS:
            return {"pi": r"\pi"}.get(actual, _mathrm(actual))
        return _mathrm(actual)
    if isinstance(node, ast.Constant):
        return str(node.value)
    raise ExpressionError(f"'{type(node).__name__}' is not supported in expressions")


def expression_to_latex(lhs: str, tree: ast.Expression, name_map: Dict[str, str]) -> str:
    return f"{_mathrm(lhs)} = {to_latex(tree.body, name_map)}"


def apply_expressions(df: pd.DataFrame, expression_texts: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply 'LHS = expr' expression strings to df in order. Later expressions may
    reference columns created by earlier ones. Assigning to an existing column name
    overwrites it in place; a new name appends a new column.
    Returns (new_dataframe, log_messages). Raises ExpressionError on the first failure.
    """
    result = df.copy()
    log = []
    for i, text in enumerate(expression_texts, start=1):
        text = text.strip()
        if not text:
            continue
        try:
            lhs, tree, name_map = parse_expression(text)
            names = referenced_names(tree.body, name_map)
            unknown = [n for n in names if n not in result.columns]
            if unknown:
                raise ExpressionError(f"Unknown variable(s): {', '.join(sorted(unknown))}")
            namespace = {n: result[n] for n in names}
            value = evaluate(tree.body, namespace, name_map)
            result[lhs] = value
            log.append(f"[{i}] {lhs} = {text.split('=', 1)[1].strip()}  ->  OK")
        except ExpressionError as e:
            raise ExpressionError(f"Expression {i} ('{text}'): {e}") from e
    return result, log


def extract_reconstruction_mapping(expression_texts: List[str]) -> Dict[str, str]:
    """
    Scan expressions for the pattern  LHS = <anything> - plain_identifier
    and return {lhs: plain_identifier} as the physics-estimate mapping.
    Only a bare (non-backtick) right operand is treated as the physics estimate;
    backtick-quoted names on the right are dataset columns, not intermediate estimates.
    """
    mapping: Dict[str, str] = {}
    for i, text in enumerate(expression_texts, start=1):
        text = text.strip()
        if not text:
            continue
        try:
            lhs, tree, name_map = parse_expression(text)
        except ExpressionError as e:
            # A genuinely malformed expression (bad syntax, unknown function, ...)
            # must not be silently dropped — that would leave the reconstruction
            # map quietly incomplete instead of naming the broken expression.
            raise ExpressionError(f"Expression {i} ('{text}'): {e}") from e
        body = tree.body
        if not (isinstance(body, ast.BinOp) and isinstance(body.op, ast.Sub)):
            continue
        right = body.right
        if isinstance(right, ast.Name) and right.id not in name_map:
            mapping[lhs] = right.id
    return mapping


def extract_measured_mapping(expression_texts: List[str]) -> Dict[str, str]:
    """
    Scan expressions for the pattern  LHS = measured_col - physics_est
    and return {lhs: measured_col} — the original dataset column that was measured.
    measured_col may be a plain identifier OR a backtick-quoted column name.
    physics_est must be a plain (non-backtick) identifier.
    This gives a reliable lookup regardless of how the residual target was named.
    """
    mapping: Dict[str, str] = {}
    for i, text in enumerate(expression_texts, start=1):
        text = text.strip()
        if not text:
            continue
        try:
            lhs, tree, name_map = parse_expression(text)
        except ExpressionError as e:
            raise ExpressionError(f"Expression {i} ('{text}'): {e}") from e
        body = tree.body
        if not (isinstance(body, ast.BinOp) and isinstance(body.op, ast.Sub)):
            continue
        right = body.right
        left  = body.left
        # Right must be a plain identifier (physics estimate, not a dataset col)
        if not (isinstance(right, ast.Name) and right.id not in name_map):
            continue
        # Left is the measured column — plain or backtick-quoted
        if isinstance(left, ast.Name):
            mapping[lhs] = name_map.get(left.id, left.id)
    return mapping


def save_physics_config(
    path: str,
    expressions: List[str],
    output_cols_text: str,
    reconstruction_map: Dict[str, str],
    original_dataset_path: str = "",
    measured_map: Dict[str, str] | None = None,
) -> None:
    import json
    from datetime import datetime
    config = {
        "version": "1.1",
        "generated_at": datetime.now().isoformat(),
        "expressions": [e for e in expressions if e.strip()],
        "output_columns_text": output_cols_text,
        "reconstruction_map": reconstruction_map,
        "measured_map": measured_map or {},
        "original_dataset_path": original_dataset_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_script_physics_config(
    path: str,
    script_path: str,
    input_vars: List[str],
    output_vars: List[str],
    constants: dict,
    reconstruction_map: Dict[str, str],
    measured_map: Dict[str, str],
    name_mapping: Dict[str, str] | None = None,
    time_col: str | None = None,
    original_dataset_path: str = "",
) -> None:
    import json
    from datetime import datetime
    config = {
        "version": "1.1",
        "mode": "script",
        "generated_at": datetime.now().isoformat(),
        "script_path": script_path,
        "input_vars": input_vars,
        "output_vars": output_vars,
        "constants": constants,
        "name_mapping": name_mapping or {},
        "time_col": time_col,
        "reconstruction_map": reconstruction_map,
        "measured_map": measured_map,
        "original_dataset_path": original_dataset_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_physics_config(path: str) -> dict:
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_output_columns(df: pd.DataFrame, columns_text: str) -> pd.DataFrame:
    """Reorder/trim df to a comma-separated list of (optionally backtick-quoted) column
    names. Blank input keeps all columns as-is."""
    columns_text = columns_text.strip()
    if not columns_text:
        return df
    names = []
    for raw in columns_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name = raw[1:-1].strip() if raw.startswith("`") and raw.endswith("`") else raw
        if name not in df.columns:
            raise ExpressionError(f"Unknown output column '{name}'")
        if name in names:
            raise ExpressionError(f"Column '{name}' is listed more than once")
        names.append(name)
    if not names:
        return df
    return df[names]
