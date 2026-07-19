# validation.py
# Shared input-validation helpers, so every user-settable knob fails the same
# way when given a suboptimal value: an immediate ValueError naming the
# parameter, raised at the public entry point of whichever module consumes it.
# Never a silent wrong answer, and never a cryptic crash deep inside a
# third-party library.
#
# The standard (enforced by tests/test_input_robustness.py): for each knob,
# {zero, negative, huge, float-where-int, wrong-type string, unknown name}
# must either work (documented/clamped behaviour) or raise through one of
# these helpers.

import numbers


def require_int_at_least(name: str, value, minimum: int = 1) -> int:
    """value must be an integer (bool excluded) >= minimum. Returns it as int.

    Floats are rejected even when integral (2.0): a float here usually means a
    formula or UI slip, and silently truncating 2.7 -> 2 would change the run.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be an integer, got {value!r} ({type(value).__name__})."
        )
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return int(value)


def require_in_range(name: str, value, low, high,
                     inclusive_low: bool = True, inclusive_high: bool = True) -> float:
    """value must be a real number within [low, high] (bounds inclusive by
    default; set the inclusive_* flags for open bounds). Returns it as float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be a number, got {value!r} ({type(value).__name__})."
        )
    ok_low = (value >= low) if inclusive_low else (value > low)
    ok_high = (value <= high) if inclusive_high else (value < high)
    if not (ok_low and ok_high):
        lo_b = "[" if inclusive_low else "("
        hi_b = "]" if inclusive_high else ")"
        raise ValueError(
            f"{name} must be in the range {lo_b}{low}, {high}{hi_b}, got {value!r}."
        )
    return float(value)


def require_choice(name: str, value, choices) -> object:
    """value must be one of `choices`. Returns it unchanged."""
    if value not in choices:
        raise ValueError(
            f"{name} must be one of {sorted(map(str, choices))}, got {value!r}."
        )
    return value
