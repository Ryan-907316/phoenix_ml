# progress.py
# Shared console/progress-log formatting helpers, so every pipeline module logs
# in one consistent style instead of each growing its own (banners, tags, and
# table rendering used to differ between steps).
#
# Conventions:
#   log_step:    one per pipeline step (Preprocessing, Training, HPO, ...) —
#                the dashed banner style the Report step always used.
#   log_substep: repeated unit inside a step (one model, one target, ...).
#   log_info:    notable fact worth a [INFO] tag (dataset shape, file saved).
#   log_warn:    something degraded/skipped but the run continues ([WARN] —
#                the single warning tag; the older [WARNING] spelling is
#                retired).
#   log_table:   any tabular result — same tabulate grid the Model Performance
#                Summary always used, so tables look identical everywhere.
#
# All output must be plain ASCII: the progress log is routinely captured to
# text files/consoles with non-UTF-8 encodings, where characters like the
# Greek lambda or an em-dash turn into mojibake. (Report/PDF text is separate
# and keeps proper typography.)

from tabulate import tabulate

_STEP_RULE_WIDTH = 60


def log_step(title: str) -> None:
    """Banner marking the start of a top-level pipeline step."""
    rule = "-" * _STEP_RULE_WIDTH
    print(f"\n{rule}\n  {title}\n{rule}")


def log_substep(title: str) -> None:
    """Marker for a repeated unit of work inside a step (a model, a target)."""
    print(f"\n=== {title} ===")


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def log_table(df, floatfmt: str = ".4f", max_col_width: int | None = None) -> None:
    """Print a DataFrame as the same tabulate grid used by the Model
    Performance Summary. `max_col_width` truncates long cell values (e.g.
    hyperparameter dicts) so the grid stays readable in a console."""
    display = df
    if max_col_width is not None:
        display = df.copy()
        for col in display.columns:
            display[col] = display[col].map(
                lambda v: (s[: max_col_width - 3] + "...")
                if len(s := str(v)) > max_col_width else s
            )
    print(tabulate(display, headers="keys", tablefmt="grid",
                   showindex=False, floatfmt=floatfmt))
