"""Tests for ui.py's _QueueStream — the one Tk-independent unit in ui.py
(~2,750 lines of customtkinter widget code bound to live Tk state, otherwise
not unit-testable without a display; see tests/ISSUES.md's "Explicitly out of
scope" note).

_QueueStream captures stdout/stderr from worker threads and routes lines to a
queue the UI thread drains — plain print()/tqdm.write() must pass through
unchanged, while tqdm's own \\r-updated progress bars are throttled (5%
milestones, or every _TIME_FLOOR seconds as a fallback) so a fast-moving bar
doesn't flood the log with hundreds of near-identical lines.
"""
import queue as queue_module

from phoenix_ml import ui as ui_module
from phoenix_ml.ui import _QueueStream


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_isatty_is_always_false():
    assert _QueueStream(queue_module.Queue()).isatty() is False


def test_plain_print_lines_pass_through_unchanged():
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("hello\nworld\n")
    assert _drain(q) == ["hello\n", "world\n"]


def test_carriage_return_is_also_treated_as_a_line_boundary():
    # tqdm.write() itself uses plain \n, but tqdm's live bar updates use \r —
    # both must split the buffer into emitted lines.
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("update 1\rupdate 2\r")
    assert _drain(q) == ["update 1\n", "update 2\n"]


def test_empty_write_is_a_no_op():
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("")
    assert q.empty()


def test_blank_lines_are_dropped_not_queued():
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("\n   \n\n")
    assert q.empty()


def test_flush_emits_a_trailing_partial_line():
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("no newline yet")
    assert q.empty()  # buffered, not yet emitted — no line boundary seen
    stream.flush()
    assert _drain(q) == ["no newline yet\n"]


def test_flush_is_a_no_op_when_the_buffer_is_already_empty():
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("already emitted\n")
    _drain(q)
    stream.flush()
    assert q.empty()


# ── tqdm bar throttling ───────────────────────────────────────────────────────
#
# A fixed, monkeypatched clock makes the _TIME_FLOOR fallback deterministic —
# real wall-clock timing in a fast test body would otherwise never cross the
# 2-second floor, silently failing to exercise that branch at all.

def _fake_clock(monkeypatch, start=0.0):
    clock = {"t": start}
    monkeypatch.setattr(ui_module._time, "monotonic", lambda: clock["t"])
    return clock


def test_tqdm_bar_is_throttled_to_5_percent_milestones(monkeypatch):
    clock = _fake_clock(monkeypatch)
    q = queue_module.Queue()
    stream = _QueueStream(q)

    for pct in [0, 1, 2, 3, 4, 5, 6, 10]:
        clock["t"] += 0.01  # negligible — stays well under the 2s time floor
        stream.write(f"Training: {pct}%|{'#' * pct}|\n")

    emitted = _drain(q)
    shown_pcts = [int(line.split(":")[1].strip().split("%")[0]) for line in emitted]
    # 0% always shown (first-ever call for this label); 1-4% share the same
    # milestone (0) as 0% and are suppressed; 5% is a new milestone; 6-9%
    # share milestone 5 and are suppressed; 10% is a new milestone.
    assert shown_pcts == [0, 5, 10]


def test_tqdm_bar_always_shows_the_final_99_percent_or_higher(monkeypatch):
    clock = _fake_clock(monkeypatch)
    q = queue_module.Queue()
    stream = _QueueStream(q)

    stream.write("Training: 0%|#|\n")
    clock["t"] += 0.01
    stream.write("Training: 97%|#####|\n")   # milestone 95, new -> shown anyway
    clock["t"] += 0.01
    stream.write("Training: 99%|######|\n")  # milestone 95, same as above, but >=99 forces it
    emitted = _drain(q)
    assert len(emitted) == 3


def test_tqdm_bar_emits_after_the_time_floor_even_without_a_new_milestone(monkeypatch):
    clock = _fake_clock(monkeypatch)
    q = queue_module.Queue()
    stream = _QueueStream(q)

    stream.write("Training: 1%|#|\n")     # first-ever call for this label -> always emitted
    stream.write("Training: 2%|##|\n")    # same milestone (0), too soon -> suppressed
    clock["t"] += 3.0                      # advance past the 2.0s time floor
    stream.write("Training: 3%|###|\n")   # same milestone, but time floor triggers -> emitted

    emitted = _drain(q)
    assert len(emitted) == 2
    assert "1%" in emitted[0]
    assert "3%" in emitted[1]


def test_a_new_bar_starting_at_0_percent_resets_its_own_milestone_tracking(monkeypatch):
    # Two independently-labelled bars (e.g. two successive tqdm loops) must
    # not share throttling state — a second bar's first 0% must always show,
    # not be suppressed by the first bar's history.
    clock = _fake_clock(monkeypatch)
    q = queue_module.Queue()
    stream = _QueueStream(q)

    for pct in [0, 50, 100]:
        clock["t"] += 0.01
        stream.write(f"Preprocessing: {pct}%|{'#' * (pct // 10)}|\n")
    _drain(q)

    clock["t"] += 0.01
    stream.write("Training: 0%|#|\n")  # different label, own bar -> must show
    emitted = _drain(q)
    assert len(emitted) == 1
    assert "Training" in emitted[0]


def test_non_tqdm_lines_are_never_treated_as_a_progress_bar():
    # A line that happens to contain "%" without the "|...|" bar syntax must
    # pass straight through, untouched by the throttling logic.
    q = queue_module.Queue()
    stream = _QueueStream(q)
    stream.write("Accuracy improved by 5%\n")
    assert _drain(q) == ["Accuracy improved by 5%\n"]
