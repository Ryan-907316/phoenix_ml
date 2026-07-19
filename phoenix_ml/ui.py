# ui.py
# customtkinter UI for Phoenix ML.

from __future__ import annotations
import copy, dataclasses, io, os, queue, sys, threading, traceback, time as _time
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from PIL import Image

import customtkinter as ctk

_THEME = os.path.join(os.path.dirname(__file__), "orange_theme.json")
ctk.set_appearance_mode("System")
ctk.set_default_color_theme(_THEME)

from phoenix_ml.workflow_steps import (
    WorkflowSession,
    run_step_preprocessing, run_step_training,
    run_step_uq_before, run_step_interpretability_before, run_step_interpretability_after,
    run_step_hpo, run_step_cv, run_step_uq_after,
    run_step_perl, run_step_report,
)
from phoenix_ml.models import models_dict as ALL_MODELS
from phoenix_ml.model_training import derive_seed, SEED_OFFSET_OUTLIER_DETECTION
from phoenix_ml.dataset_cleaning import (
    auto_classify_columns, apply_cleaning, build_issue_mask,
    ACTION_FFILL, ACTION_BFILL, ACTION_NONE, ACTION_REMOVE,
    ACTION_CAP, ACTION_INTERP, ACTION_MEAN, ACTION_MEDIAN, ACTION_DROP,
    OUTLIER_NONE, OUTLIER_IQR, OUTLIER_ZSCORE, OUTLIER_PERCENTILE,
    OUTLIER_ISOLATION_FOREST, OUTLIER_LOF, OUTLIER_MCD,
    MULTIVARIATE_OUTLIER_METHODS,
    ROLE_INPUT, ROLE_TARGET, ROLE_TIMESTAMP, ROLE_EXCLUDE,
    TYPE_DATETIME,
)
from phoenix_ml.physics_expressions import (
    parse_expression, expression_to_latex, apply_expressions, select_output_columns,
    extract_reconstruction_mapping, extract_measured_mapping,
    save_physics_config, save_script_physics_config, load_physics_config,
    ExpressionError,
)
from phoenix_ml.physics_model import (
    import_physics_script, run_physics_model, generate_residual_dataset,
)

# (key, display name in pipeline table, prerequisite keys)
STEPS = [
    ("preprocessing",           "Preprocessing",                         []),
    ("training",                "Baseline Training",                     ["preprocessing"]),
    ("uq_before",                "Uncertainty Quantification (Before HPO)", ["training"]),
    ("interpretability_before", "Interpretability (Before HPO)",         ["training"]),
    ("hpo",                     "Hyperparameter Optimisation",           ["training"]),
    ("cv",                      "Cross-Validation",                     ["training"]),
    ("uq_after",                "Uncertainty Quantification (After HPO)", ["training", "hpo"]),
    ("interpretability_after",  "Interpretability (After HPO)",          ["training", "hpo"]),
    ("perl",                    "Physics-Enhanced Residual Learning",    ["training"]),
]

STEP_FNS = {
    "preprocessing":           run_step_preprocessing,
    "training":                run_step_training,
    "uq_before":                run_step_uq_before,
    "interpretability_before": run_step_interpretability_before,
    "hpo":                     run_step_hpo,
    "cv":                      run_step_cv,
    "uq_after":                run_step_uq_after,
    "interpretability_after":  run_step_interpretability_after,
    "perl":                    run_step_perl,
}

STATUS_INFO = {
    "not_run":   ("gray60",  "Not Run"),
    "running":   ("#E07818", "Running..."),
    "done":      ("#4CAF50", "Done"),
    "error":     ("#F44336", "Error  x"),
    "skipped":   ("#7B97C7", "Skipped"),
    "cancelled": ("#9E9E9E", "Cancelled"),
}


class WorkflowCancelled(Exception):
    """Raised by a running step's checkpoint callback when the user presses Stop."""

ALL_MODEL_NAMES = list(ALL_MODELS.keys())
DEFAULT_ON = {"SVR (RBF)", "Gaussian Process Regressor", "XGBoost Regressor"}

# Cross-validation method: display label → internal key expected by workflow
_CV_METHOD_MAP = {
    "Shuffle Split":   "Shuffle Split",
    "K-Fold":          "K-Fold",
    "Repeated K-Fold": "Repeated K-Fold",
    "Leave One Out":   "LOO",
    "Leave p Out":     "LpO",
}
_CV_METHOD_DISPLAY = list(_CV_METHOD_MAP.keys())

# Sampling method: display label -> internal key expected by hyperparameter_optimisation.py
_SAMPLING_METHOD_MAP = {
    "Sobol":               "Sobol",
    "Halton":              "Halton",
    "Latin Hypercube":     "Latin Hypercube",
    "Random (Monte Carlo)":"Random",
}

# Which parameter fields each CV method needs (controls dynamic UI)
_CV_FIELDS = {
    "Shuffle Split":   ("n_splits", "test_size", "random_state"),
    "K-Fold":          ("n_splits", "random_state"),
    "Repeated K-Fold": ("n_splits", "n_repeats", "random_state"),
    "Leave One Out":   (),
    "Leave p Out":     ("p",),
}

# Label colour constants
_LBL_NORMAL   = ["gray10", "#DCE4EE"]
_LBL_DISABLED = "gray55"


import re as _re
_TQDM_PCT   = _re.compile(r'(\d+)%\|')
_TQDM_LABEL = _re.compile(r'^(.+):\s+\d+%\|')


class _QueueStream:
    """
    Captures stdout/stderr from worker threads and routes to a queue.
    tqdm bars are filtered: shown at 5% milestones, and at most once every
    _TIME_FLOOR seconds as a safety net regardless of percentage.
    Everything else (tqdm.write, plain print) passes through unchanged.
    """

    _STEP       = 5    # show every 5 percentage points
    _TIME_FLOOR = 2.0  # fallback: always show if >= this many seconds since last update

    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = ""
        self._bar_mile: dict[str, int]   = {}
        self._bar_time: dict[str, float] = {}

    def isatty(self) -> bool:
        return False

    def write(self, text: str):
        if not text:
            return
        self._buf += text
        while True:
            ni = self._buf.find('\n')
            ri = self._buf.find('\r')
            if ni == -1 and ri == -1:
                break
            if ri == -1 or (ni != -1 and ni < ri):
                idx = ni
            else:
                idx = ri
            line, self._buf = self._buf[:idx], self._buf[idx + 1:]
            self._emit(line)

    def _emit(self, line: str):
        s = line.strip()
        if not s:
            return
        m = _TQDM_PCT.search(s)
        if not m or '|' not in s:
            self._q.put(s + '\n')
            return
        pct = int(m.group(1))
        lm  = _TQDM_LABEL.match(s)
        label = lm.group(1).strip() if lm else "_bar"
        now = _time.monotonic()

        if pct == 0:
            self._bar_mile[label] = -(self._STEP)
            self._bar_time[label] = now - self._TIME_FLOOR - 1.0

        last_mile = self._bar_mile.get(label, -(self._STEP))
        last_time = self._bar_time.get(label, now - self._TIME_FLOOR - 1.0)
        milestone = (pct // self._STEP) * self._STEP

        if pct >= 99 or milestone > last_mile or (now - last_time) >= self._TIME_FLOOR:
            self._bar_mile[label] = milestone
            self._bar_time[label] = now
            self._q.put(s + '\n')

    def flush(self):
        if self._buf:
            self._emit(self._buf)
        self._buf = ""


class PhoenixApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("phoenix_ml")
        self.geometry("1150x820")
        self.minsize(950, 680)

        self.session = WorkflowSession()
        self.log_queue: queue.Queue[str] = queue.Queue()
        # Background worker threads must never touch Tk state directly (only the main
        # thread may) — a worker signals "call this on the main thread when you get a
        # chance" by putting a callable here; _poll_log (already running on the main
        # thread via self.after) drains it every tick, the same safe hand-off pattern
        # already used for log lines.
        self._callback_queue: queue.Queue = queue.Queue()
        self.step_status: dict[str, str] = {k: "not_run" for k, *_ in STEPS}
        self._monotonic_constraints: dict[str, dict[str, int]] = {}
        self._running_step: str | None = None
        self._step_timer_labels: dict[str, ctk.CTkLabel] = {}
        self._step_start_times: dict[str, float]         = {}
        self._step_elapsed:     dict[str, float]         = {}

        # Pause/Stop controls. _pause_event set == currently paused; _cancel_event set ==
        # Stop was requested. _paused_seconds accumulates, per step, time spent paused so
        # it can be subtracted back out of that step's recorded elapsed time.
        self._pause_event  = threading.Event()
        self._cancel_event = threading.Event()
        self._paused_seconds: dict[str, float] = {}

        # Re-entrancy guards for the target/monotonicity picker popups — a double-click
        # on "Select..."/"Configure..." used to stack two modals, with the second
        # stealing the grab from the first with no way back to it.
        self._target_picker_win: ctk.CTkToplevel | None = None
        self._monotonic_picker_win: ctk.CTkToplevel | None = None

        # A snapshot of every settings field at the moment each step last completed
        # successfully — compared against the live settings before a later step is
        # allowed to run, so a step can't silently run against a prerequisite whose
        # settings have since changed (see _stale_prereqs).
        self._step_settings_snapshot: dict[str, dict] = {}

        self._build_ui()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.pack(fill="both", expand=True, padx=8, pady=8)
        for name in ["Dataset Cleaning", "Physics Modelling",
                     "Home", "Models", "Preprocessing",
                     "Uncertainty Quantification", "Interpretability",
                     "Hyperparameter Optimisation", "Postprocessing"]:
            self.tabview.add(name)
        self._setup_dataset_cleaning_tab()
        self._setup_physics_modelling_tab()
        self._setup_home_tab()
        self._setup_models_tab()
        self._setup_preprocessing_tab()
        self._setup_uq_tab()
        self._setup_interpretability_tab()
        self._setup_hpo_tab()
        self._setup_cv_tab()
        self.tabview.set("Home")

    # ── Dataset Cleaning Tab ──────────────────────────────────────────────────

    def _setup_dataset_cleaning_tab(self):
        # State
        self._clean_df          = None
        self._clean_df_original = None
        self._clean_col_info    = {}
        self._clean_prev_roles  = {}
        self._clean_fix_log     = []

        tab = self.tabview.tab("Dataset Cleaning")

        # ── Top bar ───────────────────────────────────────────────────────────
        top = ctk.CTkFrame(tab)
        top.pack(side="top", fill="x", padx=4, pady=(4, 2))
        top.columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="Dataset Path:", anchor="w", width=100).grid(
            row=0, column=0, padx=(8, 4), pady=6)
        self._clean_dataset_entry = ctk.CTkEntry(top, placeholder_text="Path to raw .csv file")
        self._clean_dataset_entry.grid(row=0, column=1, padx=4, pady=6, sticky="ew")
        ctk.CTkButton(top, text="Browse", width=70,
                      command=self._browse_clean_dataset).grid(row=0, column=2, padx=4, pady=6)
        self._clean_load_btn = ctk.CTkButton(top, text="Load", width=70,
                      command=self._clean_load_dataset)
        self._clean_load_btn.grid(row=0, column=3, padx=4, pady=6)
        self._clean_apply_btn  = ctk.CTkButton(top, text="Apply", width=70, state="disabled",
                                                command=self._clean_apply)
        self._clean_apply_btn.grid(row=0, column=4, padx=4, pady=6)
        self._clean_export_btn = ctk.CTkButton(top, text="Export", width=70, state="disabled",
                                                command=self._clean_export)
        self._clean_export_btn.grid(row=0, column=5, padx=(4, 8), pady=6)

        # ── Bottom bar (settings + log) ───────────────────────────────────────
        bot = ctk.CTkFrame(tab, height=300)
        bot.pack(side="bottom", fill="x", padx=4, pady=(2, 4))
        bot.pack_propagate(False)

        settings_frame = ctk.CTkFrame(bot, width=460)
        settings_frame.pack(side="left", fill="y", padx=(0, 2), pady=4)
        settings_frame.pack_propagate(False)
        self._clean_build_settings(settings_frame)

        log_frame = ctk.CTkFrame(bot)
        log_frame.pack(side="left", fill="both", expand=True, padx=(2, 0), pady=4)
        self._clean_build_log(log_frame)

        # ── Middle (column manager | data preview) ────────────────────────────
        mid = ctk.CTkFrame(tab)
        mid.pack(side="top", fill="both", expand=True, padx=4, pady=2)

        left_pane = ctk.CTkFrame(mid, width=460)
        left_pane.pack(side="left", fill="y", padx=(0, 2))
        left_pane.pack_propagate(False)

        col_mgr_frame = ctk.CTkFrame(left_pane)
        col_mgr_frame.pack(side="top", fill="both", expand=True, padx=2, pady=(2, 1))

        detail_frame = ctk.CTkFrame(left_pane, height=170)
        detail_frame.pack(side="bottom", fill="x", padx=2, pady=(1, 2))
        detail_frame.pack_propagate(False)

        right_pane = ctk.CTkFrame(mid)
        right_pane.pack(side="left", fill="both", expand=True, padx=(2, 0))

        self._clean_build_column_manager(col_mgr_frame)
        self._clean_build_detail_panel(detail_frame)
        self._clean_build_data_preview(right_pane)
        self._clean_apply_treeview_style()

    # ── Dataset Cleaning helpers ──────────────────────────────────────────────

    def _clean_build_settings(self, parent):
        parent.columnconfigure(1, weight=1)
        ctk.CTkLabel(parent, text="Cleaning Settings",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(6, 2), sticky="w")

        def _row(label, row, width=140):
            return ctk.CTkLabel(parent, text=label, anchor="w", width=width,
                                 font=ctk.CTkFont(size=11)).grid(
                row=row, column=0, padx=(8, 2), pady=2, sticky="w")

        filter_row = ctk.CTkFrame(parent, fg_color="transparent")
        filter_row.grid(row=1, column=0, columnspan=2, padx=4, pady=(2, 4), sticky="w")
        self._clean_filter_issues = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(filter_row, text="Show only rows with issues in Data Preview",
                        variable=self._clean_filter_issues,
                        command=self._clean_populate_data_preview,
                        font=ctk.CTkFont(size=11)).pack(side="left")

        _row("Missing Values:", 2)
        self._clean_missing_action = ctk.CTkOptionMenu(
            parent, width=175,
            values=["Forward Fill", "Backward Fill", "Interpolate",
                    "Replace with Mean", "Replace with Median", "Drop Rows", "None"])
        self._clean_missing_action.set("Drop Rows")
        self._clean_missing_action.grid(row=2, column=1, padx=(2, 8), pady=2, sticky="w")

        _row("Outlier Detection:", 3)
        self._clean_outlier_method = ctk.CTkOptionMenu(
            parent, width=175,
            values=[OUTLIER_NONE, OUTLIER_IQR, OUTLIER_ZSCORE, OUTLIER_PERCENTILE,
                    OUTLIER_ISOLATION_FOREST, OUTLIER_LOF, OUTLIER_MCD],
            command=self._clean_on_outlier_method_change)
        self._clean_outlier_method.set(OUTLIER_NONE)
        self._clean_outlier_method.grid(row=3, column=1, padx=(2, 8), pady=2, sticky="w")

        self._clean_outlier_thresh_label = ctk.CTkLabel(
            parent, text="Outlier Threshold:", anchor="w", width=140,
            font=ctk.CTkFont(size=11))
        self._clean_outlier_thresh_label.grid(row=4, column=0, padx=(8, 2), pady=2, sticky="w")
        self._clean_outlier_thresh = ctk.CTkEntry(parent, width=175)
        self._clean_outlier_thresh.insert(0, "1.5")
        self._clean_outlier_thresh.grid(row=4, column=1, padx=(2, 8), pady=2, sticky="w")

        _row("Outlier Action:", 5)
        self._clean_outlier_action = ctk.CTkOptionMenu(
            parent, width=175,
            values=["Remove Rows", "Cap (Winsorise)", "Interpolate", "None"])
        self._clean_outlier_action.set("None")
        self._clean_outlier_action.grid(row=5, column=1, padx=(2, 8), pady=2, sticky="w")

        stuck_row = ctk.CTkFrame(parent, fg_color="transparent")
        stuck_row.grid(row=6, column=0, columnspan=2, padx=4, pady=(4, 0), sticky="w")
        self._clean_stuck_enabled = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(stuck_row, text="Stuck value detection  min run:",
                        variable=self._clean_stuck_enabled, width=200,
                        font=ctk.CTkFont(size=11)).pack(side="left")
        self._clean_stuck_min_run = ctk.CTkEntry(stuck_row, width=40)
        self._clean_stuck_min_run.insert(0, "10")
        self._clean_stuck_min_run.pack(side="left", padx=4)

        _row("Stuck Action:", 7)
        self._clean_stuck_action = ctk.CTkOptionMenu(
            parent, width=175, values=["None", "Interpolate", "Remove Rows"])
        self._clean_stuck_action.set("None")
        self._clean_stuck_action.grid(row=7, column=1, padx=(2, 8), pady=2, sticky="w")

        dup_row = ctk.CTkFrame(parent, fg_color="transparent")
        dup_row.grid(row=8, column=0, columnspan=2, padx=4, pady=(4, 2), sticky="w")
        self._clean_remove_dupes = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(dup_row, text="Remove exact duplicate rows",
                        variable=self._clean_remove_dupes,
                        font=ctk.CTkFont(size=11)).pack(side="left")

    def _clean_on_outlier_method_change(self, choice):
        defaults = {
            OUTLIER_NONE:              ("Outlier Threshold:",        "1.5"),
            OUTLIER_IQR:               ("Threshold (x IQR):",        "1.5"),
            OUTLIER_ZSCORE:            ("Threshold (x Std Dev):",    "3.0"),
            OUTLIER_PERCENTILE:        ("Keep Within (%):",          "95"),
            OUTLIER_ISOLATION_FOREST:  ("Contamination (0-0.5):",    "0.1"),
            OUTLIER_LOF:               ("Contamination (0-0.5):",    "0.1"),
            OUTLIER_MCD:               ("Contamination (0-0.5):",    "0.1"),
        }
        label_text, default_val = defaults.get(choice, ("Outlier Threshold:", "1.5"))
        self._clean_outlier_thresh_label.configure(text=label_text)
        self._clean_outlier_thresh.delete(0, "end")
        self._clean_outlier_thresh.insert(0, default_val)

        # Row-level multivariate methods have no per-column bounds, so
        # "Cap (Winsorise)" isn't a meaningful action for them.
        if choice in MULTIVARIATE_OUTLIER_METHODS:
            self._clean_outlier_action.configure(
                values=["Remove Rows", "Interpolate", "None"])
            if self._clean_outlier_action.get() == "Cap (Winsorise)":
                self._clean_outlier_action.set("Remove Rows")
        else:
            self._clean_outlier_action.configure(
                values=["Remove Rows", "Cap (Winsorise)", "Interpolate", "None"])

    def _clean_build_log(self, parent):
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(side="top", fill="x", padx=4, pady=(4, 2))
        ctk.CTkLabel(hdr, text="Log", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(hdr, text="Clear", width=55,
                      command=self._clean_clear_log).pack(side="right")
        self._clean_log_box = ctk.CTkTextbox(
            parent, state="disabled", font=ctk.CTkFont(family="Courier New", size=10))
        self._clean_log_box.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def _clean_build_column_manager(self, parent):
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(side="top", fill="x", padx=4, pady=(4, 1))
        ctk.CTkLabel(hdr, text="Column Manager",
                     font=ctk.CTkFont(weight="bold")).pack(side="left")

        tree_frame = ctk.CTkFrame(parent)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.rowconfigure(1, weight=0)
        tree_frame.columnconfigure(0, weight=1)

        cols = ("incl", "col", "type", "role", "missing", "issues")
        self._clean_col_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", selectmode="browse")
        self._clean_col_tree.heading("incl",    text="",        anchor="center")
        self._clean_col_tree.heading("col",     text="Column",  anchor="w")
        self._clean_col_tree.heading("type",    text="Type",    anchor="center")
        self._clean_col_tree.heading("role",    text="Role",    anchor="center")
        self._clean_col_tree.heading("missing", text="Missing", anchor="center")
        self._clean_col_tree.heading("issues",  text="Issues",  anchor="w")
        self._clean_col_tree.column("incl",    width=28,  minwidth=28,  anchor="center", stretch=False)
        self._clean_col_tree.column("col",     width=140, minwidth=80,  anchor="w",      stretch=False)
        self._clean_col_tree.column("type",    width=72,  minwidth=55,  anchor="center", stretch=False)
        self._clean_col_tree.column("role",    width=80,  minwidth=55,  anchor="center", stretch=False)
        self._clean_col_tree.column("missing", width=65,  minwidth=50,  anchor="center", stretch=False)
        self._clean_col_tree.column("issues",  width=280, minwidth=100, anchor="w",      stretch=False)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._clean_col_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                            command=self._clean_col_tree.xview)
        self._clean_col_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._clean_col_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self._clean_col_tree.bind("<<TreeviewSelect>>", self._clean_on_column_select)
        self._clean_col_tree.bind("<Double-1>",         self._clean_on_column_double_click)
        self._clean_col_tree.bind("<ButtonRelease-1>",  self._clean_on_col_tree_click)

    def _clean_build_detail_panel(self, parent):
        self._clean_detail_inner = ctk.CTkFrame(parent, fg_color="transparent")
        self._clean_detail_inner.pack(fill="both", expand=True, padx=4, pady=4)
        ctk.CTkLabel(self._clean_detail_inner,
                     text="Select a column to view details",
                     font=ctk.CTkFont(size=11)).pack(
            anchor="w", padx=6, pady=4)

    def _clean_build_data_preview(self, parent):
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(side="top", fill="x", padx=4, pady=(4, 1))
        self._clean_preview_hdr = ctk.CTkLabel(hdr, text="Data Preview (first 200 rows)",
                     font=ctk.CTkFont(weight="bold"))
        self._clean_preview_hdr.pack(side="left")

        tree_frame = ctk.CTkFrame(parent)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._clean_preview_tree = ttk.Treeview(
            tree_frame, show="headings", selectmode="none")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._clean_preview_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                            command=self._clean_preview_tree.xview)
        self._clean_preview_tree.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._clean_preview_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

    def _clean_apply_treeview_style(self):
        mode  = ctk.get_appearance_mode()
        style = ttk.Style()
        style.theme_use("clam")
        if mode == "Dark":
            bg, fg, field = "#2b2b2b", "#DCE4EE", "#2b2b2b"
            hbg, hfg      = "#1f1f1f", "#E07818"
            nan_bg, out_bg, stuck_bg = "#5a1e1e", "#4d3a00", "#1a2e4a"
        else:
            bg, fg, field = "white", "#1a1a1a", "white"
            hbg, hfg      = "#f0f0f0", "#E07818"
            nan_bg, out_bg, stuck_bg = "#ffd5d5", "#fff0c0", "#d5e8ff"

        style.configure("Treeview",
                        background=bg, foreground=fg, fieldbackground=field,
                        rowheight=22, borderwidth=0, relief="flat")
        style.configure("Treeview.Heading",
                        background=hbg, foreground=hfg, relief="flat",
                        font=("Helvetica", 9, "bold"))
        style.map("Treeview",
                  background=[("selected", "#E07818")],
                  foreground=[("selected", "white")])

        for tree in (self._clean_col_tree, self._clean_preview_tree):
            tree.tag_configure("has_nan",     background=nan_bg)
            tree.tag_configure("has_outlier", background=out_bg)
            tree.tag_configure("has_stuck",   background=stuck_bg)

        self._clean_col_tree.tag_configure("timestamp_row", foreground="gray60")
        self._clean_col_tree.tag_configure("exclude_row",   foreground="gray50")
        self._clean_col_tree.tag_configure("target_row",    foreground="#E07818")

    def _clean_load_dataset(self):
        path = self._clean_dataset_entry.get().strip()
        if not path:
            messagebox.showerror("Missing Path", "Please specify a dataset path."); return
        if not os.path.isfile(path):
            messagebox.showerror("File Not Found", f"Cannot find:\n{path}"); return

        # A large CSV (read_csv) or a dataset with many columns (auto_classify_columns,
        # which profiles every column) used to run on the main thread and freeze the
        # whole window with no progress indication or way to cancel. Both go on a
        # background thread; only the Tk widget population below stays on the main one.
        self._clean_load_btn.configure(state="disabled", text="Loading...")

        def worker():
            try:
                df = pd.read_csv(path)
                col_info = auto_classify_columns(df)
            except Exception as exc:
                self._callback_queue.put(lambda: self._finish_clean_load(path, None, None, exc))
                return
            self._callback_queue.put(lambda: self._finish_clean_load(path, df, col_info, None))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_clean_load(self, path, df, col_info, exc):
        self._clean_load_btn.configure(state="normal", text="Load")
        if exc is not None:
            messagebox.showerror("Load Error", str(exc))
            return

        self._clean_df          = df
        self._clean_df_original = df.copy()
        self._clean_col_info    = col_info
        self._clean_prev_roles  = {}
        self._clean_fix_log     = []

        self._clean_populate_column_manager()
        self._clean_populate_data_preview()

        fname   = os.path.basename(path)
        n_dupes = int(df.duplicated().sum())
        dupe_note = f"  [{n_dupes} duplicate rows detected]" if n_dupes > 0 else ""
        self._clean_log(
            f"[INFO] Loaded: {fname} - {len(df)} rows x {len(df.columns)} columns{dupe_note}"
        )

        # Log auto-excluded columns (user needs to know what was auto-decided)
        for col, info in self._clean_col_info.items():
            if info["role"] == ROLE_EXCLUDE:
                self._clean_log(f"[EXCL] '{col}': {info['reason']}")
            elif info["role"] == ROLE_TIMESTAMP:
                self._clean_log(f"[CONV] '{col}': datetime - will be converted to elapsed seconds on Apply")

        # Log columns with missing values (always a data quality issue)
        nan_cols = [
            (col, info) for col, info in self._clean_col_info.items()
            if any("missing" in iss.lower() for iss in info["issues"])
        ]
        if nan_cols:
            for col, info in nan_cols:
                miss_note = next(iss for iss in info["issues"] if "missing" in iss.lower())
                self._clean_log(f"[WARN] '{col}': {miss_note}")

        # Summary of informational notes (visible in column manager; not spammed here)
        notes = sum(1 for info in self._clean_col_info.values()
                    if info["issues"] and info["role"] not in (ROLE_EXCLUDE, ROLE_TIMESTAMP))
        if notes:
            self._clean_log(
                f"[INFO] {notes} column(s) have informational notes "
                f"(stuck values, clipping, binary flags) - review the column manager."
            )

        self._clean_apply_btn.configure(state="normal")
        self._clean_export_btn.configure(state="normal")

    def _clean_populate_column_manager(self):
        for item in self._clean_col_tree.get_children():
            self._clean_col_tree.delete(item)

        for col, info in self._clean_col_info.items():
            stats      = info["stats"]
            n_miss     = stats.get("n_missing", 0)
            pct        = stats.get("missing_pct", 0.0)
            missing_str = f"{n_miss} ({pct}%)" if n_miss > 0 else "-"
            issues_str  = "; ".join(info["issues"]) if info["issues"] else "-"

            role = info["role"]
            if role == ROLE_TIMESTAMP:
                tag = "timestamp_row"
            elif role == ROLE_EXCLUDE:
                tag = "exclude_row"
            elif role == ROLE_TARGET:
                tag = "target_row"
            else:
                tag = ""

            incl_char = "☐" if role == ROLE_EXCLUDE else "☑"
            self._clean_col_tree.insert(
                "", "end", iid=col,
                values=(incl_char, col, info["type"], role, missing_str, issues_str),
                tags=(tag,) if tag else ()
            )

    def _clean_populate_data_preview(self):
        if self._clean_df is None:
            return
        df = self._clean_df

        # Rebuild columns
        self._clean_preview_tree.configure(columns=list(df.columns))
        for col in df.columns:
            self._clean_preview_tree.heading(col, text=col, anchor="w")
            self._clean_preview_tree.column(col, width=110, minwidth=60, anchor="w")
        for item in self._clean_preview_tree.get_children():
            self._clean_preview_tree.delete(item)

        issue_mask  = build_issue_mask(df, self._clean_col_info)
        filter_on   = self._clean_filter_issues.get()
        row_indices = [i for i in range(len(df)) if issue_mask.get(i, set())] \
                      if filter_on else list(range(len(df)))
        shown_indices = row_indices[:200]

        for i in shown_indices:
            row_vals = []
            for col in df.columns:
                val = df.iloc[i][col]
                if pd.isna(val):
                    row_vals.append("")
                elif isinstance(val, float):
                    row_vals.append(round(val, 4))
                else:
                    row_vals.append(val)

            issues = issue_mask.get(i, set())
            if "nan" in issues:
                tag = "has_nan"
            elif "stuck" in issues:
                tag = "has_stuck"
            elif "outlier" in issues:
                tag = "has_outlier"
            else:
                tag = ""

            self._clean_preview_tree.insert(
                "", "end", values=row_vals,
                tags=(tag,) if tag else ()
            )

        if filter_on:
            self._clean_preview_hdr.configure(
                text=f"Data Preview ({len(row_indices)} flagged row(s), showing first "
                     f"{len(shown_indices)})")
        else:
            self._clean_preview_hdr.configure(
                text=f"Data Preview (first {len(shown_indices)} of {len(df)} rows)")

    def _clean_on_col_tree_click(self, event):
        region = self._clean_col_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col_id = self._clean_col_tree.identify_column(event.x)
        if col_id != "#1":   # incl checkbox column
            return
        item = self._clean_col_tree.identify_row(event.y)
        if not item or item not in self._clean_col_info:
            return
        current_role = self._clean_col_info[item]["role"]
        if current_role == ROLE_EXCLUDE:
            # Restore previous role (default to Input)
            self._clean_col_info[item]["role"] = self._clean_prev_roles.pop(item, ROLE_INPUT)
        else:
            # Save current role then exclude
            self._clean_prev_roles[item] = current_role
            self._clean_col_info[item]["role"] = ROLE_EXCLUDE
        self._clean_populate_column_manager()
        self._clean_col_tree.selection_set(item)
        self._clean_populate_data_preview()

    def _clean_on_column_select(self, event):
        sel = self._clean_col_tree.selection()
        if not sel:
            return
        col_name = sel[0]
        if col_name in self._clean_col_info:
            self._clean_update_detail_panel(col_name)

    def _clean_on_column_double_click(self, event):
        region = self._clean_col_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col_id = self._clean_col_tree.identify_column(event.x)
        if col_id != "#4":   # Role column (incl is #1, col is #2, type is #3, role is #4)
            return
        item = self._clean_col_tree.identify_row(event.y)
        if not item:
            return
        bbox = self._clean_col_tree.bbox(item, "#4")
        if not bbox:
            return

        roles   = [ROLE_INPUT, ROLE_TARGET, ROLE_TIMESTAMP, ROLE_EXCLUDE]
        cur_role = self._clean_col_info.get(item, {}).get("role", ROLE_INPUT)
        var     = tk.StringVar(value=cur_role)

        cb = ttk.Combobox(self._clean_col_tree, textvariable=var,
                          values=roles, state="readonly",
                          width=max(len(r) for r in roles))
        cb.place(x=bbox[0], y=bbox[1], width=bbox[2] + 2, height=bbox[3])
        cb.focus_set()
        cb.event_generate("<ButtonPress-1>")

        def _commit(evt=None):
            new_role = var.get()
            if item in self._clean_col_info:
                self._clean_col_info[item]["role"] = new_role
            cb.destroy()
            self._clean_populate_column_manager()
            self._clean_populate_data_preview()

        cb.bind("<<ComboboxSelected>>", _commit)
        cb.bind("<Return>",    _commit)
        cb.bind("<FocusOut>",  lambda e: cb.destroy())

    def _clean_update_detail_panel(self, col_name: str):
        for w in self._clean_detail_inner.winfo_children():
            w.destroy()

        info  = self._clean_col_info[col_name]
        stats = info["stats"]

        name_text = col_name if len(col_name) <= 36 else col_name[:33] + "..."
        ctk.CTkLabel(self._clean_detail_inner, text=name_text,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     anchor="w").grid(row=0, column=0, padx=8, pady=(4, 1), sticky="w")
        ctk.CTkLabel(self._clean_detail_inner,
                     text=f"Type: {info['type']}   |   Role: {info['role']}",
                     font=ctk.CTkFont(size=10),
                     anchor="w").grid(row=1, column=0, padx=8, pady=1, sticky="w")

        row = 2
        if "mean" in stats and stats["mean"] is not None:
            sf = ctk.CTkFrame(self._clean_detail_inner, fg_color="transparent")
            sf.grid(row=row, column=0, padx=8, pady=(2, 1), sticky="w")

            def _stat(label, value, r, c_lbl, c_val):
                ctk.CTkLabel(sf, text=label, anchor="e",
                             font=ctk.CTkFont(size=10), width=130).grid(
                    row=r, column=c_lbl, padx=(0, 4), pady=1, sticky="e")
                ctk.CTkLabel(sf, text=str(value), anchor="w",
                             font=ctk.CTkFont(size=10), width=75).grid(
                    row=r, column=c_val, padx=(0, 18), pady=1, sticky="w")

            _stat("Minimum:",          stats["min"],     0, 0, 1)
            _stat("Maximum:",          stats["max"],     0, 2, 3)
            _stat("Mean:",             stats["mean"],    1, 0, 1)
            _stat("Median:",           stats["median"],  1, 2, 3)
            _stat("Std. deviation:",   stats["std"],     2, 0, 1)
            missing_val = f"{stats['n_missing']} ({stats['missing_pct']}%)"
            _stat("Missing values:",   missing_val,      2, 2, 3)
            row += 1

        elif "n_unique" in stats:
            ctk.CTkLabel(self._clean_detail_inner,
                         text=f"Unique values: {stats['n_unique']}   "
                              f"Missing values: {stats['n_missing']} ({stats['missing_pct']}%)",
                         font=ctk.CTkFont(size=10),
                         anchor="w").grid(row=row, column=0, padx=8, pady=1, sticky="w")
            row += 1

        if info["issues"]:
            issue_text = "; ".join(info["issues"])
            ctk.CTkLabel(self._clean_detail_inner,
                         text=f"Issues: {issue_text}",
                         text_color="#E07818", font=ctk.CTkFont(size=10),
                         wraplength=420, justify="left",
                         anchor="w").grid(row=row, column=0, padx=8, pady=(2, 4), sticky="w")
        else:
            ctk.CTkLabel(self._clean_detail_inner, text="Issues: None",
                         font=ctk.CTkFont(size=10),
                         anchor="w").grid(row=row, column=0, padx=8, pady=(2, 4), sticky="w")

    def _clean_apply(self):
        if self._clean_df is None:
            messagebox.showwarning("No Dataset", "Load a dataset first."); return
        try:
            threshold = float(self._clean_outlier_thresh.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Outlier threshold must be a number."); return
        if self._clean_outlier_method.get() in MULTIVARIATE_OUTLIER_METHODS and not (0 < threshold <= 0.5):
            # sklearn's IsolationForest/LOF/EllipticEnvelope all require contamination in (0, 0.5]
            messagebox.showerror(
                "Invalid Input",
                "Contamination must be between 0 and 0.5 (e.g. 0.1 for 10%) for "
                f"{self._clean_outlier_method.get()}.")
            return
        try:
            min_run = int(self._clean_stuck_min_run.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Stuck min run must be an integer."); return

        # apply_cleaning (outlier detection especially, for the multivariate methods)
        # and the re-profiling auto_classify_columns pass below used to run on the main
        # thread and freeze the whole window on a slow method or a large dataset.
        # Every Tk widget value the worker needs is read here, on the main thread,
        # BEFORE spawning it — a background thread must never touch Tk widgets itself.
        self._clean_apply_btn.configure(state="disabled", text="Applying...")
        df_original         = self._clean_df_original
        col_info_snapshot   = self._clean_col_info
        missing_action      = self._clean_missing_action.get()
        outlier_method      = self._clean_outlier_method.get()
        outlier_action      = self._clean_outlier_action.get()
        stuck_enabled       = self._clean_stuck_enabled.get()
        stuck_action        = self._clean_stuck_action.get()
        remove_duplicates   = self._clean_remove_dupes.get()
        try:
            base_seed = int(self._random_seed_entry.get().strip() or 0)
        except ValueError:
            base_seed = 0
        outlier_seed = derive_seed(base_seed, SEED_OFFSET_OUTLIER_DETECTION)

        def worker():
            try:
                cleaned, log_lines = apply_cleaning(
                    df=df_original,
                    col_info=col_info_snapshot,
                    missing_action=missing_action,
                    outlier_method=outlier_method,
                    outlier_threshold=threshold,
                    outlier_action=outlier_action,
                    stuck_enabled=stuck_enabled,
                    stuck_min_run=min_run,
                    stuck_action=stuck_action,
                    remove_duplicates=remove_duplicates,
                    random_state=outlier_seed,
                )
                remaining = auto_classify_columns(cleaned)
            except Exception as exc:
                self._callback_queue.put(lambda: self._finish_clean_apply(None, None, None, exc))
                return
            self._callback_queue.put(lambda: self._finish_clean_apply(cleaned, log_lines, remaining, None))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_clean_apply(self, cleaned, log_lines, remaining, exc):
        self._clean_apply_btn.configure(state="normal", text="Apply")
        if exc is not None:
            messagebox.showerror("Cleaning Failed", f"Could not apply cleaning:\n{exc}")
            return

        self._clean_df = cleaned

        # Refresh stats for columns that survived cleaning; keep excluded columns
        # in col_info so a second Apply still knows to exclude them.
        for col, info in remaining.items():
            if col in self._clean_col_info:
                user_role = self._clean_col_info[col]["role"]
                self._clean_col_info[col] = info
                self._clean_col_info[col]["role"] = user_role
            else:
                # New column (e.g. converted timestamp now numeric)
                self._clean_col_info[col] = info

        self._clean_populate_column_manager()
        self._clean_populate_data_preview()
        for line in log_lines:
            self._clean_log(line)
            if line.startswith("[FIX "):
                self._clean_fix_log.append(line.split("]", 1)[1].strip())

    def _clean_export(self):
        df = self._clean_df if self._clean_df is not None else self._clean_df_original
        if df is None:
            messagebox.showwarning("No Dataset", "Load a dataset first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        df.to_csv(path, index=False)
        fname = os.path.basename(path)
        self._clean_log(
            f"[INFO] Exported: {fname} ({len(df)} rows x {len(df.columns)} columns)")

        original = self._clean_df_original
        self.session.cleaning_summary = {
            "rows_before": int(len(original)) if original is not None else None,
            "rows_after": int(len(df)),
            "actions": list(self._clean_fix_log),
            "export_path": os.path.abspath(path),
        }

        messagebox.showinfo("Export Complete", f"Saved to:\n{path}")

    def _clean_log(self, msg: str):
        if not hasattr(self, "_clean_log_box"):
            return
        self._clean_log_box.configure(state="normal")
        self._clean_log_box.insert("end", msg + "\n")
        self._clean_log_box.see("end")
        self._clean_log_box.configure(state="disabled")

    def _clean_clear_log(self):
        self._clean_log_box.configure(state="normal")
        self._clean_log_box.delete("1.0", "end")
        self._clean_log_box.configure(state="disabled")

    # ── Physics Modelling Tab ─────────────────────────────────────────────────

    def _setup_physics_modelling_tab(self):
        tab = self.tabview.tab("Physics Modelling")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        sf = ctk.CTkScrollableFrame(tab)
        sf.pack(fill="both", expand=True, padx=4, pady=4)
        sf.columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(sf, text="Physics Modelling",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=row, column=0, columnspan=3, padx=10, pady=(10, 2), sticky="w"); row += 1
        ctk.CTkLabel(sf,
                     text="Define physics equations to generate residual or physics-validated datasets for the ML workflow.").grid(
            row=row, column=0, columnspan=3, padx=10, pady=(0, 8), sticky="w"); row += 1

        # Input dataset
        ctk.CTkLabel(sf, text="Input Dataset",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, columnspan=3, padx=10, pady=(8, 2), sticky="w"); row += 1
        ctk.CTkLabel(sf, text="Dataset Path:", anchor="w", width=240).grid(
            row=row, column=0, padx=10, pady=4, sticky="w")
        self._phys_dataset_entry = ctk.CTkEntry(sf, placeholder_text="Path to input .csv file")
        self._phys_dataset_entry.grid(row=row, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(sf, text="Browse", width=80,
                      command=self._browse_phys_dataset).grid(
            row=row, column=2, padx=(4, 10), pady=4); row += 1

        # Mode selector
        ctk.CTkLabel(sf, text="Mode",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, columnspan=3, padx=10, pady=(12, 2), sticky="w"); row += 1
        mf = ctk.CTkFrame(sf)
        mf.grid(row=row, column=0, columnspan=3, padx=10, pady=4, sticky="ew"); row += 1
        self._phys_mode_var = tk.StringVar(value="Script")
        ctk.CTkRadioButton(mf, text="Script Mode",
                           variable=self._phys_mode_var, value="Script",
                           command=self._refresh_physics_mode).pack(side="left", padx=12, pady=6)
        ctk.CTkRadioButton(mf, text="Expression Mode",
                           variable=self._phys_mode_var, value="Expression",
                           command=self._refresh_physics_mode).pack(side="left", padx=12)

        # Mode content wrapper — panels are pack-managed inside here, one visible at a time
        self._phys_mode_frame = ctk.CTkFrame(sf, fg_color="transparent")
        self._phys_mode_frame.grid(row=row, column=0, columnspan=3, sticky="ew"); row += 1
        self._phys_mode_frame.columnconfigure(0, weight=1)

        # Script panel
        self._phys_script_panel = ctk.CTkFrame(self._phys_mode_frame, fg_color="transparent")
        self._phys_script_panel.columnconfigure(1, weight=1)
        ctk.CTkLabel(self._phys_script_panel,
                     text="Point to a Python (.py) file that defines your physics equations.\n"
                          "The script must define four things:\n"
                          "  • governing_function(inputs, constants, time)  — returns a DataFrame of physics estimates\n"
                          "  • constants = {...}   — dictionary of physical constants used in your equations\n"
                          "  • input_vars  = [...]  — list of column names from your dataset that are inputs\n"
                          "  • output_vars = [...]  — list of column names from your dataset that are outputs\n"
                          "For a worked example, run 'phoenix-ml --get-examples' and see\n"
                          "examples/DC_Motors_Dataset_Generation.py in the copied folder.",
                     wraplength=620, justify="left").grid(
            row=0, column=0, columnspan=3, padx=10, pady=(6, 6), sticky="w")
        ctk.CTkLabel(self._phys_script_panel, text="Physics Script:", anchor="w", width=140).grid(
            row=1, column=0, padx=10, pady=4, sticky="w")
        self._phys_script_entry = ctk.CTkEntry(
            self._phys_script_panel,
            placeholder_text="Path to .py file")
        self._phys_script_entry.grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(self._phys_script_panel, text="Browse", width=80,
                      command=self._browse_physics_script).grid(
            row=1, column=2, padx=(4, 10), pady=4)
        ctk.CTkButton(self._phys_script_panel, text="Validate Script", width=130,
                      command=self._phys_validate_script).grid(
            row=2, column=0, padx=10, pady=(2, 6), sticky="w")

        # Expression panel
        self._phys_expr_panel = ctk.CTkFrame(self._phys_mode_frame, fg_color="transparent")
        self._phys_expr_panel.columnconfigure(0, weight=1)
        self._phys_expr_panel.columnconfigure(1, weight=0)

        # Left: description
        desc_frame = ctk.CTkFrame(self._phys_expr_panel, fg_color="transparent")
        desc_frame.grid(row=0, column=0, padx=(10, 4), pady=(6, 6), sticky="nw")
        ctk.CTkLabel(desc_frame,
                     text="Syntax:  NewColumn = expression",
                     font=ctk.CTkFont(weight="bold"),
                     anchor="w").pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(desc_frame,
                     text="• Use dataset column names as variables — spelling must match exactly.\n"
                          "• Column names with spaces or special characters must be wrapped in\n"
                          "  backticks:  e.g.  `Column Name`  or  `dw/dt`\n"
                          "• Expressions run top-to-bottom — a later expression can use a column\n"
                          "  created by an earlier one.\n"
                          "• Assigning to an existing column name overwrites it; a new name\n"
                          "  appends a new column to the dataset.",
                     justify="left", anchor="w").pack(anchor="w")

        # Right: function reference card
        ref_frame = ctk.CTkFrame(self._phys_expr_panel, corner_radius=6)
        ref_frame.grid(row=0, column=1, padx=(0, 10), pady=(6, 6), sticky="ne")
        ctk.CTkLabel(ref_frame, text="Supported Operations",
                     font=ctk.CTkFont(weight="bold", size=11),
                     anchor="w").pack(anchor="w", padx=10, pady=(8, 2))
        ctk.CTkLabel(ref_frame,
                     text="Arithmetic:   + − * /  ** (power)\n"
                          "\n"
                          "sqrt( )    exp( )    abs( )\n"
                          "ln( )      log( )    log2( )    log10( )\n"
                          "sin( )     cos( )    tan( )\n"
                          "asin( )    acos( )   atan( )\n"
                          "sinh( )    cosh( )   tanh( )\n"
                          "floor( )   ceil( )   erf( )\n"
                          "gradient( )  ← finite difference\n"
                          "atan2(y, x)  ← full-circle arctangent\n"
                          "\n"
                          "Constant:   pi",
                     justify="left", anchor="w",
                     font=ctk.CTkFont(family="Courier New", size=11)).pack(
            anchor="w", padx=10, pady=(0, 8))

        self._phys_expressions = []
        self._phys_expr_list = ctk.CTkFrame(self._phys_expr_panel, fg_color="transparent")
        self._phys_expr_list.grid(row=1, column=0, sticky="ew")
        self._phys_expr_list.columnconfigure(0, weight=1)

        ctk.CTkButton(self._phys_expr_panel, text="+ Add Expression", width=160,
                      command=self._phys_add_expression).grid(
            row=2, column=0, padx=10, pady=(4, 10), sticky="w")

        ctk.CTkLabel(self._phys_expr_panel, text="Final Output Columns (optional):",
                     anchor="w").grid(row=3, column=0, padx=10, pady=(4, 2), sticky="w")
        self._phys_output_cols = ctk.CTkEntry(
            self._phys_expr_panel,
            placeholder_text="Comma-separated column names to keep (leave blank to keep all columns)")
        self._phys_output_cols.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")

        self._phys_add_expression()

        ctk.CTkLabel(sf, text="Output Path:", anchor="w", width=240).grid(
            row=row, column=0, padx=10, pady=4, sticky="w")
        self._phys_output_entry = ctk.CTkEntry(sf, placeholder_text="Save physics-processed .csv to...")
        self._phys_output_entry.grid(row=row, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(sf, text="Browse", width=80,
                      command=self._browse_physics_output).grid(
            row=row, column=2, padx=(4, 10), pady=4); row += 1
        self._phys_generate_btn = ctk.CTkButton(
            sf, text="Generate Dataset", width=200,
            command=self._phys_generate_dataset)
        self._phys_generate_btn.grid(
            row=row, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w"); row += 1

        # PERL config save row
        perl_row = ctk.CTkFrame(sf, fg_color="transparent")
        perl_row.grid(row=row, column=0, columnspan=3, padx=10, pady=(0, 4), sticky="ew")
        perl_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(perl_row, text="Configuration Path:", anchor="w", width=160).grid(
            row=0, column=0, padx=(0, 4), pady=4, sticky="w")
        self._phys_perl_config_entry = ctk.CTkEntry(
            perl_row, placeholder_text="Path to save the physics configuration (.json)")
        self._phys_perl_config_entry.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(perl_row, text="Browse", width=80,
                      command=self._browse_perl_config_save).grid(
            row=0, column=2, padx=(4, 0), pady=4)
        self._phys_perl_save_btn = ctk.CTkButton(
            sf, text="Save Physics Configuration", width=220,
            command=self._phys_save_perl_config)
        self._phys_perl_save_btn.grid(
            row=row + 1, column=0, columnspan=3, padx=10, pady=(0, 4), sticky="w")
        self._phys_perl_status = ctk.CTkLabel(
            sf, text="", anchor="w", text_color="#4CAF50")
        self._phys_perl_status.grid(
            row=row + 2, column=0, columnspan=3, padx=10, pady=(0, 12), sticky="w")

        self._refresh_physics_mode()

    def _refresh_physics_mode(self):
        mode = self._phys_mode_var.get()
        if mode == "Script":
            self._phys_expr_panel.pack_forget()
            self._phys_script_panel.pack(fill="x")
        else:
            self._phys_script_panel.pack_forget()
            self._phys_expr_panel.pack(fill="x")

    # ── Expression Mode: expression list ─────────────────────────────────────

    def _phys_add_expression(self, initial_text: str = ""):
        row_frame = ctk.CTkFrame(self._phys_expr_list)
        row_frame.pack(fill="x", padx=4, pady=4)
        row_frame.columnconfigure(0, weight=1)

        entry = ctk.CTkEntry(row_frame, placeholder_text="Expression 1")
        entry.grid(row=0, column=0, padx=(8, 4), pady=(8, 2), sticky="ew")
        if initial_text:
            entry.insert(0, initial_text)

        preview_lbl = ctk.CTkLabel(row_frame, text="", image=None, anchor="w")
        preview_lbl.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        expr = {"frame": row_frame, "entry": entry, "preview": preview_lbl,
                "after_id": None, "image": None}
        self._phys_expressions.append(expr)

        remove_btn = ctk.CTkButton(row_frame, text="Remove", width=70,
                                    fg_color="#8a3a3a", hover_color="#6e2e2e",
                                    command=lambda: self._phys_remove_expression(expr))
        remove_btn.grid(row=0, column=1, padx=(0, 8), pady=(8, 2))

        entry.bind("<KeyRelease>", lambda e, x=expr: self._phys_schedule_preview(x))
        self._phys_renumber_expressions()
        self._phys_schedule_preview(expr)

    def _phys_remove_expression(self, expr: dict):
        if expr.get("after_id"):
            self.after_cancel(expr["after_id"])
        expr["frame"].destroy()
        self._phys_expressions.remove(expr)
        self._phys_renumber_expressions()

    def _phys_renumber_expressions(self):
        for i, expr in enumerate(self._phys_expressions, start=1):
            expr["entry"].configure(placeholder_text=f"Expression {i}")

    def _phys_schedule_preview(self, expr: dict):
        if expr.get("after_id"):
            self.after_cancel(expr["after_id"])
        expr["after_id"] = self.after(400, lambda: self._phys_update_preview(expr))

    def _phys_update_preview(self, expr: dict):
        expr["after_id"] = None
        text = expr["entry"].get().strip()
        if not text:
            expr["preview"].configure(text="", image=None)
            return
        try:
            lhs, tree, name_map = parse_expression(text)
            latex = expression_to_latex(lhs, tree, name_map)
            img = self._phys_render_latex(latex)
            expr["image"] = img  # keep a reference, CTkImage is GC'd otherwise
            expr["preview"].configure(image=img, text="")
        except ExpressionError as e:
            expr["preview"].configure(image=None, text=f"⚠ {e}", text_color="#c0392b")

    def _phys_render_latex(self, latex_str: str) -> ctk.CTkImage:
        fig = Figure(figsize=(0.01, 0.01), dpi=200)
        fig.text(0, 0, f"${latex_str}$", fontsize=15, color="black")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15, facecolor="white")
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        w, h = img.size
        scale = 30 / h
        if w * scale > 520:
            scale = 520 / w
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)

    def _phys_validate_script(self):
        """Import the selected script and check it has all required exports."""
        script_path = self._phys_script_entry.get().strip()
        if not script_path or not os.path.isfile(script_path):
            messagebox.showerror("Physics Modelling", "Select a valid physics script (.py) first.")
            return
        try:
            module = import_physics_script(script_path)
            missing = [a for a in ("governing_function", "constants", "input_vars", "output_vars")
                       if not hasattr(module, a)]
            if missing:
                messagebox.showerror("Physics Modelling",
                                     f"Script is missing required definitions:\n  {', '.join(missing)}\n\n"
                                     "For the expected format, run 'phoenix-ml --get-examples' and see "
                                     "examples/DC_Motors_Dataset_Generation.py in the copied folder.")
                return
            gf = module.governing_function
            if not callable(gf):
                messagebox.showerror("Physics Modelling", "'governing_function' must be a callable function.")
                return
            messagebox.showinfo(
                "Physics Modelling",
                f"Script validated successfully.\n\n"
                f"  input_vars  : {module.input_vars}\n"
                f"  output_vars : {module.output_vars}\n"
                f"  constants   : {list(module.constants.keys())}")
        except Exception as e:
            messagebox.showerror("Physics Modelling", f"Failed to load script:\n{e}")

    def _phys_generate_dataset(self):
        # Reading the dataset, running the physics model/expressions, and writing the
        # output .csv used to run on the main thread and freeze the whole window on a
        # large dataset or a slow script. Every Tk widget value the worker needs is
        # read here, on the main thread, before spawning it — a background thread must
        # never touch Tk widgets itself.
        mode = self._phys_mode_var.get()
        dataset_path = self._phys_dataset_entry.get().strip()
        output_path  = self._phys_output_entry.get().strip()
        if not dataset_path or not os.path.isfile(dataset_path):
            messagebox.showerror("Physics Modelling", "Select a valid input dataset (.csv) first.")
            return
        if not output_path:
            messagebox.showerror("Physics Modelling", "Choose an output path for the generated dataset.")
            return

        if mode == "Script":
            script_path = self._phys_script_entry.get().strip()
            if not script_path or not os.path.isfile(script_path):
                messagebox.showerror("Physics Modelling", "Select a valid physics script (.py) first.")
                return

            self._phys_generate_btn.configure(state="disabled", text="Generating...")

            def worker():
                try:
                    module = import_physics_script(script_path)
                    missing = [a for a in ("governing_function", "constants", "input_vars", "output_vars")
                               if not hasattr(module, a)]
                    if missing:
                        self._callback_queue.put(lambda: self._finish_phys_generate(
                            None, None, output_path, f"Script is missing: {', '.join(missing)}"))
                        return
                    df = pd.read_csv(dataset_path)
                    name_mapping = getattr(module, "name_mapping", None)
                    time_col     = getattr(module, "time_col", None)
                    physics_df = run_physics_model(
                        df, time_col, module.governing_function, module.constants,
                        module.input_vars, module.output_vars, name_mapping,
                    )
                    result_df = generate_residual_dataset(
                        df, physics_df, module.input_vars, module.output_vars, name_mapping,
                    )
                    result_df.to_csv(output_path, index=False)
                except Exception as e:
                    self._callback_queue.put(lambda: self._finish_phys_generate(
                        None, None, output_path, f"Failed to generate dataset:\n{e}"))
                    return
                self._callback_queue.put(lambda: self._finish_phys_generate(result_df, None, output_path, None))

            threading.Thread(target=worker, daemon=True).start()
            return

        # Expression mode
        texts = [x["entry"].get() for x in self._phys_expressions if x["entry"].get().strip()]
        if not texts:
            messagebox.showerror("Physics Modelling", "Add at least one expression.")
            return
        output_cols_text = self._phys_output_cols.get()

        self._phys_generate_btn.configure(state="disabled", text="Generating...")

        def worker():
            try:
                df = pd.read_csv(dataset_path)
                result_df, log = apply_expressions(df, texts)
                result_df = select_output_columns(result_df, output_cols_text)
                result_df.to_csv(output_path, index=False)
            except ExpressionError as e:
                self._callback_queue.put(lambda: self._finish_phys_generate(None, None, output_path, str(e)))
                return
            except Exception as e:
                self._callback_queue.put(lambda: self._finish_phys_generate(
                    None, None, output_path, f"Failed to generate dataset: {e}"))
                return
            self._callback_queue.put(lambda: self._finish_phys_generate(result_df, log, output_path, None))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_phys_generate(self, result_df, log, output_path, error):
        self._phys_generate_btn.configure(state="normal", text="Generate Dataset")
        if error is not None:
            messagebox.showerror("Physics Modelling", error)
            return
        detail = ("\n\n" + "\n".join(log)) if log else ""
        messagebox.showinfo(
            "Physics Modelling",
            f"Saved {len(result_df)} rows × {len(result_df.columns)} columns to:\n{output_path}"
            + detail)

    # ── Home Tab ──────────────────────────────────────────────────────────────

    def _setup_home_tab(self):
        tab = self.tabview.tab("Home")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)

        top = ctk.CTkFrame(tab)
        top.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 3))
        top.columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="Dataset Path:", anchor="w", width=120).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        self.dataset_entry = ctk.CTkEntry(top, placeholder_text="Path to .csv file")
        self.dataset_entry.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(top, text="Browse", width=80, command=self._browse_dataset).grid(row=0, column=2, padx=(4, 8))

        ctk.CTkLabel(top, text="Output Directory:", anchor="w", width=120).grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self.output_entry = ctk.CTkEntry(top, placeholder_text="Folder for results")
        self.output_entry.grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(top, text="Browse", width=80, command=self._browse_output).grid(row=1, column=2, padx=(4, 8))

        ctk.CTkLabel(top, text="Targets:", anchor="w", width=120).grid(row=2, column=0, padx=8, pady=4, sticky="w")
        self.targets_entry = ctk.CTkEntry(top, placeholder_text="Comma-separated target column names")
        self.targets_entry.grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(top, text="Select...", width=80, command=self._open_target_picker).grid(
            row=2, column=2, padx=(4, 8))

        ctk.CTkLabel(top, text="Physics Configuration:", anchor="w", width=150).grid(row=3, column=0, padx=8, pady=4, sticky="w")
        self.perl_config_entry = ctk.CTkEntry(top, placeholder_text="(optional) Path to a saved physics configuration file")
        self.perl_config_entry.grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        self.perl_config_entry.bind("<FocusOut>", lambda e: self._on_perl_config_entry_change())
        self.perl_config_entry.bind("<Return>",   lambda e: self._on_perl_config_entry_change())
        ctk.CTkButton(top, text="Browse", width=80, command=self._browse_perl_config_load).grid(row=3, column=2, padx=(4, 8))

        ctk.CTkLabel(top, text="Random Seed:", anchor="w", width=150).grid(row=4, column=0, padx=8, pady=4, sticky="w")
        self._random_seed_entry = ctk.CTkEntry(top, placeholder_text="0")
        self._random_seed_entry.insert(0, "0")
        self._random_seed_entry.grid(row=4, column=1, padx=4, pady=4, sticky="w")
        ctk.CTkLabel(
            top, text="Random seed is applied throughout where randomness appears. "
                      "The same seed and configuration reproduces identical results.",
            # Same colour as every other label in the UI (no grey de-emphasis).
            font=ctk.CTkFont(size=11), anchor="w", text_color=_LBL_NORMAL,
        ).grid(row=5, column=0, columnspan=3, padx=8, pady=(0, 4), sticky="w")

        # Pipeline table
        pipe = ctk.CTkFrame(tab)
        pipe.grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        pipe.columnconfigure(1, weight=1)

        ctk.CTkLabel(pipe, text="Workflow Steps", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, columnspan=5, padx=8, pady=(6, 2), sticky="w")
        ctk.CTkLabel(pipe, text="Enable", width=55).grid(row=1, column=0, padx=(8, 0))
        ctk.CTkLabel(pipe, text="Step",   width=240, anchor="w").grid(row=1, column=1, padx=4)
        ctk.CTkLabel(pipe, text="Time",   width=70).grid(row=1, column=2, padx=4)
        ctk.CTkLabel(pipe, text="Status", width=100).grid(row=1, column=3, padx=8)

        self._step_enable_vars:   dict[str, tk.BooleanVar]  = {}
        self._step_status_labels: dict[str, ctk.CTkLabel]   = {}
        self._step_run_btns:      dict[str, ctk.CTkButton]  = {}

        for i, (key, name, _) in enumerate(STEPS):
            r = i + 2
            # "Before HPO" passes (UQ and Interpretability) are opt-in — useful for
            # before/after comparison, but only the "After HPO" pass is strictly
            # necessary for a normal run, so default them off along with PERL.
            var = tk.BooleanVar(value=(key not in ("perl", "uq_before", "interpretability_before")))
            self._step_enable_vars[key] = var
            ctk.CTkCheckBox(pipe, text="", variable=var, width=30).grid(row=r, column=0, padx=(12, 0), pady=3)
            ctk.CTkLabel(pipe, text=name, width=240, anchor="w").grid(row=r, column=1, padx=4, pady=3, sticky="w")
            tlbl = ctk.CTkLabel(pipe, text="--", width=70)
            tlbl.grid(row=r, column=2, padx=4, pady=3)
            self._step_timer_labels[key] = tlbl
            slbl = ctk.CTkLabel(pipe, text="Not Run", width=100)
            slbl.grid(row=r, column=3, padx=8, pady=3)
            self._step_status_labels[key] = slbl
            btn = ctk.CTkButton(pipe, text="Run", width=70, state="disabled",
                                command=lambda k=key: self._run_single_step(k))
            btn.grid(row=r, column=4, padx=(4, 10), pady=3)
            self._step_run_btns[key] = btn

        # Actions
        actions = ctk.CTkFrame(tab, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=6, pady=(4, 2))
        self.run_all_btn = ctk.CTkButton(actions, text="  Run All Enabled Steps",
                                         command=self._run_all_steps, width=220)
        self.run_all_btn.pack(side="left", padx=4)
        self.pause_btn = ctk.CTkButton(actions, text="Pause", command=self._toggle_pause,
                                       width=90, state="disabled")
        self.pause_btn.pack(side="left", padx=4)
        self.stop_btn = ctk.CTkButton(actions, text="Stop", command=self._stop_workflow,
                                      width=90, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.report_btn = ctk.CTkButton(actions, text="Generate Report",
                                        command=self._generate_report, width=160, state="disabled")
        self.report_btn.pack(side="left", padx=4)
        # Same construction pattern as the buttons above (no colour/style override) so
        # it reads as part of the same row, not a distinct "danger" action.
        self.reset_btn = ctk.CTkButton(actions, text="Reset", command=self._reset_workflow,
                                       width=90)
        self.reset_btn.pack(side="left", padx=4)
        self._report_progress = ctk.CTkProgressBar(actions, width=160, mode="indeterminate")
        self._report_status_lbl = ctk.CTkLabel(
            actions, text="", font=ctk.CTkFont(size=11), text_color="#E07818")
        # Not packed until report generation starts — see _generate_report/_finish_report

        # Log
        log_hdr = ctk.CTkFrame(tab, fg_color="transparent")
        log_hdr.grid(row=3, column=0, sticky="new", padx=6, pady=(4, 0))
        ctk.CTkLabel(log_hdr, text="Progress Log", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(log_hdr, text="Clear", width=60, command=self._clear_log).pack(side="right")

        tab.rowconfigure(4, weight=1, minsize=120)
        self.log_textbox = ctk.CTkTextbox(tab, state="disabled",
                                          font=ctk.CTkFont(family="Courier New", size=11))
        self.log_textbox.grid(row=4, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self._refresh_buttons()

    # ── Models Tab ────────────────────────────────────────────────────────────

    def _setup_models_tab(self):
        tab = self.tabview.tab("Models")
        ctk.CTkLabel(tab, text="Select Models", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(10, 4), sticky="w")

        self._model_vars: dict[str, tk.BooleanVar] = {}
        for i, name in enumerate(ALL_MODEL_NAMES):
            var = tk.BooleanVar(value=(name in DEFAULT_ON))
            self._model_vars[name] = var
            ctk.CTkCheckBox(tab, text=name, variable=var).grid(
                row=i + 1, column=0, padx=20, pady=3, sticky="w")
        self._model_vars["Gaussian Process Regressor"].trace_add(
            "write", self._refresh_gp_posterior_state
        )
        for _mono_name in ("XGBoost Regressor", "LGBM Regressor"):
            self._model_vars[_mono_name].trace_add("write", self._refresh_monotonic_button_state)

        btn_row = len(ALL_MODEL_NAMES) + 1
        bf = ctk.CTkFrame(tab, fg_color="transparent")
        bf.grid(row=btn_row, column=0, padx=10, pady=8, sticky="w")
        ctk.CTkButton(bf, text="Select All",   width=110,
                      command=lambda: [v.set(True)  for v in self._model_vars.values()]).pack(side="left", padx=4)
        ctk.CTkButton(bf, text="Deselect All", width=110,
                      command=lambda: [v.set(False) for v in self._model_vars.values()]).pack(side="left", padx=4)
        self._monotonic_btn = ctk.CTkButton(bf, text="Monotonicity Constraints...", width=190,
                                            command=self._open_monotonic_picker, state="disabled")
        self._monotonic_btn.pack(side="left", padx=4)
        self._refresh_monotonic_button_state()

        # ── Report Metrics ────────────────────────────────────────────────────
        hdr_row = btn_row + 1
        ctk.CTkLabel(tab, text="Report Metrics",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=hdr_row, column=0, columnspan=2, padx=10, pady=(10, 2), sticky="w")
        ctk.CTkLabel(tab, text="Metrics shown in the PDF training table  (Excel always exports all)",
                     font=ctk.CTkFont(size=11)).grid(
            row=hdr_row + 1, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="w")

        # (key, display label, default on)
        _report_metric_defs = [
            ("MSE",              "Mean Squared Error (MSE)",                   False),
            ("RMSE",             "Root Mean Squared Error (RMSE)",             True),
            ("NRMSE",            "Normalised Root Mean Squared Error (NRMSE)", False),
            ("MAPE",             "Mean Absolute Percentage Error (MAPE)",      False),
            ("R^2",              "R² (Coefficient of Determination)",          False),
            ("ADJUSTED R^2",     "Adjusted R²",                                True),
            ("Q^2",              "Nash-Sutcliffe Efficiency (Q²/NSE)",         True),
            ("KGE",              "Kling-Gupta Efficiency (KGE)",               False),
            ("Time Elapsed (s)", "Time Elapsed (s)",                          True),
        ]
        self._report_metric_vars: dict[str, tk.BooleanVar] = {}
        cb_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cb_frame.grid(row=hdr_row + 2, column=0, columnspan=2, padx=10, pady=2, sticky="w")
        n_cols = 2
        for idx, (key, label, default) in enumerate(_report_metric_defs):
            var = tk.BooleanVar(value=default)
            self._report_metric_vars[key] = var
            ctk.CTkCheckBox(cb_frame, text=label, variable=var, width=320).grid(
                row=idx // n_cols, column=idx % n_cols, padx=6, pady=2, sticky="w")

    # ── Preprocessing Tab ─────────────────────────────────────────────────────

    def _setup_preprocessing_tab(self):
        tab = self.tabview.tab("Preprocessing")
        tab.columnconfigure(1, weight=1)
        self._preproc_test_size   = self._lbl_entry(tab, "Test Size:",     "0.2",   0)
        self._preproc_split       = self._lbl_option(tab, "Split Method:",
                                                     ["First", "Last", "Random"], 1)
        self._preproc_split.configure(command=self._refresh_preproc_state)
        self._preproc_scaler      = self._lbl_option(tab, "Feature Scaling:",
                                                     ["Standard", "MinMax", "Robust", "None"], 2)
        ctk.CTkLabel(
            tab, text="Random split uses the Random Seed set on the Home tab.",
            font=ctk.CTkFont(size=11), anchor="w", text_color=_LBL_NORMAL,
        ).grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="w")
        self._preproc_target_vs_target   = self._lbl_check(tab, "Show Target vs Target Plot",                                    True,  4)
        self._preproc_feat_vs_target     = self._lbl_check(tab, "Show Feature vs Target Scatter Plots",                          True,  5)
        self._preproc_boxplots           = self._lbl_check(tab, "Show Boxplots",                                                 True,  6)

        self._preproc_dist_corr          = self._lbl_check(tab, "Show Distance Correlation Matrix",                              True,  7)
        self._preproc_dist_corr.trace_add("write", self._refresh_preproc_state)

        # Sub-options — inline/"tabbed" and indented to show they depend on the checkbox above.
        dist_sub = ctk.CTkFrame(tab, fg_color="transparent")
        dist_sub.grid(row=8, column=0, columnspan=2, padx=(40, 10), pady=(0, 4), sticky="w")
        self._preproc_dist_dummy = tk.BooleanVar(value=True)
        self._preproc_dist_dummy_cb = ctk.CTkCheckBox(
            dist_sub, text="Include Dummy (Noise Baseline)", variable=self._preproc_dist_dummy)
        self._preproc_dist_dummy_cb.pack(side="left", padx=(0, 16))
        self._preproc_dist_mp = tk.BooleanVar(value=False)
        self._preproc_dist_mp_cb = ctk.CTkCheckBox(
            dist_sub, text="Marchenko-Pastur Denoising", variable=self._preproc_dist_mp)
        self._preproc_dist_mp_cb.pack(side="left")

        self._preproc_multicollinearity  = self._lbl_check(tab, "Show Multicollinearity (VIF + Condition Number)",              True,  9)
        ctk.CTkLabel(
            tab, text="VIF = Variance Inflation Factor",
            font=ctk.CTkFont(size=11), anchor="w",
        ).grid(row=10, column=0, columnspan=2, padx=(40, 10), pady=(0, 4), sticky="w")

        self._preproc_feat_sel           = self._lbl_check(tab, "Show Feature Selection: Advisory Flags",                       True,  11)
        self._preproc_feat_sel.trace_add("write", self._refresh_preproc_state)
        self._preproc_feat_sel_thresh_lbl, self._preproc_feat_sel_thresh = self._lbl_entry_ref(
            tab, "    Feature Selection Redundancy Threshold:", "0.90", 12)
        self._preproc_pca                = self._lbl_check(tab, "Show Principal Component Analysis (PCA): Scree Plot and Biplot", True,  13)
        self._refresh_preproc_state()

    def _refresh_preproc_state(self, *_):
        if hasattr(self, "_preproc_feat_sel_thresh"):
            fs_on = self._preproc_feat_sel.get()
            self._preproc_feat_sel_thresh.configure(state="normal" if fs_on else "disabled")
            self._preproc_feat_sel_thresh_lbl.configure(
                text_color=("gray20" if fs_on else "gray60")
            )
        if hasattr(self, "_preproc_dist_dummy_cb"):
            dist_on = self._preproc_dist_corr.get()
            self._preproc_dist_dummy_cb.configure(state="normal" if dist_on else "disabled")
            self._preproc_dist_mp_cb.configure(state="normal" if dist_on else "disabled")

    # ── Uncertainty Quantification Tab ────────────────────────────────────────

    def _setup_uq_tab(self):
        tab = self.tabview.tab("Uncertainty Quantification")
        tab.columnconfigure(1, weight=1)

        # Method selection via checkboxes (replaces single option menu)
        ctk.CTkLabel(tab, text="Methods:", anchor="w", width=160).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        method_frame = ctk.CTkFrame(tab, fg_color="transparent")
        method_frame.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self._uq_bootstrap_var    = tk.BooleanVar(value=True)
        self._uq_conformal_var    = tk.BooleanVar(value=True)
        self._uq_gp_posterior_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(method_frame, text="Bootstrapping", variable=self._uq_bootstrap_var,
                        command=self._refresh_uq_state).pack(side="left", padx=(0, 12))
        ctk.CTkCheckBox(method_frame, text="Conformal Prediction", variable=self._uq_conformal_var,
                        command=self._refresh_uq_state).pack(side="left", padx=(0, 12))
        self._uq_gp_posterior_cb = ctk.CTkCheckBox(
            method_frame, text="Gaussian Process Posterior (GPR only)",
            variable=self._uq_gp_posterior_var, state="disabled"
        )
        self._uq_gp_posterior_cb.pack(side="left")
        self._refresh_gp_posterior_state()

        self._uq_n_bootstrap_lbl, self._uq_n_bootstrap = self._lbl_entry_ref(
            tab, "Bootstrap Samples:", "200", 1)
        self._uq_n_jobs_lbl, self._uq_n_jobs = self._lbl_entry_ref(
            tab, "Bootstrap Parallel Jobs:", "1", 2)
        ctk.CTkLabel(tab, text="(1 = off, -1 = all cores)",
                     anchor="w").grid(row=3, column=1, padx=4, sticky="w")
        self._uq_confidence  = self._lbl_entry(tab, "Confidence Interval (%):", "95",   4)
        self._uq_calib_frac_lbl, self._uq_calib_frac = self._lbl_entry_ref(
            tab, "Calibration Fraction:", "0.05", 5)
        self._uq_subsample   = self._lbl_entry(tab, "Subsample Test Size:",     "50",   6)
        self._uq_calibration_var = self._lbl_check(
            tab, "Show Calibration Reporting", True, 7)
        self._refresh_uq_state()

    def _refresh_uq_state(self, *_):
        bs_on   = self._uq_bootstrap_var.get()
        conf_on = self._uq_conformal_var.get()
        for w in (self._uq_n_bootstrap, self._uq_n_jobs):
            w.configure(state="normal" if bs_on else "disabled")
        self._uq_n_bootstrap_lbl.configure(text_color=("gray20" if bs_on else "gray60"))
        self._uq_n_jobs_lbl.configure(text_color=("gray20" if bs_on else "gray60"))
        self._uq_calib_frac.configure(state="normal" if conf_on else "disabled")
        self._uq_calib_frac_lbl.configure(text_color=("gray20" if conf_on else "gray60"))

    def _refresh_gp_posterior_state(self, *_):
        if not hasattr(self, "_uq_gp_posterior_cb"):
            return
        gpr_on = self._model_vars.get("Gaussian Process Regressor", tk.BooleanVar(value=False)).get()
        self._uq_gp_posterior_cb.configure(state="normal" if gpr_on else "disabled")
        if not gpr_on:
            self._uq_gp_posterior_var.set(False)

    def _refresh_monotonic_button_state(self, *_):
        if not hasattr(self, "_monotonic_btn"):
            return
        xgb_on = self._model_vars.get("XGBoost Regressor", tk.BooleanVar(value=False)).get()
        lgbm_on = self._model_vars.get("LGBM Regressor", tk.BooleanVar(value=False)).get()
        self._monotonic_btn.configure(state="normal" if (xgb_on or lgbm_on) else "disabled")

    # ── Interpretability Tab ──────────────────────────────────────────────────

    def _setup_interpretability_tab(self):
        tab = self.tabview.tab("Interpretability")
        tab.columnconfigure(1, weight=1)
        row = 0

        # ── Shared settings ───────────────────────────────────────────────────
        # Every selected model is profiled now (see Home tab's Interpretability
        # Before/After HPO steps) — no single "preferred" model to choose here any more.
        self._interp_test_size = self._lbl_entry(tab, "Test Sample Size:",       "1000", row); row += 1
        self._interp_bg_size   = self._lbl_entry(tab, "Background Sample Size:", "10",   row); row += 1

        # ── ICE / PDP ─────────────────────────────────────────────────────────
        ctk.CTkLabel(tab, text="Individual Conditional Expectation and Partial Dependence Plots (ICE and PDP)",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      padx=10, pady=(12, 2), sticky="w"); row += 1
        self._interp_ice_pdp_var = self._lbl_check(tab, "Show ICE and PDP Plots", True, row); row += 1
        self._interp_ice_pdp_var.trace_add("write", self._refresh_interpretability_state)
        self._interp_subsample_lbl, self._interp_subsample = self._lbl_entry_ref(
            tab, "Subsample:", "250", row); row += 1
        self._interp_grid_res_lbl, self._interp_grid_res = self._lbl_entry_ref(
            tab, "Grid Resolution:", "10", row); row += 1

        # ── ALE ───────────────────────────────────────────────────────────────
        ctk.CTkLabel(tab, text="Accumulated Local Effects (ALE)", font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      padx=10, pady=(12, 2), sticky="w"); row += 1
        self._interp_ale_var = self._lbl_check(
            tab, "Show ALE Plots",
            True, row); row += 1
        self._interp_ale_var.trace_add("write", self._refresh_interpretability_state)

        # ── SHAP ──────────────────────────────────────────────────────────────
        ctk.CTkLabel(tab, text="Shapley Additive Explanations (SHAP)", font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      padx=10, pady=(12, 2), sticky="w"); row += 1
        self._interp_shap_summary_var  = self._lbl_check(tab, "Show SHAP Summary",           True, row); row += 1
        self._interp_shap_dep_var      = self._lbl_check(tab, "Show SHAP Dependence Plots",  True, row); row += 1
        self._interp_shap_wf_var       = self._lbl_check(tab, "Show SHAP Waterfall Plots",   True, row); row += 1
        self._interp_shap_wf_var.trace_add("write", self._refresh_interpretability_state)

        # Waterfall samples count
        self._interp_wf_samples_lbl = ctk.CTkLabel(tab, text="Waterfall Samples:", anchor="w", width=240)
        self._interp_wf_samples_lbl.grid(row=row, column=0, padx=10, pady=4, sticky="w")
        self._interp_wf_n_var = tk.StringVar(value="3")
        self._interp_wf_samples = ctk.CTkEntry(tab, width=60, textvariable=self._interp_wf_n_var)
        self._interp_wf_samples.grid(row=row, column=1, padx=10, pady=4, sticky="w")
        self._interp_wf_n_err_lbl = ctk.CTkLabel(tab, text="", text_color="#CC0000",
                                                   font=ctk.CTkFont(size=11), anchor="w")
        self._interp_wf_n_err_lbl.grid(row=row, column=2, padx=(0, 10), pady=4, sticky="w")
        self._interp_wf_n_var.trace_add("write", self._on_waterfall_n_changed)
        row += 1

        # Percentile slider bank — one vertical slider per waterfall sample
        pct_header = ctk.CTkLabel(tab, text="Percentile positions  (100% = worst error, 0% = best error)",
                                  anchor="w", font=ctk.CTkFont(size=11))
        pct_header.grid(row=row, column=0, columnspan=2, padx=10, pady=(4, 0), sticky="w")
        self._interp_wf_pct_header = pct_header
        row += 1

        self._interp_wf_pct_frame = ctk.CTkFrame(tab)
        self._interp_wf_pct_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=(2, 6), sticky="ew")
        self._waterfall_pct_vars: list[tk.StringVar] = []
        self._waterfall_pct_sliders: list[ctk.CTkSlider] = []
        self._waterfall_pct_entries: list[ctk.CTkEntry] = []
        self._rebuild_waterfall_percentile_sliders(3)
        row += 1

        # ── Global Sensitivity Analysis ──────────────────────────────────────
        ctk.CTkLabel(tab, text="Global Sensitivity Analysis (Morris / Sobol / FAST)",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      padx=10, pady=(12, 2), sticky="w"); row += 1
        self._interp_sens_morris_var = self._lbl_check(
            tab, "Show Morris Screening", True, row); row += 1
        self._interp_sens_morris_traj_lbl, self._interp_sens_morris_traj = self._lbl_entry_ref(
            tab, "Morris Trajectories:", "10", row); row += 1
        self._interp_sens_morris_levels_lbl, self._interp_sens_morris_levels = self._lbl_entry_ref(
            tab, "Morris Levels:", "4", row); row += 1
        self._interp_sens_sobol_var = self._lbl_check(
            tab, "Show Sobol Indices", False, row); row += 1

        # Sobol base sample size as a linked pair: a plain number entry and a
        # power-of-two entry rendered exponent-style ("[N] or 2^[p]"). Editing either
        # updates the other; a non-power-of-two N shows its exponent rounded to at
        # most 3 decimal places. Both boxes accept non-negative integers only.
        self._interp_sens_sobol_n_lbl = ctk.CTkLabel(
            tab, text="Sobol Base Sample Size:", anchor="w", width=240)
        self._interp_sens_sobol_n_lbl.grid(row=row, column=0, padx=10, pady=3, sticky="w")
        _sobol_frame = ctk.CTkFrame(tab, fg_color="transparent")
        _sobol_frame.grid(row=row, column=1, padx=4, pady=3, sticky="w")
        self._interp_sens_sobol_n = ctk.CTkEntry(_sobol_frame, width=80)
        self._interp_sens_sobol_n.insert(0, "512")
        self._interp_sens_sobol_n.grid(row=0, column=0, sticky="s")
        self._interp_sens_sobol_or_lbl = ctk.CTkLabel(_sobol_frame, text="or  2")
        self._interp_sens_sobol_or_lbl.grid(row=0, column=1, padx=(10, 1), sticky="s")
        self._interp_sens_sobol_pow = ctk.CTkEntry(
            _sobol_frame, width=44, height=20, font=ctk.CTkFont(size=10))
        self._interp_sens_sobol_pow.insert(0, "9")
        self._interp_sens_sobol_pow.grid(row=0, column=2, sticky="n", pady=(0, 12))
        row += 1

        self._sobol_sync_guard = False

        def _sobol_sync_from_n(_event=None):
            if self._sobol_sync_guard:
                return
            self._sobol_sync_guard = True
            try:
                txt = self._interp_sens_sobol_n.get().strip()
                if txt.isdigit() and int(txt) > 0:
                    import math
                    p = math.log2(int(txt))
                    p_str = str(int(round(p))) if abs(p - round(p)) < 1e-9 \
                        else f"{p:.3f}".rstrip("0").rstrip(".")
                    self._interp_sens_sobol_pow.delete(0, "end")
                    self._interp_sens_sobol_pow.insert(0, p_str)
            finally:
                self._sobol_sync_guard = False

        def _sobol_sync_from_pow(_event=None):
            if self._sobol_sync_guard:
                return
            self._sobol_sync_guard = True
            try:
                txt = self._interp_sens_sobol_pow.get().strip()
                if txt.isdigit():  # non-negative integer powers only
                    self._interp_sens_sobol_n.delete(0, "end")
                    self._interp_sens_sobol_n.insert(0, str(2 ** int(txt)))
            finally:
                self._sobol_sync_guard = False

        self._interp_sens_sobol_n.bind("<KeyRelease>", _sobol_sync_from_n)
        self._interp_sens_sobol_pow.bind("<KeyRelease>", _sobol_sync_from_pow)

        # FAST sits directly after Sobol: both are variance-based estimators of
        # the same S1/ST indices (when both are ticked the report renders them
        # as one side-by-side agreement plot), unlike Morris which is a
        # different kind of method (screening).
        self._interp_sens_fast_var = self._lbl_check(
            tab, "Show FAST Indices (Fourier Amplitude Sensitivity Testing)", False, row); row += 1
        self._interp_sens_fast_n_lbl, self._interp_sens_fast_n = self._lbl_entry_ref(
            tab, "FAST Samples per Feature:", "512", row); row += 1

        self._interp_sens_morris_var.trace_add("write", self._refresh_interpretability_state)
        self._interp_sens_sobol_var.trace_add("write", self._refresh_interpretability_state)
        self._interp_sens_fast_var.trace_add("write", self._refresh_interpretability_state)

        self._refresh_interpretability_state()

    def _refresh_interpretability_state(self, *_):
        ice_on = self._interp_ice_pdp_var.get()
        ale_on = self._interp_ale_var.get()
        wf_on  = self._interp_shap_wf_var.get()
        self._interp_subsample.configure(state="normal" if ice_on else "disabled")
        self._interp_subsample_lbl.configure(text_color=("gray20" if ice_on else "gray60"))
        # Grid Resolution is shared between ICE/PDP and ALE, so it stays active if either is on.
        grid_res_on = ice_on or ale_on
        self._interp_grid_res.configure(state="normal" if grid_res_on else "disabled")
        self._interp_grid_res_lbl.configure(text_color=("gray20" if grid_res_on else "gray60"))
        self._interp_wf_samples.configure(state="normal" if wf_on else "disabled")
        self._interp_wf_samples_lbl.configure(text_color=("gray20" if wf_on else "gray60"))
        dim_color = "gray20" if wf_on else "gray60"
        for attr in ("_interp_wf_pct_header",):
            if hasattr(self, attr):
                getattr(self, attr).configure(text_color=dim_color)
        for slider in getattr(self, "_waterfall_pct_sliders", []):
            slider.configure(state="normal" if wf_on else "disabled")
        for entry in getattr(self, "_waterfall_pct_entries", []):
            entry.configure(state="normal" if wf_on else "disabled")

        if hasattr(self, "_interp_sens_morris_var"):
            morris_on = self._interp_sens_morris_var.get()
            for w in (self._interp_sens_morris_traj, self._interp_sens_morris_levels):
                w.configure(state="normal" if morris_on else "disabled")
            for lbl in (self._interp_sens_morris_traj_lbl, self._interp_sens_morris_levels_lbl):
                lbl.configure(text_color=("gray20" if morris_on else "gray60"))
            sobol_on = self._interp_sens_sobol_var.get()
            self._interp_sens_sobol_n.configure(state="normal" if sobol_on else "disabled")
            self._interp_sens_sobol_pow.configure(state="normal" if sobol_on else "disabled")
            dim = "gray20" if sobol_on else "gray60"
            self._interp_sens_sobol_n_lbl.configure(text_color=dim)
            self._interp_sens_sobol_or_lbl.configure(text_color=dim)
            fast_on = self._interp_sens_fast_var.get()
            self._interp_sens_fast_n.configure(state="normal" if fast_on else "disabled")
            self._interp_sens_fast_n_lbl.configure(
                text_color=("gray20" if fast_on else "gray60"))

    def _rebuild_waterfall_percentile_sliders(self, n: int) -> None:
        for w in self._interp_wf_pct_frame.winfo_children():
            w.destroy()
        self._waterfall_pct_vars.clear()
        self._waterfall_pct_sliders.clear()
        self._waterfall_pct_entries.clear()

        defaults = [round(100 * (n - 1 - i) / max(n - 1, 1)) for i in range(n)]

        for i, default in enumerate(defaults):
            col = ctk.CTkFrame(self._interp_wf_pct_frame, fg_color="transparent")
            col.pack(side="left", padx=5, pady=6)

            ctk.CTkLabel(col, text=f"S{i + 1}", font=ctk.CTkFont(size=10)).pack()

            slider = ctk.CTkSlider(
                col, from_=0, to=100, number_of_steps=100,
                orientation="vertical", height=90, width=16,
                command=lambda v, idx=i: self._on_waterfall_slider(idx, v),
            )
            slider.set(float(default))
            slider.pack()
            self._waterfall_pct_sliders.append(slider)

            var = tk.StringVar(value=str(default))
            self._waterfall_pct_vars.append(var)
            entry = ctk.CTkEntry(col, width=42, textvariable=var)
            entry.bind("<FocusOut>", lambda e, idx=i: self._on_waterfall_entry(idx))
            entry.bind("<Return>",   lambda e, idx=i: self._on_waterfall_entry(idx))
            entry.pack(pady=(2, 0))
            self._waterfall_pct_entries.append(entry)

    def _get_wf_valid_range(self, i: int) -> tuple[int, int]:
        n = len(self._waterfall_pct_vars)
        lo = int(self._waterfall_pct_vars[i + 1].get()) + 1 if i < n - 1 else 0
        hi = int(self._waterfall_pct_vars[i - 1].get()) - 1 if i > 0 else 100
        return lo, hi

    def _on_waterfall_slider(self, i: int, v: float) -> None:
        val = int(round(v))
        lo, hi = self._get_wf_valid_range(i)
        val = max(lo, min(hi, val))
        self._waterfall_pct_vars[i].set(str(val))
        self._waterfall_pct_sliders[i].set(float(val))

    def _on_waterfall_entry(self, i: int) -> None:
        raw = self._waterfall_pct_vars[i].get().strip()
        try:
            val = int(raw)
        except ValueError:
            val = int(round(self._waterfall_pct_sliders[i].get()))
            self._waterfall_pct_vars[i].set(str(val))
            return
        lo, hi = self._get_wf_valid_range(i)
        val = max(lo, min(hi, val))
        self._waterfall_pct_vars[i].set(str(val))
        self._waterfall_pct_sliders[i].set(float(val))

    def _on_waterfall_n_changed(self, *_) -> None:
        raw = self._interp_wf_n_var.get()
        try:
            n = int(raw)
        except ValueError:
            if raw:
                self._interp_wf_n_err_lbl.configure(text="Integer 1–12 required")
            return
        if 1 <= n <= 12:
            self._interp_wf_n_err_lbl.configure(text="")
            self._rebuild_waterfall_percentile_sliders(n)
        else:
            self._interp_wf_n_err_lbl.configure(text="Integer 1–12 required")

    # ── Hyperparameter Optimisation Tab ───────────────────────────────────────

    def _setup_hpo_tab(self):
        tab = self.tabview.tab("Hyperparameter Optimisation")
        sf = ctk.CTkScrollableFrame(tab)
        sf.pack(fill="both", expand=True, padx=4, pady=4)
        sf.columnconfigure(1, weight=1)
        row = 0

        # ── Optimisation Methods ──
        ctk.CTkLabel(sf, text="Optimisation Methods",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, padx=10, pady=(10, 2), sticky="w", columnspan=2); row += 1

        mf = ctk.CTkFrame(sf)
        mf.grid(row=row, column=0, columnspan=2, padx=10, pady=4, sticky="ew"); row += 1

        self._hpo_use_random   = tk.BooleanVar(value=True)
        self._hpo_use_hyperopt = tk.BooleanVar(value=True)
        self._hpo_use_skopt    = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(mf, text="Random Search (Monte Carlo)",
                        variable=self._hpo_use_random,
                        command=self._refresh_hpo_state).pack(side="left", padx=12, pady=6)
        ctk.CTkCheckBox(mf, text="Hyperopt",
                        variable=self._hpo_use_hyperopt,
                        command=self._refresh_hpo_state).pack(side="left", padx=12)
        ctk.CTkCheckBox(mf, text="Scikit-Optimize",
                        variable=self._hpo_use_skopt,
                        command=self._refresh_hpo_state).pack(side="left", padx=12)

        # ── Optimisation Metric ──
        ctk.CTkLabel(sf, text="Optimisation Metric",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, padx=10, pady=(10, 2), sticky="w", columnspan=2); row += 1

        mef = ctk.CTkFrame(sf)
        mef.grid(row=row, column=0, columnspan=2, padx=10, pady=4, sticky="ew"); row += 1

        self._hpo_metric_var = tk.StringVar(value="Q^2")
        _hpo_metrics = [
            ("Mean Squared Error",                         "MSE"),
            ("Normalised Root Mean Squared Error (NRMSE)", "NRMSE"),
            ("Mean Absolute Percentage Error (MAPE)",      "MAPE"),
            ("R²",                                         "R^2"),
            ("Adjusted R²",                                "ADJUSTED R^2"),
            ("Nash-Sutcliffe Efficiency (Q²/NSE)",         "Q^2"),
            ("Kling-Gupta Efficiency (KGE)",               "KGE"),
        ]
        for text, val in _hpo_metrics:
            ctk.CTkRadioButton(mef, text=text,
                               variable=self._hpo_metric_var, value=val).pack(side="left", padx=12, pady=6)

        # ── Settings ──
        ctk.CTkLabel(sf, text="Settings",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, padx=10, pady=(10, 2), sticky="w", columnspan=2); row += 1

        self._hpo_sampling = self._lbl_option(
            sf, "Sampling Method (Random Search):",
            ["Sobol", "Halton", "Latin Hypercube", "Random (Monte Carlo)"], row); row += 1

        # Random Search iterations — save refs for greying
        self._hpo_n_iter_lbl, self._hpo_n_iter = self._lbl_entry_ref(
            sf, "Random Search Iterations:", "500", row); row += 1

        self._hpo_sample_size = self._lbl_entry(sf, "Sample Size:", "1000", row); row += 1

        # Hyperopt evaluations — save refs
        self._hpo_evals_lbl, self._hpo_evals = self._lbl_entry_ref(
            sf, "Hyperopt Evaluations:", "50", row); row += 1

        # Scikit-Optimize calls — save refs
        self._hpo_calls_lbl, self._hpo_calls = self._lbl_entry_ref(
            sf, "Scikit-Optimize Calls:", "50", row); row += 1

        self._hpo_n_jobs = self._lbl_entry(sf, "Parallel Jobs:", "-1", row); row += 1

        # ── Early Stopping ──
        ctk.CTkLabel(sf, text="Early Stopping",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, padx=10, pady=(12, 2), sticky="w", columnspan=2); row += 1

        esf = ctk.CTkFrame(sf)
        esf.grid(row=row, column=0, columnspan=2, padx=10, pady=4, sticky="ew"); row += 1

        for col, (txt, w) in enumerate([
            ("Method", 160), ("Enable", 60), ("Patience", 90), ("Minimum Delta", 110)
        ]):
            ctk.CTkLabel(esf, text=txt, width=w,
                         font=ctk.CTkFont(weight="bold")).grid(row=0, column=col, padx=4, pady=(6, 2))

        self._es_vars: dict[str, tuple] = {}
        _es_cfg = {
            "random_search": ("Random Search",    True, "10", "1e-4"),
            "hyperopt":      ("Hyperopt",         True, "10", "1e-4"),
            "skopt":         ("Scikit-Optimize",  True, "10", "1e-4"),
        }
        # Save per-method ES row widgets for greying
        self._hpo_random_es_widgets:  list = []
        self._hpo_hyperopt_es_widgets: list = []
        self._hpo_skopt_es_widgets:   list = []

        for i, (method, (label, en, pat, md)) in enumerate(_es_cfg.items()):
            r2 = i + 1
            lbl_w = ctk.CTkLabel(esf, text=label, width=160, anchor="w")
            lbl_w.grid(row=r2, column=0, padx=4, pady=4, sticky="w")
            en_var = tk.BooleanVar(value=en)
            en_cb  = ctk.CTkCheckBox(esf, text="", variable=en_var, width=30)
            en_cb.grid(row=r2, column=1, padx=4)
            pat_e = ctk.CTkEntry(esf, width=90); pat_e.insert(0, pat)
            pat_e.grid(row=r2, column=2, padx=4, pady=4)
            md_e  = ctk.CTkEntry(esf, width=110); md_e.insert(0, md)
            md_e.grid(row=r2, column=3, padx=4, pady=4)
            self._es_vars[method] = (en_var, pat_e, md_e)
            row_widgets = [lbl_w, en_cb, pat_e, md_e]
            if method == "random_search":
                self._hpo_random_es_widgets  = row_widgets
            elif method == "hyperopt":
                self._hpo_hyperopt_es_widgets = row_widgets
            else:
                self._hpo_skopt_es_widgets   = row_widgets

        # Initial grey-out pass
        self._refresh_hpo_state()

    # ── Cross-Validation Tab ──────────────────────────────────────────────────

    def _setup_cv_tab(self):
        tab = self.tabview.tab("Postprocessing")
        tab.columnconfigure(1, weight=1)

        # ── Cross-Validation ──
        ctk.CTkLabel(tab, text="Cross-Validation",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(8, 2), sticky="w", columnspan=2)

        # Method dropdown — inline so we can pass command=
        ctk.CTkLabel(tab, text="Cross-Validation Method:", anchor="w", width=240).grid(
            row=1, column=0, padx=10, pady=4, sticky="w")
        self._cv_method = ctk.CTkOptionMenu(tab, values=_CV_METHOD_DISPLAY, width=200,
                                            command=self._on_cv_method_change)
        self._cv_method.set(_CV_METHOD_DISPLAY[0])
        self._cv_method.grid(row=1, column=1, padx=10, pady=4, sticky="w")

        # Dynamic parameter rows — use _lbl_entry_ref so we can show/hide the label too
        self._cv_n_splits_lbl,  self._cv_n_splits  = self._lbl_entry_ref(tab, "Number of Splits:", "10",  2)
        self._cv_test_size_lbl, self._cv_test_size  = self._lbl_entry_ref(tab, "Test Size:",        "0.2", 3)
        # Random State isn't user-facing here any more — it's always overridden downstream
        # by the single Random Seed field on the Home tab. Widget kept (never shown) so
        # the existing sync/field-visibility code below doesn't need restructuring.
        self._cv_rand_state_lbl,self._cv_rand_state = self._lbl_entry_ref(tab, "Random State:",     "0",   4)
        self._cv_rand_state_lbl.grid_remove()
        self._cv_rand_state.grid_remove()
        self._cv_n_repeats_lbl, self._cv_n_repeats  = self._lbl_entry_ref(tab, "Number of Repeats:","5",   5)
        self._cv_p_lbl,         self._cv_p          = self._lbl_entry_ref(tab, "p (leave-p-out):",  "2",   6)

        # Shown only for Leave One Out (which has no parameters)
        self._cv_no_params_lbl = ctk.CTkLabel(tab,
            text="No parameters required for this method.", text_color="gray")
        self._cv_no_params_lbl.grid(row=2, column=0, columnspan=2, padx=10, pady=4, sticky="w")

        # Scoring metric (fixed below the dynamic rows)
        ctk.CTkLabel(tab, text="Scoring Metric:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=8, column=0, padx=10, pady=(10, 2), sticky="w", columnspan=2)

        self._cv_metric_var = tk.StringVar(value="R^2")
        _cv_metrics = [
            ("Mean Absolute Error",                        "MAE"),
            ("Mean Squared Error",                         "MSE"),
            ("Normalised Root Mean Squared Error (NRMSE)", "NRMSE"),
            ("Mean Absolute Percentage Error (MAPE)",      "MAPE"),
            ("R²",                                         "R^2"),
            ("Adjusted R²",                                "ADJUSTED R^2"),
            ("Nash-Sutcliffe Efficiency (Q²/NSE)",         "Q^2"),
            ("Kling-Gupta Efficiency (KGE)",               "KGE"),
            ("Explained Variance",                         "Explained Variance"),
        ]
        for i, (text, val) in enumerate(_cv_metrics):
            ctk.CTkRadioButton(tab, text=text,
                               variable=self._cv_metric_var, value=val).grid(
                row=9 + i, column=0, padx=20, pady=2, sticky="w")

        # ── Postprocessing Sections ──
        ctk.CTkLabel(tab, text="Postprocessing Sections",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=18, column=0, padx=10, pady=(14, 2), sticky="w", columnspan=2)

        self._cv_show_cv_summary  = self._lbl_check(tab, "Show Cross-Validation Summary",             True, 19)
        self._cv_show_influential = self._lbl_check(tab, "Show Influential Points Analysis",           True, 20)
        self._cv_show_stat_tests  = self._lbl_check(tab, "Show Residual Statistical Tests",            True, 21)
        self._cv_show_residuals   = self._lbl_check(tab, "Show Residuals with Influential Points",     True, 22)

        self._cv_show_transforms  = self._lbl_check(tab, "Show Residual Transformations",              True, 23)
        self._cv_show_transforms.trace_add("write", self._refresh_cv_state)

        # Which transforms to try — inline/"tabbed" and indented under the checkbox above.
        transforms_sub = ctk.CTkFrame(tab, fg_color="transparent")
        transforms_sub.grid(row=24, column=0, columnspan=2, padx=(40, 10), pady=(0, 4), sticky="w")
        self._cv_transform_vars: dict[str, tk.BooleanVar] = {}
        self._cv_transform_cbs: list[ctk.CTkCheckBox] = []
        for name in ["Yeo-Johnson", "Arcsinh"]:
            var = tk.BooleanVar(value=True)
            self._cv_transform_vars[name] = var
            cb = ctk.CTkCheckBox(transforms_sub, text=name, variable=var)
            cb.pack(side="left", padx=(0, 14))
            self._cv_transform_cbs.append(cb)

        self._cv_show_perm_imp = self._lbl_check(tab, "Show Permutation Feature Importance", True, 25)
        self._cv_show_lofo = self._lbl_check(
            tab, "Show Leave-One-Feature-Out (LOFO) Importance", False, 26)

        # ── Normality Test Metrics ──
        # Which tests the report's Best Transformation Normality Metrics table shows.
        # Every test is always computed and always in the Excel export; Anderson-Darling
        # and Shapiro-Wilk are always shown (AD picks the best transform when one is
        # needed, Shapiro-Wilk decides whether "None" already passes), so neither is
        # listed here. The checkbox below controls whether the table appears at all,
        # independent of which individual tests are ticked -- unchecking every test
        # below means "show zero columns", not "hide the table" (that's this one).
        self._cv_show_normality_metrics = self._lbl_check(
            tab, "Show Best Transformation Normality Metrics", True, 27)
        norm_sub = ctk.CTkFrame(tab, fg_color="transparent")
        norm_sub.grid(row=28, column=0, columnspan=2, padx=(40, 10), pady=(0, 4), sticky="w")
        self._cv_normality_vars: dict[str, tk.BooleanVar] = {}
        for name, default in [("Shapiro-Wilk", True), ("Lilliefors", True),
                              ("Filiben", True), ("Jarque-Bera", False),
                              ("D'Agostino", False)]:
            var = tk.BooleanVar(value=default)
            self._cv_normality_vars[name] = var
            ctk.CTkCheckBox(norm_sub, text=name, variable=var).pack(side="left", padx=(0, 14))
        ctk.CTkLabel(
            tab, text="Anderson-Darling and Shapiro-Wilk are always computed and shown (they "
                      "select the best transformation). All tests are always included in the "
                      "Excel results file.",
            font=ctk.CTkFont(size=11), anchor="w", text_color=_LBL_NORMAL,
        ).grid(row=29, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="w")

        # ── Model Deployment ──
        # Which artifact files the report step writes to the Models folder. With all
        # four off, no Models folder is created and the report's Saved Models and
        # Artifacts section is omitted.
        ctk.CTkLabel(tab, text="Model Deployment",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=30, column=0, padx=10, pady=(14, 2), sticky="w", columnspan=2)
        self._deploy_pipelines = self._lbl_check(tab, "Save Per-Target Pipelines (.pkl)",   True, 31)
        self._deploy_metadata  = self._lbl_check(tab, "Save Reproducibility Metadata (.json)", True, 32)
        self._deploy_bundle    = self._lbl_check(tab, "Save Combined Bundle (.pkl)",        True, 33)
        self._deploy_predictor = self._lbl_check(tab, "Save Deployable Predictor (.pkl)",   True, 34)

        self._refresh_cv_state()

        # Apply initial visibility for the default method
        self._on_cv_method_change(_CV_METHOD_DISPLAY[0])

    def _refresh_cv_state(self, *_):
        if hasattr(self, "_cv_transform_cbs"):
            on = self._cv_show_transforms.get()
            for cb in self._cv_transform_cbs:
                cb.configure(state="normal" if on else "disabled")

    def _on_cv_method_change(self, method: str):
        visible = _CV_FIELDS.get(method, ())
        field_map = {
            "n_splits":    (self._cv_n_splits_lbl,   self._cv_n_splits),
            "test_size":   (self._cv_test_size_lbl,  self._cv_test_size),
            # random_state deliberately excluded — always hidden, see _setup_cv_tab.
            "n_repeats":   (self._cv_n_repeats_lbl,  self._cv_n_repeats),
            "p":           (self._cv_p_lbl,           self._cv_p),
        }
        for name, (lbl, ent) in field_map.items():
            if name in visible:
                lbl.grid()
                ent.grid()
            else:
                lbl.grid_remove()
                ent.grid_remove()
        if visible:
            self._cv_no_params_lbl.grid_remove()
        else:
            self._cv_no_params_lbl.grid()

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _lbl_entry(self, parent, label: str, default: str, row: int) -> ctk.CTkEntry:
        ctk.CTkLabel(parent, text=label, anchor="w", width=240).grid(
            row=row, column=0, padx=10, pady=4, sticky="w")
        e = ctk.CTkEntry(parent, width=180)
        e.insert(0, default)
        e.grid(row=row, column=1, padx=10, pady=4, sticky="w")
        return e

    def _lbl_entry_ref(self, parent, label: str, default: str,
                       row: int) -> tuple[ctk.CTkLabel, ctk.CTkEntry]:
        lbl = ctk.CTkLabel(parent, text=label, anchor="w", width=240)
        lbl.grid(row=row, column=0, padx=10, pady=4, sticky="w")
        e = ctk.CTkEntry(parent, width=180)
        e.insert(0, default)
        e.grid(row=row, column=1, padx=10, pady=4, sticky="w")
        return lbl, e

    def _lbl_option(self, parent, label: str, values: list[str], row: int) -> ctk.CTkOptionMenu:
        ctk.CTkLabel(parent, text=label, anchor="w", width=240).grid(
            row=row, column=0, padx=10, pady=4, sticky="w")
        m = ctk.CTkOptionMenu(parent, values=values, width=200)
        m.set(values[0])
        m.grid(row=row, column=1, padx=10, pady=4, sticky="w")
        return m

    def _lbl_check(self, parent, label: str, default: bool, row: int) -> tk.BooleanVar:
        var = tk.BooleanVar(value=default)
        ctk.CTkCheckBox(parent, text=label, variable=var).grid(
            row=row, column=0, padx=20, pady=4, sticky="w", columnspan=2)
        return var

    # ── File browsers ─────────────────────────────────────────────────────────

    def _browse_dataset(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, p)

    def _browse_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, p)

    def _open_target_picker(self):
        # Popup rather than an inline checklist: datasets can have dozens of columns,
        # and this field only needs occasional use, so a modal keeps the Home tab compact.
        if self._target_picker_win is not None and self._target_picker_win.winfo_exists():
            self._target_picker_win.lift()
            self._target_picker_win.focus_set()
            return
        path = self.dataset_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Missing Dataset", "Set a valid dataset path first.")
            return
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read columns:\n{exc}")
            return
        if not columns:
            messagebox.showinfo("No Columns", "This dataset has no columns.")
            return

        current = {t.strip() for t in self.targets_entry.get().split(",") if t.strip()}

        picker = ctk.CTkToplevel(self)
        self._target_picker_win = picker
        # bind() here is scoped to this specific Toplevel instance, not its children,
        # so this fires exactly once, when the picker itself is destroyed (Apply,
        # Cancel, or the window's native close button all end up calling .destroy()).
        picker.bind("<Destroy>", lambda e: setattr(self, "_target_picker_win", None))
        picker.title("Select Target Column(s)")
        picker.geometry("360x420")
        picker.transient(self)
        picker.grab_set()

        ctk.CTkLabel(picker, text="Choose one or more target columns:", anchor="w").pack(
            fill="x", padx=10, pady=(10, 4))

        scroll = ctk.CTkScrollableFrame(picker)
        scroll.pack(fill="both", expand=True, padx=10, pady=4)

        col_vars: dict[str, tk.BooleanVar] = {}
        for col in columns:
            var = tk.BooleanVar(value=(col in current))
            col_vars[col] = var
            ctk.CTkCheckBox(scroll, text=col, variable=var).pack(anchor="w", padx=4, pady=2)

        btn_row = ctk.CTkFrame(picker, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkButton(btn_row, text="Select All", width=80,
                     command=lambda: [v.set(True) for v in col_vars.values()]).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Clear", width=80,
                     command=lambda: [v.set(False) for v in col_vars.values()]).pack(side="left", padx=4)

        def _apply():
            chosen = [c for c in columns if col_vars[c].get()]
            if not chosen:
                messagebox.showwarning("No Selection", "Select at least one target column.", parent=picker)
                return
            self.targets_entry.delete(0, "end")
            self.targets_entry.insert(0, ", ".join(chosen))
            picker.destroy()

        ctk.CTkButton(btn_row, text="Apply", width=80, command=_apply).pack(side="right", padx=4)
        ctk.CTkButton(btn_row, text="Cancel", width=80, command=picker.destroy).pack(side="right", padx=4)

    _MONOTONIC_LABELS = {0: "None", 1: "Increasing", -1: "Decreasing"}
    _MONOTONIC_VALUES = {"None": 0, "Increasing": 1, "Decreasing": -1}

    def _open_monotonic_picker(self):
        # Modelled on _open_target_picker: same header-only CSV read (no preprocessing
        # run needed yet), same popup shape, but a 3-way per-row choice instead of a
        # checkbox, and rows are the prospective *feature* columns (targets excluded).
        # Per-target: a constraint direction that's physically correct for one target can
        # be wrong for another sharing the same feature, so each target gets its own set
        # of rows, switched via the dropdown at the top — all built up front (not
        # rebuilt per switch) so in-progress edits are never lost mid-edit.
        if self._monotonic_picker_win is not None and self._monotonic_picker_win.winfo_exists():
            self._monotonic_picker_win.lift()
            self._monotonic_picker_win.focus_set()
            return
        path = self.dataset_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Missing Dataset", "Set a valid dataset path first.")
            return
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read columns:\n{exc}")
            return
        target_list = [t.strip() for t in self.targets_entry.get().split(",") if t.strip()]
        if not target_list:
            messagebox.showerror("Missing Targets", "Set at least one target first.")
            return
        targets = set(target_list)
        feature_cols = [c for c in columns if c not in targets]
        if not feature_cols:
            messagebox.showinfo("No Features", "No feature columns available (check the Targets field).")
            return

        picker = ctk.CTkToplevel(self)
        self._monotonic_picker_win = picker
        picker.bind("<Destroy>", lambda e: setattr(self, "_monotonic_picker_win", None))
        picker.title("Monotonicity Constraints")
        picker.geometry("420x520")
        picker.transient(self)
        picker.grab_set()

        ctk.CTkLabel(
            picker,
            text="Only applies to XGBoost/LGBM; a no-op for any other selected model. "
                 "Set per target — a direction correct for one target can be wrong for another.",
            anchor="w", wraplength=390, font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=10, pady=(10, 4))

        target_row = ctk.CTkFrame(picker, fg_color="transparent")
        target_row.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(target_row, text="Target:", anchor="w", width=60).pack(side="left")
        target_var = tk.StringVar(value=target_list[0])
        target_menu = ctk.CTkOptionMenu(target_row, variable=target_var, values=target_list, width=200)
        target_menu.pack(side="left", padx=(4, 0))

        scroll = ctk.CTkScrollableFrame(picker)
        scroll.pack(fill="both", expand=True, padx=10, pady=4)

        # One row-var dict per target; one frame per target, only one packed at a time.
        row_vars_by_target: dict[str, dict[str, tk.StringVar]] = {}
        frame_by_target: dict[str, ctk.CTkFrame] = {}
        for target in target_list:
            frame = ctk.CTkFrame(scroll, fg_color="transparent")
            frame_by_target[target] = frame
            row_vars: dict[str, tk.StringVar] = {}
            existing = self._monotonic_constraints.get(target, {})
            for col in feature_cols:
                row = ctk.CTkFrame(frame, fg_color="transparent")
                row.pack(fill="x", padx=2, pady=2)
                ctk.CTkLabel(row, text=col, anchor="w", width=220).pack(side="left", padx=(2, 8))
                current_label = self._MONOTONIC_LABELS.get(existing.get(col, 0), "None")
                var = tk.StringVar(value=current_label)
                row_vars[col] = var
                ctk.CTkOptionMenu(row, variable=var, values=["None", "Increasing", "Decreasing"],
                                 width=130).pack(side="left")
            row_vars_by_target[target] = row_vars

        frame_by_target[target_list[0]].pack(fill="both", expand=True)

        def _on_target_change(new_target):
            for t, frame in frame_by_target.items():
                frame.pack_forget()
            frame_by_target[new_target].pack(fill="both", expand=True)

        target_menu.configure(command=_on_target_change)

        btn_row = ctk.CTkFrame(picker, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkButton(btn_row, text="Clear All (this target)", width=140,
                     command=lambda: [v.set("None") for v in row_vars_by_target[target_var.get()].values()]
                     ).pack(side="left", padx=4)

        def _apply():
            self._monotonic_constraints = {
                t: {
                    col: self._MONOTONIC_VALUES[v.get()]
                    for col, v in row_vars.items() if self._MONOTONIC_VALUES[v.get()] != 0
                }
                for t, row_vars in row_vars_by_target.items()
            }
            # Drop targets left with no active constraints, to keep the stored dict tidy.
            self._monotonic_constraints = {t: c for t, c in self._monotonic_constraints.items() if c}
            picker.destroy()

        ctk.CTkButton(btn_row, text="Apply", width=80, command=_apply).pack(side="right", padx=4)
        ctk.CTkButton(btn_row, text="Cancel", width=80, command=picker.destroy).pack(side="right", padx=4)

    def _browse_clean_dataset(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self._clean_dataset_entry.delete(0, "end")
            self._clean_dataset_entry.insert(0, p)

    def _browse_phys_dataset(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self._phys_dataset_entry.delete(0, "end")
            self._phys_dataset_entry.insert(0, p)

    def _browse_physics_script(self):
        p = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if p:
            self._phys_script_entry.delete(0, "end")
            self._phys_script_entry.insert(0, p)

    def _browse_physics_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self._phys_output_entry.delete(0, "end")
            self._phys_output_entry.insert(0, p)

    def _browse_perl_config_save(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="physics_config.json")
        if p:
            self._phys_perl_config_entry.delete(0, "end")
            self._phys_perl_config_entry.insert(0, p)

    def _browse_perl_config_load(self):
        p = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if p:
            self.perl_config_entry.delete(0, "end")
            self.perl_config_entry.insert(0, p)
            self._on_perl_config_entry_change()

    def _on_perl_config_entry_change(self):
        """Called when the Physics Config entry loses focus or the user presses Enter."""
        p = self.perl_config_entry.get().strip()
        self.session.perl_config_path = p
        self._refresh_buttons()

    def _phys_save_perl_config(self):
        config_path = self._phys_perl_config_entry.get().strip()
        if not config_path:
            messagebox.showerror("Physics Modelling", "Choose a path to save the configuration first.")
            return

        mode = self._phys_mode_var.get()
        orig_path = self._phys_dataset_entry.get().strip()

        if mode == "Script":
            script_path = self._phys_script_entry.get().strip()
            if not script_path or not os.path.isfile(script_path):
                messagebox.showerror("Physics Modelling", "Select a valid physics script (.py) first.")
                return
            try:
                module = import_physics_script(script_path)
                missing = [a for a in ("governing_function", "constants", "input_vars", "output_vars")
                           if not hasattr(module, a)]
                if missing:
                    messagebox.showerror("Physics Modelling",
                                         f"Script is missing: {', '.join(missing)}")
                    return
                input_vars   = module.input_vars
                output_vars  = module.output_vars
                constants    = module.constants
                name_mapping = getattr(module, "name_mapping", None)
                time_col     = getattr(module, "time_col", None)

                # Build maps: residual target name → physics column / measured column
                recon_map    = {f"Residual {v}": f"{v}_physics" for v in output_vars}
                measured_map = {f"Residual {v}": v for v in output_vars}

                save_script_physics_config(
                    path=config_path,
                    script_path=script_path,
                    input_vars=input_vars,
                    output_vars=output_vars,
                    constants=constants,
                    reconstruction_map=recon_map,
                    measured_map=measured_map,
                    name_mapping=name_mapping,
                    time_col=time_col,
                    original_dataset_path=orig_path,
                )
            except Exception as e:
                messagebox.showerror("Physics Modelling", f"Failed to save configuration: {e}")
                return

            n_mapped = len(output_vars)
            detail = "\n".join(f"  {k}  →  {v}" for k, v in recon_map.items())
        else:
            # Expression mode
            texts = [x["entry"].get() for x in self._phys_expressions if x["entry"].get().strip()]
            if not texts:
                messagebox.showerror("Physics Modelling", "Add at least one expression before saving.")
                return

            output_cols_text = self._phys_output_cols.get().strip()
            recon_map    = extract_reconstruction_mapping(texts)
            measured_map = extract_measured_mapping(texts)

            try:
                save_physics_config(
                    path=config_path,
                    expressions=texts,
                    output_cols_text=output_cols_text,
                    reconstruction_map=recon_map,
                    measured_map=measured_map,
                    original_dataset_path=orig_path,
                )
            except Exception as e:
                messagebox.showerror("Physics Modelling", f"Failed to save configuration: {e}")
                return

            n_mapped = len(recon_map)
            detail = "\n".join(f"  {k}  →  {v}" for k, v in recon_map.items())

        self._phys_perl_status.configure(
            text=f"Saved  ({n_mapped} target(s) mapped)  →  {os.path.basename(config_path)}",
            text_color="#4CAF50")

        # Auto-populate the Home tab Physics Configuration field if it's empty
        if hasattr(self, "perl_config_entry") and not self.perl_config_entry.get().strip():
            self.perl_config_entry.insert(0, config_path)
            self.session.perl_config_path = config_path
            self._refresh_buttons()

        messagebox.showinfo(
            "Physics Modelling",
            f"Physics configuration saved to:\n{config_path}\n\n"
            f"Reconstruction map ({n_mapped} targets):\n" + detail)

    # ── HPO grey-out ──────────────────────────────────────────────────────────

    def _refresh_hpo_state(self):
        groups = [
            (self._hpo_use_random,
             [self._hpo_n_iter_lbl, self._hpo_n_iter, self._hpo_sampling] + self._hpo_random_es_widgets),
            (self._hpo_use_hyperopt,
             [self._hpo_evals_lbl,  self._hpo_evals]  + self._hpo_hyperopt_es_widgets),
            (self._hpo_use_skopt,
             [self._hpo_calls_lbl,  self._hpo_calls]  + self._hpo_skopt_es_widgets),
        ]
        for var, widgets in groups:
            enabled = var.get()
            for w in widgets:
                if isinstance(w, ctk.CTkEntry):
                    w.configure(state="normal" if enabled else "disabled")
                elif isinstance(w, ctk.CTkCheckBox):
                    w.configure(state="normal" if enabled else "disabled")
                elif isinstance(w, ctk.CTkLabel):
                    w.configure(text_color=_LBL_NORMAL if enabled else _LBL_DISABLED)
                elif isinstance(w, ctk.CTkOptionMenu):
                    w.configure(state="normal" if enabled else "disabled")

    # ── Session sync ──────────────────────────────────────────────────────────

    def _sync_session(self) -> bool:
        s = self.session
        s.dataset_path    = self.dataset_entry.get().strip()
        s.output_dir      = self.output_entry.get().strip()
        s.targets         = [t.strip() for t in self.targets_entry.get().split(",") if t.strip()]
        s.selected_models = [n for n, v in self._model_vars.items() if v.get()]
        s.report_metric_cols = [k for k, v in self._report_metric_vars.items() if v.get()]
        s.monotonic_constraints = dict(self._monotonic_constraints)

        if not s.dataset_path:
            messagebox.showerror("Missing Input", "Please specify a dataset path."); return False
        if not s.output_dir:
            messagebox.showerror("Missing Input", "Please specify an output directory."); return False
        if not s.targets:
            messagebox.showerror("Missing Input", "Please specify at least one target variable."); return False
        if not s.selected_models:
            messagebox.showerror("Missing Input", "Please select at least one model."); return False

        # Preprocessing
        try:
            s.test_size = float(self._preproc_test_size.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Test Size must be a decimal number."); return False
        if not (0 < s.test_size < 1):
            messagebox.showerror("Invalid Input", "Test Size must be between 0 and 1 (e.g. 0.2 for 20%)."); return False
        s.split_method              = self._preproc_split.get()   # "First"/"Last"/"Random"; code uses .lower()
        s.scaler_type               = self._preproc_scaler.get()
        s.show_target_vs_target     = self._preproc_target_vs_target.get()
        s.show_features_vs_targets  = self._preproc_feat_vs_target.get()
        s.show_boxplots             = self._preproc_boxplots.get()
        s.show_distance_corr        = self._preproc_dist_corr.get()
        s.dist_corr_dummy           = self._preproc_dist_dummy.get()
        s.dist_corr_mp              = self._preproc_dist_mp.get()
        s.show_multicollinearity    = self._preproc_multicollinearity.get()
        s.feat_sel_enabled          = self._preproc_feat_sel.get()
        s.plot_pca_enabled          = self._preproc_pca.get()
        try:
            s.feat_sel_redundancy_threshold = float(self._preproc_feat_sel_thresh.get().strip())
            if not (0.0 < s.feat_sel_redundancy_threshold < 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input",
                "Feature Selection Redundancy Threshold must be a decimal between 0 and 1 (e.g. 0.90).")
            return False
        try:
            rs_val = self._random_seed_entry.get().strip()
            s.random_seed = int(rs_val) if rs_val else 0
        except ValueError:
            messagebox.showerror("Invalid Input", "Random Seed must be an integer."); return False
        s.report_source = "ui"

        # Uncertainty Quantification
        bs_on   = self._uq_bootstrap_var.get()
        conf_on = self._uq_conformal_var.get()
        if not bs_on and not conf_on:
            messagebox.showerror("Invalid Input", "Please select at least one UQ method."); return False
        if bs_on and conf_on:
            uq_method_str = "Both"
        elif bs_on:
            uq_method_str = "Bootstrapping"
        else:
            uq_method_str = "Conformal"
        try:
            s.uq_settings = dict(
                uq_method=uq_method_str,
                n_bootstrap=int(self._uq_n_bootstrap.get()),
                confidence_interval=float(self._uq_confidence.get()),
                calibration_frac=float(self._uq_calib_frac.get()),
                subsample_test_size=int(self._uq_subsample.get()),
                n_jobs=int(self._uq_n_jobs.get()),
                include_gp_posterior=self._uq_gp_posterior_var.get(),
                calibration_enabled=self._uq_calibration_var.get(),
            )
        except ValueError:
            messagebox.showerror("Invalid Input", "Uncertainty Quantification fields must be numeric."); return False
        if s.uq_settings["n_bootstrap"] < 2:
            messagebox.showerror("Invalid Input", "Number of Bootstraps must be at least 2."); return False
        if not (0 < s.uq_settings["confidence_interval"] < 100):
            messagebox.showerror("Invalid Input", "Confidence Interval must be between 0 and 100 (e.g. 95)."); return False
        if not (0 < s.uq_settings["calibration_frac"] < 1):
            messagebox.showerror("Invalid Input", "Calibration Fraction must be between 0 and 1 (e.g. 0.05)."); return False
        if s.uq_settings["subsample_test_size"] < 1:
            messagebox.showerror("Invalid Input", "Subsample Test Size must be at least 1."); return False

        # Interpretability
        try:
            s.interpretability_settings = dict(
                test_sample_size=int(self._interp_test_size.get()),
                background_sample_size=int(self._interp_bg_size.get()),
                subsample=int(self._interp_subsample.get()),
                grid_resolution=int(self._interp_grid_res.get()),
                show_ice_pdp=self._interp_ice_pdp_var.get(),
                show_ale=self._interp_ale_var.get(),
                show_shap_summary=self._interp_shap_summary_var.get(),
                show_shap_dependence=self._interp_shap_dep_var.get(),
                show_shap_waterfall=self._interp_shap_wf_var.get(),
                n_waterfall_samples=len(self._waterfall_pct_vars),
                waterfall_percentiles=[int(v.get()) / 100.0 for v in self._waterfall_pct_vars],
                show_sensitivity_morris=self._interp_sens_morris_var.get(),
                show_sensitivity_sobol=self._interp_sens_sobol_var.get(),
                show_sensitivity_fast=self._interp_sens_fast_var.get(),
                sensitivity_morris_trajectories=int(self._interp_sens_morris_traj.get()),
                sensitivity_morris_levels=int(self._interp_sens_morris_levels.get()),
                sensitivity_sobol_n=int(self._interp_sens_sobol_n.get()),
                sensitivity_fast_n=int(self._interp_sens_fast_n.get()),
            )
        except ValueError:
            messagebox.showerror("Invalid Input", "Interpretability fields must be numeric."); return False
        if s.interpretability_settings["sensitivity_sobol_n"] < 1:
            messagebox.showerror("Invalid Input",
                                 "Sobol Base Sample Size must be a positive integer."); return False
        # SALib's FAST hard-requires N > 4*M^2; this app always calls it with
        # the library default M=4, so N must be at least 65 -- caught here so
        # the run doesn't get partway through Interpretability before failing.
        if s.interpretability_settings["show_sensitivity_fast"] \
                and s.interpretability_settings["sensitivity_fast_n"] <= 64:
            messagebox.showerror(
                "Invalid Input",
                "FAST Samples per Feature must be at least 65 (SALib requires "
                "N > 4*M^2 with M=4). Increase the value or untick Show FAST Indices.")
            return False

        # Hyperparameter Optimisation methods
        methods = []
        if self._hpo_use_random.get():   methods.append("random")
        if self._hpo_use_hyperopt.get(): methods.append("hyperopt")
        if self._hpo_use_skopt.get():    methods.append("skopt")
        if not methods:
            messagebox.showerror("Invalid Input", "Please select at least one HPO method."); return False
        s.methods_to_run  = methods
        s.hpo_metric      = self._hpo_metric_var.get()
        s.sampling_method = _SAMPLING_METHOD_MAP[self._hpo_sampling.get()]
        try:
            s.n_iter       = int(self._hpo_n_iter.get())
            s.sample_size  = int(self._hpo_sample_size.get())
            s.evals        = int(self._hpo_evals.get())
            s.calls        = int(self._hpo_calls.get())
            s.n_jobs       = int(self._hpo_n_jobs.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Hyperparameter Optimisation numeric fields must be integers."); return False
        # skopt's gp_minimize hard-requires n_calls >= 10 (it raises otherwise) -
        # caught here so the run doesn't get partway through HPO before failing.
        if "skopt" in methods and s.calls < 10:
            messagebox.showerror(
                "Invalid Input",
                "Scikit-Optimize Calls must be at least 10 (gp_minimize requires "
                "n_calls >= 10). Increase the value or untick Scikit-Optimize.")
            return False

        # Early stopping
        es: dict = {}
        for method, (en_var, pat_e, md_e) in self._es_vars.items():
            if en_var.get():
                try:
                    es[method] = {"patience": int(pat_e.get()), "min_delta": float(md_e.get())}
                except ValueError:
                    messagebox.showerror("Invalid Input",
                                         f"Early stopping fields for {method} must be numeric."); return False
        s.early_stopping = es if es else None

        # Cross-Validation — collect only the fields relevant to the chosen method
        _method_display = self._cv_method.get()
        _visible = _CV_FIELDS.get(_method_display, ())
        try:
            cv_args: dict = {}
            if "n_splits"    in _visible: cv_args["n_splits"]    = int(self._cv_n_splits.get())
            if "test_size"   in _visible:
                cv_test_size = float(self._cv_test_size.get())
                if not (0 < cv_test_size < 1):
                    messagebox.showerror("Invalid Input",
                        "Cross-Validation Test Size must be between 0 and 1 (e.g. 0.2 for 20%).")
                    return False
                cv_args["test_size"] = cv_test_size
            if "random_state"in _visible: cv_args["random_state"]= int(self._cv_rand_state.get())
            if "n_repeats"   in _visible: cv_args["n_repeats"]   = int(self._cv_n_repeats.get())
            if "p"           in _visible: cv_args["p"]           = int(self._cv_p.get())
            s.cv_args = cv_args
        except ValueError:
            messagebox.showerror("Invalid Input", "Cross-Validation fields must be numeric."); return False
        s.cv_method      = _CV_METHOD_MAP[_method_display]
        s.scoring_metric = self._cv_metric_var.get()

        s.show_cv_summary            = self._cv_show_cv_summary.get()
        s.show_cooks_distance        = self._cv_show_influential.get()
        s.show_extended_diagnostics  = self._cv_show_stat_tests.get()
        s.show_residuals             = self._cv_show_residuals.get()
        s.show_transformation_plots  = self._cv_show_transforms.get()
        s.show_permutation_importance= self._cv_show_perm_imp.get()
        s.show_lofo_importance       = self._cv_show_lofo.get()
        s.transforms_to_run          = [name for name, v in self._cv_transform_vars.items() if v.get()]
        s.show_normality_metrics     = self._cv_show_normality_metrics.get()
        s.normality_tests            = [name for name, v in self._cv_normality_vars.items() if v.get()]

        # Model Deployment
        s.save_pipelines      = self._deploy_pipelines.get()
        s.save_metadata       = self._deploy_metadata.get()
        s.save_bundle         = self._deploy_bundle.get()
        s.save_predictor_file = self._deploy_predictor.get()

        # PERL
        perl_path = self.perl_config_entry.get().strip()
        s.perl_config_path = perl_path

        return True

    # Fields reset_results() clears, plus a couple of live control/bookkeeping fields —
    # everything else on WorkflowSession is a user-configured setting. Used by
    # _settings_snapshot to compare "what settings did this step run with" against
    # "what are the settings now", without result data (which legitimately differs
    # after every run) causing a false mismatch.
    _NON_SETTINGS_FIELDS = frozenset({
        "preprocessing_results", "training_results", "uq_before", "uq_after",
        "interpretability_before", "interpretability_after", "hpo_results", "cv_results",
        "perl_results", "perl_config", "perl_output_df", "cleaning_summary",
        "metrics", "params", "total_elapsed", "step_timings",
        "images_dir", "report_dir", "xlsx_path", "pdf_path", "models_dir",
        "checkpoint_fn", "_warned_stale_model_names",
    })

    def _settings_snapshot(self) -> dict:
        return {
            f.name: copy.deepcopy(getattr(self.session, f.name))
            for f in dataclasses.fields(self.session)
            if f.name not in self._NON_SETTINGS_FIELDS
        }

    def _stale_prereqs(self, prereqs: list[str]) -> list[str]:
        """Which of `prereqs` last completed under settings that no longer match the
        live ones right now. Compares whole-session settings snapshots (not a
        per-field dependency graph) — coarser than strictly necessary, but a false
        "stale" from an unrelated setting changing is far safer than a step silently
        running against a prerequisite it no longer matches."""
        current = self._settings_snapshot()
        return [p for p in prereqs
                if p in self._step_settings_snapshot and self._step_settings_snapshot[p] != current]

    # ── Button state ──────────────────────────────────────────────────────────

    def _refresh_buttons(self):
        busy = self._running_step is not None
        for key, _, prereqs in STEPS:
            prereqs_done = all(self.step_status.get(p) == "done" for p in prereqs)
            # Read the entry directly, not session.perl_config_path — that field only
            # updates on FocusOut/Return, so a "type a path, then immediately click
            # Run" sequence could hit this check before the sync ran, leaving the
            # button disabled and swallowing that first click.
            extra_ok = (key != "perl") or bool(self.perl_config_entry.get().strip())
            enabled = prereqs_done and extra_ok and not busy
            self._step_run_btns[key].configure(state="normal" if enabled else "disabled")

        self.run_all_btn.configure(state="normal" if not busy else "disabled")
        self.pause_btn.configure(state="normal" if busy else "disabled")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        if not busy:
            self.pause_btn.configure(text="Pause")

        can_report = (self.session.preprocessing_results is not None
                      and self.session.training_results is not None
                      and not busy)
        self.report_btn.configure(state="normal" if can_report else "disabled")
        self.reset_btn.configure(state="normal" if not busy else "disabled")

        self._set_settings_locked(busy)

    # Widget types worth locking — anything a user could change a setting through.
    # ttk.Treeview (Dataset Cleaning's column manager/preview) has no simple "disabled"
    # state and viewing/selecting it while busy is harmless, so it's not included.
    _LOCKABLE_WIDGET_TYPES = (
        ctk.CTkEntry, ctk.CTkButton, ctk.CTkCheckBox, ctk.CTkOptionMenu,
        ctk.CTkComboBox, ctk.CTkRadioButton, ctk.CTkSwitch,
    )

    def _set_widgets_locked(self, parent, locked: bool, exclude: frozenset = frozenset()):
        """Recursively lock/unlock every settings-affecting widget under `parent`,
        skipping (and not recursing into) anything in `exclude`. state="disabled" keeps
        a widget's current value visible (an entry still shows its text, a checkbox
        still shows checked/unchecked) — it just can't be edited, which is exactly
        "visible but not changeable", not hidden."""
        state = "disabled" if locked else "normal"
        for child in parent.winfo_children():
            if child in exclude:
                continue
            if isinstance(child, self._LOCKABLE_WIDGET_TYPES):
                try:
                    child.configure(state=state)
                except Exception:
                    pass
            self._set_widgets_locked(child, locked, exclude)

    def _set_settings_locked(self, locked: bool):
        """Lock every settings widget while a step/chain is running — _sync_session()
        only runs once, at chain start, so an edit made mid-run was previously silently
        ignored for the rest of that run with no indication anything was wrong. Values
        stay visible (see _set_widgets_locked); only editing is blocked. Excludes the
        execution controls (Run/Pause/Stop/Report/Reset, per-step Run buttons), whose
        enabled state is independently governed by prerequisites above, and any widget
        with its own unrelated async operation in flight (Dataset Cleaning Load/Apply,
        Physics Generate Dataset), which manages its own state around that operation."""
        home_exclude = frozenset(
            set(self._step_run_btns.values())
            | {self.run_all_btn, self.pause_btn, self.stop_btn, self.report_btn,
               self.reset_btn, self.log_textbox}
        )
        other_exclude = frozenset({
            getattr(self, "_clean_load_btn", None),
            getattr(self, "_clean_apply_btn", None),
            getattr(self, "_phys_generate_btn", None),
        } - {None})
        for name in ["Dataset Cleaning", "Physics Modelling", "Home", "Models",
                     "Preprocessing", "Uncertainty Quantification", "Interpretability",
                     "Hyperparameter Optimisation", "Postprocessing"]:
            tab_frame = self.tabview.tab(name)
            exclude = home_exclude if name == "Home" else other_exclude
            self._set_widgets_locked(tab_frame, locked, exclude)

    def _set_status(self, key: str, status: str):
        self.step_status[key] = status
        color, text = STATUS_INFO[status]
        self._step_status_labels[key].configure(text=text, text_color=color)
        self._refresh_buttons()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", text)
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _clear_log(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def _poll_log(self):
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        # Worker-thread "call this on the main thread" requests — see
        # _callback_queue's docstring in __init__ for why this exists instead of
        # calling self.after(...) directly from a background thread.
        try:
            while True:
                self._callback_queue.get_nowait()()
        except queue.Empty:
            pass
        self.after(50, self._poll_log)

    # ── Step execution ────────────────────────────────────────────────────────

    def _run_single_step(self, key: str):
        if not self._sync_session():
            return
        prereqs = next(p for k, _, p in STEPS if k == key)
        stale = self._stale_prereqs(prereqs)
        if stale:
            names = ", ".join(dict((k, n) for k, n, _ in STEPS)[p] for p in stale)
            messagebox.showerror(
                "Stale Prerequisites",
                f"Settings have changed since {names} last ran. Re-run "
                f"{'it' if len(stale) == 1 else 'them'} (or Run All) before running this step.",
            )
            return
        self._reset_run_controls()
        self._execute_step(key)

    def _reset_workflow(self):
        """Clear every computed result, step status, and timer — so a fresh dataset
        (or a fresh attempt on the same one) can never silently inherit a previous
        run's results, the bug this button exists to give an explicit escape hatch
        for (the automatic fix lives in run_step_preprocessing, which resets the
        same state every time preprocessing (re-)runs; this covers the case where
        the user wants a clean slate without necessarily re-running anything yet).
        Settings (dataset path, targets, model selection, all tab configuration)
        are left untouched — only results are cleared.
        """
        if self._running_step is not None:
            return
        if any(status == "done" for status in self.step_status.values()):
            if not messagebox.askyesno(
                "Reset Workflow",
                "This clears all computed results (training, HPO, UQ, interpretability, "
                "CV, PERL) and resets step statuses and timers. Your dataset, targets, "
                "and other settings are kept.\n\nContinue?",
            ):
                return

        self.session.reset_results()
        self._step_elapsed.clear()
        self._step_start_times.clear()
        self._paused_seconds.clear()
        self._step_settings_snapshot.clear()
        for key, _, _ in STEPS:
            self._set_status(key, "not_run")
            self._step_timer_labels[key].configure(text="--", text_color=_LBL_NORMAL)
        self._refresh_buttons()

    # Every step's session result field except "preprocessing" itself (which has
    # no single result field the way the others do — its output feeds
    # session.preprocessing_results, but "has preprocessing run" is really "is this
    # not None", the same shape, so it isn't included here since it's never the one
    # being reconciled — see _reconcile_step_statuses_with_session).
    _STEP_RESULT_FIELD = {
        "training":                "training_results",
        "uq_before":               "uq_before",
        "interpretability_before": "interpretability_before",
        "hpo":                     "hpo_results",
        "cv":                      "cv_results",
        "uq_after":                "uq_after",
        "interpretability_after":  "interpretability_after",
        "perl":                    "perl_results",
    }

    def _reconcile_step_statuses_with_session(self):
        """Reset the displayed status/timer/settings-snapshot for any step whose
        session result field session.reset_results() just cleared to None, but whose
        UI badge still claims "Done" (with a stale elapsed time) from before —
        called after preprocessing (re-)runs, the one step whose own function calls
        reset_results() internally. session data itself needs no fixing here; only
        the UI's own separate tracking (step_status/_step_timer_labels/_step_elapsed/
        _step_settings_snapshot) can go out of sync with it."""
        for key, field in self._STEP_RESULT_FIELD.items():
            if getattr(self.session, field) is None and self.step_status.get(key) not in (None, "not_run"):
                self._set_status(key, "not_run")
                self._step_timer_labels[key].configure(text="--", text_color=_LBL_NORMAL)
                self._step_elapsed.pop(key, None)
                self._step_settings_snapshot.pop(key, None)

    def _run_all_steps(self):
        if not self._sync_session():
            return
        enabled = [k for k, _, _ in STEPS if self._step_enable_vars[k].get()]
        if not enabled:
            messagebox.showinfo("Nothing to Run", "No steps are enabled.")
            return
        self._reset_run_controls()
        self._execute_chain(enabled)

    def _tick_timer(self, key: str):
        if self.step_status.get(key) != "running":
            return
        elapsed = _time.monotonic() - self._step_start_times[key] - self._paused_seconds.get(key, 0.0)
        self._step_timer_labels[key].configure(text=f"{elapsed:.1f} s")
        self.after(100, lambda: self._tick_timer(key))

    # ── Pause / Stop ─────────────────────────────────────────────────────────

    def _reset_run_controls(self):
        """Clear any leftover pause/cancel state from a previous (stopped) run before
        starting a new one."""
        self._cancel_event.clear()
        self._pause_event.clear()
        self.pause_btn.configure(text="Pause")

    def _toggle_pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.pause_btn.configure(text="Pause")
            self.log_queue.put("\n[Resumed]\n")
        else:
            self._pause_event.set()
            self.pause_btn.configure(text="Resume")
            self.log_queue.put(
                "\n[Paused - will pause before the next step starts, or between models "
                "during Hyperparameter Optimisation]\n"
            )

    def _stop_workflow(self):
        self._cancel_event.set()
        self._pause_event.clear()  # unblock a paused wait so cancellation takes effect now
        self.log_queue.put("\n[Stop requested - cancelling as soon as the current step allows]\n")

    def _on_close_request(self):
        """Handler for the window's close ('X') button. Without this, closing mid-run
        killed the Tk root while the daemon worker thread (which does not block process
        exit) kept writing files — a possible truncated pickle/PDF/CSV. Cooperative
        cancellation can't guarantee the write finishes either once the user chooses to
        close anyway, but an explicit warning replaces a silent, unannounced kill."""
        if self._running_step is not None:
            if not messagebox.askyesno(
                "Close phoenix_ml",
                "A step is currently running. Closing now will stop it immediately and "
                "may leave partially-written files (models, report, or exported data).\n\n"
                "Close anyway?",
                icon="warning",
            ):
                return
            self._cancel_event.set()
        self.destroy()

    def _make_checkpoint(self, key: str):
        """Cooperative pause/cancel hook passed to the session as `checkpoint_fn`. Long
        steps call this at safe boundaries (currently: HPO, once per model). Blocks here
        while paused rather than in the caller, and tracks paused time against `key` so
        it can be subtracted from that step's recorded elapsed time."""
        def checkpoint():
            while self._pause_event.is_set():
                if self._cancel_event.is_set():
                    raise WorkflowCancelled()
                t0 = _time.monotonic()
                _time.sleep(0.1)
                self._paused_seconds[key] = self._paused_seconds.get(key, 0.0) + (_time.monotonic() - t0)
            if self._cancel_event.is_set():
                raise WorkflowCancelled()
        return checkpoint

    # ── Step execution ────────────────────────────────────────────────────────

    def _execute_step(self, key: str, on_done=None):
        self._running_step = key
        self._set_status(key, "running")
        self._step_start_times[key] = _time.monotonic()
        self._paused_seconds[key] = 0.0
        self.session.checkpoint_fn = self._make_checkpoint(key)
        self._step_timer_labels[key].configure(text="0.0 s", text_color="#E07818")
        self._tick_timer(key)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            qs = _QueueStream(self.log_queue)
            sys.stdout = sys.stderr = qs
            exc_info = None
            cancelled = False
            try:
                STEP_FNS[key](self.session)
            except WorkflowCancelled:
                cancelled = True
                self.log_queue.put(f"\n[{key}] Cancelled by user.\n")
            except Exception:
                exc_info = traceback.format_exc()
                self.log_queue.put(f"\n[ERROR in {key}]\n{exc_info}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            # Not self.after(...) — this line runs on the worker thread; only
            # _poll_log (main thread) may touch Tk state. See _callback_queue.
            self._callback_queue.put(
                lambda: self._finish_step(key, exc_info, on_done, cancelled=cancelled))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_step(self, key: str, exc_info, on_done, cancelled: bool = False):
        elapsed = (_time.monotonic() - self._step_start_times.get(key, _time.monotonic())
                  - self._paused_seconds.get(key, 0.0))
        self._step_elapsed[key] = elapsed
        if cancelled:
            done_color = "#9E9E9E"
        elif exc_info:
            done_color = "#F44336"
        else:
            done_color = "gray60"
        self._step_timer_labels[key].configure(text=f"{elapsed:.1f} s", text_color=done_color)
        self._running_step = None
        self.session.checkpoint_fn = None
        if cancelled:
            self._set_status(key, "cancelled")
        else:
            self._set_status(key, "error" if exc_info else "done")
            if exc_info is None:
                # Record what settings this step actually ran with, so a later step
                # can tell whether they've since changed (see _stale_prereqs).
                self._step_settings_snapshot[key] = self._settings_snapshot()
                if key == "preprocessing":
                    # run_step_preprocessing calls session.reset_results() internally
                    # (the automatic fix for stale-results-across-datasets) — that
                    # clears every OTHER step's session data, but nothing told the UI,
                    # so e.g. "Training: Done" could keep showing (with its old
                    # elapsed time) after a dataset change even though
                    # session.training_results is now None. Reconcile every other
                    # step's displayed status against what session actually holds.
                    self._reconcile_step_statuses_with_session()
        if cancelled:
            return  # chain stops here; _run_all_steps/_run_single_step reset cancel state next time
        if exc_info is None and on_done:
            on_done()

    def _execute_chain(self, keys: list[str]):
        if not keys:
            return
        if self._cancel_event.is_set():
            # Nothing currently running (the cancelled step already cleaned up in
            # _finish_step) — just stop here without starting the next step.
            return
        if self._pause_event.is_set():
            self.after(200, lambda: self._execute_chain(keys))
            return
        key, *rest = keys
        prereqs = next(p for k, _, p in STEPS if k == key)
        if not all(self.step_status.get(p) == "done" for p in prereqs):
            self.log_queue.put(f"\n[Skipping {key}: prerequisites not complete]\n")
            self._set_status(key, "skipped")
            self._execute_chain(rest)
            return
        stale = self._stale_prereqs(prereqs)
        if stale:
            names = ", ".join(dict((k, n) for k, n, _ in STEPS)[p] for p in stale)
            self.log_queue.put(
                f"\n[Skipping {key}: settings changed since {names} last ran - "
                f"re-run {'it' if len(stale) == 1 else 'them'} first]\n"
            )
            self._set_status(key, "skipped")
            self._execute_chain(rest)
            return
        if key == "perl" and not self.session.perl_config_path:
            self.log_queue.put("\n[Skipping Physics-Enhanced Residual Learning: no physics configuration loaded]\n")
            self._set_status(key, "skipped")
            self._execute_chain(rest)
            return
        self._execute_step(key, on_done=lambda: self._execute_chain(rest))

    def _generate_report(self):
        if not self._sync_session():
            return
        self.session.step_timings = [
            (name, self._step_elapsed[key])
            for key, name, _ in STEPS
            if key in self._step_elapsed
        ]
        self.session.total_elapsed = sum(self._step_elapsed.values())
        self._running_step = "report"
        self._refresh_buttons()
        self.report_btn.configure(text="Generating...")
        self._report_status_lbl.configure(
            text="Generating report - this may take a while for long reports...")
        self._report_status_lbl.pack(side="left", padx=(8, 4))
        self._report_progress.pack(side="left", padx=4)
        self._report_progress.start()
        # (The "Generating Report" banner itself comes from run_step_report, like
        # every other step's banner.)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            qs = _QueueStream(self.log_queue)
            sys.stdout = sys.stderr = qs
            exc_info = None
            try:
                run_step_report(self.session)
            except Exception:
                exc_info = traceback.format_exc()
                self.log_queue.put(f"\n[ERROR generating report]\n{exc_info}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            self._callback_queue.put(lambda: self._finish_report(exc_info))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_report(self, exc_info):
        self._running_step = None
        self._refresh_buttons()
        self.report_btn.configure(text="Generate Report")
        self._report_progress.stop()
        self._report_progress.pack_forget()
        self._report_status_lbl.pack_forget()
        if exc_info is None:
            messagebox.showinfo("Report Ready", f"Saved to:\n{self.session.pdf_path}")
        else:
            messagebox.showerror("Report Error",
                                 "Report generation failed. See the progress log for details.")


def launch():
    app = PhoenixApp()
    app.mainloop()
