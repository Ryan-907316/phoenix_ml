# ui.py
# customtkinter UI for Phoenix ML.

from __future__ import annotations
import io, os, queue, sys, threading, traceback, time as _time
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
    run_step_uq_before, run_step_interpretability,
    run_step_hpo, run_step_cv, run_step_uq_after,
    run_step_perl, run_step_report,
)
from phoenix_ml.models import models_dict as ALL_MODELS
from phoenix_ml.dataset_cleaning import (
    auto_classify_columns, apply_cleaning, build_issue_mask,
    ACTION_FFILL, ACTION_BFILL, ACTION_NONE, ACTION_REMOVE,
    ACTION_CAP, ACTION_INTERP, ACTION_MEAN, ACTION_MEDIAN, ACTION_DROP,
    OUTLIER_NONE, OUTLIER_IQR, OUTLIER_ZSCORE, OUTLIER_PERCENTILE,
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
    ("preprocessing",    "Preprocessing",                         []),
    ("training",         "Baseline Training",                     ["preprocessing"]),
    ("uq_before",        "Uncertainty Quantification (Before)",   ["training"]),
    ("interpretability", "Interpretability",                      ["training"]),
    ("hpo",              "Hyperparameter Optimisation",           ["training"]),
    ("cv",               "Cross-Validation",                     ["training"]),
    ("uq_after",         "Uncertainty Quantification (After)",    ["training", "hpo"]),
    ("perl",             "Physics-Enhanced Machine Learning",     ["training"]),
]

STEP_FNS = {
    "preprocessing":    run_step_preprocessing,
    "training":         run_step_training,
    "uq_before":        run_step_uq_before,
    "interpretability": run_step_interpretability,
    "hpo":              run_step_hpo,
    "cv":               run_step_cv,
    "uq_after":         run_step_uq_after,
    "perl":             run_step_perl,
}

STATUS_INFO = {
    "not_run": ("gray60",  "Not Run"),
    "running": ("#E07818", "Running..."),
    "done":    ("#4CAF50", "Done"),
    "error":   ("#F44336", "Error  x"),
    "skipped": ("#7B97C7", "Skipped"),
}

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
        self.step_status: dict[str, str] = {k: "not_run" for k, *_ in STEPS}
        self._running_step: str | None = None
        self._step_timer_labels: dict[str, ctk.CTkLabel] = {}
        self._step_start_times: dict[str, float]         = {}
        self._step_elapsed:     dict[str, float]         = {}

        self._build_ui()
        self._poll_log()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.pack(fill="both", expand=True, padx=8, pady=8)
        for name in ["Dataset Cleaning", "Physics Modelling",
                     "Home", "Models", "Preprocessing",
                     "Uncertainty Quantification", "Interpretability",
                     "Hyperparameter Optimisation", "Cross-Validation"]:
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
        ctk.CTkButton(top, text="Load", width=70,
                      command=self._clean_load_dataset).grid(row=0, column=3, padx=4, pady=6)
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
            values=[OUTLIER_NONE, OUTLIER_IQR, OUTLIER_ZSCORE, OUTLIER_PERCENTILE],
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
            OUTLIER_NONE:       ("Outlier Threshold:",        "1.5"),
            OUTLIER_IQR:        ("Threshold (x IQR):",        "1.5"),
            OUTLIER_ZSCORE:     ("Threshold (x Std Dev):",    "3.0"),
            OUTLIER_PERCENTILE: ("Keep Within (%):",          "95"),
        }
        label_text, default_val = defaults.get(choice, ("Outlier Threshold:", "1.5"))
        self._clean_outlier_thresh_label.configure(text=label_text)
        self._clean_outlier_thresh.delete(0, "end")
        self._clean_outlier_thresh.insert(0, default_val)

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
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc)); return

        self._clean_df          = df
        self._clean_df_original = df.copy()
        self._clean_col_info    = auto_classify_columns(df)
        self._clean_prev_roles  = {}

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
        try:
            min_run = int(self._clean_stuck_min_run.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Stuck min run must be an integer."); return

        cleaned, log_lines = apply_cleaning(
            df=self._clean_df_original,
            col_info=self._clean_col_info,
            missing_action=self._clean_missing_action.get(),
            outlier_method=self._clean_outlier_method.get(),
            outlier_threshold=threshold,
            outlier_action=self._clean_outlier_action.get(),
            stuck_enabled=self._clean_stuck_enabled.get(),
            stuck_min_run=min_run,
            stuck_action=self._clean_stuck_action.get(),
            remove_duplicates=self._clean_remove_dupes.get(),
        )
        self._clean_df = cleaned

        # Refresh stats for columns that survived cleaning; keep excluded columns
        # in col_info so a second Apply still knows to exclude them.
        remaining = auto_classify_columns(cleaned)
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
                          "See examples/DC_Motors_Dataset_Generation.py for a worked example.",
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
                          "floor( )   ceil( )\n"
                          "gradient( )  ← finite difference\n"
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
                                     "See examples/DC_Motors_Dataset_Generation.py for the expected format.")
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
            try:
                module = import_physics_script(script_path)
                missing = [a for a in ("governing_function", "constants", "input_vars", "output_vars")
                           if not hasattr(module, a)]
                if missing:
                    messagebox.showerror("Physics Modelling",
                                         f"Script is missing: {', '.join(missing)}")
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
                messagebox.showerror("Physics Modelling", f"Failed to generate dataset:\n{e}")
                return
            messagebox.showinfo(
                "Physics Modelling",
                f"Saved {len(result_df)} rows × {len(result_df.columns)} columns to:\n{output_path}")
            return

        # Expression mode
        texts = [x["entry"].get() for x in self._phys_expressions if x["entry"].get().strip()]
        if not texts:
            messagebox.showerror("Physics Modelling", "Add at least one expression.")
            return

        try:
            df = pd.read_csv(dataset_path)
            result_df, log = apply_expressions(df, texts)
            result_df = select_output_columns(result_df, self._phys_output_cols.get())
            result_df.to_csv(output_path, index=False)
        except ExpressionError as e:
            messagebox.showerror("Physics Modelling", str(e))
            return
        except Exception as e:
            messagebox.showerror("Physics Modelling", f"Failed to generate dataset: {e}")
            return

        messagebox.showinfo(
            "Physics Modelling",
            f"Saved {len(result_df)} rows × {len(result_df.columns)} columns to:\n{output_path}\n\n"
            + "\n".join(log))

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

        ctk.CTkLabel(top, text="Physics Configuration:", anchor="w", width=150).grid(row=3, column=0, padx=8, pady=4, sticky="w")
        self.perl_config_entry = ctk.CTkEntry(top, placeholder_text="(optional) Path to a saved physics configuration file")
        self.perl_config_entry.grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        self.perl_config_entry.bind("<FocusOut>", lambda e: self._on_perl_config_entry_change())
        self.perl_config_entry.bind("<Return>",   lambda e: self._on_perl_config_entry_change())
        ctk.CTkButton(top, text="Browse", width=80, command=self._browse_perl_config_load).grid(row=3, column=2, padx=(4, 8))

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
            var = tk.BooleanVar(value=(key != "perl"))
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
        self.report_btn = ctk.CTkButton(actions, text="Generate Report",
                                        command=self._generate_report, width=160, state="disabled")
        self.report_btn.pack(side="left", padx=4)

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

        btn_row = len(ALL_MODEL_NAMES) + 1
        bf = ctk.CTkFrame(tab, fg_color="transparent")
        bf.grid(row=btn_row, column=0, padx=10, pady=8, sticky="w")
        ctk.CTkButton(bf, text="Select All",   width=110,
                      command=lambda: [v.set(True)  for v in self._model_vars.values()]).pack(side="left", padx=4)
        ctk.CTkButton(bf, text="Deselect All", width=110,
                      command=lambda: [v.set(False) for v in self._model_vars.values()]).pack(side="left", padx=4)

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
        self._preproc_random_state_lbl, self._preproc_random_state = self._lbl_entry_ref(
            tab, "Random State (Random split):", "0", 3)
        self._preproc_show_plots  = self._lbl_check(tab, "Show Preprocessing Plots",                True,  4)
        self._preproc_dist_dummy  = self._lbl_check(tab, "Distance Correlation: Include Dummy",      True,  5)
        self._preproc_dist_mp     = self._lbl_check(tab, "Distance Correlation: Marchenko-Pastur",   False, 6)
        self._refresh_preproc_state()

    def _refresh_preproc_state(self, *_):
        is_random = self._preproc_split.get().lower() == "random"
        state = "normal" if is_random else "disabled"
        self._preproc_random_state.configure(state=state)
        self._preproc_random_state_lbl.configure(
            text_color=("gray20" if is_random else "gray60")
        )

    # ── Uncertainty Quantification Tab ────────────────────────────────────────

    def _setup_uq_tab(self):
        tab = self.tabview.tab("Uncertainty Quantification")
        tab.columnconfigure(1, weight=1)

        # Method selection via checkboxes (replaces single option menu)
        ctk.CTkLabel(tab, text="Methods:", anchor="w", width=160).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        method_frame = ctk.CTkFrame(tab, fg_color="transparent")
        method_frame.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self._uq_bootstrap_var = tk.BooleanVar(value=True)
        self._uq_conformal_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(method_frame, text="Bootstrapping", variable=self._uq_bootstrap_var,
                        command=self._refresh_uq_state).pack(side="left", padx=(0, 12))
        ctk.CTkCheckBox(method_frame, text="Conformal Prediction", variable=self._uq_conformal_var,
                        command=self._refresh_uq_state).pack(side="left")

        self._uq_n_bootstrap_lbl, self._uq_n_bootstrap = self._lbl_entry_ref(
            tab, "Bootstrap Samples:", "200", 1)
        self._uq_confidence  = self._lbl_entry(tab, "Confidence Interval (%):", "95",   2)
        self._uq_calib_frac_lbl, self._uq_calib_frac = self._lbl_entry_ref(
            tab, "Calibration Fraction:", "0.05", 3)
        self._uq_subsample   = self._lbl_entry(tab, "Subsample Test Size:",     "50",   4)
        self._uq_n_jobs_lbl, self._uq_n_jobs = self._lbl_entry_ref(
            tab, "Bootstrap Parallel Jobs:", "1", 5)
        ctk.CTkLabel(tab, text="(1 = off, -1 = all cores)",
                     anchor="w").grid(row=6, column=1, padx=4, sticky="w")
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

    # ── Interpretability Tab ──────────────────────────────────────────────────

    def _setup_interpretability_tab(self):
        tab = self.tabview.tab("Interpretability")
        tab.columnconfigure(1, weight=1)
        self._interp_model     = self._lbl_option(tab, "Preferred Model:", ALL_MODEL_NAMES, 0)
        self._interp_model.set("XGBoost Regressor")
        self._interp_test_size = self._lbl_entry(tab, "Test Sample Size:",       "1000", 1)
        self._interp_bg_size   = self._lbl_entry(tab, "Background Sample Size:", "10",   2)
        self._interp_subsample = self._lbl_entry(tab, "Subsample:",              "250",  3)
        self._interp_grid_res  = self._lbl_entry(tab, "Grid Resolution:",        "10",   4)

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
        ctk.CTkCheckBox(mf, text="Random Search",
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
            ("Mean Squared Error", "MSE"),
            ("R²",            "R^2"),
            ("Adjusted R²",   "ADJUSTED R^2"),
            ("Q²",            "Q^2"),
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
            ["Sobol", "Halton", "Latin Hypercube", "Random"], row); row += 1

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
        tab = self.tabview.tab("Cross-Validation")
        tab.columnconfigure(1, weight=1)

        # Method dropdown — inline so we can pass command=
        ctk.CTkLabel(tab, text="Cross-Validation Method:", anchor="w", width=240).grid(
            row=0, column=0, padx=10, pady=4, sticky="w")
        self._cv_method = ctk.CTkOptionMenu(tab, values=_CV_METHOD_DISPLAY, width=200,
                                            command=self._on_cv_method_change)
        self._cv_method.set(_CV_METHOD_DISPLAY[0])
        self._cv_method.grid(row=0, column=1, padx=10, pady=4, sticky="w")

        # Dynamic parameter rows — use _lbl_entry_ref so we can show/hide the label too
        self._cv_n_splits_lbl,  self._cv_n_splits  = self._lbl_entry_ref(tab, "Number of Splits:", "10",  1)
        self._cv_test_size_lbl, self._cv_test_size  = self._lbl_entry_ref(tab, "Test Size:",        "0.2", 2)
        self._cv_rand_state_lbl,self._cv_rand_state = self._lbl_entry_ref(tab, "Random State:",     "0",   3)
        self._cv_n_repeats_lbl, self._cv_n_repeats  = self._lbl_entry_ref(tab, "Number of Repeats:","5",   4)
        self._cv_p_lbl,         self._cv_p          = self._lbl_entry_ref(tab, "p (leave-p-out):",  "2",   5)

        # Shown only for Leave One Out (which has no parameters)
        self._cv_no_params_lbl = ctk.CTkLabel(tab,
            text="No parameters required for this method.", text_color="gray")
        self._cv_no_params_lbl.grid(row=1, column=0, columnspan=2, padx=10, pady=4, sticky="w")

        # Scoring metric (fixed below the dynamic rows)
        ctk.CTkLabel(tab, text="Scoring Metric:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=7, column=0, padx=10, pady=(10, 2), sticky="w", columnspan=2)

        self._cv_metric_var = tk.StringVar(value="R^2")
        _cv_metrics = [
            ("Mean Absolute Error", "MAE"),
            ("Mean Squared Error",  "MSE"),
            ("R²",             "R^2"),
            ("Adjusted R²",    "ADJUSTED R^2"),
            ("Q²",             "Q^2"),
            ("Explained Variance",  "Explained Variance"),
        ]
        for i, (text, val) in enumerate(_cv_metrics):
            ctk.CTkRadioButton(tab, text=text,
                               variable=self._cv_metric_var, value=val).grid(
                row=8 + i, column=0, padx=20, pady=2, sticky="w")

        # Apply initial visibility for the default method
        self._on_cv_method_change(_CV_METHOD_DISPLAY[0])

    def _on_cv_method_change(self, method: str):
        visible = _CV_FIELDS.get(method, ())
        field_map = {
            "n_splits":    (self._cv_n_splits_lbl,   self._cv_n_splits),
            "test_size":   (self._cv_test_size_lbl,  self._cv_test_size),
            "random_state":(self._cv_rand_state_lbl, self._cv_rand_state),
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

    def _browse_clean_dataset(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self._clean_dataset_entry.delete(0, "end")
            self._clean_dataset_entry.insert(0, p)

    def _browse_clean_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self._clean_output_entry.delete(0, "end")
            self._clean_output_entry.insert(0, p)

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
        s.split_method         = self._preproc_split.get()   # "First"/"Last"/"Random"; code uses .lower()
        s.scaler_type          = self._preproc_scaler.get()
        s.show_preproc_plots   = self._preproc_show_plots.get()
        s.dist_corr_dummy      = self._preproc_dist_dummy.get()
        s.dist_corr_mp         = self._preproc_dist_mp.get()
        if s.split_method.lower() == "random":
            try:
                rs_val = self._preproc_random_state.get().strip()
                s.split_random_state = int(rs_val) if rs_val else None
            except ValueError:
                messagebox.showerror("Invalid Input", "Random State must be an integer."); return False
        else:
            s.split_random_state = None
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
            )
        except ValueError:
            messagebox.showerror("Invalid Input", "Uncertainty Quantification fields must be numeric."); return False

        # Interpretability
        try:
            s.interpretability_settings = dict(
                preferred_model_name=self._interp_model.get(),
                test_sample_size=int(self._interp_test_size.get()),
                background_sample_size=int(self._interp_bg_size.get()),
                subsample=int(self._interp_subsample.get()),
                grid_resolution=int(self._interp_grid_res.get()),
            )
        except ValueError:
            messagebox.showerror("Invalid Input", "Interpretability fields must be integers."); return False

        # Hyperparameter Optimisation methods
        methods = []
        if self._hpo_use_random.get():   methods.append("random")
        if self._hpo_use_hyperopt.get(): methods.append("hyperopt")
        if self._hpo_use_skopt.get():    methods.append("skopt")
        if not methods:
            messagebox.showerror("Invalid Input", "Please select at least one HPO method."); return False
        s.methods_to_run  = methods
        s.hpo_metric      = self._hpo_metric_var.get()
        s.sampling_method = self._hpo_sampling.get()
        try:
            s.n_iter       = int(self._hpo_n_iter.get())
            s.sample_size  = int(self._hpo_sample_size.get())
            s.evals        = int(self._hpo_evals.get())
            s.calls        = int(self._hpo_calls.get())
            s.n_jobs       = int(self._hpo_n_jobs.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Hyperparameter Optimisation numeric fields must be integers."); return False

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
            if "test_size"   in _visible: cv_args["test_size"]   = float(self._cv_test_size.get())
            if "random_state"in _visible: cv_args["random_state"]= int(self._cv_rand_state.get())
            if "n_repeats"   in _visible: cv_args["n_repeats"]   = int(self._cv_n_repeats.get())
            if "p"           in _visible: cv_args["p"]           = int(self._cv_p.get())
            s.cv_args = cv_args
        except ValueError:
            messagebox.showerror("Invalid Input", "Cross-Validation fields must be numeric."); return False
        s.cv_method      = _CV_METHOD_MAP[_method_display]
        s.scoring_metric = self._cv_metric_var.get()

        # PERL
        perl_path = self.perl_config_entry.get().strip()
        s.perl_config_path = perl_path

        return True

    # ── Button state ──────────────────────────────────────────────────────────

    def _refresh_buttons(self):
        busy = self._running_step is not None
        for key, _, prereqs in STEPS:
            prereqs_done = all(self.step_status.get(p) == "done" for p in prereqs)
            extra_ok = (key != "perl") or bool(self.session.perl_config_path)
            enabled = prereqs_done and extra_ok and not busy
            self._step_run_btns[key].configure(state="normal" if enabled else "disabled")

        self.run_all_btn.configure(state="normal" if not busy else "disabled")

        can_report = (self.session.preprocessing_results is not None
                      and self.session.training_results is not None
                      and not busy)
        self.report_btn.configure(state="normal" if can_report else "disabled")

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
        self.after(50, self._poll_log)

    # ── Step execution ────────────────────────────────────────────────────────

    def _run_single_step(self, key: str):
        if not self._sync_session():
            return
        self._execute_step(key)

    def _run_all_steps(self):
        if not self._sync_session():
            return
        enabled = [k for k, _, _ in STEPS if self._step_enable_vars[k].get()]
        if not enabled:
            messagebox.showinfo("Nothing to Run", "No steps are enabled.")
            return
        self._execute_chain(enabled)

    def _tick_timer(self, key: str):
        if self.step_status.get(key) != "running":
            return
        elapsed = _time.monotonic() - self._step_start_times[key]
        self._step_timer_labels[key].configure(text=f"{elapsed:.1f} s")
        self.after(100, lambda: self._tick_timer(key))

    def _execute_step(self, key: str, on_done=None):
        self._running_step = key
        self._set_status(key, "running")
        self._step_start_times[key] = _time.monotonic()
        self._step_timer_labels[key].configure(text="0.0 s", text_color="#E07818")
        self._tick_timer(key)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            qs = _QueueStream(self.log_queue)
            sys.stdout = sys.stderr = qs
            exc_info = None
            try:
                STEP_FNS[key](self.session)
            except Exception:
                exc_info = traceback.format_exc()
                self.log_queue.put(f"\n[ERROR in {key}]\n{exc_info}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            self.after(0, lambda: self._finish_step(key, exc_info, on_done))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_step(self, key: str, exc_info, on_done):
        elapsed = _time.monotonic() - self._step_start_times.get(key, _time.monotonic())
        self._step_elapsed[key] = elapsed
        done_color = "#F44336" if exc_info else "gray60"
        self._step_timer_labels[key].configure(text=f"{elapsed:.1f} s", text_color=done_color)
        self._running_step = None
        self._set_status(key, "error" if exc_info else "done")
        if exc_info is None and on_done:
            on_done()

    def _execute_chain(self, keys: list[str]):
        if not keys:
            return
        key, *rest = keys
        prereqs = next(p for k, _, p in STEPS if k == key)
        if not all(self.step_status.get(p) == "done" for p in prereqs):
            self.log_queue.put(f"\n[Skipping {key}: prerequisites not complete]\n")
            self._set_status(key, "skipped")
            self._execute_chain(rest)
            return
        if key == "perl" and not self.session.perl_config_path:
            self.log_queue.put("\n[Skipping Physics-Enhanced Machine Learning: no physics configuration loaded]\n")
            self._set_status(key, "skipped")
            self._execute_chain(rest)
            return
        self._execute_step(key, on_done=lambda: self._execute_chain(rest))

    def _generate_report(self):
        if not self._sync_session():
            return
        self.session.total_elapsed = sum(self._step_elapsed.values())
        self._running_step = "report"
        self._refresh_buttons()
        self.log_queue.put("\n" + "-" * 60 + "\n  Generating Report\n" + "-" * 60 + "\n")

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
            self.after(0, lambda: self._finish_report(exc_info))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_report(self, exc_info):
        self._running_step = None
        self._refresh_buttons()
        if exc_info is None:
            messagebox.showinfo("Report Ready", f"Saved to:\n{self.session.pdf_path}")
        else:
            messagebox.showerror("Report Error",
                                 "Report generation failed. See the progress log for details.")


def launch():
    app = PhoenixApp()
    app.mainloop()
