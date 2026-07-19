# pareto_analysis.py
# Pareto front analysis for performance vs training-time trade-offs across models.
# Non-dominated solutions are highlighted; auto log-scale applied when range ratio > 10.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from phoenix_ml.hyperparameter_optimisation import _HPO_LOWER_IS_BETTER


def _is_higher_better(metric_name: str) -> bool:
    # Reuses hyperparameter_optimisation's set rather than hand-duplicating it —
    # a metric added to one but not the other used to silently drift out of sync.
    return metric_name not in _HPO_LOWER_IS_BETTER


def _auto_log(values, threshold: float = 10.0) -> bool:
    """Return True if log scale is warranted: all values positive, max/min > threshold."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if len(valid) < 2:
        return False
    return (valid.max() / valid.min()) > threshold


# Relative tolerance for "strictly better" in dominance comparisons — without it,
# two solutions that are really tied but differ by float noise (e.g. the same
# training run's elapsed_time measured with jitter, or a metric accumulated in a
# different operation order across machines) could flip which one is classified
# as dominating, non-deterministically.
_DOMINANCE_EPS = 1e-9


def _approx_gt(a, b, eps=_DOMINANCE_EPS):
    return a > b + eps * max(abs(a), abs(b), 1.0)


def _approx_lt(a, b, eps=_DOMINANCE_EPS):
    return a < b - eps * max(abs(a), abs(b), 1.0)


def compute_pareto_front(perf_values, time_values, higher_is_better: bool = True):
    """
    Return a boolean array indicating which solutions are non-dominated.

    Dominance rule: solution A dominates B when A is at least as good on
    both performance and training time, and strictly better on at least one.
    Goal: maximise performance (or minimise for MSE/MAE), minimise training time.
    "At least as good" / "strictly better" both allow a small relative tolerance
    (_DOMINANCE_EPS) so float noise can't flip the classification.
    """
    perf = np.asarray(perf_values, dtype=float)
    time = np.asarray(time_values, dtype=float)
    n = len(perf)
    non_dominated = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Does j dominate i?
            if higher_is_better:
                j_perf_ok     = not _approx_lt(perf[j], perf[i])   # not meaningfully worse
                j_perf_strict = _approx_gt(perf[j], perf[i])
            else:
                j_perf_ok     = not _approx_gt(perf[j], perf[i])
                j_perf_strict = _approx_lt(perf[j], perf[i])
            j_time_ok     = not _approx_gt(time[j], time[i])   # time is always lower-is-better
            j_time_strict = _approx_lt(time[j], time[i])
            if j_perf_ok and j_time_ok and (j_perf_strict or j_time_strict):
                non_dominated[i] = False
                break

    return non_dominated


def _separate_overlapping_pts(ax, x_data, y_data, min_dist_pts=22, max_iter=200):
    """
    Nudge scatter points apart in display space so numbered circles don't overlap.
    Operates in pixel coordinates (so log/linear scale is handled transparently),
    then converts back to data coordinates. Returns adjusted (x, y) arrays.
    min_dist_pts: minimum centre-to-centre distance in display points; should be
                  at least the marker diameter (sqrt(s/pi)*2 ≈ 18 pt for s=260).
    """
    x = np.asarray(x_data, dtype=float).copy()
    y = np.asarray(y_data, dtype=float).copy()
    if len(x) <= 1:
        return x, y

    trans = ax.transData
    pts = trans.transform(np.column_stack([x, y]))   # (n, 2) in display coords

    for _ in range(max_iter):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.hypot(dx, dy)
                if dist < min_dist_pts:
                    moved = True
                    if dist < 1e-6:
                        # Exactly coincident: deterministic vertical push
                        pts[i, 1] += min_dist_pts / 2 + 1
                        pts[j, 1] -= min_dist_pts / 2 + 1
                    else:
                        push = (min_dist_pts - dist) / 2 + 0.5
                        nx, ny = dx / dist, dy / dist
                        pts[i, 0] += push * nx
                        pts[i, 1] += push * ny
                        pts[j, 0] -= push * nx
                        pts[j, 1] -= push * ny
        if not moved:
            break

    adj = trans.inverted().transform(pts)
    return adj[:, 0], adj[:, 1]


def plot_pareto_front(models, perf_values, time_values, target_var, perf_metric,
                      higher_is_better: bool = True, model_numbers: dict = None):
    """
    Scatter plot of all models in (training time, performance) space.
    Each model is represented by a numbered circle (number = UI selection order).
    A legend on the right lists the mapping; Pareto-optimal names are bold.
    Axis scales are chosen automatically per-axis: log if range ratio > 10×.
    Overlapping circles are nudged apart in display space for readability; the
    Pareto front line is drawn at the true (unmodified) positions.
    """
    models = list(models)
    perf = np.asarray(perf_values, dtype=float)
    time = np.asarray(time_values, dtype=float)
    n = len(models)

    non_dominated = compute_pareto_front(perf, time, higher_is_better)

    use_log_x = _auto_log(time)
    # _auto_log filters to positive values, so this works for both higher-is-better
    # (Q²/R² bounded 0-1, rarely triggers) and minimisation metrics (MSE always positive)
    use_log_y = _auto_log(perf)

    # Wider figure: left portion for the scatter, right portion for the model legend
    fig, ax = plt.subplots(figsize=(11, 6))

    # Raise axes bottom to leave room below for the outside legend
    ax.set_position([0.09, 0.22, 0.56, 0.66])

    dom_idx  = np.where(~non_dominated)[0]
    ndom_idx = np.where(non_dominated)[0]

    _MARKER = 260  # large enough to hold a two-digit number

    sc_dom = None
    if len(dom_idx):
        sc_dom = ax.scatter(time[dom_idx], perf[dom_idx],
                            color='steelblue', s=_MARKER, zorder=3, alpha=0.85,
                            label='Dominated')

    sc_ndom = ax.scatter(time[ndom_idx], perf[ndom_idx],
                         color='#E07818', s=_MARKER, zorder=4,
                         edgecolors='black', linewidths=1.2, label='Pareto-optimal')

    # Numbers inside circles — stored for later position adjustment
    text_objs = []
    for i, model_name in enumerate(models):
        num = model_numbers.get(model_name, i + 1) if model_numbers else i + 1
        text_objs.append(ax.text(
            time[i], perf[i], str(num),
            ha='center', va='center', fontsize=8, fontweight='bold',
            color='white', zorder=5,
        ))

    # Pareto front line
    if len(ndom_idx) > 1:
        order = ndom_idx[np.argsort(time[ndom_idx])]
        ax.plot(time[order], perf[order],
                color='#E07818', linestyle='--', linewidth=1.5,
                zorder=2, alpha=0.85, label='Pareto front')

    _METRIC_LABEL = {
        "MSE": "MSE", "MAE": "MAE", "RMSE": "RMSE",
        "NRMSE": "NRMSE", "MAPE": "MAPE (%)",
        "R^2": "R²", "ADJUSTED R^2": "Adjusted R²", "Q^2": "Q²/NSE",
    }
    perf_metric_label = _METRIC_LABEL.get(perf_metric, perf_metric)
    x_label = 'Training Time (s, log)' if use_log_x else 'Training Time (s)'
    y_label = f'{perf_metric_label} (log)' if use_log_y else perf_metric_label
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'Performance vs Training Time: {target_var}', fontsize=12)

    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    # Ideal direction marker — placed in a corner using axes fraction so it never
    # distorts the data range. Top-left for higher-is-better, bottom-left otherwise.
    ideal_y_frac = 0.95 if higher_is_better else 0.05
    ideal_va     = "top"    if higher_is_better else "bottom"
    ax.annotate(
        "★", xy=(0.03, ideal_y_frac), xycoords="axes fraction",
        fontsize=17, color="gold", ha="left", va=ideal_va, zorder=7,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.75),
    )
    ideal_handle = Line2D(
        [], [], marker="*", color="gold", markersize=10,
        markeredgecolor="#888888", markeredgewidth=0.5, linestyle="None",
        label="Ideal direction",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [ideal_handle], labels + ["Ideal direction"],
              fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=4, markerscale=0.6, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

    # Guard against extreme outliers (e.g. GPR with Q²=-10000) blowing up the
    # y-axis range, which makes bbox_inches='tight' produce a multi-thousand-inch image.
    if not use_log_y:
        finite_perf = perf[np.isfinite(perf)]
        if len(finite_perf) >= 2:
            lo, hi = np.percentile(finite_perf, [5, 95])
            span = hi - lo if hi != lo else abs(hi) * 0.1 + 1e-6
            y_lo, y_hi = ax.get_ylim()
            if (y_hi - y_lo) > max(span * 20, abs(hi) * 5 + 1):
                margin = span * 0.3
                ax.set_ylim(lo - margin, hi + margin)
                ax.text(0.5, 0.02, "y-axis clipped, extreme outlier values present",
                        transform=ax.transAxes, fontsize=7, color='gray',
                        ha='center', style='italic')

    # ── Right-side numbered model legend ─────────────────────────────────────
    legend_x = 0.68   # figure-coordinate x for the legend block
    top_y    = 0.90
    # Scale line spacing so all models fit within the figure height
    line_h   = min(0.072, 0.82 / max(n, 1))

    fig.text(legend_x, top_y, 'Models:', fontsize=9, fontweight='bold',
             transform=fig.transFigure, va='top', color='#222222')

    for i, model_name in enumerate(models):
        num    = model_numbers.get(model_name, i + 1) if model_numbers else i + 1
        is_nd  = non_dominated[i]
        label  = f"{num}.  {model_name}"
        weight = 'bold' if is_nd else 'normal'
        color  = '#C06010' if is_nd else '#444444'
        fig.text(legend_x, top_y - (i + 1) * line_h, label,
                 fontsize=8, fontweight=weight, color=color,
                 transform=fig.transFigure, va='top')

    # ── Nudge overlapping circles apart for readability ───────────────────────
    # The Pareto front LINE stays at true data positions; only circles/numbers move.
    try:
        fig.canvas.draw()   # lock transforms so display coords are valid
        time_adj, perf_adj = _separate_overlapping_pts(ax, time, perf)

        if sc_dom is not None:
            sc_dom.set_offsets(np.column_stack([time_adj[dom_idx], perf_adj[dom_idx]]))
        sc_ndom.set_offsets(np.column_stack([time_adj[ndom_idx], perf_adj[ndom_idx]]))
        for i, t in enumerate(text_objs):
            t.set_position((time_adj[i], perf_adj[i]))

        # Freeze axis limits: adjusted points must not auto-expand the axes
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    except Exception:
        pass  # fall back to original positions if anything goes wrong

    plt.close(fig)
    return fig


def run_pareto_analysis(session_metrics: dict, target_columns: list,
                        perf_metric: str, selected_models: list) -> dict:
    """
    Build one Pareto front figure per target variable.

    Performance value: best across all HPO methods (random / hyperopt / skopt),
    falling back to baseline default if HPO was not run.
    Training time: baseline elapsed_time from the default training run for THIS
    target specifically (not averaged across targets — training time is nearly
    identical per target in practice, so this is a simplification, not a bug).

    Returns dict {target_var: matplotlib.figure.Figure}.
    Skips targets where fewer than 2 models have complete data.
    """
    higher = _is_higher_better(perf_metric)
    figures = {}

    for target_var in target_columns:
        models, perfs, times = [], [], []

        for model_name in selected_models:
            # Baseline training time
            default_entry = (session_metrics
                             .get("default", {})
                             .get(model_name, {})
                             .get(target_var, {}))
            train_time = default_entry.get("elapsed_time")
            if train_time is None:
                continue

            # Best performance: scan HPO methods first, fall back to default
            best_perf = None
            for method in ("random", "hyperopt", "skopt"):
                entry = (session_metrics
                         .get(method, {})
                         .get(model_name, {})
                         .get(target_var, {}))
                val = entry.get(perf_metric)
                if val is None:
                    continue
                if best_perf is None:
                    best_perf = val
                elif higher and val > best_perf:
                    best_perf = val
                elif not higher and val < best_perf:
                    best_perf = val

            if best_perf is None:
                best_perf = default_entry.get(perf_metric)

            if best_perf is None:
                continue

            models.append(model_name)
            perfs.append(best_perf)
            times.append(train_time)

        if len(models) < 2:
            continue

        # Numbers follow the UI selection order (selected_models index + 1)
        model_numbers = {name: i + 1 for i, name in enumerate(selected_models)}
        fig = plot_pareto_front(models, perfs, times, target_var, perf_metric,
                                higher, model_numbers=model_numbers)
        figures[target_var] = fig

    return figures


def save_pareto_plots(figures: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for target_var, fig in figures.items():
        safe = target_var.lower().replace(' ', '_').replace('/', '_')
        path = os.path.join(output_dir, f"pareto_{safe}.png")
        try:
            fig.savefig(path, dpi=150)
        except Exception as e:
            print(f"[Pareto] Warning: could not save figure for '{target_var}': {e}")
            plt.close(fig)
            continue
        plt.close(fig)
        paths[target_var] = path
    return paths
