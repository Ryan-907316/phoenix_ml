# pareto_analysis.py
# Pareto front analysis for performance vs training-time trade-offs across models.
# Non-dominated solutions are highlighted; auto log-scale applied when range ratio > 10.

import os
import numpy as np
import matplotlib.pyplot as plt


def _is_higher_better(metric_name: str) -> bool:
    return metric_name not in ("MSE", "MAE", "RMSE")


def _auto_log(values, threshold: float = 10.0) -> bool:
    """Return True if log scale is warranted: all values positive, max/min > threshold."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if len(valid) < 2:
        return False
    return (valid.max() / valid.min()) > threshold


def compute_pareto_front(perf_values, time_values, higher_is_better: bool = True):
    """
    Return a boolean array indicating which solutions are non-dominated.

    Dominance rule: solution A dominates B when A is at least as good on
    both performance and training time, and strictly better on at least one.
    Goal: maximise performance (or minimise for MSE/MAE), minimise training time.
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
            j_perf_ok = perf[j] >= perf[i] if higher_is_better else perf[j] <= perf[i]
            j_time_ok = time[j] <= time[i]
            j_perf_strict = perf[j] > perf[i] if higher_is_better else perf[j] < perf[i]
            j_time_strict = time[j] < time[i]
            if j_perf_ok and j_time_ok and (j_perf_strict or j_time_strict):
                non_dominated[i] = False
                break

    return non_dominated


def plot_pareto_front(models, perf_values, time_values, target_var, perf_metric,
                      higher_is_better: bool = True, model_numbers: dict = None):
    """
    Scatter plot of all models in (training time, performance) space.
    Each model is represented by a numbered circle (number = UI selection order).
    A legend on the right lists the mapping; Pareto-optimal names are bold.
    Axis scales are chosen automatically per-axis: log if range ratio > 10×.
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

    # Shrink the axes rightward to leave room for the numbered legend
    ax.set_position([0.09, 0.12, 0.56, 0.76])

    dom_idx  = np.where(~non_dominated)[0]
    ndom_idx = np.where(non_dominated)[0]

    _MARKER = 260  # large enough to hold a two-digit number

    if len(dom_idx):
        ax.scatter(time[dom_idx], perf[dom_idx],
                   color='steelblue', s=_MARKER, zorder=3, alpha=0.85, label='Dominated')

    ax.scatter(time[ndom_idx], perf[ndom_idx],
               color='#E07818', s=_MARKER, zorder=4,
               edgecolors='black', linewidths=1.2, label='Pareto-optimal')

    # Numbers inside circles
    for i, model_name in enumerate(models):
        num = model_numbers.get(model_name, i + 1) if model_numbers else i + 1
        ax.text(time[i], perf[i], str(num),
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=5)

    # Pareto front line
    if len(ndom_idx) > 1:
        order = ndom_idx[np.argsort(time[ndom_idx])]
        ax.plot(time[order], perf[order],
                color='#E07818', linestyle='--', linewidth=1.5,
                zorder=2, alpha=0.85, label='Pareto front')

    x_label = 'Training Time (s, log scale)' if use_log_x else 'Training Time (s)'
    y_label = f'{perf_metric} (log scale)' if use_log_y else perf_metric
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'Performance vs Training Time — Target: {target_var}', fontsize=12)

    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    ax.legend(fontsize=9, loc='lower right', markerscale=0.6)
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
                ax.text(0.5, 0.02, "y-axis clipped — extreme outlier values present",
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

    plt.close(fig)
    return fig


def run_pareto_analysis(session_metrics: dict, target_columns: list,
                        perf_metric: str, selected_models: list) -> dict:
    """
    Build one Pareto front figure per target variable.

    Performance value: best across all HPO methods (random / hyperopt / skopt),
    falling back to baseline default if HPO was not run.
    Training time: baseline elapsed_time from the default training run (per model,
    averaged across targets for the X axis since training time per target is nearly
    identical in practice).

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
