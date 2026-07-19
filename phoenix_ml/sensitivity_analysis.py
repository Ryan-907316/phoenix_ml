# sensitivity_analysis.py
# Global sensitivity analysis (Sobol, Morris) via SALib, treating the fitted model as a
# black-box function and sampling across the observed range of each feature. This is a
# different signal from SHAP/permutation/LOFO importance: those measure contribution to
# accuracy on the actual test-set distribution, whereas Sobol/Morris measure how much
# each input drives *output variance* across the whole sampled input space, including
# combinations that may be under-represented in the test set.
#
# Bounds are taken from the observed min/max of the sampled training data (in the same
# scaled space the model was fit on) — there is no other bounds-specification mechanism
# in phoenix_ml, so this is a simplification worth stating plainly wherever these results
# are shown: it describes sensitivity within the observed data's range, not a
# domain-expert-specified physical range.

import numpy as np
import matplotlib.pyplot as plt

try:
    from SALib.sample.sobol import sample as _sobol_sample
    from SALib.analyze.sobol import analyze as _sobol_analyze
    from SALib.sample.morris import sample as _morris_sample
    from SALib.analyze.morris import analyze as _morris_analyze
    from SALib.sample.fast_sampler import sample as _fast_sample
    from SALib.analyze.fast import analyze as _fast_analyze
    _SALIB_AVAILABLE = True
except Exception:
    _SALIB_AVAILABLE = False

from phoenix_ml.validation import require_int_at_least


def _build_problem(X_sampled, feature_names):
    """SALib "problem" dict: per-feature (min, max) bounds from the observed sample,
    in feature order. A tiny epsilon nudge on zero-variance features avoids SALib
    producing degenerate all-equal samples for that column (shouldn't normally occur
    post-preprocessing, but guarded defensively)."""
    X = np.asarray(X_sampled, dtype=float)
    bounds = []
    for i in range(X.shape[1]):
        lo, hi = float(X[:, i].min()), float(X[:, i].max())
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        bounds.append([lo, hi])
    return {"num_vars": len(feature_names), "names": list(feature_names), "bounds": bounds}


def compute_morris_indices(model, X_sampled, feature_names, n_trajectories=10, num_levels=4,
                           random_state=None):
    """Morris elementary-effects screening: cheap (r*(D+1) evaluations), good for a
    first-pass ranking of which features matter at all. Returns None (with a printed
    warning) if SALib isn't installed."""
    if not _SALIB_AVAILABLE:
        print("  [WARN] SALib is not installed - skipping Morris sensitivity analysis "
             "(pip install \"SALib>=1.4.5,<1.5\").")
        return None

    problem = _build_problem(X_sampled, feature_names)
    # optimal_trajectories left at its default (None): turning it on evaluates a larger
    # candidate pool before selecting the best r trajectories, which would balloon the
    # up-front cost well past the cheap r*(D+1) this method is chosen for.
    # None resolves to seed 0 (package-wide convention) so trajectories are
    # reproducible by default.
    param_values = _morris_sample(problem, N=n_trajectories, num_levels=num_levels,
                                  seed=0 if random_state is None else random_state)
    Y = model.predict(param_values)  # single batched call, not looped
    Si = _morris_analyze(problem, param_values, Y, num_levels=num_levels)
    return {
        "mu_star": np.asarray(Si["mu_star"]),
        "sigma": np.asarray(Si["sigma"]),
        "feature_names": list(feature_names),
        "n_evaluations": len(param_values),
    }


def compute_sobol_indices(model, X_sampled, feature_names, n_base=512, calc_second_order=False,
                          random_state=None):
    """Sobol variance-based indices: more expensive (N*(D+2) evaluations with
    calc_second_order=False) but decomposes output variance attributable to each
    feature more precisely than Morris. Returns None (with a printed warning) if
    SALib isn't installed."""
    if not _SALIB_AVAILABLE:
        print("  [WARN] SALib is not installed - skipping Sobol sensitivity analysis "
             "(pip install \"SALib>=1.4.5,<1.5\").")
        return None

    problem = _build_problem(X_sampled, feature_names)
    # None resolves to seed 0 (package-wide convention), matching Morris above.
    param_values = _sobol_sample(problem, n_base, calc_second_order=calc_second_order,
                                 seed=0 if random_state is None else random_state)
    Y = model.predict(param_values)  # single batched call, not looped
    Si = _sobol_analyze(problem, Y, calc_second_order=calc_second_order)
    return {
        "S1": np.asarray(Si["S1"]),
        "ST": np.asarray(Si["ST"]),
        "feature_names": list(feature_names),
        "n_evaluations": len(param_values),
    }


def compute_fast_indices(model, X_sampled, feature_names, n_samples=256, M=4,
                         random_state=None):
    """eFAST (extended Fourier Amplitude Sensitivity Testing): variance-based
    indices estimating the same quantities as Sobol (first-order S1 and
    total-order ST) via a different mechanism — Fourier analysis of the model
    output along a space-filling periodic search curve. Its value alongside
    Sobol is corroboration: two independent estimators agreeing raises
    confidence in the decomposition; disagreement flags sampling problems.
    Returns None (with a printed warning) if SALib isn't installed.

    N must exceed 4*M^2 (SALib hard requirement; 65 with the default M=4).

    Seeding note: SALib < 1.5's fast_sampler accepted a `seed` kwarg but did
    NOT honour it (verified empirically — two seeded calls differed), because
    the random phase shift drew from numpy's legacy global state; that had to
    be seeded (and restored afterwards) around the sample call as a
    workaround. SALib >= 1.5 (required by this package, see pyproject.toml)
    rewrote fast_sampler to build its own `numpy.random.Generator` from the
    `seed` argument directly, independent of the legacy global state -- so
    `seed` is now passed straight through to sampling instead.

    analyze()'s confidence-interval bootstrap is a separate matter: it still
    draws from numpy's *legacy* global RandomState regardless of version, and
    does so even when no seed is given -- left alone, that permanently
    advances the global state as a side effect, leaking into whatever
    unrelated code runs next. The save/restore below guards against that
    leak; it doesn't need to seed deterministically since S1/ST themselves
    don't depend on the bootstrap draws (only the confidence-interval bounds
    this function doesn't return would).
    """
    if not _SALIB_AVAILABLE:
        print("  [WARN] SALib is not installed - skipping FAST sensitivity analysis "
             "(pip install \"SALib>=1.5\").")
        return None

    require_int_at_least("sensitivity_fast_n", n_samples, minimum=4 * M * M + 1)
    problem = _build_problem(X_sampled, feature_names)
    # None resolves to seed 0 (package-wide convention), matching Morris/Sobol.
    seed = 0 if random_state is None else random_state
    param_values = _fast_sample(problem, n_samples, M=M, seed=seed)
    Y = model.predict(param_values)  # single batched call, not looped
    state = np.random.get_state()
    try:
        Si = _fast_analyze(problem, Y, M=M)
    finally:
        np.random.set_state(state)
    return {
        "S1": np.asarray(Si["S1"]),
        "ST": np.asarray(Si["ST"]),
        "feature_names": list(feature_names),
        "n_evaluations": len(param_values),
    }


def plot_sensitivity_indices(results, target_var, model_name, method_label):
    """Horizontal bar chart of sensitivity indices, sorted descending. `results` is the
    dict returned by compute_morris_indices() (uses mu_star) or compute_sobol_indices()/
    compute_fast_indices() (uses ST, the total-order index — accounts for interactions,
    more conservative than S1 for ranking overall importance)."""
    names = results["feature_names"]
    values = results["mu_star"] if "mu_star" in results else results["ST"]
    value_label = "Morris μ*" if "mu_star" in results else f"{method_label} total-order index (ST)"

    order = np.argsort(values)[::-1]
    sorted_names = [names[i] for i in order]
    sorted_values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, max(3, 0.5 * len(names) + 1.5)))
    ax.barh(sorted_names, sorted_values, color="#E07818", alpha=0.85)
    ax.set_xlabel(value_label)
    ax.set_title(f"Global Sensitivity ({method_label}): {target_var} ({model_name})", fontweight="bold")
    ax.invert_yaxis()

    x_max = max(sorted_values) if sorted_values else 1.0
    ax.set_xlim(right=x_max * 1.2 if x_max > 0 else 1.0)
    for i, v in enumerate(sorted_values):
        # A slightly negative index (sampling noise on a near-zero-effect
        # feature) must still be labelled to the RIGHT of its bar, not at the
        # bar's own tip (which would land inside/left of it) -- anchor at the
        # zero baseline instead whenever the value itself is negative.
        label_x = v + x_max * 0.02 if v >= 0 else x_max * 0.02
        ax.text(label_x, i, f"{v:.3f}", va="center", ha="left", fontsize=7, color="#333333")

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_sobol_fast_comparison(sobol_results, fast_results, target_var, model_name):
    """Side-by-side Sobol vs FAST comparison — one figure, two panels (first-order
    S1 left, total-order ST right), paired bars per feature. Only produced when BOTH
    methods ran: they estimate the same indices, so showing them apart would just be
    two redundant charts, whereas paired bars make their agreement (the point of
    running both) directly visible. Features sorted by Sobol ST descending."""
    names = sobol_results["feature_names"]
    fast_by_name_s1 = dict(zip(fast_results["feature_names"], fast_results["S1"]))
    fast_by_name_st = dict(zip(fast_results["feature_names"], fast_results["ST"]))

    order = np.argsort(sobol_results["ST"])[::-1]
    sorted_names = [names[i] for i in order]
    panels = [
        ("S1 Index by Feature", "First-order index (S1)",
         [sobol_results["S1"][i] for i in order],
         [fast_by_name_s1.get(names[i], np.nan) for i in order]),
        ("ST Index by Feature", "Total-order index (ST)",
         [sobol_results["ST"][i] for i in order],
         [fast_by_name_st.get(names[i], np.nan) for i in order]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3, 0.55 * len(names) + 1.5)),
                             sharey=True)
    y_pos = np.arange(len(sorted_names))
    bar_h = 0.38
    for ax, (title, xlabel, sobol_vals, fast_vals) in zip(axes, panels):
        sobol_bars = ax.barh(y_pos - bar_h / 2, sobol_vals, height=bar_h,
                             color="#E07818", alpha=0.85, label="Sobol")
        fast_bars = ax.barh(y_pos + bar_h / 2, fast_vals, height=bar_h,
                            color="#1878E0", alpha=0.85, label="FAST")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.grid(True, axis="x", linewidth=0.4)

        finite_vals = [v for v in list(sobol_vals) + list(fast_vals) if np.isfinite(v)]
        x_max = max(finite_vals) if finite_vals else 1.0
        ax.set_xlim(right=x_max * 1.25 if x_max > 0 else 1.0)
        for bars, vals in ((sobol_bars, sobol_vals), (fast_bars, fast_vals)):
            for bar, v in zip(bars, vals):
                if not np.isfinite(v):
                    continue
                # Same negative-value handling as plot_sensitivity_indices:
                # anchor at the zero baseline, not the bar's own tip, so the
                # label always sits clear to the right of the bar.
                label_x = v + x_max * 0.02 if v >= 0 else x_max * 0.02
                ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                       va="center", ha="left", fontsize=6.5, color="#333333")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Global Sensitivity, Sobol vs FAST: {target_var} ({model_name})",
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig
