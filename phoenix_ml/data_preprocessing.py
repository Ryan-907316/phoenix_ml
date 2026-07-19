# data_preprocessing.py
# This module serves as the analysis of the dataset provided before it undergoes any model evaluation.
# This includes the test/train split, features-target scatter plots, boxplots, and distance correlation. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import seaborn as sns
import dcor
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_preprocess_data(filepath, test_size, split_method="random", target_columns=None,
                            scaler_type="Standard", random_state=None):
    """
    Load a CSV, choose targets, split into train/test by a chosen method, and standardize features.

    Args:
        filepath (str): Path to CSV.
        test_size (float): Proportion of rows in the test split (0–1).
        split_method (str): 'random' | 'first' | 'last'.
        target_columns (list[str] | None): Columns to treat as targets. If None, uses last column.
        scaler_type (str): Name of the scaler to apply to features (see the scaler lookup).
        random_state (int | None): Seed for the 'random' split method.

    Returns:
        tuple: (df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler,
                target_columns, feature_names)
    """
    df = pd.read_csv(filepath)

    # Automatically make the last column the target variable if none are specified
    if target_columns is None:
        target_columns = df.columns[-1:]

    missing_targets = [t for t in target_columns if t not in df.columns]
    if missing_targets:
        raise ValueError(
            f"Target column(s) not found in the dataset: {missing_targets}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Inclusive bounds, not exclusive: 0 and 1 are handled by the existing -0/whole-frame
    # clamp below for "first"/"last" (tested, deliberate behaviour) — this check exists to
    # catch genuinely nonsensical values (e.g. a user typing 50 meaning "50%"), which
    # "random" already rejects via sklearn's train_test_split but "first"/"last" used to
    # silently clamp into an almost-empty split with no error at all.
    if not (0 <= test_size <= 1):
        raise ValueError(
            f"test_size must be between 0 and 1 (e.g. 0.2 for 20%); got {test_size!r}."
        )

    # Split into features (X) and target variables (y)
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    feature_names = X.columns.tolist()

    # Real-world CSVs (especially raw sensor exports) often carry text columns,
    # missing readings, or inf values. The scaler and models reject these with
    # cryptic errors deep in sklearn, so check up front and name the columns —
    # pointing the user at the Dataset Cleaning tab instead of a traceback.
    # Targets are checked too, not just features: a text/categorical target used
    # to sail past this exact check while the NaN/Inf checks below (which scan
    # `df`, not just `X`) already covered it — an inconsistently narrower check
    # for the same failure mode (found via a systematic failure-mode sweep).
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric column(s): {non_numeric}. Exclude or convert them "
            f"first (the Dataset Cleaning tab can do this)."
        )
    bad_nan = [c for c in df.columns if df[c].isna().any()]
    if bad_nan:
        raise ValueError(
            f"Column(s) contain missing values: {bad_nan}. Handle them first "
            f"(the Dataset Cleaning tab offers interpolation, fill, and row-drop options)."
        )
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bad_inf = [c for c in numeric_cols if np.isinf(df[c].to_numpy()).any()]
    if bad_inf:
        raise ValueError(
            f"Column(s) contain infinite values: {bad_inf}. Replace or remove them first."
        )

    # Split data based on the chosen method
    if split_method.lower() == "random":
        # None resolves to seed 0 (package-wide convention): a "random" split is
        # still reproducible by default unless the caller supplies its own seed.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=0 if random_state is None else random_state,
        )
    elif split_method.lower() in ("first", "last"):
        # Clamp so both splits are always non-empty. Guards the test_size=0 edge
        # where ceil() gives 0 and X.iloc[-0:] would return the WHOLE frame as the
        # test set (Python's -0 == 0), leaving an empty training set.
        test_count = int(np.ceil(test_size * len(X)))
        test_count = min(max(test_count, 1), len(X) - 1)
        if len(X) < 2:
            raise ValueError("Dataset needs at least 2 rows to split into train and test sets.")
        if split_method.lower() == "first":
            X_test,  y_test  = X.iloc[:test_count],  y.iloc[:test_count]
            X_train, y_train = X.iloc[test_count:],  y.iloc[test_count:]
        else:
            X_test,  y_test  = X.iloc[-test_count:], y.iloc[-test_count:]
            X_train, y_train = X.iloc[:-test_count], y.iloc[:-test_count]
    else:
        raise ValueError("split_method must be 'random', 'first', or 'last'.")

    _scalers = {
        "Standard":  StandardScaler,
        "MinMax":    MinMaxScaler,
        "Robust":    RobustScaler,
        # The UI offers "None" (no scaling) in its Feature Scaling dropdown; without
        # this entry .get()'s default silently substituted StandardScaler for it.
        "None":      None,
    }
    scaler_cls = _scalers.get(scaler_type, StandardScaler)
    scaler = scaler_cls() if scaler_cls is not None else None
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled  = X_test.values

    return df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, target_columns, feature_names

def plot_target_vs_target(y_train, y_test, target_columns):
    """
    Scatter plot of first two target variables, colored by train/test.
    """
    if len(target_columns) < 2:
        print("Not enough target variables specified to plot graph of target variables.")
        return

    # Scatter plot of target variables
    target1, target2 = target_columns[:2]  # Use first two for now (maybe add more in later versions?)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_train[target1], y_train[target2], color='black', alpha=0.5, s=10, label=f'Training Data (n={len(y_train)})')
    ax.scatter(y_test[target1], y_test[target2], color='red', alpha=0.5, s=10, label=f'Testing Data (n={len(y_test)})')
    ax.set_xlabel(target1)
    ax.set_ylabel(target2)
    ax.set_title(f'Train/Test Split: {target1} vs {target2}', fontweight='bold')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig

# Max subplot rows per figure for wide 3-col layouts (5"×4" subplots, 15" figure width).
# 4 rows × 4" = 16" → ~558pt display, safely under the 652pt A4 content limit.
_MAX_ROWS_PER_FIG = 4

def plot_features_vs_targets(X_train, y_train, target_columns):
    """
    For each target, create grid scatter plots of every feature vs the target.
    Returns {target: [fig, ...]} where each figure has at most _MAX_ROWS_PER_FIG rows.
    """
    figs = {}
    if X_train.shape[1] == 0:
        # Reachable by default: target_columns defaults to the last two columns
        # when unspecified, so any 2-column dataset leaves zero feature columns
        # here. num_cols would be 0, making range(0, 0, 0) raise ValueError deeper
        # in this function — nothing to plot, so just skip instead.
        print("No feature columns to plot against targets.")
        return figs

    for target_var in target_columns:
        col_list = list(X_train.columns)
        num_features = len(col_list)
        num_cols = min(num_features, 3)
        subplot_w, subplot_h = 5.0, 4.0
        features_per_fig = _MAX_ROWS_PER_FIG * num_cols

        chunks = [col_list[i:i + features_per_fig] for i in range(0, num_features, features_per_fig)]
        n_parts = len(chunks)
        target_figs = []

        for part_idx, chunk_cols in enumerate(chunks):
            chunk_n = len(chunk_cols)
            num_rows = (chunk_n + num_cols - 1) // num_cols

            fig, axes = plt.subplots(
                num_rows, num_cols,
                figsize=(num_cols * subplot_w, num_rows * subplot_h),
                squeeze=False,
            )
            part_label = f" (Part {part_idx + 1} of {n_parts})" if n_parts > 1 else ""
            fig.suptitle(f"Scatter Plots of Features Against {target_var}{part_label}",
                         fontsize=14, fontweight='bold')
            axes_flat = axes.flatten()

            for i, column in enumerate(chunk_cols):
                ax = axes_flat[i]
                ax.scatter(X_train[column], y_train[target_var], alpha=0.5, s=8)
                try:
                    slope, intercept = np.polyfit(X_train[column], y_train[target_var], 1)
                    ax.plot(X_train[column], slope * X_train[column] + intercept, color='red')
                except (np.linalg.LinAlgError, ValueError):
                    pass
                ax.set_xlabel(column, fontsize=10)
                ax.set_ylabel(target_var, fontsize=10)
                ax.set_title(f'{column} vs {target_var}', fontsize=11)
                ax.tick_params(labelsize=9)

            for j in range(i + 1, len(axes_flat)):
                fig.delaxes(axes_flat[j])

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            target_figs.append(fig)

        figs[target_var] = target_figs

    return figs

def plot_boxplots(df, target_columns):
    """
    Boxplots for all features + targets.
    Returns a list of figures, each with at most _MAX_ROWS_PER_FIG rows.
    """
    features = df.drop(columns=target_columns)
    targets = df[target_columns]
    combined_df = pd.concat([features, targets], axis=1)
    all_columns = combined_df.columns.tolist()
    num_columns = len(all_columns)

    num_cols = min(num_columns, 3)
    subplot_w, subplot_h = 5.0, 3.5
    cols_per_fig = _MAX_ROWS_PER_FIG * num_cols

    chunks = [all_columns[i:i + cols_per_fig] for i in range(0, num_columns, cols_per_fig)]
    n_parts = len(chunks)
    figs = []

    for part_idx, chunk_cols in enumerate(chunks):
        chunk_n = len(chunk_cols)
        num_rows = (chunk_n + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(num_cols * subplot_w, num_rows * subplot_h),
            squeeze=False,
        )
        part_label = f" (Part {part_idx + 1} of {n_parts})" if n_parts > 1 else ""
        fig.suptitle(f"Box Plots of Features and Target Variables{part_label}",
                     fontsize=14, fontweight='bold')
        axes_flat = axes.flatten()

        for i, column in enumerate(chunk_cols):
            ax = axes_flat[i]
            data = combined_df[column]

            ax.boxplot(data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='black'),
                       flierprops=dict(marker='o', color='red', alpha=0.5))

            median = data.median()
            mean = data.mean()
            q1, q3 = data.quantile([0.25, 0.75])
            p5, p95 = data.quantile([0.05, 0.95])
            min_val, max_val = data.min(), data.max()

            ax.axvline(mean, color='red', linestyle='-', linewidth=1.5)
            ax.axvline(median, color='orange', linestyle='-', linewidth=1.5)
            ax.axvline(p5, color='green', linestyle='--', linewidth=1)
            ax.axvline(p95, color='green', linestyle='--', linewidth=1)

            ax.set_title(column, fontsize=11)
            ax.set_xlabel("Value", fontsize=10)
            ax.set_yticks([])
            ax.tick_params(labelsize=9)

            legend_handles = [
                mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label=f"Mean: {mean:.2f}"),
                mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, label=f"Median: {median:.2f}"),
                mpatches.Patch(facecolor='lightblue', edgecolor='black', label=f"IQR: {q1:.2f} to {q3:.2f}"),
                mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label=f"5th/95th: {p5:.2f}, {p95:.2f}"),
                mlines.Line2D([], [], color='black', linestyle='-', linewidth=1, label=f"Min/Max: {min_val:.2f}, {max_val:.2f}")
            ]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8, title="Statistics", title_fontsize=9)

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        figs.append(fig)

    return figs

def _mp_denoise(matrix, n_samples):
    """
    Denoise a symmetric matrix using Marchenko-Pastur thresholding.

    Eigenvalues below the MP upper bound λ+ = (1 + sqrt(p/n))^2 are treated as
    noise and zeroed out; the matrix is reconstructed from signal eigenvectors only.

    Returns (denoised_matrix, n_signal_components, lambda_plus).
    """
    p = matrix.shape[0]
    q = p / n_samples
    lambda_plus = (1.0 + np.sqrt(q)) ** 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    signal_mask = eigenvalues > lambda_plus
    n_signal = int(np.sum(signal_mask))
    if n_signal == 0:
        # All noise — return zero matrix with one kept component to avoid blank plot
        n_signal = 1
        signal_mask[np.argmax(eigenvalues)] = True
    denoised = eigenvectors[:, signal_mask] @ np.diag(eigenvalues[signal_mask]) @ eigenvectors[:, signal_mask].T
    denoised = np.clip(denoised, 0.0, 1.0)
    return denoised, n_signal, lambda_plus


def plot_distance_correlation_matrix(df, title="Distance Correlation Matrix", cmap='RdYlGn',
                                     dummy=False, mp_threshold=False, annotate=True):
    """
    Compute & plot a distance-correlation heatmap for numeric columns.

    Args:
        df (pd.DataFrame): Numeric columns only.
        title (str): Figure title.
        cmap (str): Matplotlib colormap.
        dummy (bool): If True, append a random dummy column as a noise baseline.
        mp_threshold (bool): If True, apply Marchenko-Pastur denoising to the DC matrix.
        annotate (bool): Show numeric values on the heatmap.

    Returns:
        (pd.DataFrame, matplotlib.figure.Figure): distance corr matrix and the figure.
    """
    n_samples = len(df)

    if dummy:
        df = df.copy()
        if "Dummy" in df.columns:
            # A real feature happens to be named "Dummy" — do NOT silently
            # overwrite it with synthetic noise (that would corrupt a real
            # column's data and quietly poison the noise-baseline flagging
            # downstream in compute_feature_selection_recommendations, which
            # would treat the real feature's values as the noise floor).
            # Skip the synthetic column; feature selection just runs without
            # a noise baseline for this run instead.
            print("[WARN] Dataset already has a column named 'Dummy' - skipping "
                  "the synthetic noise baseline to avoid overwriting real data.")
        else:
            # Seeded so the noise baseline (and the feature-selection flags derived
            # from it) is identical between runs on the same dataset.
            df["Dummy"] = np.random.default_rng(0).normal(size=n_samples)

    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All columns in the dataset must be numeric for distance correlation calculation.")

    features = df.columns
    n = len(features)
    dist_corr_matrix = np.zeros((n, n))

    # dcor is symmetric and dcor(x, x) == 1, so only the upper triangle needs computing
    for i in range(n):
        dist_corr_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            v = dcor.distance_correlation(df[features[i]], df[features[j]])
            dist_corr_matrix[i, j] = v
            dist_corr_matrix[j, i] = v

    dist_corr_df = pd.DataFrame(dist_corr_matrix, index=features, columns=features)

    # Optionally apply MP denoising
    plot_df = dist_corr_df
    n_signal, lp = None, None
    if mp_threshold:
        denoised, n_signal, lp = _mp_denoise(dist_corr_matrix, n_samples)
        plot_df = pd.DataFrame(denoised, index=features, columns=features)

    # Consistent per-cell sizing: 0.65" per feature, minimum 8"
    cell_in = 0.65
    fig_size = max(8, n * cell_in)

    # Scale annotation font inversely with feature count; hide above threshold
    _ANNOT_THRESHOLD = 25
    show_annot = annotate and (n <= _ANNOT_THRESHOLD)
    annot_size = max(5, min(9, int(80 / n))) if show_annot else 8
    tick_fontsize = max(7, min(10, int(90 / n)))

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = plt.gca()
    sns.heatmap(
        plot_df, annot=show_annot, cmap=cmap, square=True,
        linewidths=0.3 if n <= 20 else 0.1, fmt=".4f",
        annot_kws={"size": annot_size}, cbar_kws={"shrink": 0.8}, ax=ax,
        vmin=0.0, vmax=1.0,
    )
    plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    _title = title if show_annot else f"{title}\n(values not annotated, {n} features)"

    if mp_threshold:
        comp_line = f"MP denoised: {n_signal}/{n} signal components retained"
        lp_line   = f"Noise threshold λ⁺ = {lp:.3f}"
        plt.title(f"{_title}\n{comp_line}\n{lp_line}", fontsize=10, fontweight='bold', pad=12)
    else:
        plt.title(_title, fontsize=13, fontweight='bold', pad=8)

    plt.tight_layout()

    return dist_corr_df, fig

def compute_feature_selection_recommendations(
    dist_corr_df, X_train, y_train, feature_names, target_columns,
    dummy_col="Dummy", redundancy_threshold=0.90,
):
    """
    Flag features for user review based on the distance correlation matrix.

    Two advisory flags (nothing is dropped automatically):
      - Noise-level: feature's max dcor with any target ≤ the Dummy noise baseline
      - Redundant:   inter-feature dcor > redundancy_threshold

    Returns a dict of results, or None if inputs are insufficient.
    """
    # dcor lives in [0, 1], so a threshold outside it silently degenerates:
    # negative flagged every feature pair as redundant, > 1 disabled the check
    # entirely with nothing saying so.
    from phoenix_ml.validation import require_in_range
    require_in_range("feat_sel_redundancy_threshold", redundancy_threshold, 0, 1)
    if dist_corr_df is None:
        return None

    feat_cols = [f for f in feature_names if f in X_train.columns]
    tgt_cols = list(target_columns)
    if not feat_cols or not tgt_cols:
        return None

    # Compute feature-to-target distance correlations
    ft_data = {}
    for feat in feat_cols:
        ft_data[feat] = {
            tgt: float(dcor.distance_correlation(X_train[feat].values, y_train[tgt].values))
            for tgt in tgt_cols
        }
    relevance = pd.DataFrame(ft_data).T   # shape: (n_features, n_targets)
    max_relevance = relevance.max(axis=1)

    # Noise baseline: max dcor of Dummy with any real feature in the dc matrix
    dc_feats = [f for f in feat_cols if f in dist_corr_df.columns]
    has_dummy = dummy_col in dist_corr_df.index
    noise_baseline = None
    if has_dummy and dc_feats:
        noise_baseline = float(dist_corr_df.loc[dummy_col, dc_feats].max())

    # Flag 1: max feature-target dcor at or below the noise floor
    noise_flagged = (
        [f for f in feat_cols if max_relevance.get(f, 0.0) <= noise_baseline]
        if noise_baseline is not None else []
    )

    # Flag 2: inter-feature redundancy above threshold
    feat_to_feat = dist_corr_df.loc[dc_feats, dc_feats]
    redundant_pairs, recommended_drop, checked = [], set(), set()
    for i, fa in enumerate(dc_feats):
        for fb in dc_feats[i + 1:]:
            if (fa, fb) in checked:
                continue
            checked.add((fa, fb))
            val = float(feat_to_feat.loc[fa, fb])
            if val > redundancy_threshold:
                # Flag the one with lower max relevance as the one to consider dropping
                drop = fb if max_relevance.get(fa, 0.0) >= max_relevance.get(fb, 0.0) else fa
                redundant_pairs.append((fa, fb, round(val, 4)))
                recommended_drop.add(drop)

    # Build summary DataFrame
    rows = []
    for f in feat_cols:
        per_tgt = {f"dcor({t})": round(float(relevance.loc[f, t]), 4) for t in tgt_cols}
        redundant_with = [
            f"{fb} ({dv:.3f})" for fa, fb, dv in redundant_pairs if fa == f
        ] + [
            f"{fa} ({dv:.3f})" for fa, fb, dv in redundant_pairs if fb == f
        ]
        flags = (["Noise-level"] if f in noise_flagged else []) + (
            ["Redundant"] if f in recommended_drop else []
        )
        rows.append({
            "Feature": f,
            "Max dcor (target)": round(float(max_relevance.get(f, 0.0)), 4),
            **per_tgt,
            "Noise Flagged": f in noise_flagged,
            "Redundant With": ", ".join(redundant_with),
            "Status": ", ".join(flags) if flags else "No flags",
        })

    return {
        "feature_relevance": relevance,
        "max_relevance": max_relevance,
        "noise_baseline": noise_baseline,
        "noise_flagged": noise_flagged,
        "redundant_pairs": redundant_pairs,
        "recommended_drop": sorted(recommended_drop),
        "summary_df": pd.DataFrame(rows),
    }


def plot_feature_selection_bar(feat_sel_results, feature_names):
    """
    Horizontal bar chart of max feature-target dcor with a noise-baseline reference line.
    Bars are coloured green (no flags), orange (redundant), or red (noise-level).
    """
    if feat_sel_results is None:
        return None

    summary_df = feat_sel_results["summary_df"].copy()
    noise_baseline = feat_sel_results["noise_baseline"]
    noise_flagged = set(feat_sel_results["noise_flagged"])
    recommended_drop = set(feat_sel_results["recommended_drop"])

    summary_df = summary_df.sort_values("Max dcor (target)", ascending=True)
    features = summary_df["Feature"].tolist()
    values = summary_df["Max dcor (target)"].tolist()
    n = len(features)

    fig, ax = plt.subplots(figsize=(10, max(3.5, n * 0.42 + 1.5)))

    bar_colors = []
    for f in features:
        if f in noise_flagged:
            bar_colors.append("#D94F3D")
        elif f in recommended_drop:
            bar_colors.append("#E07818")
        else:
            bar_colors.append("#4C9B6B")

    bars = ax.barh(features, values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    if noise_baseline is not None:
        ax.axvline(noise_baseline, color="#D94F3D", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Max distance correlation with any target", fontsize=10)
    ax.set_title("Feature Relevance: Distance Correlation with Targets", fontweight="bold", fontsize=12)
    x_max = max(values) * 1.2 if values else 1.0
    ax.set_xlim(0, min(1.0, x_max))

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#4C9B6B", alpha=0.85, label="No flags"),
        mpatches.Patch(color="#E07818", alpha=0.85, label="Flagged for review: redundant"),
        mpatches.Patch(color="#D94F3D", alpha=0.85, label="Flagged for review: at/below noise baseline"),
    ]
    if noise_baseline is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="#D94F3D", linestyle="--", linewidth=1.5,
                          label=f"Noise baseline (Dummy max): {noise_baseline:.3f}")
        )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_pca_analysis(X_train_scaled, feature_names, target_columns=None, y_train=None):
    """
    Returns (fig_scree, biplot_dict).

    Scree: individual + cumulative explained variance with 80/90/95% reference lines.
    biplot_dict: {target_name: fig} for each target (one plot per target, coloured by that
                 target's values), or {"": fig} if no targets given.
    Returns (None, {}) if fewer than 1 feature; biplot_dict is {} if fewer than 2 features.
    """
    n_features = len(feature_names)
    if n_features < 1:
        return None, {}

    pca = PCA()
    scores = pca.fit_transform(X_train_scaled)
    ev = pca.explained_variance_ratio_ * 100
    cumev = np.cumsum(ev)
    n_show = min(n_features, 15)

    # ── Scree plot ─────────────────────────────────────────────────────────────
    fig_scree, ax1 = plt.subplots(figsize=(10, 4.5))
    comp_labels = [f"PC{i+1}" for i in range(n_show)]

    ax1.bar(comp_labels, ev[:n_show], color="#4472C4", alpha=0.80, label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Explained Variance (%)", fontsize=11, color="#4472C4")
    ax1.tick_params(axis="y", labelcolor="#4472C4")
    ax1.set_title("PCA Scree Plot: Explained Variance per Component",
                  fontweight="bold", fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(comp_labels, cumev[:n_show], color="#E07818", marker="o", markersize=5,
             linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=11, color="#E07818")
    ax2.tick_params(axis="y", labelcolor="#E07818")
    ax2.set_ylim(0, 105)

    for pct, ls in [(80, ":"), (90, "--"), (95, "-.")]:
        ax2.axhline(pct, color="grey", linestyle=ls, linewidth=1.0, alpha=0.7,
                    label=f"{pct}% threshold")

    idx_90 = next((i for i, c in enumerate(cumev) if c >= 90.0), None)
    if idx_90 is not None and idx_90 < n_show:
        ax2.axvline(idx_90, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
        ax2.text(idx_90 + 0.15, 91.5, f"90% @ PC{idx_90+1}", fontsize=8, color="red")

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="center right")
    fig_scree.tight_layout()
    plt.close(fig_scree)

    # ── Biplot(s): one per target variable ────────────────────────────────────
    if n_features < 2:
        return fig_scree, {}

    pc1_var, pc2_var = ev[0], ev[1]
    loadings = pca.components_[:2].T   # (n_features, 2)

    rng = np.random.default_rng(0)
    max_pts = 800
    sample_idx = (rng.choice(len(scores), max_pts, replace=False)
                  if len(scores) > max_pts else np.arange(len(scores)))
    sc = scores[sample_idx]

    # Scale arrows to 75% of each axis range independently, then take the tighter
    # of the two — prevents labels from overflowing on the shorter axis (e.g. PC2
    # when PC1 variance is much larger).
    range_x = max(np.abs(sc[:, 0]).max(), 1e-9)
    range_y = max(np.abs(sc[:, 1]).max(), 1e-9)
    max_lx = max(np.abs(loadings[:, 0]).max(), 1e-9)
    max_ly = max(np.abs(loadings[:, 1]).max(), 1e-9)
    scale = min(range_x * 0.75 / max_lx, range_y * 0.75 / max_ly)

    tgt_list = list(target_columns) if target_columns is not None and len(target_columns) > 0 else [""]

    biplot_dict = {}
    for tgt_name in tgt_list:
        fig_biplot, ax = plt.subplots(figsize=(9, 7))

        if tgt_name and y_train is not None:
            tgt_arr = (
                y_train[tgt_name].values if hasattr(y_train, "columns")
                else np.asarray(y_train)[:, 0]
            )[sample_idx]
            scatter = ax.scatter(sc[:, 0], sc[:, 1], c=tgt_arr, cmap="coolwarm",
                                 alpha=0.45, s=14, edgecolors="none")
            cb = fig_biplot.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
            cb.set_label(tgt_name, fontsize=9)
            title = (f"PCA Biplot: PC1 × PC2 ({pc1_var + pc2_var:.1f}% variance),"
                     f" coloured by {tgt_name}")
        else:
            ax.scatter(sc[:, 0], sc[:, 1], alpha=0.35, s=14, color="steelblue", edgecolors="none")
            title = f"PCA Biplot: PC1 × PC2 ({pc1_var + pc2_var:.1f}% of total variance)"

        texts = []
        for i, feat in enumerate(feature_names):
            lx, ly = loadings[i, 0] * scale, loadings[i, 1] * scale
            ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.1, alpha=0.8))
            texts.append(ax.text(lx * 1.10, ly * 1.10, feat, fontsize=7,
                                 ha="center", va="center", color="#222222", alpha=0.85))

        ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.18, linestyle="--")

        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888888",
                                                       lw=0.6, alpha=0.6))
        except ImportError:
            pass

        fig_biplot.tight_layout()
        plt.close(fig_biplot)

        biplot_dict[tgt_name] = fig_biplot

    return fig_scree, biplot_dict


def compute_multicollinearity(X_train, X_train_scaled, feature_names):
    """
    Compute per-feature VIF (on raw features) plus the condition number and matrix
    rank (both on scaled features). Returns (vif_df, condition_number, matrix_rank).

    Rank complements the condition number: the condition number is a continuous
    ill-conditioning measure needing thresholds to interpret, while rank vs. the
    feature count is a binary, self-explanatory flag for EXACT linear dependence —
    e.g. a derived Expression Mode column that is precisely the sum of two existing
    features, or more features than training rows.
    """
    X_arr = np.array(X_train, dtype=float)
    X_const = np.column_stack([np.ones(len(X_arr)), X_arr])

    vif_values = []
    for i in range(1, X_const.shape[1]):
        try:
            vif = variance_inflation_factor(X_const, i)
        except Exception:
            vif = float("nan")
        vif_values.append(vif)

    vif_df = pd.DataFrame({
        "Feature": list(feature_names),
        "VIF": [round(float(v), 3) for v in vif_values],
    })

    X_scaled_arr = np.array(X_train_scaled, dtype=float)
    cond_number = float(np.linalg.cond(X_scaled_arr))
    matrix_rank = int(np.linalg.matrix_rank(X_scaled_arr))
    return vif_df, cond_number, matrix_rank


# Function to actually run the program
def run_preprocessing_workflow(
    file_path,
    test_size=0.2,
    split_method="random",
    target_columns=None,
    plot_target_vs_target_enabled=True,
    plot_features_vs_targets_enabled=True,
    plot_boxplots_enabled=True,
    plot_distance_corr_enabled=True,
    dist_corr_dummy=True,
    dist_corr_mp=False,
    show_multicollinearity=True,
    plot_pca_enabled=True,
    feat_sel_enabled=True,
    feat_sel_redundancy_threshold=0.90,
    scaler_type="Standard",
    random_state=None,
    figures=None,
):
    if figures is None:
        figures = {}
    else:
        # A caller-supplied dict must not silently accumulate stale entries across
        # reconfigured runs (e.g. a plot type disabled since the last call would
        # otherwise leave its old figure behind) — this function always builds a
        # complete, fresh figure set for the current call.
        figures.clear()
    df_preview = pd.read_csv(file_path)
    print(f"[INFO] Available columns in the dataset: {df_preview.columns.tolist()}")

    # Default to last 2 columns if not specified
    if target_columns is None:
        target_columns = df_preview.columns[-2:].tolist()

    (
        df, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
        target_columns, feature_names
    ) = load_and_preprocess_data(
        file_path, test_size=test_size, split_method=split_method,
        target_columns=target_columns, scaler_type=scaler_type, random_state=random_state,
    )

    # Dataset metadata for the report
    n_rows, n_cols = df.shape
    features = [c for c in df.columns if c not in target_columns]
    train_n, test_n = len(X_train), len(X_test)
    meta = {
        "dataset_path": file_path,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "targets": list(target_columns),
        "features": features,
        "n_features": len(features),
        "split_method": split_method,
        "test_size_param": test_size,
        "train_count": train_n,
        "test_count": test_n,
        "train_prop": train_n / n_rows if n_rows else 0.0,
        "test_prop": test_n / n_rows if n_rows else 0.0,
        "scaler_name": type(scaler).__name__ if scaler is not None else "None",
    }

    print(f"[INFO] Dataset has {n_rows} rows and {n_cols} columns "
          f"({train_n} train / {test_n} test).")
    print(f"[INFO] Target columns: {target_columns}")

    X_train_df = pd.DataFrame(X_train, columns=df.columns.drop(target_columns))

    if plot_target_vs_target_enabled:
        print("\nGenerating Target vs Target plot...")
        fig = plot_target_vs_target(y_train, y_test, target_columns)
        if fig: figures["Target vs Target"] = fig

    if plot_features_vs_targets_enabled:
        print("\nGenerating Feature vs Target scatter plots...")
        feature_vs_target_figs = plot_features_vs_targets(X_train_df, y_train, target_columns)
        for target, fig_list in feature_vs_target_figs.items():
            for part_idx, fig in enumerate(fig_list):
                key = f"Features vs {target}" if len(fig_list) == 1 else f"Features vs {target} Part {part_idx + 1}"
                figures[key] = fig

    if plot_boxplots_enabled:
        print("\nGenerating Box plots...")
        box_figs = plot_boxplots(df, target_columns)
        for part_idx, fig in enumerate(box_figs):
            key = "Boxplots" if len(box_figs) == 1 else f"Boxplots Part {part_idx + 1}"
            figures[key] = fig

    dist_corr_df = None
    if plot_distance_corr_enabled:
        print("\nGenerating Distance Correlation Matrix...")
        features_df = df.drop(columns=target_columns)
        dist_corr_df, fig = plot_distance_correlation_matrix(
            features_df, dummy=dist_corr_dummy, mp_threshold=False)
        if fig:
            figures["Distance Correlation"] = fig
        if dist_corr_mp:
            print("  Applying Marchenko-Pastur denoising...")
            _, fig_mp = plot_distance_correlation_matrix(
                features_df, dummy=dist_corr_dummy, mp_threshold=True,
                title="Distance Correlation Matrix (MP Denoised)")
            if fig_mp:
                figures["Distance Correlation (MP Denoised)"] = fig_mp

    # ── Multicollinearity (VIF + condition number + rank) ──────────────────────
    multicollinearity_vif_df = None
    multicollinearity_cond = None
    multicollinearity_rank = None
    if show_multicollinearity:
        print("\nComputing multicollinearity diagnostics (VIF + condition number + rank)...")
        multicollinearity_vif_df, multicollinearity_cond, multicollinearity_rank = compute_multicollinearity(
            X_train, X_train_scaled, feature_names
        )

    # ── Feature selection recommendations (advisory flags only) ────────────────
    feat_sel_results = None
    if feat_sel_enabled and dist_corr_df is not None:
        print("\nComputing feature selection recommendations...")
        feat_sel_results = compute_feature_selection_recommendations(
            dist_corr_df, X_train, y_train, feature_names, target_columns,
            redundancy_threshold=feat_sel_redundancy_threshold,
        )
        if feat_sel_results is not None:
            fig_bar = plot_feature_selection_bar(feat_sel_results, feature_names)
            if fig_bar:
                figures["Feature Selection"] = fig_bar
            if feat_sel_results["noise_flagged"]:
                print(f"  [INFO] Features flagged at noise level: {feat_sel_results['noise_flagged']}")
            if feat_sel_results["redundant_pairs"]:
                pairs = [(a, b) for a, b, _ in feat_sel_results["redundant_pairs"]]
                print(f"  [INFO] Redundant feature pairs (dcor > {feat_sel_redundancy_threshold}): {pairs}")
            if not feat_sel_results["noise_flagged"] and not feat_sel_results["redundant_pairs"]:
                print("  [INFO] No features flagged.")

    # ── PCA visualisation (scree + one biplot per target) ──────────────────────
    if plot_pca_enabled:
        print("\nGenerating PCA analysis...")
        fig_scree, biplot_dict = plot_pca_analysis(
            X_train_scaled, feature_names,
            target_columns=target_columns, y_train=y_train,
        )
        if fig_scree:
            figures["PCA Scree"] = fig_scree
        for tgt_name, fig_bp in biplot_dict.items():
            key = f"PCA Biplot {tgt_name}" if tgt_name else "PCA Biplot"
            figures[key] = fig_bp

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "target_columns": target_columns,
        "feature_names": feature_names,
        "distance_corr_matrix": dist_corr_df,
        "multicollinearity": {
            "vif_df": multicollinearity_vif_df,
            "condition_number": multicollinearity_cond,
            "matrix_rank": multicollinearity_rank,
            "n_features": len(feature_names),
        } if multicollinearity_vif_df is not None else None,
        "feature_selection": feat_sel_results,
        "figures": figures,
        "meta": meta,
    }
