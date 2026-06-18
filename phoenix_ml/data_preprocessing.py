# data_preprocessing.py
# This module serves as the analysis of the dataset provided before it undergoes any model evaluation.
# This includes the test/train split, features-target scatter plots, boxplots, and distance correlation. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import seaborn as sns
import dcor

def load_and_preprocess_data(filepath, test_size, split_method="random", target_columns=None):
    """
    Load a CSV, choose targets, split into train/test by a chosen method, and standardize features.

    Args:
        filepath (str): Path to CSV.
        test_size (float): Proportion of rows in the test split (0–1).
        split_method (str): 'random' | 'first' | 'last'.
        target_columns (list[str] | None): Columns to treat as targets. If None, uses last column.

    Returns:
        tuple: (df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler,
                target_columns, feature_names)
    """
    df = pd.read_csv(filepath)

    # Automatically make the last column the target variable if none are specified
    if target_columns is None:
        target_columns = df.columns[-1:]

    # Split into features (X) and target variables (y)
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    feature_names = X.columns.tolist()

    # Split data based on the chosen method
    if split_method.lower() == "random":
        # Random split using scikit-learn's train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    elif split_method.lower() == "first":
        # Use the first 'test_size' proportion of rows as the test set
        test_count = int(np.ceil(test_size * len(X)))
        X_test = X.iloc[:test_count]
        y_test = y.iloc[:test_count]
        X_train = X.iloc[test_count:]
        y_train = y.iloc[test_count:]
    elif split_method.lower() == "last":
        # Use the last 'test_size' proportion of rows as the test set
        test_count = int(np.ceil(test_size * len(X)))
        X_test = X.iloc[-test_count:]
        y_test = y.iloc[-test_count:]
        X_train = X.iloc[:-test_count]
        y_train = y.iloc[:-test_count]
    else:
        raise ValueError("split_method must be 'random', 'first', or 'last'.")

    # Standardise features (mean = 0, variance = 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    ax.scatter(y_train[target1], y_train[target2], color='black', alpha=0.5, label=f'Training Data (n={len(y_train)})')
    ax.scatter(y_test[target1], y_test[target2], color='red', alpha=0.5, label=f'Testing Data (n={len(y_test)})')
    ax.set_xlabel(target1)
    ax.set_ylabel(target2)
    ax.set_title(f'Train/Test Split of {target1} vs {target2}')
    ax.legend()
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
            fig.suptitle(f"Scatter Plots of features against {target_var}{part_label}", fontsize=14)
            axes_flat = axes.flatten()

            for i, column in enumerate(chunk_cols):
                ax = axes_flat[i]
                ax.scatter(X_train[column], y_train[target_var], alpha=0.5, s=18)
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
        fig.suptitle(f"Box Plots of Features and Target Variables{part_label}", fontsize=14)
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
        df["Dummy"] = np.random.normal(size=n_samples)

    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All columns in the dataset must be numeric for distance correlation calculation.")

    features = df.columns
    n = len(features)
    dist_corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_corr_matrix[i, j] = dcor.distance_correlation(df[features[i]], df[features[j]])

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
    _title = title if show_annot else f"{title}\n(values not annotated — {n} features)"

    if mp_threshold:
        comp_line = f"MP denoised: {n_signal}/{n} signal components retained"
        lp_line   = f"λ⁺ = {lp:.3f}"
        plt.title(f"{_title}\n{comp_line}\n{lp_line}", fontsize=10, pad=12)
    else:
        plt.title(_title, fontsize=13, pad=8)

    plt.tight_layout()

    return dist_corr_df, fig

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
    figures=None,
):
    if figures is None:
        figures = {}
    print("\nAvailable columns in the dataset:")
    df_preview = pd.read_csv(file_path)
    print(df_preview.columns.tolist())

    # Default to last 2 columns if not specified
    if target_columns is None:
        target_columns = df_preview.columns[-2:].tolist()

    (
        df, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
        target_columns, feature_names
    ) = load_and_preprocess_data(
        file_path, test_size=test_size, split_method=split_method, target_columns=target_columns
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
        "scaler_name": type(scaler).__name__,
    }

    print(f"\nDataset has {n_rows} rows and {n_cols} columns.")
    print(f"Using the following target columns: {target_columns}")

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
        dist_corr_df, fig = plot_distance_correlation_matrix(
            df.drop(columns=target_columns), dummy=dist_corr_dummy, mp_threshold=dist_corr_mp)
        if fig: figures["Distance Correlation"] = fig

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
        "figures": figures,
        "meta": meta,
    }
