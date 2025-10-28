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

def plot_features_vs_targets(X_train, y_train, target_columns):
    """
    For each target, create a grid of scatter plots of every feature vs the target,
    with a simple fitted line (np.polyfit).
    """
    figs = {}

    for target_var in target_columns:
        num_features = X_train.shape[1]
        num_cols = 3  # Fixed number of columns
        num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows dynamically

        # Dynamically adjust figure size based on the number of rows
        fig_width = 15  # Fixed width
        row_height = 4  # Height per row
        fig_height = min(row_height * num_rows, 50)  # Prevent excessive figure size

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f"Scatter Plots of features against {target_var}", fontsize=16)
        axes = axes.flatten()

        for i, column in enumerate(X_train.columns):
            ax = axes[i]
            ax.scatter(X_train[column], y_train[target_var], alpha=0.5)

            # Fit and plot a regression line
            slope, intercept = np.polyfit(X_train[column], y_train[target_var], 1)
            ax.plot(X_train[column], slope * X_train[column] + intercept, color='red')

            ax.set_xlabel(column)
            ax.set_ylabel(target_var)
            ax.set_title(f'{column} vs {target_var}')

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        figs[target_var] = fig  # Store the figure with key = target name

    return figs

def plot_boxplots(df, target_columns):
    """
    Boxplots for all features + targets, with overlayed mean/median/percentiles legends.
    """ 
    # Separate features and targets
    features = df.drop(columns=target_columns)
    targets = df[target_columns]

    # Combine features and targets for plotting
    combined_df = pd.concat([features, targets], axis=1)

    # Get column names for the combined data
    all_columns = combined_df.columns.tolist()
    num_columns = len(all_columns)

    # Dynamic layout
    num_cols = 3  # Number of columns for subplots
    num_rows = (num_columns + num_cols - 1) // num_cols  # Dynamically calculate rows
    max_height_per_row = 5  # Maximum height for each row of subplots

    # Adjust figure size dynamically
    fig_height = min(max_height_per_row * num_rows, 50)  # Limit overall height
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, fig_height), sharey=True)
    fig.suptitle("Box Plots of Features and Target Variables", fontsize=16)
    axes = axes.flatten()

    for i, column in enumerate(all_columns):
        ax = axes[i]
        data = combined_df[column]

        # Plot the box plot
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        flierprops=dict(marker='o', color='red', alpha=0.5))

        # Add statistical information
        median = data.median()
        mean = data.mean()
        q1, q3 = data.quantile([0.25, 0.75])
        p5, p95 = data.quantile([0.05, 0.95])
        min_val, max_val = data.min(), data.max()

        # Draw vertical lines for the mean, median, and percentiles
        ax.axvline(mean, color='red', linestyle='-', linewidth=1.5)
        ax.axvline(median, color='orange', linestyle='-', linewidth=1.5)
        ax.axvline(p5, color='green', linestyle='--', linewidth=1)
        ax.axvline(p95, color='green', linestyle='--', linewidth=1)

        ax.set_title(column, fontsize=12)
        ax.set_xlabel("Value")
        ax.set_yticks([])

        # Custom legend handles (using matplotlib.lines and patches)
        legend_handles = [
            mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label=f"Mean: {mean:.2f}"),
            mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, label=f"Median: {median:.2f}"),
            mpatches.Patch(facecolor='lightblue', edgecolor='black', label=f"IQR: {q1:.2f} to {q3:.2f}"),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label=f"5th/95th: {p5:.2f}, {p95:.2f}"),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=1, label=f"Min/Max: {min_val:.2f}, {max_val:.2f}")
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, title="Statistics", title_fontsize='9')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_distance_correlation_matrix(df, title="Distance Correlation Matrix", cmap='RdYlGn', dummy=False, annotate=True):
    """
    Compute & plot a distance-correlation heatmap for numeric columns.

    Args:
        df (pd.DataFrame): Numeric columns only.
        title (str): Figure title.
        cmap (str): Matplotlib colormap.
        dummy (bool): If True, append a random dummy column to illustrate.
        annotate (bool): Show numeric values on the heatmap.

    Returns:
        (pd.DataFrame, matplotlib.figure.Figure): distance corr matrix and the figure.
    """
    if dummy:
        df = df.copy()
        df["Dummy"] = np.random.normal(size=len(df))

    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All columns in the dataset must be numeric for distance correlation calculation.")

    features = df.columns
    n = len(features)
    dist_corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_corr_matrix[i, j] = dcor.distance_correlation(df[features[i]], df[features[j]])

    dist_corr_df = pd.DataFrame(dist_corr_matrix, index=features, columns=features)

    fig_width = max(12, n * 0.5)
    fig_height = max(10, n * 0.5)

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()  # Get the current axes
    sns.heatmap(dist_corr_df, annot=annotate, cmap=cmap, square=True, linewidths=0.5, fmt=".4f",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(title, fontsize=14)
    
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
    figures = {}
):
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
        figures.update({f"Features vs {target}": fig for target, fig in feature_vs_target_figs.items()})

    if plot_boxplots_enabled:
        print("\nGenerating Box plots...")
        fig = plot_boxplots(df, target_columns)
        if fig: figures["Boxplots"] = fig

    dist_corr_df = None
    if plot_distance_corr_enabled:
        print("\nGenerating Distance Correlation Matrix...")
        dist_corr_df, fig = plot_distance_correlation_matrix(df.drop(columns=target_columns), dummy=True)
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
