"""EDA plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_missingness(df: pd.DataFrame, output_path: Path) -> None:
    """Plot missing value percentages by column."""
    missing = df.isna().mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.xlabel("Missing Percentage")
    plt.ylabel("Feature")
    plt.title("Missing Values by Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    """Plot the target class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, output_path: Path, max_columns: int = 9) -> None:
    """Plot histograms for the first numeric columns."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()[:max_columns]
    if not numeric_columns:
        return

    axes = df[numeric_columns].hist(figsize=(14, 10), bins=25)
    for ax in axes.flatten():
        ax.set_ylabel("Count")
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot a heatmap of numeric feature correlations."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return

    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
