"""Smart visualization engine for statistical plots.

Auto-selects appropriate plot types based on data types and analysis context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from astats.data.models import ColumnType, DataProfile, Dataset
from astats.logging import get_logger

matplotlib.use("Agg")
logger = get_logger("viz.engine")

# Professional color palette
ASTATS_PALETTE = [
    "#6366f1",  # Indigo
    "#f43f5e",  # Rose
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#3b82f6",  # Blue
    "#8b5cf6",  # Violet
    "#ec4899",  # Pink
    "#14b8a6",  # Teal
]

# Professional style
PLOT_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#e5e7eb",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#d1d5db",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
}


class VizEngine:
    """Smart visualization engine with professional styling.

    Usage:
        engine = VizEngine(output_dir="./output/plots")
        engine.auto_visualize(dataset)
    """

    def __init__(
        self,
        output_dir: str | Path = "./output/plots",
        dpi: int = 150,
        format: str = "png",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.format = format
        self._counter = 0
        self._apply_style()

    def _apply_style(self) -> None:
        """Apply professional AStats plotting style."""
        plt.rcParams.update(PLOT_STYLE)
        sns.set_palette(ASTATS_PALETTE)

    def _save_plot(self, fig: plt.Figure, name: str) -> str:
        """Save a plot and return the path."""
        self._counter += 1
        filename = f"{self._counter:03d}_{name}.{self.format}"
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"[success]Plot saved[/success]: {path}")
        return str(path)

    def auto_visualize(self, dataset: Dataset) -> list[str]:
        """Automatically generate appropriate visualizations for a dataset.

        Args:
            dataset: Dataset with profile.

        Returns:
            List of saved plot paths.
        """
        if dataset.profile is None:
            logger.warning("Dataset has no profile. Run AutoProfiler first.")
            return []

        plots = []
        profile = dataset.profile
        df = dataset.df

        # 1. Numeric distributions
        numeric_cols = [
            cp.name for cp in profile.column_profiles
            if cp.semantic_type in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE)
        ]
        if numeric_cols:
            plots.append(self.plot_distributions(df, numeric_cols[:8]))

        # 2. Correlation heatmap
        if len(numeric_cols) >= 2 and profile.correlations:
            plots.append(self.plot_correlation_heatmap(
                profile.correlations.matrix,
                title=f"Correlation Matrix — {dataset.name}",
            ))

        # 3. Missing data
        if profile.missing_report.missing_pct > 0:
            plots.append(self.plot_missing_data(df, dataset.name))

        # 4. Categorical distributions
        cat_cols = [
            cp.name for cp in profile.column_profiles
            if cp.semantic_type in (ColumnType.CATEGORICAL, ColumnType.BINARY)
            and cp.unique_count <= 20
        ]
        if cat_cols:
            plots.append(self.plot_categorical_summary(df, cat_cols[:6]))

        # 5. Box plots for outlier visibility
        if numeric_cols:
            plots.append(self.plot_box_comparison(df, numeric_cols[:8]))

        logger.info(f"[success]Auto-generated {len(plots)} visualizations[/success]")
        return plots

    def plot_distributions(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str = "Numeric Distributions",
    ) -> str:
        """Plot distribution histograms for numeric columns."""
        n = len(columns)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(columns):
            ax = axes[i]
            data = df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color=ASTATS_PALETTE[i % len(ASTATS_PALETTE)])
            ax.set_title(col)
            ax.set_xlabel("")

            # Add stats annotation
            mean, std = data.mean(), data.std()
            ax.axvline(mean, color="#ef4444", linestyle="--", alpha=0.7, label=f"μ={mean:.2f}")
            ax.legend(fontsize=9)

        # Hide empty subplots
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return self._save_plot(fig, "distributions")

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
    ) -> str:
        """Plot a correlation heatmap."""
        n = len(corr_matrix)
        size = max(8, n * 0.8)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_plot(fig, "correlation_heatmap")

    def plot_missing_data(
        self,
        df: pd.DataFrame,
        dataset_name: str = "",
    ) -> str:
        """Plot missing data patterns."""
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if len(missing) == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No missing data!", ha="center", va="center", fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return self._save_plot(fig, "missing_data")

        fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
        colors = [ASTATS_PALETTE[0] if v / len(df) < 0.3 else "#ef4444" for v in missing.values]
        bars = ax.barh(missing.index, missing.values, color=colors)

        # Add percentage labels
        for bar, val in zip(bars, missing.values):
            pct = val / len(df) * 100
            ax.text(
                bar.get_width() + len(df) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("Missing Count")
        ax.set_title(f"Missing Data — {dataset_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_plot(fig, "missing_data")

    def plot_categorical_summary(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> str:
        """Plot bar charts for categorical columns."""
        n = len(columns)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle("Categorical Distributions", fontsize=16, fontweight="bold", y=1.02)

        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(columns):
            ax = axes[i]
            vc = df[col].value_counts().head(10)
            vc.plot.bar(ax=ax, color=ASTATS_PALETTE[i % len(ASTATS_PALETTE)], alpha=0.85)
            ax.set_title(col)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return self._save_plot(fig, "categorical_distributions")

    def plot_box_comparison(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> str:
        """Plot box plots for outlier comparison."""
        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.2), 6))

        # Normalize for comparison
        normalized = df[columns].apply(lambda x: (x - x.mean()) / x.std())
        sns.boxplot(data=normalized, ax=ax, palette=ASTATS_PALETTE)
        ax.set_title("Normalized Box Plots (Outlier Comparison)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Standardized Values")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return self._save_plot(fig, "box_comparison")
