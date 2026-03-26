#!/usr/bin/env python3
"""AStats Example: Exploratory Data Analysis on the Iris Dataset.

This script demonstrates the complete EDA pipeline using AStats on Fisher's
classic Iris dataset (150 samples, 4 numeric features, 1 categorical target).

Statistical Methodology:
    1. Auto-profiling: Column type inference, univariate statistics, missing data
    2. Distribution analysis: Histograms with KDE, box plots per species
    3. Correlation analysis: Pearson correlation matrix across numeric features
    4. Outlier detection: IQR-based flagging
    5. Group comparisons: Distribution differences across species

Usage:
    # CLI mode (recommended):
    astats analyze examples/data/iris.csv -q "Compare petal measurements across species"

    # Python API:
    python examples/eda_iris_example.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astats.data.ingestion import DataLoader
from astats.data.profiler import AutoProfiler
from astats.viz.engine import VizEngine


def main() -> None:
    """Run a full EDA pipeline on the Iris dataset."""
    print("=" * 60)
    print("  AStats Example: EDA on Fisher's Iris Dataset")
    print("=" * 60)

    # ---- Step 1: Load Data ----
    loader = DataLoader()
    dataset = loader.load("examples/data/iris.csv")
    print(f"\n✅ Loaded: {dataset.name} — {dataset.shape[0]} rows × {dataset.shape[1]} cols")

    # ---- Step 2: Auto-Profile ----
    profiler = AutoProfiler()
    profiler.profile(dataset)
    print("\n📊 Statistical Profile:")
    print(dataset.profile.summary_for_llm())

    # ---- Step 3: Visualize ----
    viz = VizEngine(output_dir="output/examples/iris")
    plots = viz.auto_visualize(dataset)
    print(f"\n📈 Generated {len(plots)} visualizations → output/examples/iris/")

    # ---- Step 4: Key Insights (programmatic) ----
    df = dataset.df
    print("\n🔍 Quick Insights:")
    print(f"  • Species counts: {df['species'].value_counts().to_dict()}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols].corr()
    # Find highest correlation pair (excluding self-correlations)
    import numpy as np
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    max_corr = corr.where(mask).stack().abs().idxmax()
    max_val = corr.loc[max_corr[0], max_corr[1]]
    print(f"  • Strongest correlation: {max_corr[0]} ↔ {max_corr[1]} (r={max_val:.3f})")

    # Group means
    group_means = df.groupby("species")[numeric_cols].mean()
    print(f"\n📋 Mean Feature Values by Species:")
    print(group_means.round(2).to_string())

    print("\n✨ EDA complete! Check output/examples/iris/ for plots.")


if __name__ == "__main__":
    main()
