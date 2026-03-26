#!/usr/bin/env python3
"""AStats Example: Hypothesis Testing on the Titanic Dataset.

This script demonstrates the hypothesis testing pipeline using AStats on the
Titanic survival dataset (100 samples, mixed types, binary outcome).

Statistical Methodology:
    1. Exploratory profiling: Missing data analysis, class distributions
    2. Chi-square test: Association between sex and survival (categorical × categorical)
    3. Independent t-test: Age differences between survivors vs non-survivors
    4. Mann-Whitney U test: Non-parametric fare comparison by survival
    5. ANOVA: Fare differences across passenger classes
    6. Effect size: Cohen's d and Cramér's V for practical significance

Usage:
    # CLI mode:
    astats analyze examples/data/titanic.csv -q "Test which factors significantly predict survival"

    # Python API:
    python examples/hypothesis_titanic_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from astats.data.ingestion import DataLoader
from astats.data.profiler import AutoProfiler
from astats.viz.engine import VizEngine


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.std()**2 + (n2 - 1) * group2.std()**2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0.0


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calculate Cramér's V for effect size of chi-square test."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0


def main() -> None:
    """Run a hypothesis testing pipeline on Titanic data."""
    print("=" * 60)
    print("  AStats Example: Hypothesis Testing on Titanic Dataset")
    print("=" * 60)

    alpha = 0.05

    # ---- Step 1: Load & Profile ----
    dataset = DataLoader().load("examples/data/titanic.csv")
    AutoProfiler().profile(dataset)
    df = dataset.df
    print(f"\n✅ Loaded: {dataset.shape[0]} rows × {dataset.shape[1]} cols")
    print(f"   Survival rate: {df['survived'].mean():.1%}")

    # Visualizations
    viz = VizEngine(output_dir="output/examples/titanic")
    viz.auto_visualize(dataset)
    print(f"📈 Visualizations → output/examples/titanic/\n")

    # ---- Test 1: Chi-Square — Sex vs Survival ----
    print("─" * 50)
    print("  Test 1: Chi-Square — Sex × Survival")
    print("─" * 50)
    ct = pd.crosstab(df["sex"], df["survived"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    v = cramers_v(ct)
    print(f"  χ²({dof}) = {chi2:.4f}, p = {p:.6f}")
    print(f"  Cramér's V = {v:.4f}")
    print(f"  → {'Significant' if p < alpha else 'Not significant'}: Sex {'is' if p < alpha else 'is not'} associated with survival")

    # ---- Test 2: Independent t-test — Age by Survival ----
    print(f"\n{'─' * 50}")
    print("  Test 2: Independent t-test — Age by Survival")
    print("─" * 50)
    survived = df[df["survived"] == 1]["age"].dropna()
    died = df[df["survived"] == 0]["age"].dropna()
    t_stat, p_val = stats.ttest_ind(survived, died)
    d = cohens_d(survived, died)
    print(f"  Survivors: μ={survived.mean():.1f} (n={len(survived)})")
    print(f"  Non-surv:  μ={died.mean():.1f} (n={len(died)})")
    print(f"  t = {t_stat:.4f}, p = {p_val:.4f}, Cohen's d = {d:.4f}")
    print(f"  → {'Significant' if p_val < alpha else 'Not significant'} age difference between groups")

    # ---- Test 3: Mann-Whitney U — Fare by Survival ----
    print(f"\n{'─' * 50}")
    print("  Test 3: Mann-Whitney U — Fare by Survival")
    print("─" * 50)
    fare_surv = df[df["survived"] == 1]["fare"].dropna()
    fare_died = df[df["survived"] == 0]["fare"].dropna()
    u_stat, p_mw = stats.mannwhitneyu(fare_surv, fare_died, alternative="two-sided")
    print(f"  Survivors: median fare = {fare_surv.median():.2f}")
    print(f"  Non-surv:  median fare = {fare_died.median():.2f}")
    print(f"  U = {u_stat:.1f}, p = {p_mw:.6f}")
    print(f"  → {'Significant' if p_mw < alpha else 'Not significant'} fare difference")

    # ---- Test 4: One-Way ANOVA — Fare across Classes ----
    print(f"\n{'─' * 50}")
    print("  Test 4: One-Way ANOVA — Fare by Passenger Class")
    print("─" * 50)
    groups = [g["fare"].dropna().values for _, g in df.groupby("pclass")]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"  Class 1 median fare: {df[df['pclass']==1]['fare'].median():.2f}")
    print(f"  Class 2 median fare: {df[df['pclass']==2]['fare'].median():.2f}")
    print(f"  Class 3 median fare: {df[df['pclass']==3]['fare'].median():.2f}")
    print(f"  F = {f_stat:.4f}, p = {p_anova:.6f}")
    print(f"  → {'Significant' if p_anova < alpha else 'Not significant'} fare differences across classes")

    # ---- Summary ----
    print(f"\n{'=' * 50}")
    print("  Summary of Hypothesis Tests (α=0.05)")
    print("=" * 50)
    results = [
        ("Sex × Survival (χ²)", p, p < alpha),
        ("Age × Survival (t-test)", p_val, p_val < alpha),
        ("Fare × Survival (Mann-Whitney)", p_mw, p_mw < alpha),
        ("Fare × Class (ANOVA)", p_anova, p_anova < alpha),
    ]
    for name, pv, sig in results:
        icon = "✅" if sig else "❌"
        print(f"  {icon} {name}: p={pv:.4f} {'(Significant)' if sig else '(Not Significant)'}")

    print("\n✨ Hypothesis testing complete!")


if __name__ == "__main__":
    main()
