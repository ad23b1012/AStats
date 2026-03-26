#!/usr/bin/env python3
"""AStats Example: Regression Analysis on the Diabetes Dataset.

This script demonstrates a regression workflow using AStats on the sklearn
Diabetes dataset (442 samples, 10 baseline features, 1 continuous target).

Statistical Methodology:
    1. Data profiling: Feature distributions, multicollinearity screening
    2. OLS Regression: Ordinary Least Squares with assumption diagnostics
    3. Model diagnostics: Residual analysis, VIF for multicollinearity
    4. Regularization comparison: Ridge vs Lasso for feature selection
    5. Feature importance: Random Forest permutation importance

Usage:
    # CLI mode:
    astats analyze examples/data/diabetes.csv -q "Build a regression model to predict disease progression"

    # Python API:
    python examples/regression_diabetes_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from astats.data.ingestion import DataLoader
from astats.data.profiler import AutoProfiler
from astats.viz.engine import VizEngine


def main() -> None:
    """Run a complete regression analysis pipeline."""
    print("=" * 60)
    print("  AStats Example: Regression on Diabetes Dataset")
    print("=" * 60)

    # ---- Step 1: Load & Profile ----
    dataset = DataLoader().load("examples/data/diabetes.csv")
    AutoProfiler().profile(dataset)
    print(f"\n✅ Loaded: {dataset.shape[0]} rows × {dataset.shape[1]} cols")
    print(f"   Target: 'target' (disease progression after 1 year)")

    df = dataset.df
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    # ---- Step 2: Visualizations ----
    viz = VizEngine(output_dir="output/examples/diabetes")
    viz.auto_visualize(dataset)
    print(f"\n📈 Visualizations saved → output/examples/diabetes/")

    # ---- Step 3: Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Split: {len(X_train)} train / {len(X_test)} test samples")

    # ---- Step 4: OLS Regression ----
    print("\n" + "─" * 40)
    print("  OLS Regression Results")
    print("─" * 40)

    import statsmodels.api as sm
    X_train_c = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_c).fit()

    print(f"  R² (train):  {ols_model.rsquared:.4f}")
    print(f"  Adj R²:      {ols_model.rsquared_adj:.4f}")
    print(f"  F-statistic: {ols_model.fvalue:.2f} (p={ols_model.f_pvalue:.2e})")

    # Significant predictors
    sig_vars = ols_model.pvalues[ols_model.pvalues < 0.05].index.tolist()
    print(f"  Significant predictors (p<0.05): {sig_vars}")

    # VIF for multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame({
        "Feature": feature_cols,
        "VIF": [variance_inflation_factor(X_train.values, i) for i in range(len(feature_cols))],
    })
    high_vif = vif_data[vif_data["VIF"] > 5]
    if len(high_vif) > 0:
        print(f"\n  ⚠️ High VIF (>5) — potential multicollinearity:")
        for _, row in high_vif.iterrows():
            print(f"    • {row['Feature']}: VIF={row['VIF']:.1f}")

    # ---- Step 5: Regularization Comparison ----
    print("\n" + "─" * 40)
    print("  Regularized Models Comparison")
    print("─" * 40)

    for name, model in [("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=1.0))]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
        print(f"  {name:6s}: R²={r2:.4f}, RMSE={rmse:.1f}, Non-zero features={n_nonzero}/{len(feature_cols)}")

    # ---- Step 6: Normality of Residuals ----
    X_test_c = sm.add_constant(X_test)
    residuals = y_test - ols_model.predict(X_test_c)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\n  Shapiro-Wilk (residuals): W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
    print(f"  Residuals normal: {'Yes' if shapiro_p > 0.05 else 'No (violates OLS assumption)'}")

    print("\n✨ Regression analysis complete!")


if __name__ == "__main__":
    main()
