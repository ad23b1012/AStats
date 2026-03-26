# AStats Statistical Methodology

This document outlines the rigorous mathematical and statistical workflows enforced by the AStats agentic framework. By enforcing these workflows through a restricted `ToolRegistry` and systematic `BaseAgent` loop, we prevent LLM "hallucinations" and ensure analyses meet the standards of applied statistical practitioners.

## 1. Exploratory Data Analysis (EDA) Workflow

Before any hypothesis testing or modeling occurs, the `EDAAgent` executes a structured exploration phase.

### Data Profiling
- **Type Inference:** Variables are rigorously classified (continuous, discrete, categorical, binary, datetime).
- **Univariate Statistics:** Mean, median, standard deviation, variance, skewness, and kurtosis are calculated for continuous variables.
- **Missing Data:** Pattern detection (MCAR vs MAR vs MNAR proxy detection) using missingness correlation matrices.

### Distribution & Outlier Detection
- **Normality Screening:** Preliminary Shapiro-Wilk or D'Agostino's K-squared tests on continuous distributions.
- **Outliers:** Identified using the standard Interquartile Range (IQR) method (values outside $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$) or Z-scores ($|Z| > 3$) for normally distributed data.

### Bivariate Associations
- **Numeric-Numeric:** Pearson correlation ($r$) for linear relationships; Spearman rank correlation ($\rho$) for monotonic non-linear relationships.
- **Categorical-Categorical:** Contingency tables and Cramér's V for effect size.
- **Numeric-Categorical:** Grouped means, standard deviations, and visualizations (box/violin plots).

---

## 2. Hypothesis Testing Framework

The `HypothesisAgent` dynamically selects the appropriate statistical test based on the data profile (e.g., sample size, normality, variance homogeneity).

### Parametric Tests (Default if assumptions met)
- **Independent t-test:** Used to compare means of two independent groups (assumes normality and equal variance; Welch's t-test used if Levene's test for equality of variances fails).
- **One-Way ANOVA:** Used to compare means across three or more groups. Followed by Tukey's HSD post-hoc test if significant.
- **Pearson Correlation Test:** To test the significance of a linear relationship between two continuous variables.

### Non-Parametric Alternatives (Fallback if assumptions violated)
- **Mann-Whitney U Test:** Non-parametric alternative to the independent t-test (comparing medians/rank-sums).
- **Kruskal-Wallis H Test:** Non-parametric alternative to One-Way ANOVA.
- **Chi-Square Test of Independence:** Used for categorical association.
- **Spearman Rank Correlation:** Used for ordinal data or non-linear continuous relationships.

### Effect Sizes
Statistical significance ($p < \alpha$) is always paired with practical significance (effect sizes):
- **Cohen's $d$:** For t-tests.
- **Eta-squared ($\eta^2$):** For ANOVA.
- **Cramér's $V$:** For Chi-Square.

---

## 3. Regression Modeling Pipeline

The `RegressionAgent` follows a strict progression from Ordinary Least Squares (OLS) to regularized/robust models depending on diagnostic checks.

### OLS Assumptions & Diagnostics
- **Linearity:** Checked via residual vs. fitted value plots.
- **Independence:** Durbin-Watson statistic (target range 1.5 - 2.5).
- **Homoscedasticity:** Breusch-Pagan test. If violated, Heteroskedasticity-Consistent (HC) standard errors are used.
- **Normality of Residuals:** Shapiro-Wilk test or Q-Q plots.
- **Multicollinearity:** Variance Inflation Factor (VIF). Features with VIF > 5-10 are flagged for removal or regularization.

### Regularization (Ridge / Lasso)
If multicollinearity is high, the agent shifts from `statsmodels.OLS` to `sklearn.linear_model.Ridge` or `Lasso` to apply L2/L1 penalties, improving out-of-sample generalization.

---

## 4. Design Philosophy: Reliability over Adaptability

AStats priorities **Decision Reliability** above all. 

Unlike open-ended LLM wrappers (like ChatGPT Advanced Data Analysis), AStats agents do not simply write untested `scikit-learn` scripts. They run code within an abstract execution layer (`tools.py`) that returns raw stack traces to the agent. If an OLS regression fails due to a singular matrix (perfect collinearity), the agent catches the `LinAlgError`, reflects on the VIF profile, drops the collinear column, and re-executes the procedure. 

This creates a highly robust pipeline that mirrors the actual daily workflow of an applied statistician.
