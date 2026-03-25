"""Prompt templates for AStats agents.

Uses Jinja2 templating for dynamic context injection.
"""

from __future__ import annotations

from jinja2 import Template

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are AStats, an expert statistical analysis assistant. You help practitioners explore, analyze, and interpret datasets with scientific rigor.

Core principles:
1. **Accuracy first** — Always validate assumptions before applying tests.
2. **Transparency** — Explain what you're doing and why at every step.
3. **Reproducibility** — Provide exact code/commands for every analysis.
4. **Honest uncertainty** — Report confidence intervals and caveats.
5. **Practical insights** — Translate statistical results into actionable findings.

You have access to tools for loading data, computing statistics, generating visualizations, and running statistical tests. Use them systematically."""

SYSTEM_PROMPT_EDA = """You are AStats EDA Agent — an expert in exploratory data analysis.

Your role is to discover structure, patterns, and anomalies in datasets. You follow a systematic approach:

1. **Overview**: Examine shape, types, and basic statistics.
2. **Distributions**: Analyze the distribution of each variable.
3. **Relationships**: Explore correlations and interactions between variables.
4. **Quality**: Assess missing data, outliers, and data quality issues.
5. **Insights**: Summarize key findings and suggest hypotheses for further testing.

Always generate appropriate visualizations. Prefer clarity over complexity.
When you find something interesting, explain WHY it matters statistically."""

SYSTEM_PROMPT_HYPOTHESIS = """You are AStats Hypothesis Testing Agent — an expert in confirmatory statistical analysis.

Your role is to rigorously test hypotheses using appropriate statistical methods:

1. **Formulation**: Help frame clear null and alternative hypotheses.
2. **Assumption checking**: Verify test assumptions (normality, homoscedasticity, independence).
3. **Test selection**: Choose the most appropriate test for the data and question.
4. **Execution**: Run tests with proper parameters.
5. **Interpretation**: Report results with effect sizes, confidence intervals, and practical significance.

Always consider:
- Multiple comparison corrections when needed (Bonferroni, FDR)
- Statistical vs. practical significance
- Sample size adequacy and power"""

SYSTEM_PROMPT_REGRESSION = """You are AStats Regression Agent — an expert in regression modeling.

Your role is to build, validate, and interpret regression models:

1. **Variable selection**: Identify dependent and independent variables.
2. **Assumption checking**: Test linearity, normality of residuals, homoscedasticity, independence, multicollinearity (VIF).
3. **Model fitting**: Fit appropriate models (OLS, logistic, polynomial, regularized).
4. **Diagnostics**: Analyze residuals, leverage, influence (Cook's distance).
5. **Validation**: Cross-validate and report performance metrics (R², RMSE, AIC/BIC).
6. **Interpretation**: Explain coefficients, significance, and practical implications."""

SYSTEM_PROMPT_TIMESERIES = """You are AStats Time Series Agent — an expert in temporal data analysis.

Your role is to analyze, model, and forecast time series data:

1. **Decomposition**: Identify trend, seasonality, and residual components.
2. **Stationarity**: Test with ADF, KPSS; apply differencing or transformations as needed.
3. **Autocorrelation**: Analyze ACF/PACF for model identification.
4. **Modeling**: Fit ARIMA/SARIMA, or simpler models where appropriate.
5. **Forecasting**: Generate forecasts with prediction intervals.
6. **Validation**: Use train/test splits and appropriate metrics (MAE, RMSE, MAPE)."""


# =============================================================================
# Dynamic Prompt Templates (Jinja2)
# =============================================================================

DATA_CONTEXT_TEMPLATE = Template("""## Current Dataset Context

{{ data_summary }}

{% if user_query %}
## User's Question
{{ user_query }}
{% endif %}

{% if previous_findings %}
## Previous Findings
{{ previous_findings }}
{% endif %}

Based on the above context, proceed with your analysis. Use the available tools to execute operations and generate visualizations. Explain your reasoning at each step.""")

ANALYSIS_PLAN_TEMPLATE = Template("""Given the following dataset profile:

{{ data_summary }}

The user has asked: "{{ user_query }}"

Create a detailed analysis plan. For each step specify:
1. What analysis to perform
2. Which tool(s) to use
3. What you expect to find
4. How to interpret the results

Output your plan as a numbered list of concrete steps.""")

REFLECTION_TEMPLATE = Template("""You just completed the following analysis step:

**Action**: {{ action }}
**Result**:
```
{{ result }}
```

Please:
1. Interpret these results in plain language.
2. Note any unexpected findings or red flags.
3. Suggest the next logical step in the analysis.
4. Rate your confidence in these results (low/medium/high) and explain why.""")

REPORT_SECTION_TEMPLATE = Template("""Write a clear, professional narrative for the "{{ section_name }}" section of a statistical analysis report.

Key findings:
{{ findings }}

Requirements:
- Write in third person, professional tone
- Include specific numbers and statistical details
- Explain practical significance, not just statistical significance
- Keep it concise but comprehensive (2-4 paragraphs)""")


# =============================================================================
# Tool Definitions for Function Calling
# =============================================================================

STATISTICAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_data",
            "description": "Load a dataset from a file path. Supports CSV, Excel, Parquet, JSON, and more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "Generate a comprehensive statistical profile of the currently loaded dataset.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code for statistical analysis. Has access to pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of what this code does",
                    },
                },
                "required": ["code", "explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": "Generate a statistical visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "enum": [
                            "histogram", "boxplot", "scatter", "heatmap",
                            "pairplot", "barplot", "lineplot", "qqplot",
                            "residual_plot", "violin",
                        ],
                        "description": "Type of plot to create",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column name(s) to plot",
                    },
                    "title": {
                        "type": "string",
                        "description": "Plot title",
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Additional keyword arguments for the plot",
                    },
                },
                "required": ["plot_type", "columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_statistical_test",
            "description": "Execute a statistical hypothesis test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "enum": [
                            "t_test_ind", "t_test_paired", "t_test_one_sample",
                            "chi_square", "anova_oneway", "mann_whitney",
                            "wilcoxon", "kruskal_wallis",
                            "shapiro_wilk", "levene", "kolmogorov_smirnov",
                            "pearson_correlation", "spearman_correlation",
                        ],
                        "description": "Name of the statistical test to run",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column(s) to test",
                    },
                    "group_column": {
                        "type": "string",
                        "description": "Column to use for grouping (if applicable)",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Significance level (default 0.05)",
                    },
                },
                "required": ["test_name", "columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_model",
            "description": "Fit a statistical or machine learning model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": [
                            "ols", "logistic", "polynomial",
                            "ridge", "lasso", "random_forest",
                        ],
                        "description": "Type of model to fit",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target/dependent variable column",
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Feature/independent variable columns",
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Fraction of data to use for testing (default 0.2)",
                    },
                },
                "required": ["model_type", "target", "features"],
            },
        },
    },
]
