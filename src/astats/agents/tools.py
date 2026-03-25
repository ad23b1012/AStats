"""Tool registry for agentic statistical operations.

Each tool is a callable that the LLM can invoke through function calling.
Tools are registered with name, description, parameter schema, and execute function.
"""

from __future__ import annotations

import io
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from astats.data.models import Dataset
from astats.logging import get_logger

matplotlib.use("Agg")  # Non-interactive backend

logger = get_logger("agents.tools")


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: str
    data: Any = None
    plots: list[str] = field(default_factory=list)  # Paths to generated plots


@dataclass
class Tool:
    """A registered tool that agents can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]
    execute_fn: Callable[..., ToolResult]


class ToolRegistry:
    """Registry of tools available to agents.

    Tools are callable operations for data analysis, visualization,
    statistical testing, and model fitting.
    """

    def __init__(self, output_dir: str = "./output") -> None:
        self._tools: dict[str, Tool] = {}
        self._dataset: Dataset | None = None
        self._output_dir = output_dir
        self._plot_counter = 0

        # Register built-in tools
        self._register_builtins()

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    def set_dataset(self, dataset: Dataset) -> None:
        """Set the active dataset for tools to operate on."""
        self._dataset = dataset
        logger.info(f"[data]Active dataset[/data]: {dataset.name} ({dataset.shape})")

    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools in function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}",
            )

        try:
            logger.info(f"[agent]Executing tool[/agent]: {tool_name}")
            result = tool.execute_fn(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolResult(
                success=False,
                output=f"Tool execution failed: {type(e).__name__}: {str(e)}",
            )

    # ---- Built-in Tools ----

    def _register_builtins(self) -> None:
        """Register all built-in statistical tools."""
        self.register(Tool(
            name="describe_data",
            description="Generate summary statistics for the dataset or specific columns",
            parameters={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific columns to describe (default: all)",
                    },
                },
            },
            execute_fn=self._describe_data,
        ))

        self.register(Tool(
            name="run_code",
            description="Execute Python code for analysis. Has pandas, numpy, scipy, statsmodels, sklearn available. Use 'df' to access the dataset.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "explanation": {"type": "string", "description": "What this code does"},
                },
                "required": ["code"],
            },
            execute_fn=self._run_code,
        ))

        self.register(Tool(
            name="create_plot",
            description="Generate a statistical visualization",
            parameters={
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "enum": ["histogram", "boxplot", "scatter", "heatmap", "pairplot",
                                 "barplot", "lineplot", "qqplot", "violin", "countplot"],
                    },
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": "string"},
                    "hue": {"type": "string", "description": "Column for color grouping"},
                },
                "required": ["plot_type", "columns"],
            },
            execute_fn=self._create_plot,
        ))

        self.register(Tool(
            name="run_statistical_test",
            description="Run a statistical hypothesis test",
            parameters={
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "enum": ["t_test_ind", "t_test_paired", "t_test_one_sample",
                                 "chi_square", "anova_oneway", "mann_whitney",
                                 "wilcoxon", "kruskal_wallis", "shapiro_wilk",
                                 "levene", "pearson_correlation", "spearman_correlation"],
                    },
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "group_column": {"type": "string"},
                    "alpha": {"type": "number", "default": 0.05},
                },
                "required": ["test_name", "columns"],
            },
            execute_fn=self._run_test,
        ))

        self.register(Tool(
            name="fit_model",
            description="Fit a regression or classification model",
            parameters={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["ols", "logistic", "ridge", "lasso", "random_forest"],
                    },
                    "target": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "test_size": {"type": "number", "default": 0.2},
                },
                "required": ["model_type", "target", "features"],
            },
            execute_fn=self._fit_model,
        ))

    def _require_dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            raise ValueError("No dataset loaded. Load a dataset first.")
        return self._dataset.df

    def _describe_data(self, columns: list[str] | None = None) -> ToolResult:
        """Describe the dataset or specific columns."""
        df = self._require_dataset()
        if columns:
            df = df[columns]

        desc = df.describe(include="all").to_string()
        dtypes = df.dtypes.to_string()
        missing = df.isna().sum()
        missing_str = missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values"

        output = f"Shape: {df.shape}\n\nData Types:\n{dtypes}\n\nDescriptive Statistics:\n{desc}\n\nMissing Values:\n{missing_str}"
        return ToolResult(success=True, output=output)

    def _run_code(self, code: str, explanation: str = "") -> ToolResult:
        """Execute Python code in a sandboxed namespace."""
        df = self._require_dataset()

        # Set up execution namespace
        namespace: dict[str, Any] = {
            "df": df,
            "pd": pd,
            "np": np,
            "stats": stats,
            "plt": plt,
            "sns": sns,
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            exec(code, namespace)  # noqa: S102
            output = buffer.getvalue()

            # Check if there's a result variable
            if "result" in namespace and namespace["result"] is not None:
                result_val = namespace["result"]
                if isinstance(result_val, pd.DataFrame):
                    output += "\n" + result_val.to_string()
                else:
                    output += "\n" + str(result_val)

            if not output.strip():
                output = "Code executed successfully (no output)"

            return ToolResult(success=True, output=output)
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                output=f"Code execution error: {type(e).__name__}: {e}\n{tb}",
            )
        finally:
            sys.stdout = old_stdout

    def _create_plot(
        self,
        plot_type: str,
        columns: list[str],
        title: str | None = None,
        hue: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Generate a visualization."""
        df = self._require_dataset()
        self._plot_counter += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_title = title or f"{plot_type.title()} of {', '.join(columns)}"

        try:
            if plot_type == "histogram":
                for col in columns:
                    sns.histplot(data=df, x=col, kde=True, ax=ax, label=col)
                if len(columns) > 1:
                    ax.legend()
            elif plot_type == "boxplot":
                if hue and len(columns) == 1:
                    sns.boxplot(data=df, x=hue, y=columns[0], ax=ax)
                else:
                    df[columns].boxplot(ax=ax)
            elif plot_type == "scatter":
                if len(columns) >= 2:
                    sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=hue, ax=ax)
            elif plot_type == "heatmap":
                corr = df[columns].corr() if columns else df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
            elif plot_type == "pairplot":
                plt.close(fig)
                g = sns.pairplot(df[columns], hue=hue)
                fig = g.figure
                ax = None
            elif plot_type == "barplot":
                if len(columns) >= 2:
                    sns.barplot(data=df, x=columns[0], y=columns[1], hue=hue, ax=ax)
                else:
                    df[columns[0]].value_counts().head(20).plot.bar(ax=ax)
            elif plot_type == "lineplot":
                for col in columns:
                    sns.lineplot(data=df, y=col, ax=ax, label=col)
                if len(columns) > 1:
                    ax.legend()
            elif plot_type == "violin":
                if hue and len(columns) == 1:
                    sns.violinplot(data=df, x=hue, y=columns[0], ax=ax)
                else:
                    df[columns].plot.box(ax=ax)
            elif plot_type == "qqplot":
                from statsmodels.graphics.gofplots import qqplot
                qqplot(df[columns[0]].dropna(), line="45", ax=ax)
            elif plot_type == "countplot":
                sns.countplot(data=df, x=columns[0], hue=hue, ax=ax)

            if ax is not None:
                ax.set_title(plot_title, fontsize=14, fontweight="bold")
                plt.tight_layout()

            # Save plot
            from pathlib import Path
            out_dir = Path(self._output_dir) / "plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_path = str(out_dir / f"plot_{self._plot_counter:03d}_{plot_type}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ToolResult(
                success=True,
                output=f"Plot saved: {plot_path}",
                plots=[plot_path],
            )
        except Exception as e:
            plt.close(fig)
            return ToolResult(success=False, output=f"Plot error: {e}")

    def _run_test(
        self,
        test_name: str,
        columns: list[str],
        group_column: str | None = None,
        alpha: float = 0.05,
    ) -> ToolResult:
        """Run a statistical test."""
        df = self._require_dataset()

        try:
            result_lines = [f"Statistical Test: {test_name}", f"Alpha: {alpha}", ""]

            if test_name == "t_test_ind":
                if group_column and len(columns) == 1:
                    groups = df[group_column].unique()
                    if len(groups) != 2:
                        return ToolResult(False, f"Need exactly 2 groups, got {len(groups)}")
                    g1 = df[df[group_column] == groups[0]][columns[0]].dropna()
                    g2 = df[df[group_column] == groups[1]][columns[0]].dropna()
                elif len(columns) == 2:
                    g1 = df[columns[0]].dropna()
                    g2 = df[columns[1]].dropna()
                else:
                    return ToolResult(False, "Need 2 columns or 1 column + group_column")
                stat, pval = stats.ttest_ind(g1, g2)
                effect_d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
                result_lines.extend([
                    f"t-statistic: {stat:.4f}",
                    f"p-value: {pval:.6f}",
                    f"Cohen's d: {effect_d:.4f}",
                    f"Significant: {'Yes' if pval < alpha else 'No'}",
                ])

            elif test_name == "shapiro_wilk":
                data = df[columns[0]].dropna()
                sample = data.sample(min(5000, len(data)))  # Shapiro limit
                stat, pval = stats.shapiro(sample)
                result_lines.extend([
                    f"W-statistic: {stat:.4f}",
                    f"p-value: {pval:.6f}",
                    f"Normal: {'Yes (fail to reject H0)' if pval >= alpha else 'No (reject H0)'}",
                ])

            elif test_name == "anova_oneway":
                if not group_column:
                    return ToolResult(False, "ANOVA requires group_column")
                groups = [g[columns[0]].dropna().values for _, g in df.groupby(group_column)]
                stat, pval = stats.f_oneway(*groups)
                result_lines.extend([
                    f"F-statistic: {stat:.4f}",
                    f"p-value: {pval:.6f}",
                    f"Significant: {'Yes' if pval < alpha else 'No'}",
                ])

            elif test_name == "chi_square":
                if len(columns) == 2:
                    ct = pd.crosstab(df[columns[0]], df[columns[1]])
                    stat, pval, dof, expected = stats.chi2_contingency(ct)
                    result_lines.extend([
                        f"Chi-square statistic: {stat:.4f}",
                        f"p-value: {pval:.6f}",
                        f"Degrees of freedom: {dof}",
                        f"Significant: {'Yes' if pval < alpha else 'No'}",
                    ])

            elif test_name == "mann_whitney":
                g1 = df[columns[0]].dropna()
                g2 = df[columns[1]].dropna() if len(columns) > 1 else None
                if g2 is None and group_column:
                    groups = df[group_column].unique()
                    g1 = df[df[group_column] == groups[0]][columns[0]].dropna()
                    g2 = df[df[group_column] == groups[1]][columns[0]].dropna()
                if g2 is None:
                    return ToolResult(False, "Need 2 groups for Mann-Whitney")
                stat, pval = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                result_lines.extend([
                    f"U-statistic: {stat:.4f}",
                    f"p-value: {pval:.6f}",
                    f"Significant: {'Yes' if pval < alpha else 'No'}",
                ])

            elif test_name == "pearson_correlation":
                if len(columns) < 2:
                    return ToolResult(False, "Need 2 columns for correlation")
                stat, pval = stats.pearsonr(df[columns[0]].dropna(), df[columns[1]].dropna())
                result_lines.extend([
                    f"Pearson r: {stat:.4f}",
                    f"p-value: {pval:.6f}",
                    f"Significant: {'Yes' if pval < alpha else 'No'}",
                ])

            elif test_name == "spearman_correlation":
                if len(columns) < 2:
                    return ToolResult(False, "Need 2 columns for correlation")
                stat, pval = stats.spearmanr(df[columns[0]].dropna(), df[columns[1]].dropna())
                result_lines.extend([
                    f"Spearman rho: {stat:.4f}",
                    f"p-value: {pval:.6f}",
                    f"Significant: {'Yes' if pval < alpha else 'No'}",
                ])

            elif test_name == "levene":
                if group_column and len(columns) == 1:
                    groups = [g[columns[0]].dropna().values for _, g in df.groupby(group_column)]
                    stat, pval = stats.levene(*groups)
                    result_lines.extend([
                        f"Levene statistic: {stat:.4f}",
                        f"p-value: {pval:.6f}",
                        f"Equal variances: {'Yes' if pval >= alpha else 'No'}",
                    ])
                else:
                    return ToolResult(False, "Levene test requires group_column + 1 column")

            else:
                return ToolResult(False, f"Test '{test_name}' not implemented yet")

            return ToolResult(success=True, output="\n".join(result_lines))
        except Exception as e:
            return ToolResult(False, f"Test failed: {type(e).__name__}: {e}")

    def _fit_model(
        self,
        model_type: str,
        target: str,
        features: list[str],
        test_size: float = 0.2,
    ) -> ToolResult:
        """Fit a statistical model."""
        df = self._require_dataset()

        try:
            from sklearn.model_selection import train_test_split

            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            mask = y.notna()
            X = X[mask]
            y = y[mask]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            result_lines = [
                f"Model: {model_type}",
                f"Target: {target}",
                f"Features: {features}",
                f"Train size: {len(X_train)}, Test size: {len(X_test)}",
                "",
            ]

            if model_type == "ols":
                import statsmodels.api as sm

                X_train_c = sm.add_constant(X_train)
                model = sm.OLS(y_train, X_train_c).fit()
                result_lines.append(model.summary().as_text())

                # VIF
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                if len(features) > 1:
                    vif_data = pd.DataFrame()
                    vif_data["Feature"] = features
                    vif_data["VIF"] = [
                        variance_inflation_factor(X_train.values, i)
                        for i in range(len(features))
                    ]
                    result_lines.extend(["", "VIF (Multicollinearity Check):", vif_data.to_string()])

            elif model_type == "logistic":
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import classification_report

                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred)
                result_lines.extend(["Classification Report:", report])

            elif model_type in ("ridge", "lasso"):
                from sklearn.linear_model import Lasso, Ridge
                from sklearn.metrics import mean_squared_error, r2_score

                model_cls = Ridge if model_type == "ridge" else Lasso
                model = model_cls(alpha=1.0, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                result_lines.extend([
                    f"R²: {r2:.4f}",
                    f"RMSE: {rmse:.4f}",
                    "",
                    "Coefficients:",
                ])
                for feat, coef in zip(features, model.coef_):
                    result_lines.append(f"  {feat}: {coef:.4f}")

            elif model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_squared_error, r2_score

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                result_lines.extend([
                    f"R²: {r2:.4f}",
                    f"RMSE: {rmse:.4f}",
                    "",
                    "Feature Importance:",
                ])
                for feat, imp in sorted(
                    zip(features, model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    result_lines.append(f"  {feat}: {imp:.4f}")

            return ToolResult(success=True, output="\n".join(result_lines))
        except Exception as e:
            return ToolResult(False, f"Model fitting failed: {type(e).__name__}: {e}")
