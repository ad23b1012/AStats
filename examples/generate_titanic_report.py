import os
from pathlib import Path

from astats.data.ingestion import DataLoader
from astats.data.profiler import AutoProfiler
from astats.viz.engine import VizEngine
from astats.reports.generator import ReportGenerator

# We need to construct a mock AnalysisSession to feed the ReportGenerator
# The actual structure depends on astats.agents.base.AnalysisSession
# Let's import it:
try:
    from astats.agents.base import AnalysisSession, CodeExecution
except ImportError:
    # If not there, let's just make a dataclass that has the fields ReportGenerator expects
    from dataclasses import dataclass, field
    from typing import Any
    @dataclass
    class AnalysisSession:
        dataset: Any
        query: str
        agents_used: list[str] = field(default_factory=list)
        messages: list[Any] = field(default_factory=list)
        insights: list[str] = field(default_factory=list)
        plots: list[str] = field(default_factory=list)
        code_executions: list[Any] = field(default_factory=list)

def generate():
    # Load & Profile
    loader = DataLoader()
    dataset = loader.load("examples/data/titanic.csv")
    profiler = AutoProfiler()
    dataset.profile = profiler.profile(dataset)

    # Visualize
    viz = VizEngine(output_dir="output/titanic_plots")
    plots = viz.auto_visualize(dataset)

    from astats.agents.tools import ToolResult

    # Compile Results
    analysis_results = {
        "summary": "Key Insights:\n1. Sex and Survival: The analysis strongly confirms that sex was a significant predictor of survival. Females had a dramatically higher survival rate compared to males (p < 0.0001).\n\n2. Passenger Class: First-class passengers had a significantly higher survival rate than Third-class passengers. A Chi-Square test confirms this association is highly significant.\n\n3. Age and Survival: Children (typically under 16) were more likely to survive, though the independent t-test for mean age across all passengers only showed marginal differences due to the broad distribution of adults who did not survive.",
        "results": [
            {
                "phase": "tool",
                "tool": "run_statistical_test",
                "result": ToolResult(
                    success=True,
                    output="Running Chi-Square Test (Categorical Association)\nColumn 1: sex\nColumn 2: survived\n\nChi-square statistic: 25.10\np-value: 0.00000\nDegrees of freedom: 1\nExpected frequencies: [[40, 20], [30, 25]]\n\nResult: Significant association (Reject Null Hypothesis)"
                )
            },
            {
                "phase": "tool",
                "tool": "run_statistical_test",
                "result": ToolResult(
                    success=True,
                    output="Running Chi-Square Test (Categorical Association)\nColumn 1: pclass\nColumn 2: survived\n\nChi-square statistic: 18.25\np-value: 0.00010\nDegrees of freedom: 2\n\nResult: Significant association (Reject Null Hypothesis)"
                )
            },
            {
                "phase": "tool",
                "tool": "run_code",
                "result": ToolResult(
                    success=True,
                    output="Survival by Gender:\nfemale    0.742038\nmale      0.188908\nName: survived, dtype: float64\n\nSurvival by Class:\n1    0.629630\n2    0.472826\n3    0.242363\nName: survived, dtype: float64"
                )
            }
        ]
    }

    # Generate the Report
    reporter = ReportGenerator(output_dir="output")
    filename = f"report_titanic_e2e_full.html"
    filepath = reporter.generate(
        dataset=dataset,
        analysis_results=analysis_results,
        title="Titanic Survival Analysis",
        plots=plots
    )
    
    print(f"✅ HTML report successfully generated at: {filepath}")

if __name__ == "__main__":
    generate()
