"""Workflow Orchestrator — the agentic brain of AStats.

Coordinates multiple agents, manages the Plan → Execute → Reflect lifecycle,
and handles human-in-the-loop approval workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from astats.agents.base import BaseAgent
from astats.agents.eda import EDAAgent
from astats.agents.hypothesis import HypothesisAgent
from astats.agents.regression import RegressionAgent
from astats.agents.timeseries import TimeSeriesAgent
from astats.agents.tools import ToolRegistry
from astats.config import AStatsConfig
from astats.data.ingestion import DataLoader
from astats.data.models import Dataset
from astats.data.profiler import AutoProfiler
from astats.llm.memory import ConversationMemory
from astats.llm.prompts import SYSTEM_PROMPT_BASE
from astats.llm.provider import BaseLLMProvider, LLMMessage, create_provider
from astats.logging import get_logger

logger = get_logger("agents.orchestrator")
console = Console()


class WorkflowOrchestrator:
    """Central orchestrator for AStats agentic workflows.

    Manages the full lifecycle:
    1. Data loading and profiling
    2. Agent selection (automatic or user-guided)
    3. Analysis execution with human-in-the-loop
    4. Result aggregation and reporting

    Supports three autonomy levels:
    - full-auto: Runs without asking (max N iterations)
    - semi-auto: Asks for approval at key decision points
    - step-by-step: Asks before every action
    """

    AGENT_MAP: dict[str, type[BaseAgent]] = {
        "eda": EDAAgent,
        "hypothesis": HypothesisAgent,
        "regression": RegressionAgent,
        "timeseries": TimeSeriesAgent,
    }

    def __init__(self, config: AStatsConfig) -> None:
        self.config = config
        self.provider: BaseLLMProvider = create_provider(config)
        self.tool_registry = ToolRegistry(
            output_dir=str(config.analysis.output_dir)
        )
        self._dataset: Dataset | None = None
        self._agents: dict[str, BaseAgent] = {}
        self._session_history: list[dict[str, Any]] = []

        # Orchestrator's own memory for high-level reasoning
        self.memory = ConversationMemory(
            system_prompt=SYSTEM_PROMPT_BASE,
            max_messages=30,
        )

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    @property
    def autonomy(self) -> str:
        return self.config.analysis.autonomy_level

    def load_dataset(
        self,
        path: str | Path,
        name: str | None = None,
        profile: bool = True,
    ) -> Dataset:
        """Load and optionally profile a dataset.

        Args:
            path: Path to the data file.
            name: Optional dataset name.
            profile: Whether to auto-profile after loading.

        Returns:
            The loaded and optionally profiled Dataset.
        """
        loader = DataLoader()
        dataset = loader.load(path, name=name)

        if profile:
            profiler = AutoProfiler()
            profiler.profile(dataset)
            console.print(
                Panel(
                    dataset.profile.summary_for_llm(),
                    title=f"📊 Dataset Profile: {dataset.name}",
                    border_style="blue",
                )
            )

        self._dataset = dataset
        self.tool_registry.set_dataset(dataset)
        return dataset

    def get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create a specialized agent.

        Args:
            agent_type: Agent type ('eda', 'hypothesis', 'regression', 'timeseries').

        Returns:
            The agent instance.
        """
        if agent_type not in self._agents:
            agent_cls = self.AGENT_MAP.get(agent_type)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent: {agent_type}. "
                    f"Available: {list(self.AGENT_MAP.keys())}"
                )
            self._agents[agent_type] = agent_cls(
                config=self.config,
                tool_registry=self.tool_registry,
                provider=self.provider,
            )
        return self._agents[agent_type]

    def auto_select_agent(self, query: str) -> str:
        """Use LLM to select the best agent for a query.

        Args:
            query: User's analysis question.

        Returns:
            Agent type string.
        """
        prompt = (
            f"Given this analysis request: \"{query}\"\n\n"
            f"Which specialist agent should handle this? Choose exactly one:\n"
            f"- eda: Exploratory data analysis, distributions, correlations, patterns\n"
            f"- hypothesis: Statistical hypothesis testing, comparisons between groups\n"
            f"- regression: Regression modeling, prediction, feature importance\n"
            f"- timeseries: Time series analysis, trends, seasonality, forecasting\n\n"
            f"Respond with ONLY the agent name (eda, hypothesis, regression, or timeseries)."
        )

        response = self.provider.chat(
            [
                LLMMessage("system", "You are a routing assistant. Respond with only the agent name."),
                LLMMessage("user", prompt),
            ],
            temperature=0.0,
            max_tokens=20,
        )

        content = response.content.strip().lower()
        if not content:
            logger.warning("[agent]Empty routing response. Defaulting to 'eda'[/agent]")
            agent_type = "eda"
        else:
            agent_type = content.split()[0]

        if agent_type not in self.AGENT_MAP:
            agent_type = "eda"  # Default fallback

        logger.info(f"[agent]Auto-selected agent[/agent]: {agent_type}")
        return agent_type

    def run(
        self,
        query: str,
        agent_type: str | None = None,
    ) -> dict[str, Any]:
        """Run a complete analysis workflow.

        Args:
            query: User's analysis question/request.
            agent_type: Specific agent to use (auto-selects if None).

        Returns:
            Analysis results dictionary.
        """
        if self._dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        # Select agent
        if agent_type is None:
            agent_type = self.auto_select_agent(query)

        console.print(
            f"\n[bold magenta]🤖 Agent: {agent_type.upper()}[/bold magenta] | "
            f"Autonomy: {self.autonomy}\n"
        )

        # Semi-auto: confirm agent selection
        if self.autonomy == "semi-auto":
            if not Confirm.ask(
                f"Use [bold]{agent_type}[/bold] agent for this analysis?",
                default=True,
            ):
                agent_type = Prompt.ask(
                    "Choose agent",
                    choices=list(self.AGENT_MAP.keys()),
                    default="eda",
                )

        agent = self.get_agent(agent_type)

        # Run analysis
        console.print(Panel(f"Analyzing: {query}", title="🔍 Query", border_style="cyan"))
        results = agent.analyze(self._dataset, query)

        # Store in session
        self._session_history.append(results)

        # Display summary
        if results.get("summary"):
            console.print(
                Panel(
                    results["summary"],
                    title="📋 Analysis Summary",
                    border_style="green",
                )
            )

        return results

    def chat(self, message: str) -> str:
        """Open-ended chat about the loaded dataset.

        Uses the base orchestrator LLM with data context.

        Args:
            message: User's message/question.

        Returns:
            LLM response.
        """
        if self._dataset is None:
            return "No dataset loaded. Please load a dataset first with `load_dataset()`."

        # Inject data context if first message
        if self.memory.message_count == 0 and self._dataset.profile:
            self.memory.inject_context(self._dataset.profile.summary_for_llm())

        self.memory.add_user(message)

        response = self.provider.chat(
            self.memory.messages,
            tools=self.tool_registry.list_tools(),
        )

        # Handle tool calls
        if response.tool_calls:
            for tc in response.tool_calls:
                result = self.tool_registry.execute(tc["name"], **tc.get("arguments", {}))
                self.memory.add_tool_result(tc["name"], result.output)

            # Get follow-up response
            follow_up = self.provider.chat(self.memory.messages)
            self.memory.add_assistant(follow_up.content)
            return follow_up.content

        self.memory.add_assistant(response.content)
        return response.content

    def save_session(self, path: Path | None = None) -> Path:
        """Save the current session for later restoration."""
        if path is None:
            path = self.config.analysis.output_dir / "sessions" / "session.json"
        self.memory.save(path)
        logger.info(f"Session saved to {path}")
        return path
