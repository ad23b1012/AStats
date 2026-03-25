"""Base agent class for all specialized statistical agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from astats.agents.tools import ToolRegistry, ToolResult
from astats.config import AStatsConfig
from astats.data.models import Dataset
from astats.llm.memory import ConversationMemory
from astats.llm.provider import BaseLLMProvider, LLMMessage, LLMResponse, create_provider
from astats.logging import get_logger

logger = get_logger("agents.base")


class BaseAgent(ABC):
    """Abstract base class for all AStats agents.

    Implements the Plan → Execute → Reflect loop.
    """

    agent_name: str = "base"
    system_prompt: str = ""

    def __init__(
        self,
        config: AStatsConfig,
        tool_registry: ToolRegistry,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self.config = config
        self.tools = tool_registry
        self.provider = provider or create_provider(config)
        self.memory = ConversationMemory(
            system_prompt=self.system_prompt,
            max_messages=50,
        )
        self._iteration = 0
        self._results: list[dict[str, Any]] = []

    @property
    def max_iterations(self) -> int:
        return self.config.analysis.max_iterations

    def analyze(self, dataset: Dataset, query: str) -> dict[str, Any]:
        """Run a full analysis cycle on a dataset.

        Args:
            dataset: The dataset to analyze.
            query: User's analysis question/request.

        Returns:
            Dictionary with analysis results.
        """
        self.tools.set_dataset(dataset)
        self._results.clear()
        self._iteration = 0

        # Inject data context
        if dataset.profile:
            context = dataset.profile.summary_for_llm()
        else:
            context = f"Dataset '{dataset.name}': {dataset.shape[0]} rows × {dataset.shape[1]} columns\n"
            context += f"Columns: {list(dataset.df.columns)}\n"
            context += f"First rows:\n{dataset.head_str()}"

        self.memory.inject_context(context)

        # Plan → Execute → Reflect loop
        logger.info(f"[agent]{self.agent_name}[/agent] starting analysis: '{query[:80]}'")

        plan = self.plan(query)
        self._results.append({"phase": "plan", "content": plan})

        while self._iteration < self.max_iterations:
            self._iteration += 1
            logger.info(
                f"[agent]{self.agent_name}[/agent] iteration {self._iteration}/{self.max_iterations}"
            )

            # Execute: ask LLM to decide next action
            exec_result = self.execute(query)
            self._results.append({"phase": "execute", "iteration": self._iteration, "content": exec_result})

            # Check if LLM made tool calls
            if exec_result.tool_calls:
                for tc in exec_result.tool_calls:
                    tool_result = self.tools.execute(tc["name"], **tc.get("arguments", {}))
                    self.memory.add_tool_result(tc["name"], tool_result.output)
                    self._results.append({
                        "phase": "tool",
                        "tool": tc["name"],
                        "result": tool_result,
                    })
            else:
                # No tool calls — LLM is responding with analysis/conclusion
                reflection = self.reflect(exec_result.content)
                self._results.append({"phase": "reflect", "content": reflection})

                # Check if the agent wants to continue or is done
                if self._should_stop(reflection):
                    break

        summary = self.summarize()
        return {
            "agent": self.agent_name,
            "query": query,
            "iterations": self._iteration,
            "results": self._results,
            "summary": summary,
        }

    def plan(self, query: str) -> str:
        """Generate an analysis plan."""
        from astats.llm.prompts import ANALYSIS_PLAN_TEMPLATE

        data_summary = ""
        if self.tools.dataset and self.tools.dataset.profile:
            data_summary = self.tools.dataset.profile.summary_for_llm()

        plan_prompt = ANALYSIS_PLAN_TEMPLATE.render(
            data_summary=data_summary,
            user_query=query,
        )

        self.memory.add_user(plan_prompt)
        response = self.provider.chat(self.memory.messages)
        self.memory.add_assistant(response.content)

        logger.info(f"[agent]Plan generated[/agent] ({response.total_tokens} tokens)")
        return response.content

    def execute(self, query: str) -> LLMResponse:
        """Execute the next step — LLM decides which tool to call."""
        prompt = (
            f"Continue the analysis. The user asked: \"{query}\"\n\n"
            "Based on the plan and results so far, perform the next step. "
            "Use tools to execute operations. If you have gathered enough "
            "information, provide your analysis conclusions instead."
        )
        self.memory.add_user(prompt)

        response = self.provider.chat(
            self.memory.messages,
            tools=self.tools.list_tools(),
        )
        self.memory.add_assistant(response.content)

        return response

    def reflect(self, result: str) -> str:
        """Reflect on the latest result."""
        from astats.llm.prompts import REFLECTION_TEMPLATE

        prompt = REFLECTION_TEMPLATE.render(
            action="analysis step",
            result=result[:2000],
        )
        self.memory.add_user(prompt)
        response = self.provider.chat(self.memory.messages)
        self.memory.add_assistant(response.content)

        return response.content

    def summarize(self) -> str:
        """Generate a final summary of all findings."""
        self.memory.add_user(
            "Please provide a comprehensive summary of the entire analysis. "
            "Include: key findings, statistical evidence, recommendations, "
            "and any limitations or caveats."
        )
        response = self.provider.chat(self.memory.messages)
        return response.content

    def _should_stop(self, reflection: str) -> bool:
        """Determine if the agent should stop iterating."""
        stop_indicators = [
            "analysis is complete",
            "no further",
            "sufficient evidence",
            "conclude",
            "final summary",
            "confidence: high",
        ]
        lower = reflection.lower()
        return any(indicator in lower for indicator in stop_indicators)

    @abstractmethod
    def get_agent_description(self) -> str:
        """Return a description of this agent's capabilities."""
        ...
