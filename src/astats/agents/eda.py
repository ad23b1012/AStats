"""EDA Agent — Exploratory Data Analysis specialist."""

from __future__ import annotations

from astats.agents.base import BaseAgent
from astats.llm.prompts import SYSTEM_PROMPT_EDA


class EDAAgent(BaseAgent):
    """Agent specialized in exploratory data analysis.

    Systematically explores datasets to discover patterns, distributions,
    relationships, and anomalies.
    """

    agent_name = "eda"
    system_prompt = SYSTEM_PROMPT_EDA

    def get_agent_description(self) -> str:
        return (
            "EDA Agent: Performs comprehensive exploratory data analysis including "
            "distribution analysis, correlation exploration, outlier detection, "
            "and pattern discovery with automated visualizations."
        )
