"""Hypothesis Testing Agent — confirmatory statistical analysis."""

from __future__ import annotations

from astats.agents.base import BaseAgent
from astats.llm.prompts import SYSTEM_PROMPT_HYPOTHESIS


class HypothesisAgent(BaseAgent):
    """Agent specialized in hypothesis testing.

    Guides practitioners through proper test selection, assumption validation,
    execution, and interpretation of statistical tests.
    """

    agent_name = "hypothesis"
    system_prompt = SYSTEM_PROMPT_HYPOTHESIS

    def get_agent_description(self) -> str:
        return (
            "Hypothesis Testing Agent: Guides selection and execution of statistical "
            "tests (t-test, ANOVA, chi-square, Mann-Whitney, etc.) with proper "
            "assumption checking, effect sizes, and multiple comparison corrections."
        )
