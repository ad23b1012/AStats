"""Regression Agent — regression modeling specialist."""

from __future__ import annotations

from astats.agents.base import BaseAgent
from astats.llm.prompts import SYSTEM_PROMPT_REGRESSION


class RegressionAgent(BaseAgent):
    """Agent specialized in regression analysis.

    Builds, validates, and interprets regression models with proper
    diagnostics and assumption checking.
    """

    agent_name = "regression"
    system_prompt = SYSTEM_PROMPT_REGRESSION

    def get_agent_description(self) -> str:
        return (
            "Regression Agent: Fits and validates regression models (OLS, logistic, "
            "ridge, lasso) with assumption checking, VIF analysis, residual diagnostics, "
            "cross-validation, and interpretable coefficient reporting."
        )
