"""Time Series Agent — temporal data analysis specialist."""

from __future__ import annotations

from astats.agents.base import BaseAgent
from astats.llm.prompts import SYSTEM_PROMPT_TIMESERIES


class TimeSeriesAgent(BaseAgent):
    """Agent specialized in time series analysis.

    Analyzes temporal data including decomposition, stationarity testing,
    ARIMA modeling, and forecasting.
    """

    agent_name = "timeseries"
    system_prompt = SYSTEM_PROMPT_TIMESERIES

    def get_agent_description(self) -> str:
        return (
            "Time Series Agent: Performs trend/seasonality decomposition, "
            "stationarity tests (ADF, KPSS), ARIMA/SARIMA model fitting, "
            "and forecasting with prediction intervals."
        )
