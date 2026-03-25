"""Tests for tool registry."""

import numpy as np
import pandas as pd
import pytest

from astats.agents.tools import ToolRegistry, ToolResult
from astats.data.models import Dataset


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    @pytest.fixture
    def registry(self, tmp_path) -> ToolRegistry:
        reg = ToolRegistry(output_dir=str(tmp_path / "output"))
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100) + 2,
            "group": np.random.choice(["A", "B"], 100),
        })
        dataset = Dataset(df=df, name="test")
        reg.set_dataset(dataset)
        return reg

    def test_list_tools(self, registry: ToolRegistry) -> None:
        """Test that built-in tools are registered."""
        tools = registry.list_tools()
        assert len(tools) > 0
        names = {t["function"]["name"] for t in tools}
        assert "describe_data" in names
        assert "run_code" in names
        assert "create_plot" in names
        assert "run_statistical_test" in names
        assert "fit_model" in names

    def test_describe_data(self, registry: ToolRegistry) -> None:
        """Test describe_data tool."""
        result = registry.execute("describe_data")
        assert result.success
        assert "mean" in result.output.lower() or "count" in result.output.lower()

    def test_describe_specific_columns(self, registry: ToolRegistry) -> None:
        """Test describing specific columns."""
        result = registry.execute("describe_data", columns=["x"])
        assert result.success

    def test_run_code(self, registry: ToolRegistry) -> None:
        """Test code execution tool."""
        result = registry.execute("run_code", code="print(df.shape)")
        assert result.success
        assert "100" in result.output

    def test_run_code_error(self, registry: ToolRegistry) -> None:
        """Test code execution error handling."""
        result = registry.execute("run_code", code="1/0")
        assert not result.success
        assert "ZeroDivision" in result.output

    def test_create_plot(self, registry: ToolRegistry) -> None:
        """Test plot generation."""
        result = registry.execute(
            "create_plot",
            plot_type="histogram",
            columns=["x"],
        )
        assert result.success
        assert len(result.plots) == 1

    def test_statistical_test(self, registry: ToolRegistry) -> None:
        """Test running a statistical test."""
        result = registry.execute(
            "run_statistical_test",
            test_name="t_test_ind",
            columns=["x", "y"],
        )
        assert result.success
        assert "p-value" in result.output

    def test_unknown_tool(self, registry: ToolRegistry) -> None:
        """Test error for unknown tool."""
        result = registry.execute("nonexistent_tool")
        assert not result.success

    def test_no_dataset(self) -> None:
        """Test error when no dataset is loaded."""
        registry = ToolRegistry()
        result = registry.execute("describe_data")
        assert not result.success
