"""Tests for auto-profiler module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astats.data.ingestion import load_data
from astats.data.models import ColumnType, DataProfile, Dataset
from astats.data.profiler import AutoProfiler, profile_dataset

SAMPLE_CSV = Path(__file__).parent.parent / "examples" / "data" / "sample_sales.csv"


class TestAutoProfiler:
    """Tests for the AutoProfiler class."""

    @pytest.fixture
    def dataset(self) -> Dataset:
        return load_data(SAMPLE_CSV)

    @pytest.fixture
    def profile(self, dataset: Dataset) -> DataProfile:
        profiler = AutoProfiler()
        return profiler.profile(dataset)

    def test_profile_basic(self, profile: DataProfile) -> None:
        """Test basic profile properties."""
        assert profile.n_rows > 0
        assert profile.n_cols > 0
        assert profile.memory_usage_mb > 0
        assert profile.profiling_time_seconds >= 0

    def test_column_profiles(self, profile: DataProfile) -> None:
        """Test that all columns are profiled."""
        assert len(profile.column_profiles) == profile.n_cols
        for cp in profile.column_profiles:
            assert cp.name
            assert cp.count > 0
            assert 0 <= cp.missing_pct <= 1.0

    def test_numeric_columns_have_stats(self, profile: DataProfile) -> None:
        """Test numeric columns have proper statistics."""
        numeric_profiles = [
            cp for cp in profile.column_profiles
            if cp.semantic_type in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE)
        ]
        assert len(numeric_profiles) > 0
        for cp in numeric_profiles:
            assert cp.mean is not None
            assert cp.std is not None
            assert cp.min is not None
            assert cp.max is not None
            assert cp.min <= cp.max  # type: ignore[operator]

    def test_categorical_columns_have_top_values(self, profile: DataProfile) -> None:
        """Test categorical columns have value distributions."""
        cat_profiles = [
            cp for cp in profile.column_profiles
            if cp.semantic_type == ColumnType.CATEGORICAL
        ]
        assert len(cat_profiles) > 0
        for cp in cat_profiles:
            assert len(cp.top_values) > 0

    def test_missing_report(self, profile: DataProfile) -> None:
        """Test missing data report."""
        mr = profile.missing_report
        assert mr.total_cells > 0
        assert 0 <= mr.missing_pct <= 1.0
        assert mr.complete_rows >= 0
        assert 0 <= mr.complete_rows_pct <= 1.0

    def test_correlations(self, profile: DataProfile) -> None:
        """Test correlation computation."""
        assert profile.correlations is not None
        assert not profile.correlations.matrix.empty
        assert isinstance(profile.correlations.top_pairs, list)

    def test_llm_summary(self, profile: DataProfile) -> None:
        """Test LLM-friendly summary generation."""
        summary = profile.summary_for_llm()
        assert isinstance(summary, str)
        assert len(summary) > 50
        assert "rows" in summary.lower()
        assert "columns" in summary.lower()

    def test_to_dict(self, profile: DataProfile) -> None:
        """Test serialization."""
        d = profile.to_dict()
        assert d["n_rows"] > 0
        assert d["n_cols"] > 0
        assert isinstance(d["columns"], list)

    def test_convenience_function(self, dataset: Dataset) -> None:
        """Test the profile_dataset convenience function."""
        profile = profile_dataset(dataset)
        assert isinstance(profile, DataProfile)
        assert profile.n_rows > 0

    def test_profile_attached_to_dataset(self, dataset: Dataset) -> None:
        """Test that profiling attaches result to dataset."""
        assert dataset.profile is None
        profiler = AutoProfiler()
        profiler.profile(dataset)
        assert dataset.profile is not None


class TestColumnTypeClassification:
    """Test column type detection."""

    def test_classify_numeric(self) -> None:
        df = pd.DataFrame({"x": np.random.randn(100)})
        dataset = Dataset(df=df, name="test")
        profiler = AutoProfiler()
        profile = profiler.profile(dataset)
        assert profile.column_profiles[0].semantic_type == ColumnType.NUMERIC_CONTINUOUS

    def test_classify_categorical(self) -> None:
        df = pd.DataFrame({"x": np.random.choice(["A", "B", "C"], 100)})
        dataset = Dataset(df=df, name="test")
        profiler = AutoProfiler()
        profile = profiler.profile(dataset)
        assert profile.column_profiles[0].semantic_type == ColumnType.CATEGORICAL

    def test_classify_binary(self) -> None:
        df = pd.DataFrame({"x": np.random.choice([0, 1], 100)})
        dataset = Dataset(df=df, name="test")
        profiler = AutoProfiler()
        profile = profiler.profile(dataset)
        assert profile.column_profiles[0].semantic_type == ColumnType.BINARY
