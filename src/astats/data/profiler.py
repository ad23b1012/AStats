"""Automated statistical profiling engine.

Generates comprehensive profiles of DataFrames including:
- Column type detection and classification
- Univariate statistics (mean, std, skew, kurtosis, outliers)
- Missing data analysis
- Correlation analysis with top pairs
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from astats.data.models import (
    ColumnProfile,
    ColumnType,
    CorrelationResult,
    DataProfile,
    Dataset,
    MissingDataReport,
)
from astats.logging import get_logger

logger = get_logger("data.profiler")


class AutoProfiler:
    """Generates comprehensive statistical profiles of datasets.

    Usage:
        profiler = AutoProfiler()
        profile = profiler.profile(dataset)
        print(profile.summary_for_llm())
    """

    def __init__(
        self,
        correlation_method: str = "pearson",
        outlier_method: str = "iqr",
        top_n_correlations: int = 10,
        top_n_categories: int = 10,
    ) -> None:
        self.correlation_method = correlation_method
        self.outlier_method = outlier_method
        self.top_n_correlations = top_n_correlations
        self.top_n_categories = top_n_categories

    def profile(self, dataset: Dataset) -> DataProfile:
        """Generate a comprehensive profile for a dataset.

        Args:
            dataset: The dataset to profile.

        Returns:
            DataProfile with all computed statistics.
        """
        logger.info(f"[agent]Profiling[/agent] dataset '{dataset.name}'...")
        start = time.perf_counter()

        df = dataset.df

        # Profile each column
        column_profiles = [
            self._profile_column(df[col]) for col in df.columns
        ]

        # Missing data report
        missing_report = self._analyze_missing(df)

        # Correlation analysis
        correlations = self._compute_correlations(df)

        elapsed = time.perf_counter() - start

        profile = DataProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            column_profiles=column_profiles,
            missing_report=missing_report,
            correlations=correlations,
            profiling_time_seconds=elapsed,
        )

        # Attach profile to dataset
        dataset.profile = profile

        logger.info(
            f"[success]Profiling complete[/success] in {elapsed:.2f}s — "
            f"{profile.n_rows:,} rows, {profile.n_cols} columns, "
            f"{missing_report.missing_pct:.1%} missing"
        )
        return profile

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column."""
        name = str(series.name)
        count = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = missing_count / count if count > 0 else 0.0
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        cardinality = unique_count / len(non_null) if len(non_null) > 0 else 0.0

        # Determine semantic type
        semantic_type = self._classify_column(series, unique_count, cardinality)

        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            count=count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            cardinality_ratio=cardinality,
        )

        # Compute type-specific statistics
        if semantic_type in (
            ColumnType.NUMERIC_CONTINUOUS,
            ColumnType.NUMERIC_DISCRETE,
        ):
            self._add_numeric_stats(profile, non_null)
        elif semantic_type in (ColumnType.CATEGORICAL, ColumnType.BINARY):
            self._add_categorical_stats(profile, non_null)

        return profile

    def _classify_column(
        self,
        series: pd.Series,
        unique_count: int,
        cardinality: float,
    ) -> ColumnType:
        """Classify a column into a semantic type."""
        dtype = series.dtype

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME

        # Numeric
        if pd.api.types.is_numeric_dtype(dtype):
            if unique_count == 2:
                return ColumnType.BINARY
            if pd.api.types.is_integer_dtype(dtype) and unique_count < 20:
                return ColumnType.NUMERIC_DISCRETE
            return ColumnType.NUMERIC_CONTINUOUS

        # Boolean
        if pd.api.types.is_bool_dtype(dtype):
            return ColumnType.BINARY

        # String/Object
        if pd.api.types.is_string_dtype(dtype) or dtype == object:
            if unique_count == 2:
                return ColumnType.BINARY
            if cardinality > 0.9 and unique_count > 50:
                return ColumnType.IDENTIFIER
            non_null = series.dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 100:
                    return ColumnType.TEXT
            if unique_count <= 50 or cardinality < 0.05:
                return ColumnType.CATEGORICAL
            return ColumnType.TEXT

        return ColumnType.UNKNOWN

    def _add_numeric_stats(
        self, profile: ColumnProfile, series: pd.Series
    ) -> None:
        """Compute numeric statistics for a column."""
        if len(series) == 0:
            return

        desc = series.describe()
        profile.mean = float(desc.get("mean", np.nan))
        profile.std = float(desc.get("std", np.nan))
        profile.min = float(desc.get("min", np.nan))
        profile.q25 = float(desc.get("25%", np.nan))
        profile.median = float(desc.get("50%", np.nan))
        profile.q75 = float(desc.get("75%", np.nan))
        profile.max = float(desc.get("max", np.nan))

        # Skewness and kurtosis
        try:
            profile.skewness = float(series.skew())
            profile.kurtosis = float(series.kurtosis())
        except Exception:
            pass

        # Outlier detection (IQR method)
        profile.outlier_count = self._count_outliers_iqr(series)

    def _add_categorical_stats(
        self, profile: ColumnProfile, series: pd.Series
    ) -> None:
        """Compute categorical statistics."""
        if len(series) == 0:
            return

        value_counts = series.value_counts()
        total = len(series)
        profile.top_values = [
            {
                "value": str(val),
                "count": int(cnt),
                "pct": cnt / total,
            }
            for val, cnt in value_counts.head(self.top_n_categories).items()
        ]

    @staticmethod
    def _count_outliers_iqr(series: pd.Series) -> int:
        """Count outliers using the IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((series < lower) | (series > upper)).sum())

    def _analyze_missing(self, df: pd.DataFrame) -> MissingDataReport:
        """Analyze missing data patterns."""
        total_cells = df.shape[0] * df.shape[1]
        missing_per_col = df.isna().sum()
        total_missing = int(missing_per_col.sum())
        complete_rows = int(df.dropna().shape[0])

        columns_with_missing = [
            {
                "column": str(col),
                "count": int(cnt),
                "pct": cnt / len(df) if len(df) > 0 else 0.0,
            }
            for col, cnt in missing_per_col.items()
            if cnt > 0
        ]

        # Sort by most missing
        columns_with_missing.sort(key=lambda x: x["count"], reverse=True)

        return MissingDataReport(
            total_missing=total_missing,
            total_cells=total_cells,
            missing_pct=total_missing / total_cells if total_cells > 0 else 0.0,
            columns_with_missing=columns_with_missing,
            complete_rows=complete_rows,
            complete_rows_pct=complete_rows / len(df) if len(df) > 0 else 0.0,
        )

    def _compute_correlations(
        self, df: pd.DataFrame
    ) -> CorrelationResult | None:
        """Compute correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None

        try:
            corr_matrix = df[numeric_cols].corr(method=self.correlation_method)
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return None

        # Extract top correlated pairs (excluding self-correlation)
        pairs: list[dict[str, Any]] = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                val = corr_matrix.loc[col1, col2]
                if not np.isnan(val):
                    pairs.append(
                        {
                            "col1": col1,
                            "col2": col2,
                            "correlation": round(float(val), 4),
                        }
                    )

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return CorrelationResult(
            method=self.correlation_method,
            matrix=corr_matrix,
            top_pairs=pairs[: self.top_n_correlations],
        )


def profile_dataset(dataset: Dataset, **kwargs: Any) -> DataProfile:
    """Convenience function to profile a dataset.

    Args:
        dataset: Dataset to profile.
        **kwargs: Passed to AutoProfiler constructor.

    Returns:
        DataProfile with computed statistics.
    """
    profiler = AutoProfiler(**kwargs)
    return profiler.profile(dataset)
