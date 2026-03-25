"""Data models for AStats datasets and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ColumnType(str, Enum):
    """Semantic column type classification."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    dtype: str
    semantic_type: ColumnType
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    cardinality_ratio: float  # unique / count

    # Numeric stats (None for non-numeric)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int | None = None

    # Categorical stats
    top_values: list[dict[str, Any]] = field(default_factory=list)  # [{value, count, pct}]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type.value,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_pct": round(self.missing_pct, 4),
            "unique_count": self.unique_count,
        }
        if self.mean is not None:
            result.update(
                {
                    "mean": _safe_round(self.mean),
                    "std": _safe_round(self.std),
                    "min": _safe_round(self.min),
                    "q25": _safe_round(self.q25),
                    "median": _safe_round(self.median),
                    "q75": _safe_round(self.q75),
                    "max": _safe_round(self.max),
                    "skewness": _safe_round(self.skewness),
                    "kurtosis": _safe_round(self.kurtosis),
                    "outlier_count": self.outlier_count,
                }
            )
        if self.top_values:
            result["top_values"] = self.top_values[:10]
        return result


@dataclass
class MissingDataReport:
    """Report on missing data patterns."""

    total_missing: int
    total_cells: int
    missing_pct: float
    columns_with_missing: list[dict[str, Any]]  # [{column, count, pct}]
    complete_rows: int
    complete_rows_pct: float


@dataclass
class CorrelationResult:
    """Pairwise correlation results."""

    method: str  # pearson, spearman, kendall
    matrix: pd.DataFrame
    top_pairs: list[dict[str, Any]]  # [{col1, col2, correlation}]


@dataclass
class DataProfile:
    """Comprehensive statistical profile of a dataset."""

    n_rows: int
    n_cols: int
    memory_usage_mb: float
    column_profiles: list[ColumnProfile]
    missing_report: MissingDataReport
    correlations: CorrelationResult | None = None
    profiling_time_seconds: float = 0.0

    def summary_for_llm(self) -> str:
        """Generate a concise text summary suitable for LLM context."""
        lines = [
            f"Dataset: {self.n_rows} rows × {self.n_cols} columns "
            f"({self.memory_usage_mb:.1f} MB)",
            "",
            "Column Summary:",
        ]

        for cp in self.column_profiles:
            line = f"  - {cp.name} ({cp.semantic_type.value}): "
            if cp.missing_pct > 0:
                line += f"{cp.missing_pct:.1%} missing, "
            if cp.mean is not None:
                line += f"mean={_safe_round(cp.mean)}, std={_safe_round(cp.std)}, "
                line += f"range=[{_safe_round(cp.min)}, {_safe_round(cp.max)}]"
                if cp.outlier_count and cp.outlier_count > 0:
                    line += f", {cp.outlier_count} outliers"
            elif cp.top_values:
                top = cp.top_values[:3]
                vals = ", ".join(f"'{v['value']}' ({v['pct']:.1%})" for v in top)
                line += f"top: {vals}"
            else:
                line += f"{cp.unique_count} unique values"
            lines.append(line)

        if self.missing_report.missing_pct > 0:
            lines.extend([
                "",
                f"Missing Data: {self.missing_report.missing_pct:.1%} overall, "
                f"{self.missing_report.complete_rows_pct:.1%} complete rows",
            ])

        if self.correlations and self.correlations.top_pairs:
            lines.extend(["", "Top Correlations:"])
            for pair in self.correlations.top_pairs[:5]:
                lines.append(
                    f"  - {pair['col1']} ↔ {pair['col2']}: "
                    f"r={pair['correlation']:.3f}"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "columns": [cp.to_dict() for cp in self.column_profiles],
            "missing": {
                "total_pct": round(self.missing_report.missing_pct, 4),
                "complete_rows_pct": round(self.missing_report.complete_rows_pct, 4),
            },
            "profiling_time_seconds": round(self.profiling_time_seconds, 2),
        }


@dataclass
class Dataset:
    """A loaded dataset with metadata."""

    df: pd.DataFrame
    name: str
    source_path: Path | None = None
    loaded_at: datetime = field(default_factory=datetime.now)
    profile: DataProfile | None = None
    file_size_bytes: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def head_str(self, n: int = 5) -> str:
        """String representation of first n rows."""
        return self.df.head(n).to_string()


def _safe_round(value: float | None, decimals: int = 4) -> float | None:
    """Safely round a value, handling None and NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), decimals)
