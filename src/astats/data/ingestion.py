"""Intelligent data ingestion with auto-detection.

Supports: CSV, TSV, Excel, Parquet, JSON, Feather, Stata (.dta), SPSS (.sav).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from astats.data.models import Dataset
from astats.logging import get_logger

logger = get_logger("data.ingestion")

# File extension → loader mapping
_LOADERS: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".feather": "feather",
    ".ftr": "feather",
    ".dta": "stata",
    ".sav": "spss",
    ".sas7bdat": "sas",
}


class DataLoader:
    """Intelligent data loader with format auto-detection.

    Usage:
        loader = DataLoader()
        dataset = loader.load("path/to/data.csv")
    """

    def __init__(self, max_sample_rows: int | None = None) -> None:
        """Initialize the data loader.

        Args:
            max_sample_rows: If set, only load first N rows (useful for large files).
        """
        self.max_sample_rows = max_sample_rows

    def load(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load a dataset from file with auto-detection.

        Args:
            path: Path to the data file.
            name: Optional dataset name (defaults to filename stem).
            **kwargs: Additional arguments passed to the pandas reader.

        Returns:
            A Dataset object with loaded data and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is unsupported.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        loader_type = _LOADERS.get(suffix)
        if loader_type is None:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported: {', '.join(sorted(_LOADERS.keys()))}"
            )

        logger.info(f"Loading [data]{path.name}[/data] (format: {loader_type})")
        start = time.perf_counter()

        df = self._dispatch_load(path, loader_type, **kwargs)

        if self.max_sample_rows and len(df) > self.max_sample_rows:
            logger.info(
                f"Sampling {self.max_sample_rows:,} rows from {len(df):,} total"
            )
            df = df.head(self.max_sample_rows)

        elapsed = time.perf_counter() - start
        file_size = path.stat().st_size

        dataset = Dataset(
            df=df,
            name=name or path.stem,
            source_path=path,
            file_size_bytes=file_size,
        )

        logger.info(
            f"[success]Loaded[/success] {dataset.shape[0]:,} rows × "
            f"{dataset.shape[1]} columns in {elapsed:.2f}s "
            f"({file_size / 1024 / 1024:.1f} MB)"
        )
        return dataset

    def _dispatch_load(
        self, path: Path, loader_type: str, **kwargs: Any
    ) -> pd.DataFrame:
        """Dispatch to the appropriate pandas reader."""
        loaders = {
            "csv": self._load_csv,
            "excel": self._load_excel,
            "parquet": self._load_parquet,
            "json": self._load_json,
            "jsonl": self._load_jsonl,
            "feather": self._load_feather,
            "stata": self._load_stata,
            "spss": self._load_spss,
            "sas": self._load_sas,
        }
        loader_fn = loaders[loader_type]
        return loader_fn(path, **kwargs)

    def _load_csv(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load CSV with intelligent delimiter and encoding detection."""
        encoding = kwargs.pop("encoding", None)
        if encoding is None:
            encoding = self._detect_encoding(path)

        # Try to sniff delimiter
        sep = kwargs.pop("sep", None)
        if sep is None:
            sep = self._sniff_delimiter(path, encoding)

        nrows = kwargs.pop("nrows", self.max_sample_rows)
        return pd.read_csv(path, sep=sep, encoding=encoding, nrows=nrows, **kwargs)

    def _load_excel(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        nrows = kwargs.pop("nrows", self.max_sample_rows)
        return pd.read_excel(path, nrows=nrows, **kwargs)

    def _load_parquet(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)

    def _load_json(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        return pd.read_json(path, **kwargs)

    def _load_jsonl(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        nrows = kwargs.pop("nrows", self.max_sample_rows)
        return pd.read_json(path, lines=True, nrows=nrows, **kwargs)

    def _load_feather(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        return pd.read_feather(path, **kwargs)

    def _load_stata(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        return pd.read_stata(path, **kwargs)

    def _load_spss(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        try:
            import pyreadstat
        except ImportError:
            raise ImportError(
                "SPSS support requires pyreadstat. "
                "Install with: pip install astats[spss]"
            )
        df, _meta = pyreadstat.read_sav(str(path), **kwargs)
        return df

    def _load_sas(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        return pd.read_sas(path, **kwargs)

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            import chardet

            with open(path, "rb") as f:
                raw = f.read(min(100_000, path.stat().st_size))
            result = chardet.detect(raw)
            encoding = result.get("encoding", "utf-8") or "utf-8"
            confidence = result.get("confidence", 0)
            if confidence < 0.5:
                encoding = "utf-8"
            logger.debug(
                f"Detected encoding: {encoding} (confidence: {confidence:.1%})"
            )
            return encoding
        except Exception:
            return "utf-8"

    @staticmethod
    def _sniff_delimiter(path: Path, encoding: str) -> str:
        """Sniff the CSV delimiter from the first few lines."""
        import csv

        try:
            with open(path, encoding=encoding) as f:
                sample = f.read(8192)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except Exception:
            return ","


def load_data(path: str | Path, **kwargs: Any) -> Dataset:
    """Convenience function to load a dataset.

    Args:
        path: Path to the data file.
        **kwargs: Passed through to DataLoader.load().

    Returns:
        Dataset object.
    """
    return DataLoader().load(path, **kwargs)
