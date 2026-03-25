"""Tests for data ingestion module."""

from pathlib import Path

import pytest

from astats.data.ingestion import DataLoader, load_data
from astats.data.models import Dataset

SAMPLE_CSV = Path(__file__).parent.parent / "examples" / "data" / "sample_sales.csv"


class TestDataLoader:
    """Tests for the DataLoader class."""

    def test_load_csv(self) -> None:
        """Test loading a CSV file."""
        loader = DataLoader()
        dataset = loader.load(SAMPLE_CSV)
        assert isinstance(dataset, Dataset)
        assert dataset.shape[0] > 0
        assert dataset.shape[1] > 0
        assert dataset.name == "sample_sales"

    def test_load_csv_with_name(self) -> None:
        """Test loading with a custom name."""
        loader = DataLoader()
        dataset = loader.load(SAMPLE_CSV, name="my_data")
        assert dataset.name == "my_data"

    def test_load_csv_max_rows(self) -> None:
        """Test sampling with max rows."""
        loader = DataLoader(max_sample_rows=10)
        dataset = loader.load(SAMPLE_CSV)
        assert dataset.shape[0] <= 10

    def test_load_nonexistent_file(self) -> None:
        """Test error for missing file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/data.csv")

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """Test error for unsupported format."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("hello")
        loader = DataLoader()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(bad_file)

    def test_convenience_function(self) -> None:
        """Test the load_data convenience function."""
        dataset = load_data(SAMPLE_CSV)
        assert isinstance(dataset, Dataset)
        assert dataset.shape[0] > 0

    def test_dataset_metadata(self) -> None:
        """Test that dataset metadata is populated."""
        loader = DataLoader()
        dataset = loader.load(SAMPLE_CSV)
        assert dataset.source_path is not None
        assert dataset.source_path.exists()
        assert dataset.file_size_bytes > 0
        assert dataset.loaded_at is not None

    def test_head_str(self) -> None:
        """Test string representation of head."""
        dataset = load_data(SAMPLE_CSV)
        head = dataset.head_str(3)
        assert isinstance(head, str)
        assert len(head) > 0
