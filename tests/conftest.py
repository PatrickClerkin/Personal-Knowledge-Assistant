"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def test_corpus_dir() -> Path:
    """Path to test corpus directory."""
    return Path(__file__).parent.parent / "data" / "test_corpus"


@pytest.fixture
def sample_pdf_path(test_corpus_dir) -> Path:
    """Path to sample PDF for testing."""
    pdf_path = test_corpus_dir / "aswdCompositionLab2.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture
def temp_index_dir(tmp_path) -> Path:
    """Temporary directory for vector index files."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir
