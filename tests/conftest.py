"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_corpus_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "test_corpus"

@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR

@pytest.fixture
def sample_pdf_path(test_corpus_dir) -> Path:
    pdf_path = test_corpus_dir / "aswdCompositionLab2.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path

@pytest.fixture
def sample_txt_path(fixtures_dir) -> Path:
    """Path to the sample .txt test file."""
    txt_path = fixtures_dir / "sample.txt"
    if not txt_path.exists():
        pytest.skip(f"Test file not found: {txt_path}")
    return txt_path

@pytest.fixture
def sample_md_path(fixtures_dir) -> Path:
    """Path to the sample .md test file."""
    md_path = fixtures_dir / "sample.md"
    if not md_path.exists():
        pytest.skip(f"Test file not found: {md_path}")
    return md_path

@pytest.fixture
def sample_docx_path(fixtures_dir) -> Path:
    """Path to the sample .docx test file."""
    docx_path = fixtures_dir / "sample.docx"
    if not docx_path.exists():
        pytest.skip(f"Test file not found: {docx_path}")
    return docx_path

@pytest.fixture
def temp_index_dir(tmp_path) -> Path:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir
