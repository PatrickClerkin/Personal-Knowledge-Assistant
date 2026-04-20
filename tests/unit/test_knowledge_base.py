"""
Unit tests for KnowledgeBase — the central class of the system.

Tests ingestion, search, document management, and persistence
using mocked components to avoid slow embedding/FAISS operations.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.ingestion.knowledge_base import KnowledgeBase
from src.ingestion.chunking.chunk import Chunk
from src.ingestion.storage.vector_store import SearchResult
from src.ingestion.storage.document_registry import DocumentRecord


def _make_chunk(
    chunk_id: str = "chunk_1",
    doc_id: str = "test_doc",
    content: str = "Test content about neural networks.",
    source: str = "test.pdf",
    page: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        source_doc_title=source,
        page_number=page,
        metadata={},
    )


def _make_result(chunk: Chunk = None, score: float = 0.85, rank: int = 1) -> SearchResult:
    return SearchResult(
        chunk=chunk or _make_chunk(),
        score=score,
        rank=rank,
    )


class TestKnowledgeBaseInit:
    """Test KnowledgeBase initialisation."""

    def test_creates_with_defaults(self):
        """KB can be created with no arguments."""
        kb = KnowledgeBase()
        assert kb.size == 0
        assert kb.document_ids == []

    def test_empty_kb_has_zero_size(self):
        kb = KnowledgeBase()
        assert len(kb) == 0

    def test_repr_shows_size(self):
        kb = KnowledgeBase()
        assert "size=0" in repr(kb)

    def test_supported_formats_includes_pdf(self):
        kb = KnowledgeBase()
        assert ".pdf" in kb.supported_formats

    def test_supported_formats_includes_docx(self):
        kb = KnowledgeBase()
        assert ".docx" in kb.supported_formats

    def test_supported_formats_includes_txt(self):
        kb = KnowledgeBase()
        assert ".txt" in kb.supported_formats

    def test_supported_formats_includes_md(self):
        kb = KnowledgeBase()
        assert ".md" in kb.supported_formats


class TestKnowledgeBaseIngest:
    """Test document ingestion."""

    def test_ingest_txt_file(self, tmp_path):
        """Ingest a plain text file and verify chunks are created."""
        txt = tmp_path / "test.txt"
        txt.write_text(
            "First paragraph about neural networks.\n\n"
            "Second paragraph about machine learning."
        )
        kb = KnowledgeBase()
        n = kb.ingest(txt)
        assert n > 0
        assert kb.size > 0

    def test_ingest_returns_chunk_count(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Some test content for ingestion.")
        kb = KnowledgeBase()
        n = kb.ingest(txt)
        assert isinstance(n, int)
        assert n == kb.size

    def test_ingest_creates_document_id(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content for document ID test.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        assert "test" in kb.document_ids

    def test_ingest_same_file_twice_skips(self, tmp_path):
        """Second ingest of unchanged file should return 0 (skipped)."""
        txt = tmp_path / "test.txt"
        txt.write_text("Unchanged content.")
        kb = KnowledgeBase()
        first = kb.ingest(txt)
        second = kb.ingest(txt)
        assert first > 0
        assert second == 0

    def test_ingest_changed_file_re_ingests(self, tmp_path):
        """Re-ingesting a changed file should replace old chunks."""
        txt = tmp_path / "test.txt"
        txt.write_text("Original content.")
        kb = KnowledgeBase()
        first = kb.ingest(txt)
        assert first > 0

        txt.write_text("Updated content that is different.")
        second = kb.ingest(txt)
        assert second > 0

    def test_ingest_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "test.xyz"
        bad.write_text("fake")
        kb = KnowledgeBase()
        with pytest.raises(ValueError, match="No parser available"):
            kb.ingest(bad)

    def test_ingest_nonexistent_raises(self):
        kb = KnowledgeBase()
        with pytest.raises(FileNotFoundError):
            kb.ingest(Path("/nonexistent/file.txt"))

    def test_ingest_md_file(self, tmp_path):
        md = tmp_path / "notes.md"
        md.write_text("# Heading\n\nSome markdown content.")
        kb = KnowledgeBase()
        n = kb.ingest(md)
        assert n > 0


class TestKnowledgeBaseSearch:
    """Test search functionality."""

    def test_search_empty_index_returns_empty(self):
        kb = KnowledgeBase()
        results = kb.search("neural networks")
        assert results == []

    def test_search_returns_results(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text(
            "Neural networks are computational models inspired by "
            "the human brain. They consist of layers of neurons."
        )
        kb = KnowledgeBase()
        kb.ingest(txt)
        results = kb.search("neural networks", top_k=3)
        assert len(results) > 0

    def test_search_respects_top_k(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content about programming and software design.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        results = kb.search("programming", top_k=1)
        assert len(results) <= 1

    def test_search_results_have_score(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Object oriented programming uses classes.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        results = kb.search("programming", top_k=1)
        if results:
            assert results[0].score > 0


class TestKnowledgeBaseDocumentManagement:
    """Test document deletion and info."""

    def test_get_document_chunks_empty(self):
        kb = KnowledgeBase()
        assert kb.get_document_chunks("nonexistent") == []

    def test_get_document_chunks_returns_chunks(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content for chunk retrieval test.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        chunks = kb.get_document_chunks("test")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_delete_document_removes_chunks(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content to delete.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        assert kb.size > 0

        deleted = kb.delete_document("test")
        assert deleted > 0
        assert kb.size == 0
        assert "test" not in kb.document_ids

    def test_delete_nonexistent_returns_zero(self):
        kb = KnowledgeBase()
        assert kb.delete_document("ghost") == 0

    def test_clear_removes_everything(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content to clear.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        kb.clear()
        assert kb.size == 0
        assert kb.document_ids == []

    def test_get_document_info_returns_record(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content for info test.")
        kb = KnowledgeBase()
        kb.ingest(txt)
        info = kb.get_document_info("test")
        assert info is not None
        assert isinstance(info, DocumentRecord)
        assert info.filename == "test.txt"

    def test_get_document_info_nonexistent_returns_none(self):
        kb = KnowledgeBase()
        assert kb.get_document_info("missing") is None

    def test_list_documents_returns_all(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("Document A content.")
        b.write_text("Document B content.")
        kb = KnowledgeBase()
        kb.ingest(a)
        kb.ingest(b)
        records = kb.list_documents()
        assert len(records) == 2


class TestKnowledgeBasePersistence:
    """Test save and load."""

    def test_save_and_load_roundtrip(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content for persistence test.")
        index_path = tmp_path / "index" / "test"

        kb1 = KnowledgeBase(index_path=str(index_path))
        kb1.ingest(txt)
        original_size = kb1.size
        assert original_size > 0

        kb2 = KnowledgeBase(index_path=str(index_path))
        assert kb2.size == original_size

    def test_save_without_path_raises(self):
        kb = KnowledgeBase()
        with pytest.raises(ValueError, match="No save path"):
            kb.save()


class TestKnowledgeBaseDirectoryIngest:
    """Test directory ingestion."""

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Content A.")
        (tmp_path / "b.txt").write_text("Content B.")
        (tmp_path / "c.xyz").write_text("Unsupported format.")

        kb = KnowledgeBase()
        stats = kb.ingest_directory(tmp_path)

        assert stats["files_processed"] == 2
        assert stats["files_failed"] == 0
        assert stats["total_chunks"] > 0

    def test_ingest_directory_tracks_skipped(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Content.")

        kb = KnowledgeBase()
        kb.ingest_directory(tmp_path)  # first time
        stats = kb.ingest_directory(tmp_path)  # second time — should skip
        assert stats["files_skipped"] == 1
        assert stats["files_processed"] == 0