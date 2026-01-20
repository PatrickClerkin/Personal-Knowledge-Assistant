"""Unit tests for chunking strategies."""

import pytest
from src.ingestion.document import Document, DocumentMetadata, DocumentSection
from src.ingestion.chunking import ChunkManager, Chunk
from pathlib import Path
from datetime import datetime


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    metadata = DocumentMetadata(
        doc_id="test_doc_001",
        title="Test Document",
        source_path=Path("/fake/path/test.pdf"),
        file_type="pdf",
        created_date=datetime.now(),
    )

    content = """This is the first section of the document. It contains some text about programming.

Object-oriented programming uses concepts like inheritance and composition.
Composition is when objects contain other objects as members.
This is a fundamental concept in software design.

This is the second section. It talks about different topics.
Machine learning is a subset of artificial intelligence.
Neural networks can learn patterns from data.
Deep learning uses multiple layers of neural networks."""

    sections = [
        DocumentSection(
            section_id="page_1",
            title="Page 1",
            content=content[:300],
            level=1,
            page_number=1,
            start_char=0,
            end_char=300,
        ),
        DocumentSection(
            section_id="page_2",
            title="Page 2",
            content=content[300:],
            level=1,
            page_number=2,
            start_char=300,
            end_char=len(content),
        ),
    ]

    return Document(metadata=metadata, content=content, sections=sections)


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_creates_chunks(self, sample_document):
        """Test that fixed-size chunker creates chunks."""
        chunker = ChunkManager(strategy="fixed", chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_respects_chunk_size(self, sample_document):
        """Test that chunks don't exceed the max size significantly."""
        chunk_size = 200
        chunker = ChunkManager(strategy="fixed", chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_document)

        # Allow some tolerance for word boundaries
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size + 50

    def test_assigns_chunk_metadata(self, sample_document):
        """Test that chunks have proper metadata."""
        chunker = ChunkManager(strategy="fixed", chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_document)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id is not None
            assert chunk.doc_id == sample_document.metadata.doc_id
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_creates_chunks(self, sample_document):
        """Test that sentence chunker creates chunks."""
        chunker = ChunkManager(strategy="sentence", chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0

    def test_preserves_sentences(self, sample_document):
        """Test that sentence boundaries are generally preserved."""
        chunker = ChunkManager(strategy="sentence", chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(sample_document)

        # Check that chunks tend to end with sentence-ending punctuation
        for chunk in chunks[:-1]:  # Last chunk might not end with punctuation
            content = chunk.content.strip()
            # Most chunks should end with proper punctuation
            if len(content) > 20:
                assert any(content.endswith(p) for p in ['.', '!', '?', '"', "'"])


class TestSemanticChunker:
    """Tests for SemanticChunker (structure-based)."""

    def test_creates_chunks(self, sample_document):
        """Test that semantic chunker creates chunks."""
        chunker = ChunkManager(strategy="semantic", chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0

    def test_respects_section_boundaries(self, sample_document):
        """Test that chunks include section metadata."""
        chunker = ChunkManager(strategy="semantic", chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(sample_document)

        # Each chunk should have page/section info
        for chunk in chunks:
            assert chunk.section_id is not None or chunk.page_number is not None


class TestChunkManager:
    """Tests for ChunkManager."""

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkManager(strategy="invalid_strategy")

    def test_chunk_multiple_documents(self, sample_document):
        """Test chunking multiple documents."""
        chunker = ChunkManager(strategy="fixed", chunk_size=200)
        documents = [sample_document, sample_document]

        all_chunks = chunker.chunk_documents(documents)

        # Should have chunks from both documents
        assert len(all_chunks) > len(chunker.chunk_document(sample_document))
