"""Unit tests for vector storage."""

import pytest
import numpy as np
from src.ingestion.storage import FAISSVectorStore
from src.ingestion.chunking import Chunk


@pytest.fixture
def vector_store():
    """Create a FAISS vector store instance."""
    return FAISSVectorStore(embedding_dim=384)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id=f"chunk_{i}",
            doc_id="doc_1",
            content=f"This is chunk number {i}",
            source_doc_title="Test Document",
            page_number=1,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def test_add_chunks(self, vector_store, sample_chunks, sample_embeddings):
        """Test adding chunks to the store."""
        vector_store.add(sample_chunks, sample_embeddings)

        assert vector_store.size == 5

    def test_search_returns_results(self, vector_store, sample_chunks, sample_embeddings):
        """Test that search returns results."""
        vector_store.add(sample_chunks, sample_embeddings)

        query = np.random.randn(384).astype(np.float32)
        results = vector_store.search(query, top_k=3)

        assert len(results) == 3
        assert all(hasattr(r, 'chunk') for r in results)
        assert all(hasattr(r, 'score') for r in results)

    def test_search_respects_top_k(self, vector_store, sample_chunks, sample_embeddings):
        """Test that search returns correct number of results."""
        vector_store.add(sample_chunks, sample_embeddings)

        results = vector_store.search(sample_embeddings[0], top_k=2)
        assert len(results) == 2

    def test_search_results_ordered_by_score(
        self, vector_store, sample_chunks, sample_embeddings
    ):
        """Test that search results are ordered by similarity."""
        vector_store.add(sample_chunks, sample_embeddings)

        results = vector_store.search(sample_embeddings[0], top_k=5)
        scores = [r.score for r in results]

        assert scores == sorted(scores, reverse=True)

    def test_search_with_document_filter(self, vector_store, sample_embeddings):
        """Test filtering search results by document ID."""
        chunks_doc1 = [
            Chunk(chunk_id=f"d1_chunk_{i}", doc_id="doc_1", content=f"Doc 1 chunk {i}",
                  source_doc_title="Doc 1")
            for i in range(3)
        ]
        chunks_doc2 = [
            Chunk(chunk_id=f"d2_chunk_{i}", doc_id="doc_2", content=f"Doc 2 chunk {i}",
                  source_doc_title="Doc 2")
            for i in range(2)
        ]

        all_chunks = chunks_doc1 + chunks_doc2
        vector_store.add(all_chunks, sample_embeddings)

        results = vector_store.search(
            sample_embeddings[0], top_k=10, filter_doc_id="doc_1"
        )

        assert all(r.chunk.doc_id == "doc_1" for r in results)

    def test_delete_document(self, vector_store, sample_embeddings):
        """Test deleting all chunks from a document."""
        chunks_doc1 = [
            Chunk(chunk_id=f"d1_chunk_{i}", doc_id="doc_1", content=f"Doc 1 chunk {i}",
                  source_doc_title="Doc 1")
            for i in range(3)
        ]
        chunks_doc2 = [
            Chunk(chunk_id=f"d2_chunk_{i}", doc_id="doc_2", content=f"Doc 2 chunk {i}",
                  source_doc_title="Doc 2")
            for i in range(2)
        ]

        all_chunks = chunks_doc1 + chunks_doc2
        vector_store.add(all_chunks, sample_embeddings)
        assert vector_store.size == 5

        deleted = vector_store.delete_document("doc_1")

        assert deleted == 3
        assert vector_store.size == 2
        assert "doc_1" not in vector_store.get_document_ids()

    def test_save_and_load(self, vector_store, sample_chunks, sample_embeddings, tmp_path):
        """Test saving and loading the vector store."""
        vector_store.add(sample_chunks, sample_embeddings)

        save_path = tmp_path / "test_index"
        vector_store.save(str(save_path))

        # Create new store and load
        new_store = FAISSVectorStore(embedding_dim=384)
        new_store.load(str(save_path))

        assert new_store.size == vector_store.size
        assert len(new_store.get_all_chunks()) == len(sample_chunks)

    def test_clear(self, vector_store, sample_chunks, sample_embeddings):
        """Test clearing the vector store."""
        vector_store.add(sample_chunks, sample_embeddings)
        assert vector_store.size == 5

        vector_store.clear()

        assert vector_store.size == 0
        assert len(vector_store.get_all_chunks()) == 0

    def test_get_chunk_by_id(self, vector_store, sample_chunks, sample_embeddings):
        """Test retrieving a chunk by its ID."""
        vector_store.add(sample_chunks, sample_embeddings)

        chunk = vector_store.get_chunk_by_id("chunk_2")

        assert chunk is not None
        assert chunk.chunk_id == "chunk_2"

    def test_empty_store_search(self, vector_store):
        """Test searching an empty store."""
        query = np.random.randn(384).astype(np.float32)
        results = vector_store.search(query, top_k=5)

        assert results == []
