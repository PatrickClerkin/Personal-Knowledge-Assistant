"""Unit tests for the embedding service."""

import pytest
import numpy as np
from src.ingestion.embeddings import EmbeddingService
from src.ingestion.chunking import Chunk


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def embedding_service(self):
        """Create an embedding service instance."""
        return EmbeddingService()

    def test_embed_text_returns_array(self, embedding_service):
        """Test that embed_text returns a numpy array."""
        embedding = embedding_service.embed_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1

    def test_embedding_dimension(self, embedding_service):
        """Test that embeddings have the expected dimension."""
        embedding = embedding_service.embed_text("Test sentence")

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert embedding.shape[0] == 384
        assert embedding_service.embedding_dimension == 384

    def test_embed_multiple_texts(self, embedding_service):
        """Test embedding multiple texts at once."""
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = embedding_service.embed_texts(texts)

        assert embeddings.shape == (3, 384)

    def test_similar_texts_have_high_similarity(self, embedding_service):
        """Test that semantically similar texts have high cosine similarity."""
        text1 = "Object-oriented programming uses classes and objects"
        text2 = "OOP relies on classes to structure code"
        text3 = "The weather is sunny today"

        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        emb3 = embedding_service.embed_text(text3)

        sim_related = embedding_service.compute_similarity(emb1, emb2)
        sim_unrelated = embedding_service.compute_similarity(emb1, emb3)

        # Related texts should have higher similarity
        assert sim_related > sim_unrelated
        assert sim_related > 0.3  # Should be reasonably high

    def test_embed_chunks(self, embedding_service):
        """Test embedding a list of chunks."""
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                content="This is the first chunk about programming.",
                source_doc_title="Test Doc",
            ),
            Chunk(
                chunk_id="chunk_2",
                doc_id="doc_1",
                content="This is the second chunk about data science.",
                source_doc_title="Test Doc",
            ),
        ]

        embeddings = embedding_service.embed_chunks(chunks, store_in_chunks=True)

        assert embeddings.shape == (2, 384)
        assert chunks[0].has_embedding
        assert chunks[1].has_embedding

    def test_find_similar(self, embedding_service):
        """Test finding similar embeddings."""
        texts = [
            "Python is a programming language",
            "Java is also a programming language",
            "The cat sat on the mat",
            "Dogs are popular pets",
        ]
        embeddings = embedding_service.embed_texts(texts)

        query = embedding_service.embed_text("What programming languages exist?")
        results = embedding_service.find_similar(query, embeddings, top_k=2)

        # Top results should be the programming-related texts (indices 0 and 1)
        top_indices = [idx for idx, score in results]
        assert 0 in top_indices
        assert 1 in top_indices
