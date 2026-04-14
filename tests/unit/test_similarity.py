"""Unit tests for DocumentSimilarity."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.similarity import DocumentSimilarity, SimilarityResult, SimilarityMatrix


def _make_chunk(content: str, source: str = "doc_a", doc_id: str = "doc_a"):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    chunk.doc_id = doc_id
    chunk.embedding = None
    return chunk


def _make_kb(doc_map: dict):
    """doc_map: {doc_id: [chunk_content, ...]}"""
    kb = MagicMock()
    kb.document_ids = list(doc_map.keys())

    def get_chunks(doc_id):
        contents = doc_map.get(doc_id, [])
        return [_make_chunk(c, source=doc_id, doc_id=doc_id) for c in contents]

    kb.get_document_chunks.side_effect = get_chunks
    return kb


def _make_embedder(dim: int = 8):
    """Returns an embedder that gives deterministic embeddings per text."""
    embedder = MagicMock()
    def embed_text(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.random(dim).astype(np.float32)
        return v / np.linalg.norm(v)
    embedder.embed_text.side_effect = embed_text
    return embedder


def _make_similarity(doc_map=None):
    doc_map = doc_map or {
        "doc_a": ["Neural networks learn from data.", "They use backpropagation."],
        "doc_b": ["Deep learning is a subset of ML.", "CNNs are used for images."],
        "doc_c": ["Cooking pasta requires boiling water.", "Add salt to the water."],
    }
    kb = _make_kb(doc_map)
    embedder = _make_embedder()
    return DocumentSimilarity(knowledge_base=kb, embedder=embedder)


class TestDocumentSimilarity:

    def test_find_similar_returns_list(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a")
        assert isinstance(results, list)

    def test_find_similar_excludes_self(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a", exclude_self=True)
        assert all(r.doc_id != "doc_a" for r in results)

    def test_find_similar_includes_self_when_not_excluded(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a", exclude_self=False)
        ids = [r.doc_id for r in results]
        assert "doc_a" in ids

    def test_find_similar_sorted_descending(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a")
        scores = [r.similarity for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_respects_top_k(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a", top_k=1)
        assert len(results) <= 1

    def test_find_similar_missing_doc_returns_empty(self):
        ds = _make_similarity()
        results = ds.find_similar("nonexistent_doc")
        assert results == []

    def test_result_has_required_fields(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a")
        if results:
            r = results[0]
            assert hasattr(r, "doc_id")
            assert hasattr(r, "title")
            assert hasattr(r, "similarity")
            assert hasattr(r, "chunk_count")

    def test_similarity_score_between_zero_and_one(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a")
        for r in results:
            assert 0.0 <= r.similarity <= 1.0

    def test_to_dict_has_required_keys(self):
        ds = _make_similarity()
        results = ds.find_similar("doc_a")
        if results:
            d = results[0].to_dict()
            assert "doc_id" in d
            assert "title" in d
            assert "similarity" in d
            assert "chunk_count" in d

    def test_find_similar_to_query_returns_list(self):
        ds = _make_similarity()
        results = ds.find_similar_to_query("neural networks")
        assert isinstance(results, list)

    def test_find_similar_to_query_sorted_descending(self):
        ds = _make_similarity()
        results = ds.find_similar_to_query("machine learning")
        scores = [r.similarity for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_compute_matrix_returns_matrix(self):
        ds = _make_similarity()
        matrix = ds.compute_matrix()
        assert isinstance(matrix, SimilarityMatrix)

    def test_matrix_diagonal_is_one(self):
        ds = _make_similarity()
        matrix = ds.compute_matrix()
        for i in range(len(matrix.doc_ids)):
            assert matrix.matrix[i][i] == 1.0

    def test_matrix_is_symmetric(self):
        ds = _make_similarity()
        matrix = ds.compute_matrix()
        n = len(matrix.doc_ids)
        for i in range(n):
            for j in range(n):
                assert abs(matrix.matrix[i][j] - matrix.matrix[j][i]) < 1e-6

    def test_matrix_to_dict_has_required_keys(self):
        ds = _make_similarity()
        matrix = ds.compute_matrix()
        d = matrix.to_dict()
        assert "documents" in d
        assert "matrix" in d
        assert "doc_ids" in d

    def test_empty_kb_returns_empty_matrix(self):
        kb = MagicMock()
        kb.document_ids = []
        kb.get_document_chunks.return_value = []
        embedder = _make_embedder()
        ds = DocumentSimilarity(knowledge_base=kb, embedder=embedder)
        matrix = ds.compute_matrix()
        assert matrix.matrix == []
        assert matrix.doc_ids == []

    def test_cosine_similarity_identical_vectors(self):
        ds = _make_similarity()
        v = np.array([1.0, 0.0, 0.0])
        assert abs(ds._cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        ds = _make_similarity()
        v = np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, 0.0])
        assert ds._cosine_similarity(v, z) == 0.0

    def test_get_document_embedding_returns_array(self):
        ds = _make_similarity()
        emb = ds.get_document_embedding("doc_a")
        assert emb is not None
        assert isinstance(emb, np.ndarray)

    def test_get_document_embedding_missing_doc_returns_none(self):
        ds = _make_similarity()
        emb = ds.get_document_embedding("nonexistent")
        assert emb is None