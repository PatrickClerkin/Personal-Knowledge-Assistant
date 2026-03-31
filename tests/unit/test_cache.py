"""Unit tests for SemanticCache."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.rag.cache import SemanticCache, CacheEntry


def _make_response(answer: str = "Test answer"):
    """Build a minimal mock RAGResponse."""
    response = MagicMock()
    response.answer = answer
    response.cache_hit = False
    return response


def _make_cache(threshold: float = 0.92) -> SemanticCache:
    """Build a SemanticCache with a mocked embedding service."""
    cache = SemanticCache(threshold=threshold)
    # Replace embedder with a mock that returns deterministic vectors
    cache._embedder = MagicMock()
    return cache


def _set_embed(cache: SemanticCache, vectors: dict):
    """Configure mock embedder to return specific vectors per query."""
    def embed_side_effect(text):
        for key, vec in vectors.items():
            if key in text:
                return np.array(vec, dtype=np.float32)
        return np.array([0.5, 0.5, 0.0], dtype=np.float32)
    cache._embedder.embed_text.side_effect = embed_side_effect


class TestSemanticCacheInit:

    def test_default_threshold(self):
        cache = _make_cache()
        assert cache.threshold == 0.92

    def test_custom_threshold(self):
        cache = SemanticCache.__new__(SemanticCache)
        cache.threshold = 0.85
        assert cache.threshold == 0.85

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=0.0)

    def test_invalid_max_size_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=0.9, max_size=0)

    def test_starts_empty(self):
        cache = _make_cache()
        assert cache.size == 0


class TestSemanticCacheGetStore:

    def test_miss_on_empty_cache(self):
        cache = _make_cache()
        result = cache.get("what is backpropagation?")
        assert result is None

    def test_exact_match_is_hit(self):
        cache = _make_cache(threshold=0.90)
        vec = [1.0, 0.0, 0.0]
        _set_embed(cache, {"backpropagation": vec})

        response = _make_response("Backpropagation answer")
        cache.store("what is backpropagation?", response)
        result = cache.get("backpropagation")
        assert result is not None
        assert result.answer == "Backpropagation answer"

    def test_dissimilar_query_is_miss(self):
        cache = _make_cache(threshold=0.92)
        _set_embed(cache, {
            "backpropagation": [1.0, 0.0, 0.0],
            "neural": [0.0, 1.0, 0.0],
        })

        cache.store("what is backpropagation?", _make_response("BP answer"))
        result = cache.get("neural network layers")
        assert result is None

    def test_store_increases_size(self):
        cache = _make_cache()
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        cache.store("query one", _make_response())
        assert cache.size == 1

    def test_hit_increments_hit_count(self):
        cache = _make_cache(threshold=0.5)
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        cache.store("test query", _make_response())
        cache.get("test query")
        assert cache._entries[0].hits == 1

    def test_clear_empties_cache(self):
        cache = _make_cache()
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        cache.store("query", _make_response())
        cache.clear()
        assert cache.size == 0

    def test_invalidate_removes_entry(self):
        cache = _make_cache()
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        cache.store("exact query", _make_response())
        removed = cache.invalidate("exact query")
        assert removed is True
        assert cache.size == 0

    def test_invalidate_nonexistent_returns_false(self):
        cache = _make_cache()
        assert cache.invalidate("not in cache") is False


class TestSemanticCacheEviction:

    def test_evicts_lru_when_full(self):
        cache = _make_cache(threshold=0.99)
        cache.max_size = 2
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])

        cache.store("first query", _make_response("first"))
        cache.store("second query", _make_response("second"))
        cache.store("third query", _make_response("third"))

        assert cache.size == 2
        queries = [e.query for e in cache._entries]
        assert "first query" not in queries

    def test_total_hits_counts_across_entries(self):
        cache = _make_cache(threshold=0.5)
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        cache.store("query", _make_response())
        cache.get("query")
        cache.get("query")
        assert cache.total_hits == 2


class TestSemanticCacheStats:

    def test_stats_returns_dict(self):
        cache = _make_cache()
        stats = cache.stats
        assert "size" in stats
        assert "threshold" in stats
        assert "total_hits" in stats
        assert "entries" in stats

    def test_stats_size_matches(self):
        cache = _make_cache()
        cache._embedder.embed_text.return_value = np.array([1.0, 0.0, 0.0])
        cache.store("q", _make_response())
        assert cache.stats["size"] == 1
