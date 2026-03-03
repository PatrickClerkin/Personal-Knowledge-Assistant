"""Unit tests for BM25Store and Hybrid Search (RRF)."""

import pytest
import numpy as np
from src.ingestion.chunking.chunk import Chunk
from src.ingestion.storage.vector_store import SearchResult


def _chunk(chunk_id, doc_id="doc_1", content="test content"):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, content=content, source_doc_title="Test")


def _result(chunk_id, score, rank, content="test"):
    return SearchResult(chunk=_chunk(chunk_id, content=content), score=score, rank=rank)


# ── BM25Store ──────────────────────────────────────────────────────────

class TestBM25Store:

    @pytest.fixture
    def store(self):
        from src.ingestion.storage.bm25_store import BM25Store
        return BM25Store()

    @pytest.fixture
    def populated(self, store):
        chunks = [
            _chunk("c1", content="Python is a programming language used for machine learning"),
            _chunk("c2", content="Java is an object oriented programming language"),
            _chunk("c3", content="The cat sat on the mat"),
            _chunk("c4", content="Neural networks learn patterns from data"),
            _chunk("c5", content="Python decorators and metaclasses are advanced features"),
        ]
        store.add(chunks)
        return store

    def test_add_increases_size(self, store):
        store.add([_chunk("c1", content="hello world")])
        assert store.size == 1

    def test_search_returns_results(self, populated):
        results = populated.search("Python programming", top_k=3)
        assert len(results) > 0

    def test_keyword_relevance(self, populated):
        results = populated.search("Python programming language", top_k=5)
        ids = [r.chunk.chunk_id for r in results]
        assert "c1" in ids[:2] or "c5" in ids[:2]

    def test_respects_top_k(self, populated):
        results = populated.search("programming", top_k=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self, populated):
        results = populated.search("xyzzy quux frobnicate", top_k=5)
        assert results == []

    def test_filter_by_doc_id(self, store):
        store.add([_chunk("c1", doc_id="doc_1", content="Python programming language")])
        store.add([_chunk("c2", doc_id="doc_2", content="Python scripting and automation")])
        results = store.search("Python", top_k=5, filter_doc_id="doc_1")
        assert all(r.chunk.doc_id == "doc_1" for r in results)

    def test_delete_document(self, populated):
        deleted = populated.delete_document("doc_1")
        assert deleted == 5
        assert populated.size == 0

    def test_delete_nonexistent(self, populated):
        deleted = populated.delete_document("nonexistent")
        assert deleted == 0
        assert populated.size == 5

    def test_clear(self, populated):
        populated.clear()
        assert populated.size == 0

    def test_results_sorted_by_score(self, populated):
        results = populated.search("Python programming", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_have_rank(self, populated):
        results = populated.search("Python", top_k=3)
        for i, r in enumerate(results, 1):
            assert r.rank == i

    def test_save_and_load(self, populated, tmp_path):
        from src.ingestion.storage.bm25_store import BM25Store
        path = str(tmp_path / "test_index")
        populated.save(path)
        new_store = BM25Store()
        new_store.load(path)
        assert new_store.size == populated.size
        results = new_store.search("Python", top_k=3)
        assert len(results) > 0

    def test_empty_store_returns_empty(self, store):
        assert store.search("anything", top_k=5) == []

    def test_add_empty_list(self, store):
        store.add([])
        assert store.size == 0

    def test_get_document_ids(self, store):
        store.add([_chunk("c1", doc_id="doc_a", content="text one")])
        store.add([_chunk("c2", doc_id="doc_b", content="text two")])
        ids = store.get_document_ids()
        assert "doc_a" in ids
        assert "doc_b" in ids


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────

class TestRRF:

    def test_single_list(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        results = [_result("c1", 0.9, 1), _result("c2", 0.7, 2), _result("c3", 0.5, 3)]
        merged = reciprocal_rank_fusion(results, top_k=3)
        assert len(merged) == 3

    def test_two_lists_merge(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        faiss = [_result("c1", 0.9, 1), _result("c2", 0.7, 2)]
        bm25 = [_result("c3", 5.0, 1), _result("c1", 3.0, 2)]
        merged = reciprocal_rank_fusion(faiss, bm25, top_k=3)
        # c1 appears in both lists — should rank highest
        assert merged[0].chunk.chunk_id == "c1"

    def test_respects_top_k(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        results = [_result(f"c{i}", 1.0 / i, i) for i in range(1, 11)]
        merged = reciprocal_rank_fusion(results, top_k=3)
        assert len(merged) == 3

    def test_scores_decrease(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        faiss = [_result(f"c{i}", 1.0 / i, i) for i in range(1, 6)]
        bm25 = [_result(f"c{i}", 5.0 / i, i) for i in range(1, 6)]
        merged = reciprocal_rank_fusion(faiss, bm25, top_k=5)
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_empty_lists(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([], [], top_k=5) == []

    def test_sequential_ranks(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        results = [_result(f"c{i}", 1.0 / i, i) for i in range(1, 4)]
        merged = reciprocal_rank_fusion(results, top_k=3)
        for i, r in enumerate(merged, 1):
            assert r.rank == i

    def test_document_in_both_lists_scores_higher(self):
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        # c_shared appears rank-1 in both lists; c_solo only in one
        faiss = [_result("c_shared", 0.9, 1), _result("c_solo", 0.8, 2)]
        bm25 = [_result("c_shared", 5.0, 1), _result("c_other", 4.0, 2)]
        merged = reciprocal_rank_fusion(faiss, bm25, top_k=3)
        assert merged[0].chunk.chunk_id == "c_shared"


# ── HybridSearcher integration ─────────────────────────────────────────

class TestHybridSearcher:

    @pytest.fixture
    def searcher_and_embedder(self):
        from src.ingestion.storage.bm25_store import BM25Store
        from src.ingestion.storage.faiss_store import FAISSVectorStore
        from src.retrieval.hybrid_search import HybridSearcher
        from src.ingestion.embeddings.embedding_service import EmbeddingService

        embedder = EmbeddingService()
        faiss = FAISSVectorStore(embedding_dim=384)
        bm25 = BM25Store()

        chunks = [
            _chunk("c1", content="Python is a programming language for machine learning"),
            _chunk("c2", content="Java is an object oriented programming language"),
            _chunk("c3", content="The cat sat on the mat"),
        ]
        embeddings = embedder.embed_texts([c.content for c in chunks])
        faiss.add(chunks, embeddings)
        bm25.add(chunks)

        return HybridSearcher(faiss, bm25), embedder

    def test_returns_results(self, searcher_and_embedder):
        searcher, embedder = searcher_and_embedder
        emb = embedder.embed_text("Python programming")
        results = searcher.search("Python programming", emb, top_k=3)
        assert len(results) > 0

    def test_respects_top_k(self, searcher_and_embedder):
        searcher, embedder = searcher_and_embedder
        emb = embedder.embed_text("programming")
        results = searcher.search("programming", emb, top_k=2)
        assert len(results) <= 2

    def test_sequential_ranks(self, searcher_and_embedder):
        searcher, embedder = searcher_and_embedder
        emb = embedder.embed_text("Python")
        results = searcher.search("Python", emb, top_k=3)
        for i, r in enumerate(results, 1):
            assert r.rank == i