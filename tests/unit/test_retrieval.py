"""Tests for the retrieval quality module: query expansion and evaluation."""

import pytest
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.evaluation import (
    RetrievalEvaluator,
    RelevanceJudgment,
    EvaluationResult,
)
from src.ingestion.storage.vector_store import SearchResult
from src.ingestion.chunking.chunk import Chunk


# ─── Fixtures ──────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, content: str = "test content") -> Chunk:
    """Helper to create a Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        content=content,
        source_doc_title="Test Doc",
    )


def _make_result(chunk_id: str, score: float, rank: int) -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(
        chunk=_make_chunk(chunk_id),
        score=score,
        rank=rank,
    )


# ─── Query Expansion Tests ─────────────────────────────────────────

class TestQueryExpanderInit:
    """Test QueryExpander initialisation."""

    def test_valid_strategy(self):
        for strategy in QueryExpander.STRATEGIES:
            expander = QueryExpander(strategy=strategy)
            assert expander.strategy == strategy

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            QueryExpander(strategy="nonexistent")


class TestSynonymExpansion:
    """Test synonym-based query expansion."""

    def test_returns_original_query(self):
        expander = QueryExpander(strategy="synonym")
        results = expander.expand("dependency injection")
        assert results[0] == "dependency injection"

    def test_returns_multiple_variants(self):
        expander = QueryExpander(strategy="synonym")
        results = expander.expand("software design patterns")
        assert len(results) >= 1  # At least the original

    def test_respects_n_expansions(self):
        expander = QueryExpander(strategy="synonym")
        results = expander.expand("testing strategies", n_expansions=2)
        assert len(results) <= 3  # Original + up to 2 expansions


class TestMultiQueryExpansion:
    """Test multi-query expansion."""

    def test_returns_original_query(self):
        expander = QueryExpander(strategy="multi_query")
        results = expander.expand("What is dependency injection?")
        assert results[0] == "What is dependency injection?"

    def test_generates_variants_from_question(self):
        expander = QueryExpander(strategy="multi_query")
        results = expander.expand("What is dependency injection?")
        assert len(results) >= 2

    def test_extracts_key_terms(self):
        expander = QueryExpander(strategy="multi_query")
        results = expander.expand("How does garbage collection work in Python?")
        # Should have at least a key-terms variant
        assert any("garbage" in r and "collection" in r for r in results)


class TestHyDEExpansion:
    """Test Hypothetical Document Embeddings expansion."""

    def test_returns_original_and_hypothetical(self):
        expander = QueryExpander(strategy="hyde")
        results = expander.expand("What is SOLID?")
        assert len(results) == 2
        assert results[0] == "What is SOLID?"
        assert "explanation" in results[1].lower()


# ─── Evaluation Tests ──────────────────────────────────────────────

class TestEvaluationMetrics:
    """Test individual IR metrics."""

    def test_perfect_precision(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c1", "c2", "c3"}
        )
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
            _make_result("c3", 0.7, 3),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=3)
        assert ev.precision_at_k == 1.0

    def test_zero_precision(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c10", "c11"}
        )
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=2)
        assert ev.precision_at_k == 0.0

    def test_partial_precision(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c1", "c3"}
        )
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
            _make_result("c3", 0.7, 3),
            _make_result("c4", 0.6, 4),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=4)
        assert ev.precision_at_k == 0.5

    def test_recall(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c1", "c2", "c3", "c4"}
        )
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=2)
        assert ev.recall_at_k == 0.5

    def test_reciprocal_rank_first(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c1"}
        )
        results = [_make_result("c1", 0.9, 1)]
        ev = evaluator.evaluate_query(results, judgment, k=1)
        assert ev.reciprocal_rank == 1.0

    def test_reciprocal_rank_third(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c3"}
        )
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
            _make_result("c3", 0.7, 3),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=3)
        assert abs(ev.reciprocal_rank - 1.0 / 3) < 0.001

    def test_reciprocal_rank_no_hit(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c10"}
        )
        results = [_make_result("c1", 0.9, 1)]
        ev = evaluator.evaluate_query(results, judgment, k=1)
        assert ev.reciprocal_rank == 0.0

    def test_ndcg_perfect(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test",
            relevant_ids={"c1", "c2"},
            relevance_scores={"c1": 3, "c2": 2},
        )
        # Perfect ordering: highest relevance first
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=2)
        assert ev.ndcg_at_k == 1.0

    def test_average_precision(self):
        evaluator = RetrievalEvaluator()
        judgment = RelevanceJudgment(
            query="test", relevant_ids={"c1", "c3"}
        )
        results = [
            _make_result("c1", 0.9, 1),
            _make_result("c2", 0.8, 2),
            _make_result("c3", 0.7, 3),
        ]
        ev = evaluator.evaluate_query(results, judgment, k=3)
        # AP = (1/1 + 2/3) / 2 = 0.8333
        assert abs(ev.average_precision - 0.8333) < 0.01


class TestEvaluatorJudgments:
    """Test judgment management."""

    def test_add_judgment(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_judgment("test query", {"c1", "c2"})
        assert len(evaluator.judgments) == 1
        assert evaluator.judgments[0].query == "test query"

    def test_save_and_load_judgments(self, tmp_path):
        evaluator = RetrievalEvaluator()
        evaluator.add_judgment("query 1", {"c1", "c2"})
        evaluator.add_judgment("query 2", {"c3"}, {"c3": 2.0})

        path = tmp_path / "judgments.json"
        evaluator.save_judgments(path)

        loaded = RetrievalEvaluator()
        count = loaded.load_judgments(path)
        assert count == 2
        assert loaded.judgments[0].query == "query 1"
        assert "c3" in loaded.judgments[1].relevant_ids

    def test_empty_evaluation(self):
        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(None, top_k=5)
        assert result["queries"] == []
