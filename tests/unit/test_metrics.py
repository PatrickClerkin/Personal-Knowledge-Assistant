"""Unit tests for IR evaluation metrics."""

import pytest
from src.evaluation.metrics import precision_at_k, mean_reciprocal_rank, ndcg_at_k


RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT  = {"a", "c"}


class TestPrecisionAtK:

    def test_all_relevant(self):
        assert precision_at_k(["a", "c"], {"a", "c"}, k=2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_partial_overlap(self):
        assert precision_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(2 / 5)

    def test_k_larger_than_retrieved(self):
        # Pads with misses — a and c in top 2, then 3 misses
        assert precision_at_k(["a", "c"], RELEVANT, k=5) == pytest.approx(2 / 5)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            precision_at_k(RETRIEVED, RELEVANT, k=0)


class TestMeanReciprocalRank:

    def test_first_result_relevant(self):
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mean_reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_no_relevant_result(self):
        assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], {"a"}) == 0.0


class TestNdcgAtK:

    def test_perfect_ranking(self):
        # Both relevant docs at top
        score = ndcg_at_k(["a", "c", "x", "y", "z"], {"a", "c"}, k=5)
        assert score == pytest.approx(1.0)

    def test_no_relevant_retrieved(self):
        assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_partial_ranking(self):
        # Relevant doc at rank 3 instead of rank 1 — score < 1
        score = ndcg_at_k(["x", "y", "a", "z", "w"], {"a"}, k=5)
        assert 0.0 < score < 1.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            ndcg_at_k(RETRIEVED, RELEVANT, k=0)

    def test_empty_relevant_set(self):
        assert ndcg_at_k(RETRIEVED, set(), k=5) == 0.0