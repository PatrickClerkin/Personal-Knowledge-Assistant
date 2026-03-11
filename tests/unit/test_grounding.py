"""Unit tests for GroundingScorer."""

import pytest
from src.rag.grounding import GroundingScorer, GroundingResult, ChunkGroundingScore
from unittest.mock import MagicMock


def _make_result(content: str, title: str = "doc.txt", score: float = 0.8):
    """Helper to build a mock SearchResult."""
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = title
    chunk.page_number = None
    result = MagicMock()
    result.chunk = chunk
    result.score = score
    return result


class TestGroundingScorer:

    def test_well_grounded_answer(self):
        scorer = GroundingScorer(threshold=0.2)
        answer = "Dependency injection allows classes to receive dependencies from outside."
        sources = [_make_result(
            "Dependency injection is a technique where classes receive their "
            "dependencies from external sources rather than creating them."
        )]
        result = scorer.score(answer, sources)
        assert result.is_well_grounded
        assert result.overall_confidence > 0.0
        assert len(result.chunk_scores) == 1
        assert result.chunk_scores[0].is_grounded

    def test_ungrounded_answer(self):
        scorer = GroundingScorer(threshold=0.5)
        answer = "Quantum entanglement enables faster than light communication."
        sources = [_make_result(
            "Python is a high level programming language with simple syntax."
        )]
        result = scorer.score(answer, sources)
        assert not result.is_well_grounded
        assert result.overall_confidence == 0.0

    def test_matched_terms_nonempty_when_grounded(self):
        scorer = GroundingScorer(threshold=0.1)
        answer = "Neural networks learn through backpropagation training."
        sources = [_make_result(
            "Neural networks are trained using backpropagation algorithms."
        )]
        result = scorer.score(answer, sources)
        assert len(result.chunk_scores[0].matched_terms) > 0

    def test_scores_sorted_descending(self):
        scorer = GroundingScorer(threshold=0.1)
        answer = "Python supports object oriented programming with classes."
        sources = [
            _make_result("Java uses object oriented class hierarchies."),
            _make_result("Python classes support object oriented programming features."),
        ]
        result = scorer.score(answer, sources)
        scores = [c.grounding_score for c in result.chunk_scores]
        assert scores == sorted(scores, reverse=True)

    def test_empty_sources_returns_no_scores(self):
        scorer = GroundingScorer()
        result = scorer.score("Some answer text here.", [])
        assert result.chunk_scores == []
        assert result.overall_confidence == 0.0
        assert not result.is_well_grounded

    def test_answer_terms_excludes_stopwords(self):
        scorer = GroundingScorer()
        result = scorer.score("This is a test of the system.", [])
        # "This", "is", "a", "of", "the" are stopwords; "test" len=4 ok, "system" ok
        assert "this" not in result.answer_terms
        assert "the" not in result.answer_terms

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            GroundingScorer(threshold=0.0)
        with pytest.raises(ValueError):
            GroundingScorer(threshold=1.0)

    def test_multiple_chunks_confidence_is_mean_of_grounded(self):
        scorer = GroundingScorer(threshold=0.1)
        answer = "Machine learning models require training data and validation."
        sources = [
            _make_result("Machine learning models need large training datasets."),
            _make_result("The weather today is sunny with light winds."),
        ]
        result = scorer.score(answer, sources)
        grounded = [c for c in result.chunk_scores if c.is_grounded]
        if grounded:
            expected = round(
                sum(c.grounding_score for c in grounded) / len(grounded), 4
            )
            assert result.overall_confidence == expected