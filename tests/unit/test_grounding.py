"""Tests for the embedding-based GroundingScorer."""

import numpy as np
import pytest

from src.rag.grounding import (
    GroundingScorer,
    GroundingResult,
    SentenceGrounding,
    ChunkGroundingScore,  # backwards-compat alias, import should work
)
from src.ingestion.storage.vector_store import SearchResult
from src.ingestion.chunking.chunk import Chunk


def _make_chunk(chunk_id: str, content: str, embedding=None) -> Chunk:
    """Build a Chunk with the minimum fields populated for scoring."""
    chunk = Chunk(
        chunk_id=chunk_id,
        content=content,
        doc_id="test_doc",
        source_doc_title="Test Doc",
    )
    if embedding is not None:
        chunk.embedding = np.asarray(embedding, dtype=np.float32)
    return chunk


def _make_result(chunk_id: str, content: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=_make_chunk(chunk_id, content),
        score=score,
        rank=1,
    )


class TestGroundingScorer:
    """Exercises the embedding-based grounding scorer.

    These tests use the real embedding model, so they're slower than
    pure unit tests but they verify the actual scoring behaviour.
    """

    def setup_method(self):
        self.scorer = GroundingScorer()

    def test_well_grounded_answer(self):
        sources = [
            _make_result("c1", "Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes called neurons."),
        ]
        answer = "Neural networks are computing systems inspired by the brain."
        result = self.scorer.score(answer, sources)
        # Paraphrase of the source should produce a high confidence.
        assert result.overall_confidence >= 0.5
        assert len(result.sentences) == 1
        assert result.sentences[0].confidence >= 0.5

    def test_ungrounded_answer(self):
        sources = [
            _make_result("c1", "The recipe for banana bread requires flour, sugar, and ripe bananas."),
        ]
        answer = "Quantum entanglement describes correlations between distant particles."
        result = self.scorer.score(answer, sources)
        # Totally unrelated — should score low.
        assert result.overall_confidence < 0.4

    def test_scores_sorted_by_sentence_order(self):
        """Sentences are reported in the order they appear in the answer."""
        sources = [_make_result("c1", "Python is a programming language used widely in industry.")]
        answer = "Python is a programming language. It is used in industry."
        result = self.scorer.score(answer, sources)
        assert len(result.sentences) == 2
        # First reported sentence should be the first sentence of the answer
        assert result.sentences[0].sentence.startswith("Python")

    def test_empty_sources_returns_no_scores(self):
        result = self.scorer.score("Some answer text.", [])
        assert result.sentences == []
        assert result.overall_confidence == 0.0

    def test_empty_answer_returns_no_scores(self):
        sources = [_make_result("c1", "Some chunk content.")]
        result = self.scorer.score("", sources)
        assert result.sentences == []
        assert result.overall_confidence == 0.0

    def test_invalid_min_confidence_raises(self):
        with pytest.raises(ValueError):
            GroundingScorer(min_confidence=-0.1)
        with pytest.raises(ValueError):
            GroundingScorer(min_confidence=1.5)

    def test_result_has_method_field(self):
        sources = [_make_result("c1", "Relevant content about something.")]
        result = self.scorer.score("This is an answer.", sources)
        assert result.method == "embedding_cosine"

    def test_best_chunk_id_is_populated(self):
        sources = [
            _make_result("c1", "Completely unrelated content about cooking."),
            _make_result("c2", "Neural networks learn patterns from data through training."),
        ]
        answer = "Neural networks learn from training data."
        result = self.scorer.score(answer, sources)
        # The best match should be c2, not c1.
        assert result.sentences[0].best_chunk_id == "c2"

    def test_multiple_sentences_each_scored(self):
        sources = [_make_result("c1", "Machine learning models require training data to learn patterns.")]
        answer = "Machine learning needs data. Training improves the model."
        result = self.scorer.score(answer, sources)
        assert len(result.sentences) == 2
        for s in result.sentences:
            assert 0.0 <= s.confidence <= 1.0

    def test_backwards_compat_alias_exists(self):
        """ChunkGroundingScore should still import as an alias of SentenceGrounding."""
        assert ChunkGroundingScore is SentenceGrounding

    def test_confidence_in_valid_range(self):
        sources = [_make_result("c1", "Some text about a topic.")]
        result = self.scorer.score("An answer about the topic.", sources)
        assert 0.0 <= result.overall_confidence <= 1.0
        for s in result.sentences:
            assert 0.0 <= s.confidence <= 1.0