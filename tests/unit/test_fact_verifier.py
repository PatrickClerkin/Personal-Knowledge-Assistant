"""Tests for the embedding-based FactVerifier."""

import numpy as np
import pytest

from src.rag.fact_verifier import (
    FactVerifier,
    VerificationResult,
    ClaimVerification,
    SentenceVerdict,  # backwards-compat alias, import should work
)
from src.ingestion.storage.vector_store import SearchResult
from src.ingestion.chunking.chunk import Chunk


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        doc_id="test_doc",
        source_doc_title="Test Doc",
    )


def _make_result(chunk_id: str, content: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=_make_chunk(chunk_id, content),
        score=score,
        rank=1,
    )


class TestFactVerifier:
    """Exercises the embedding-based fact verifier with real embeddings."""

    def setup_method(self):
        self.verifier = FactVerifier()

    def test_returns_verification_result(self):
        sources = [_make_result("c1", "Neural networks process information using layered artificial neurons.")]
        result = self.verifier.verify(
            "Neural networks process information through layers.", sources
        )
        assert isinstance(result, VerificationResult)

    def test_empty_sources_returns_empty(self):
        result = self.verifier.verify("Some claim text here.", [])
        assert result.claims == []
        assert result.overall_verification_score == 0.0

    def test_empty_answer_returns_empty_claims(self):
        sources = [_make_result("c1", "Some chunk content here.")]
        result = self.verifier.verify("", sources)
        assert len(result.claims) == 0
        # Backwards-compat alias still works.
        assert len(result.verdicts) == 0

    def test_well_grounded_sentence_is_supported_or_partial(self):
        """A close paraphrase of source content should score as supported or partial."""
        sources = [_make_result(
            "c1",
            "Machine learning is a subfield of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
        )]
        answer = "Machine learning is part of AI and lets computers learn from data automatically."
        result = self.verifier.verify(answer, sources)
        assert len(result.claims) == 1
        # Should be supported or partial, not unverified.
        assert result.claims[0].status in ("supported", "partial")

    def test_unrelated_sentence_is_unverified(self):
        """A claim with no semantic connection to the source should be unverified."""
        sources = [_make_result(
            "c1",
            "The recipe calls for two cups of flour and one cup of sugar.",
        )]
        answer = "Quantum entanglement is a fundamental principle of modern physics research."
        result = self.verifier.verify(answer, sources)
        assert len(result.claims) == 1
        assert result.claims[0].status == "unverified"

    def test_short_sentences_skipped(self):
        """Very short 'filler' sentences should be skipped."""
        sources = [_make_result("c1", "Some chunk content about the topic at hand.")]
        result = self.verifier.verify("Yes. No. OK.", sources)
        # All three are below the filler-length threshold.
        assert len(result.claims) == 0

    def test_claim_has_required_fields(self):
        sources = [_make_result(
            "c1",
            "Python is a high-level general-purpose programming language.",
        )]
        result = self.verifier.verify(
            "Python is a popular programming language used for general tasks.",
            sources,
        )
        assert len(result.claims) >= 1
        claim = result.claims[0]
        assert isinstance(claim.claim, str)
        assert claim.status in ("supported", "partial", "unverified")
        assert 0.0 <= claim.score <= 1.0
        assert claim.best_chunk_id == "c1"
        assert claim.best_chunk_preview  # truthy string

    def test_overall_score_in_valid_range(self):
        sources = [_make_result(
            "c1",
            "Neural networks consist of layers of interconnected nodes that process information.",
        )]
        result = self.verifier.verify(
            "Neural networks have layers of connected nodes. They process inputs.",
            sources,
        )
        assert 0.0 <= result.overall_verification_score <= 1.0

    def test_counts_sum_to_total_claims(self):
        sources = [_make_result(
            "c1",
            "Databases store structured data. SQL is used to query relational databases.",
        )]
        answer = (
            "Databases hold structured data. SQL queries relational stores. "
            "Meanwhile, bananas grow on trees in tropical climates."
        )
        result = self.verifier.verify(answer, sources)
        total = (
            result.supported_count
            + result.partial_count
            + result.unverified_count
        )
        assert total == len(result.claims)

    def test_to_dict_has_required_keys(self):
        sources = [_make_result("c1", "Programming languages include Python, Java, and JavaScript.")]
        result = self.verifier.verify(
            "Python is a programming language that is widely used.",
            sources,
        )
        data = result.to_dict()
        assert "claims" in data
        assert "supported_count" in data
        assert "partial_count" in data
        assert "unverified_count" in data
        assert "overall_verification_score" in data

    def test_best_chunk_is_populated(self):
        sources = [_make_result("c1", "Python is a widely used programming language.")]
        result = self.verifier.verify(
            "Python is a programming language that many developers use.",
            sources,
        )
        assert len(result.claims) >= 1
        assert result.claims[0].best_chunk_id == "c1"

    def test_invalid_thresholds_raise(self):
        """Out-of-range or inverted thresholds must raise ValueError."""
        with pytest.raises(ValueError):
            FactVerifier(supported_threshold=-0.1)
        with pytest.raises(ValueError):
            FactVerifier(supported_threshold=1.5)
        with pytest.raises(ValueError):
            FactVerifier(partial_threshold=-0.1)
        with pytest.raises(ValueError):
            FactVerifier(partial_threshold=1.5)
        # partial > supported is nonsensical.
        with pytest.raises(ValueError):
            FactVerifier(supported_threshold=0.4, partial_threshold=0.6)

    def test_multiple_sources_picks_best(self):
        """Among multiple sources, the verifier should match against the closest one."""
        sources = [
            _make_result("c1", "Completely irrelevant content about cooking recipes."),
            _make_result("c2", "Machine learning algorithms learn patterns from training data."),
        ]
        result = self.verifier.verify(
            "Machine learning algorithms learn from training data.",
            sources,
        )
        assert len(result.claims) >= 1
        # Best match should be c2, not c1.
        assert result.claims[0].best_chunk_id == "c2"

    def test_split_sentences_basic(self):
        """The internal sentence splitter should handle simple punctuation."""
        sentences = FactVerifier._split_sentences(
            "First sentence. Second sentence. Third one!"
        )
        assert len(sentences) == 3

    def test_backwards_compat_alias_exists(self):
        """SentenceVerdict must still import as an alias of ClaimVerification."""
        assert SentenceVerdict is ClaimVerification