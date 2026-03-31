"""Unit tests for FactVerifier."""

import pytest
from unittest.mock import MagicMock
from src.rag.fact_verifier import FactVerifier, VerificationResult, SentenceVerdict


def _make_source(content: str, source: str = "doc", page: int = 1):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    chunk.page_number = page
    result = MagicMock()
    result.chunk = chunk
    result.score = 0.9
    return result


_RICH_CHUNK = (
    "Neural networks are computational models inspired by the human brain. "
    "They consist of layers of interconnected neurons that process information. "
    "Backpropagation is the algorithm used to train neural networks by adjusting weights. "
    "The loss function measures the difference between predicted and actual outputs."
)


class TestFactVerifier:

    def setup_method(self):
        self.verifier = FactVerifier(
            supported_threshold=0.35,
            partial_threshold=0.12,
        )

    def test_returns_verification_result(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify("Neural networks process information.", sources)
        assert isinstance(result, VerificationResult)

    def test_empty_sources_returns_unverified(self):
        result = self.verifier.verify(
            "Neural networks are great for classification tasks.", []
        )
        assert result.supported_count == 0
        assert result.overall_verification_score == 0.0

    def test_well_grounded_sentence_is_supported(self):
        sources = [_make_source(_RICH_CHUNK)]
        answer = (
            "Neural networks consist of layers of interconnected neurons "
            "that process information using backpropagation."
        )
        result = self.verifier.verify(answer, sources)
        assert any(v.status == "supported" for v in result.verdicts)

    def test_unrelated_sentence_is_unverified(self):
        sources = [_make_source(_RICH_CHUNK)]
        answer = "The weather today is sunny and warm outside in Ireland."
        result = self.verifier.verify(answer, sources)
        assert all(v.status == "unverified" for v in result.verdicts)

    def test_short_sentences_skipped(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify("Yes. No. OK.", sources)
        assert len(result.verdicts) == 0

    def test_verdict_has_required_fields(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify(
            "Neural networks use backpropagation to learn weights.", sources
        )
        for v in result.verdicts:
            assert hasattr(v, "sentence")
            assert hasattr(v, "status")
            assert hasattr(v, "best_score")
            assert v.status in ("supported", "partial", "unverified")

    def test_overall_score_between_zero_and_one(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify(
            "Neural networks process information. "
            "The moon is made of cheese today.",
            sources,
        )
        assert 0.0 <= result.overall_verification_score <= 1.0

    def test_counts_sum_to_total_verdicts(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify(
            "Neural networks use backpropagation. "
            "Loss functions measure prediction error. "
            "Bananas are yellow fruit from tropical regions.",
            sources,
        )
        total = (
            result.supported_count
            + result.partial_count
            + result.unverified_count
        )
        assert total == len(result.verdicts)

    def test_to_dict_has_required_keys(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify(
            "Neural networks learn through backpropagation.", sources
        )
        d = result.to_dict()
        assert "supported_count" in d
        assert "partial_count" in d
        assert "unverified_count" in d
        assert "overall_verification_score" in d
        assert "verdicts" in d

    def test_best_source_populated_on_hit(self):
        sources = [_make_source(_RICH_CHUNK, source="lecture_5", page=3)]
        result = self.verifier.verify(
            "Neural networks consist of layers processing information.", sources
        )
        supported = [v for v in result.verdicts if v.status in ("supported", "partial")]
        if supported:
            assert supported[0].best_source == "lecture_5"

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            FactVerifier(supported_threshold=0.1, partial_threshold=0.5)

    def test_multiple_sources_picks_best(self):
        weak_source = _make_source("The sky is blue and clouds are white.")
        strong_source = _make_source(_RICH_CHUNK)
        result = self.verifier.verify(
            "Neural networks use backpropagation to train weights.", [weak_source, strong_source]
        )
        # Should find a match in the strong source
        scores = [v.best_score for v in result.verdicts]
        assert max(scores) > 0.0

    def test_split_sentences_basic(self):
        sentences = self.verifier._split_sentences(
            "First sentence. Second sentence. Third sentence."
        )
        assert len(sentences) == 3

    def test_empty_answer_returns_empty_verdicts(self):
        sources = [_make_source(_RICH_CHUNK)]
        result = self.verifier.verify("", sources)
        assert len(result.verdicts) == 0