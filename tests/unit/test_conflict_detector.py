"""Unit tests for ConflictDetector."""

import pytest
from unittest.mock import MagicMock
from src.rag.conflict_detector import ConflictDetector, ConflictReport, Conflict
from src.rag.llm import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-test",
        usage={"input_tokens": 10, "output_tokens": 30},
        stop_reason="end_turn",
    )


def _make_result(content: str, source: str, page: int = 1, score: float = 0.9):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    chunk.page_number = page
    result = MagicMock()
    result.chunk = chunk
    result.score = score
    return result


_NO_CONFLICT_JSON = '''{
  "has_conflict": false,
  "conflict_type": "none",
  "severity": "none",
  "description": "No conflict found.",
  "excerpt_a_claim": "",
  "excerpt_b_claim": ""
}'''

_CONFLICT_JSON = '''{
  "has_conflict": true,
  "conflict_type": "factual",
  "severity": "high",
  "description": "Document A states neural networks require large datasets while Document B states they can work with small datasets.",
  "excerpt_a_claim": "Neural networks require large datasets to train effectively.",
  "excerpt_b_claim": "Neural networks can achieve good results with small datasets using transfer learning."
}'''


def _make_detector(llm_json=None, search_results=None):
    kb = MagicMock()
    kb.search.return_value = search_results or [
        _make_result("Neural networks require large datasets.", "doc_a"),
        _make_result("Neural networks work well with small data.", "doc_b"),
    ]

    llm = MagicMock()
    llm.is_available.return_value = True
    llm.generate.return_value = _make_llm_response(llm_json or _CONFLICT_JSON)

    return ConflictDetector(
        knowledge_base=kb,
        llm_provider=llm,
        top_k=5,
        max_pairs=5,
    )


class TestConflictDetector:

    def test_returns_conflict_report(self):
        detector = _make_detector()
        report = detector.detect("neural networks")
        assert isinstance(report, ConflictReport)

    def test_report_has_correct_topic(self):
        detector = _make_detector()
        report = detector.detect("neural networks")
        assert report.topic == "neural networks"

    def test_empty_kb_returns_empty_report(self):
        kb = MagicMock()
        kb.search.return_value = []
        llm = MagicMock()
        llm.is_available.return_value = True
        detector = ConflictDetector(knowledge_base=kb, llm_provider=llm)
        report = detector.detect("neural networks")
        assert len(report.conflicts) == 0
        assert report.documents_analysed == 0

    def test_single_source_no_pairs(self):
        kb = MagicMock()
        kb.search.return_value = [
            _make_result("Content about neural networks.", "only_doc"),
        ]
        llm = MagicMock()
        llm.is_available.return_value = True
        detector = ConflictDetector(knowledge_base=kb, llm_provider=llm)
        report = detector.detect("neural networks")
        assert report.pairs_compared == 0
        assert len(report.conflicts) == 0

    def test_conflict_detected_when_llm_says_yes(self):
        detector = _make_detector(llm_json=_CONFLICT_JSON)
        report = detector.detect("neural networks")
        assert len(report.conflicts) > 0

    def test_no_conflict_when_llm_says_no(self):
        detector = _make_detector(llm_json=_NO_CONFLICT_JSON)
        report = detector.detect("neural networks")
        assert len(report.conflicts) == 0

    def test_conflict_has_required_fields(self):
        detector = _make_detector(llm_json=_CONFLICT_JSON)
        report = detector.detect("neural networks")
        if report.conflicts:
            c = report.conflicts[0]
            assert hasattr(c, "source_a")
            assert hasattr(c, "source_b")
            assert hasattr(c, "conflict_type")
            assert hasattr(c, "severity")
            assert hasattr(c, "description")
            assert hasattr(c, "claim_a")
            assert hasattr(c, "claim_b")

    def test_severity_counts_correct(self):
        detector = _make_detector(llm_json=_CONFLICT_JSON)
        report = detector.detect("neural networks")
        total = report.high_count + report.medium_count + report.low_count
        assert total == len(report.conflicts)

    def test_to_dict_has_required_keys(self):
        detector = _make_detector()
        report = detector.detect("neural networks")
        d = report.to_dict()
        assert "topic" in d
        assert "conflicts" in d
        assert "total_conflicts" in d
        assert "documents_analysed" in d
        assert "pairs_compared" in d
        assert "high_count" in d
        assert "medium_count" in d
        assert "low_count" in d

    def test_llm_unavailable_returns_no_conflicts(self):
        kb = MagicMock()
        kb.search.return_value = [
            _make_result("Content A", "doc_a"),
            _make_result("Content B", "doc_b"),
        ]
        llm = MagicMock()
        llm.is_available.return_value = False
        detector = ConflictDetector(knowledge_base=kb, llm_provider=llm)
        report = detector.detect("neural networks")
        assert len(report.conflicts) == 0

    def test_invalid_json_handled_gracefully(self):
        kb = MagicMock()
        kb.search.return_value = [
            _make_result("Content A", "doc_a"),
            _make_result("Content B", "doc_b"),
        ]
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = _make_llm_response("not valid json")
        detector = ConflictDetector(knowledge_base=kb, llm_provider=llm)
        report = detector.detect("neural networks")
        assert isinstance(report, ConflictReport)

    def test_conflict_to_dict_structure(self):
        detector = _make_detector(llm_json=_CONFLICT_JSON)
        report = detector.detect("neural networks")
        if report.conflicts:
            d = report.conflicts[0].to_dict()
            assert "source_a" in d
            assert "source_b" in d
            assert "severity" in d
            assert "conflict_type" in d
            assert "description" in d

    def test_max_pairs_respected(self):
        results = [
            _make_result(f"Content {i}", f"doc_{i}", score=0.9 - i * 0.05)
            for i in range(6)
        ]
        kb = MagicMock()
        kb.search.return_value = results
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = _make_llm_response(_NO_CONFLICT_JSON)

        detector = ConflictDetector(
            knowledge_base=kb,
            llm_provider=llm,
            max_pairs=3,
        )
        report = detector.detect("topic")
        assert report.pairs_compared <= 3

    def test_conflicts_sorted_by_severity(self):
        detector = _make_detector(llm_json=_CONFLICT_JSON)
        report = detector.detect("neural networks")
        if len(report.conflicts) > 1:
            order = {"high": 0, "medium": 1, "low": 2}
            scores = [order.get(c.severity, 3) for c in report.conflicts]
            assert scores == sorted(scores)