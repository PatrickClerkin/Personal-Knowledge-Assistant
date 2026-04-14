"""Unit tests for QueryHistory."""

import pytest
import json
from pathlib import Path
from src.rag.query_history import QueryHistory, QueryRecord


def _make_record(**kwargs) -> QueryRecord:
    defaults = {
        "query": "What is backpropagation?",
        "answer_preview": "Backpropagation is an algorithm...",
        "confidence": 0.75,
        "cache_hit": False,
        "retrieval_attempts": 1,
        "input_tokens": 150,
        "output_tokens": 80,
        "sources": ["lecture_1", "lecture_2"],
        "verification_score": 0.8,
    }
    defaults.update(kwargs)
    return QueryRecord(**defaults)


class TestQueryRecord:

    def test_to_dict_has_required_keys(self):
        r = _make_record()
        d = r.to_dict()
        assert "query" in d
        assert "confidence" in d
        assert "cache_hit" in d
        assert "retrieval_attempts" in d
        assert "input_tokens" in d
        assert "output_tokens" in d
        assert "sources" in d
        assert "timestamp" in d

    def test_from_dict_roundtrip(self):
        r = _make_record()
        d = r.to_dict()
        r2 = QueryRecord.from_dict(d)
        assert r2.query == r.query
        assert r2.confidence == r.confidence
        assert r2.cache_hit == r.cache_hit

    def test_timestamp_set_automatically(self):
        r = _make_record()
        assert r.timestamp is not None
        assert "T" in r.timestamp  # ISO format


class TestQueryHistory:

    def test_starts_empty(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        assert h.total_queries == 0

    def test_record_increases_count(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record())
        assert h.total_queries == 1

    def test_get_recent_returns_most_recent_first(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(query="first"))
        h.record(_make_record(query="second"))
        recent = h.get_recent(2)
        assert recent[0].query == "second"
        assert recent[1].query == "first"

    def test_get_recent_respects_n(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        for i in range(10):
            h.record(_make_record(query=f"query {i}"))
        assert len(h.get_recent(3)) == 3

    def test_analytics_empty_returns_zeros(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        a = h.get_analytics()
        assert a["total_queries"] == 0
        assert a["cache_hit_rate"] == 0.0

    def test_analytics_cache_hit_rate(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(cache_hit=True))
        h.record(_make_record(cache_hit=False))
        a = h.get_analytics()
        assert a["cache_hit_rate"] == 0.5

    def test_analytics_avg_confidence(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(confidence=0.8))
        h.record(_make_record(confidence=0.4))
        a = h.get_analytics()
        assert abs(a["avg_confidence"] - 0.6) < 0.01

    def test_analytics_confidence_distribution(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(confidence=0.05))   # low
        h.record(_make_record(confidence=0.25))   # medium
        h.record(_make_record(confidence=0.75))   # high
        a = h.get_analytics()
        assert a["confidence_distribution"]["low"] == 1
        assert a["confidence_distribution"]["medium"] == 1
        assert a["confidence_distribution"]["high"] == 1

    def test_analytics_top_sources(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(sources=["doc_a", "doc_b"]))
        h.record(_make_record(sources=["doc_a"]))
        a = h.get_analytics()
        assert a["top_sources"][0]["source"] == "doc_a"
        assert a["top_sources"][0]["count"] == 2

    def test_analytics_total_tokens(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(input_tokens=100, output_tokens=50))
        h.record(_make_record(input_tokens=200, output_tokens=100))
        a = h.get_analytics()
        assert a["total_tokens"] == 450

    def test_persistence_survives_reload(self, tmp_path):
        path = str(tmp_path / "history.json")
        h1 = QueryHistory(persist_path=path)
        h1.record(_make_record(query="persisted query"))

        h2 = QueryHistory(persist_path=path)
        assert h2.total_queries == 1
        assert h2.get_recent(1)[0].query == "persisted query"

    def test_clear_resets_history(self, tmp_path):
        path = str(tmp_path / "history.json")
        h = QueryHistory(persist_path=path)
        h.record(_make_record())
        h.clear()
        assert h.total_queries == 0
        assert not Path(path).exists()

    def test_max_records_evicts_oldest(self, tmp_path):
        h = QueryHistory(
            persist_path=str(tmp_path / "history.json"),
            max_records=3,
        )
        for i in range(5):
            h.record(_make_record(query=f"query {i}"))
        assert h.total_queries == 3
        assert h.get_recent(3)[-1].query == "query 2"

    def test_multi_attempt_rate(self, tmp_path):
        h = QueryHistory(persist_path=str(tmp_path / "history.json"))
        h.record(_make_record(retrieval_attempts=1))
        h.record(_make_record(retrieval_attempts=2))
        a = h.get_analytics()
        assert a["multi_attempt_rate"] == 0.5