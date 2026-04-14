"""
Query history and analytics for the RAG pipeline.

Records every query made to the pipeline — including the question,
answer summary, confidence score, cache status, retrieval attempts,
token usage, and sources used. Persists to disk as JSON so history
survives server restarts.

Provides aggregated analytics: query volume over time, average
confidence, cache hit rate, most queried topics, and token usage.

Design Pattern: Repository Pattern — QueryHistory is the single
source of truth for all query records.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryRecord:
    """A single recorded query/response interaction.

    Attributes:
        query: The original question asked.
        answer_preview: First 200 chars of the answer.
        confidence: Grounding confidence score (0-1).
        cache_hit: Whether the response came from cache.
        retrieval_attempts: Number of retrieval attempts made.
        input_tokens: LLM input tokens used.
        output_tokens: LLM output tokens used.
        sources: List of source document titles used.
        timestamp: When the query was made (UTC).
        verification_score: Fact verification score if available.
    """
    query: str
    answer_preview: str
    confidence: float
    cache_hit: bool
    retrieval_attempts: int
    input_tokens: int
    output_tokens: int
    sources: List[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verification_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer_preview": self.answer_preview,
            "confidence": round(self.confidence, 4),
            "cache_hit": self.cache_hit,
            "retrieval_attempts": self.retrieval_attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "sources": self.sources,
            "timestamp": self.timestamp,
            "verification_score": round(self.verification_score, 4)
                if self.verification_score is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryRecord":
        return cls(
            query=data.get("query", ""),
            answer_preview=data.get("answer_preview", ""),
            confidence=data.get("confidence", 0.0),
            cache_hit=data.get("cache_hit", False),
            retrieval_attempts=data.get("retrieval_attempts", 1),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            sources=data.get("sources", []),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            verification_score=data.get("verification_score"),
        )


class QueryHistory:
    """Persistent query history with analytics.

    Stores all query records in memory and persists them to a JSON
    file on disk. Provides analytics aggregated across all records.

    Args:
        persist_path: Path to the JSON file for persistence.
            Defaults to 'data/history/query_history.json'.
        max_records: Maximum records to keep in memory and on disk.
            Oldest records are evicted when the limit is reached.
    """

    def __init__(
        self,
        persist_path: str = "data/history/query_history.json",
        max_records: int = 1000,
    ):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self._records: List[QueryRecord] = []
        self._load()

    def record(self, record: QueryRecord) -> None:
        """Add a query record to history and persist it.

        Args:
            record: The QueryRecord to store.
        """
        self._records.append(record)

        # Evict oldest records if over limit
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        self._save()
        logger.debug(
            "Recorded query: '%s' (confidence=%.3f, cache=%s)",
            record.query[:40], record.confidence, record.cache_hit,
        )

    def get_recent(self, n: int = 20) -> List[QueryRecord]:
        """Return the n most recent query records.

        Args:
            n: Number of records to return (default 20).

        Returns:
            List of QueryRecord, most recent first.
        """
        return list(reversed(self._records[-n:]))

    def get_analytics(self) -> dict:
        """Compute aggregated analytics across all records.

        Returns a dict with:
            - total_queries: Total number of queries made.
            - cache_hit_rate: Fraction of queries served from cache.
            - avg_confidence: Mean grounding confidence score.
            - avg_retrieval_attempts: Mean retrieval attempts per query.
            - avg_input_tokens: Mean input tokens per query.
            - avg_output_tokens: Mean output tokens per query.
            - total_tokens: Total tokens consumed.
            - top_sources: Most frequently cited source documents.
            - confidence_distribution: Count in low/medium/high buckets.
            - queries_by_day: Query count per date (last 30 days).
            - multi_attempt_rate: Fraction needing >1 retrieval attempt.
        """
        if not self._records:
            return self._empty_analytics()

        n = len(self._records)
        cache_hits = sum(1 for r in self._records if r.cache_hit)
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        multi_attempt = sum(
            1 for r in self._records if r.retrieval_attempts > 1
        )

        avg_confidence = sum(r.confidence for r in self._records) / n
        avg_attempts = sum(r.retrieval_attempts for r in self._records) / n

        # Confidence distribution
        low    = sum(1 for r in self._records if r.confidence < 0.15)
        medium = sum(1 for r in self._records if 0.15 <= r.confidence < 0.40)
        high   = sum(1 for r in self._records if r.confidence >= 0.40)

        # Top sources
        source_counter: Counter = Counter()
        for r in self._records:
            for src in r.sources:
                source_counter[src] += 1
        top_sources = [
            {"source": src, "count": cnt}
            for src, cnt in source_counter.most_common(10)
        ]

        # Queries by day (last 30 days)
        day_counter: Counter = Counter()
        for r in self._records:
            try:
                day = r.timestamp[:10]  # YYYY-MM-DD
                day_counter[day] += 1
            except Exception:
                pass
        queries_by_day = [
            {"date": day, "count": cnt}
            for day, cnt in sorted(day_counter.items())[-30:]
        ]

        # Average verification score (where available)
        verified = [
            r.verification_score for r in self._records
            if r.verification_score is not None
        ]
        avg_verification = sum(verified) / len(verified) if verified else None

        return {
            "total_queries": n,
            "cache_hit_rate": round(cache_hits / n, 4),
            "avg_confidence": round(avg_confidence, 4),
            "avg_retrieval_attempts": round(avg_attempts, 3),
            "avg_input_tokens": round(total_input / n, 1),
            "avg_output_tokens": round(total_output / n, 1),
            "total_tokens": total_input + total_output,
            "multi_attempt_rate": round(multi_attempt / n, 4),
            "avg_verification_score": round(avg_verification, 4)
                if avg_verification is not None else None,
            "confidence_distribution": {
                "low": low,
                "medium": medium,
                "high": high,
            },
            "top_sources": top_sources,
            "queries_by_day": queries_by_day,
        }

    def clear(self) -> None:
        """Clear all history records and delete the persist file."""
        self._records.clear()
        if self.persist_path.exists():
            self.persist_path.unlink()
        logger.info("Query history cleared.")

    @property
    def total_queries(self) -> int:
        """Total number of recorded queries."""
        return len(self._records)

    def _save(self) -> None:
        """Persist records to JSON file."""
        try:
            data = [r.to_dict() for r in self._records]
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save query history: %s", e)

    def _load(self) -> None:
        """Load records from JSON file if it exists."""
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            self._records = [QueryRecord.from_dict(d) for d in data]
            logger.info(
                "Loaded %d query history records from %s",
                len(self._records), self.persist_path,
            )
        except Exception as e:
            logger.warning("Failed to load query history: %s", e)
            self._records = []

    def _empty_analytics(self) -> dict:
        """Return zeroed analytics when no records exist."""
        return {
            "total_queries": 0,
            "cache_hit_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_retrieval_attempts": 1.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "total_tokens": 0,
            "multi_attempt_rate": 0.0,
            "avg_verification_score": None,
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
            "top_sources": [],
            "queries_by_day": [],
        }