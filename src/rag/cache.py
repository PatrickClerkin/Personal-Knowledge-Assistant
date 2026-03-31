"""
Semantic query cache for the RAG pipeline.

Caches answers to previous queries and returns them instantly when
a new query is semantically similar enough, avoiding redundant LLM
calls and reducing API costs.

Similarity is measured by cosine similarity between query embeddings.
A configurable threshold controls how aggressively the cache matches —
higher values require near-identical queries; lower values allow
broader semantic matches.

Design Pattern: Proxy Pattern — SemanticCache acts as a transparent
proxy in front of the retrieval+generation pipeline.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING

from ..ingestion.embeddings import EmbeddingService
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .pipeline import RAGResponse

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A single cached query/response pair.

    Attributes:
        query: The original query string.
        embedding: Normalised query embedding vector.
        response: The cached RAGResponse.
        hits: Number of times this entry has been served from cache.
        created_at: When this entry was first cached.
        last_hit_at: When this entry was last retrieved from cache.
    """
    query: str
    embedding: np.ndarray
    response: "RAGResponse"
    hits: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_hit_at: Optional[datetime] = None


class SemanticCache:
    """Semantic similarity cache for RAG query/response pairs.

    Stores previous queries as embeddings and serves cached responses
    when a new query is semantically close enough to a stored one.

    Attributes:
        threshold: Cosine similarity threshold for a cache hit
            (default 0.92). Range [0, 1] — higher = stricter.
        max_size: Maximum number of entries to store. When full,
            the least-recently-used entry is evicted (default 100).

    Usage:
        cache = SemanticCache(threshold=0.92)
        hit = cache.get("what is backpropagation?")
        if hit:
            return hit  # instant response
        response = pipeline.generate(...)
        cache.store("what is backpropagation?", response)
    """

    def __init__(
        self,
        threshold: float = 0.92,
        max_size: int = 100,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self.threshold = threshold
        self.max_size = max_size
        self._embedder = embedding_service or EmbeddingService()
        self._entries: List[CacheEntry] = []

    def get(self, query: str) -> Optional["RAGResponse"]:
        """Return a cached response if a similar query exists.

        Embeds the query and computes cosine similarity against all
        cached entries. Returns the response for the most similar
        entry if it meets the threshold.

        Args:
            query: The incoming query string.

        Returns:
            A cached RAGResponse, or None if no hit.
        """
        if not self._entries:
            return None

        query_embedding = self._embed(query)
        if query_embedding is None:
            return None

        best_similarity = -1.0
        best_entry: Optional[CacheEntry] = None

        for entry in self._entries:
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_similarity >= self.threshold and best_entry is not None:
            best_entry.hits += 1
            best_entry.last_hit_at = datetime.now(timezone.utc)
            logger.info(
                "Cache hit (similarity=%.4f >= %.4f): '%s' matched '%s'",
                best_similarity, self.threshold,
                query[:50], best_entry.query[:50],
            )
            return best_entry.response

        logger.debug(
            "Cache miss (best similarity=%.4f < %.4f) for: '%s'",
            best_similarity, self.threshold, query[:50],
        )
        return None

    def store(self, query: str, response: "RAGResponse") -> bool:
        """Cache a query/response pair.

        If the cache is full, evicts the least-recently-used entry
        before storing the new one.

        Args:
            query: The query string to cache.
            response: The RAGResponse to associate with the query.

        Returns:
            True if stored successfully, False if embedding failed.
        """
        embedding = self._embed(query)
        if embedding is None:
            return False

        if len(self._entries) >= self.max_size:
            self._evict_lru()

        entry = CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
        )
        self._entries.append(entry)
        logger.debug("Cached query (%d/%d): '%s'", len(self._entries), self.max_size, query[:50])
        return True

    def clear(self) -> None:
        """Remove all cached entries."""
        count = len(self._entries)
        self._entries.clear()
        logger.debug("Cache cleared (%d entries removed)", count)

    def invalidate(self, query: str) -> bool:
        """Remove a specific cached entry by exact query match.

        Args:
            query: The exact query string to remove.

        Returns:
            True if an entry was removed, False if not found.
        """
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.query != query]
        removed = before - len(self._entries)
        if removed:
            logger.debug("Invalidated cache entry: '%s'", query[:50])
        return removed > 0

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._entries)

    @property
    def total_hits(self) -> int:
        """Total number of cache hits served across all entries."""
        return sum(e.hits for e in self._entries)

    @property
    def stats(self) -> dict:
        """Summary statistics for the cache."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "threshold": self.threshold,
            "total_hits": self.total_hits,
            "entries": [
                {
                    "query": e.query[:60],
                    "hits": e.hits,
                    "created_at": e.created_at.isoformat(),
                }
                for e in self._entries
            ],
        }

    # ─── Private helpers ────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Embed and L2-normalise a text string."""
        try:
            vector = self._embedder.embed_text(text)
            norm = np.linalg.norm(vector)
            if norm == 0:
                return None
            return vector / norm
        except Exception as e:
            logger.warning("Cache embedding failed: %s", e)
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised vectors."""
        return float(np.dot(a, b))

    def _evict_lru(self) -> None:
        """Evict the least-recently-used cache entry."""
        if not self._entries:
            return

        def last_used(entry: CacheEntry) -> datetime:
            return entry.last_hit_at or entry.created_at

        lru = min(self._entries, key=last_used)
        self._entries.remove(lru)
        logger.debug("Evicted LRU cache entry: '%s'", lru.query[:50])