"""Entity-aware result boosting."""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _entity_texts(entities: List[dict]) -> set[str]:
    """Return a lower-cased set of entity text strings for fast intersection."""
    return {e["text"].lower() for e in entities if isinstance(e, dict) and "text" in e}


def boost_by_entities(
    results: List[Tuple[Any, float]],
    query_entities: List[dict],
    boost_weight: float = 0.05,
    label_filter: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    """Re-rank *results* using entity overlap with *query_entities*.

    Parameters
    ----------
    results:
        List of ``(chunk, score)`` tuples as returned by any search method.
    query_entities:
        Entities extracted from the user query by :class:`NERExtractor`.
    boost_weight:
        Score increment added per overlapping entity (default ``0.05``).
    label_filter:
        If provided (e.g. ``"PERSON"``), only chunks that contain at least
        one entity with this label survive the filter.

    Returns
    -------
    List of ``(chunk, boosted_score)`` tuples sorted descending by score.
    """
    if label_filter:
        results = [
            (chunk, score)
            for chunk, score in results
            if any(
                e.get("label") == label_filter
                for e in chunk.metadata.get("entities", [])
            )
        ]

    if not query_entities:
        return results

    query_texts = _entity_texts(query_entities)
    boosted: List[Tuple[Any, float]] = []

    for chunk, score in results:
        chunk_entities: List[dict] = chunk.metadata.get("entities", [])
        overlap = len(query_texts & _entity_texts(chunk_entities))
        boosted.append((chunk, score + overlap * boost_weight))

    boosted.sort(key=lambda t: t[1], reverse=True)
    return boosted