"""
Information Retrieval evaluation metrics.

Implements the three standard metrics used to evaluate ranked retrieval:
    - Precision@K   : fraction of top-K results that are relevant
    - MRR           : reciprocal rank of the first relevant result
    - nDCG@K        : normalised discounted cumulative gain

All functions are pure (no side effects) and operate on lists of
document IDs, making them easy to unit test independently.
"""

import math
from typing import List, Set


def precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Fraction of the top-K retrieved docs that are relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of known-relevant document IDs.
        k: Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def mean_reciprocal_rank(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """Reciprocal rank of the first relevant result.

    Returns 0 if no relevant document appears in the retrieved list.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of known-relevant document IDs.

    Returns:
        Float in [0, 1].
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise). Normalises
    against the ideal ranking where all relevant docs appear first.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of known-relevant document IDs.
        k: Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    def dcg(doc_ids: List[str], at_k: int) -> float:
        score = 0.0
        for i, doc_id in enumerate(doc_ids[:at_k], start=1):
            if doc_id in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved, k)

    # Ideal: place all relevant docs first
    ideal_retrieved = list(relevant) + [
        d for d in retrieved if d not in relevant
    ]
    ideal_dcg = dcg(ideal_retrieved, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0