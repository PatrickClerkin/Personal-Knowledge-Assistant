"""
Cross-encoder reranker for improving retrieval precision.

Initial retrieval with bi-encoders (FAISS) is fast but approximate.
Cross-encoders score query-document pairs jointly, producing more
accurate relevance scores at the cost of speed. This two-stage
pipeline (retrieve then rerank) combines the best of both approaches.

Architecture:
    Query → FAISS (top-N candidates) → CrossEncoder (rerank) → top-K results
"""

from typing import List, Optional
from ..ingestion.storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model.

    Cross-encoders process (query, document) pairs jointly through
    a transformer, producing more accurate relevance scores than
    the bi-encoder approach used for initial retrieval.

    Attributes:
        model_name: HuggingFace model identifier for the cross-encoder.
        top_k: Number of results to return after reranking.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: int = 5,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_k = top_k
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder: %s", self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank search results using the cross-encoder.

        Args:
            query: The original search query.
            results: Initial search results from FAISS retrieval.
            top_k: Override the default number of results to return.

        Returns:
            Reranked list of SearchResult objects with updated scores.
        """
        if not results:
            return []

        top_k = top_k or self.top_k

        # Create (query, document) pairs for cross-encoder scoring
        pairs = [(query, r.chunk.content) for r in results]

        logger.debug(
            "Reranking %d candidates for query: '%s'",
            len(pairs), query[:50],
        )

        # Score all pairs
        scores = self.model.predict(pairs)

        # Pair scores with original results
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Build reranked results with new scores and ranks
        reranked = []
        for rank, (score, original) in enumerate(scored_results[:top_k], 1):
            reranked.append(
                SearchResult(
                    chunk=original.chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        logger.debug(
            "Reranking complete: top score %.4f, bottom score %.4f",
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )

        return reranked
