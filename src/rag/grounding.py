"""
Answer grounding analysis.

Scores how well an LLM-generated answer is supported by the source
chunks that were retrieved.  Used by the RAG pipeline to:

  - Flag low-confidence answers that may be hallucinated.
  - Trigger adaptive re-retrieval when the initial chunks did not
    yield a well-grounded answer.
  - Surface per-sentence confidence scores in the UI so users can
    see which parts of an answer are well-supported and which are
    speculative.

Scoring method
--------------
Each sentence of the answer is embedded with the shared
``EmbeddingService`` and compared against the embeddings of each
source chunk using cosine similarity.  A sentence's confidence is
the similarity to its best-matching chunk.  The overall answer
confidence is the mean sentence confidence, weighted by sentence
length so longer, content-bearing sentences dominate.

This is a substantial upgrade over the original term-overlap
approach, which rewarded trivial vocabulary matches and produced
inflated scores (often 1.000 for any answer that shared common
nouns with its sources).  Embedding-based scoring captures actual
semantic support, including paraphrases, and gives meaningful
mid-range values that discriminate between well-grounded answers
and hallucinations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..ingestion.embeddings.embedding_service import EmbeddingService
from ..ingestion.storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Sentence splitter — simple but robust enough for LLM outputs.
# Splits on ., !, ? followed by whitespace + capital letter, while
# leaving abbreviations alone as much as possible.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'`])")


@dataclass
class SentenceGrounding:
    """Grounding analysis for a single answer sentence.

    Attributes:
        sentence: The answer sentence text.
        confidence: Cosine similarity (0..1) to the best-matching
            source chunk.  Higher means better grounded.
        best_chunk_id: chunk_id of the best-matching source chunk,
            or None if there are no sources.
    """

    sentence: str
    confidence: float
    best_chunk_id: Optional[str] = None


# Backwards-compat alias — earlier versions of this module exported
# the per-item grounding result as ``ChunkGroundingScore``.  External
# modules and tests still import that name, so keep it available.
ChunkGroundingScore = SentenceGrounding


@dataclass
class GroundingResult:
    """Full grounding analysis for an answer.

    Attributes:
        sentences: Per-sentence grounding breakdown.
        overall_confidence: Weighted mean sentence confidence (0..1).
        method: Which scoring algorithm was used — always
            ``"embedding_cosine"`` with the current implementation,
            retained for backward compatibility with downstream
            consumers that may inspect the field.
    """

    sentences: List[SentenceGrounding] = field(default_factory=list)
    overall_confidence: float = 0.0
    method: str = "embedding_cosine"


class GroundingScorer:
    """Scores answer sentences against their source chunks.

    Uses cosine similarity between sentence embeddings and chunk
    embeddings via the shared ``EmbeddingService``.  The embedding
    model loads lazily on first use and is reused across calls.

    Args:
        embedding_service: Optional pre-built embedding service to
            reuse.  If omitted, a new one is created with the
            default ``all-MiniLM-L6-v2`` model.  Pass the same
            service used by the KnowledgeBase to avoid loading the
            model twice.
        min_confidence: Optional floor for the per-sentence
            confidence.  Sentences scoring below this cut-off are
            still included in the result but contribute their raw
            score to the aggregate — this parameter exists for
            backward compatibility with code that used to pass a
            "threshold" keyword.  It does not affect the new
            embedding-based scoring in any meaningful way and can
            be safely ignored.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        min_confidence: float = 0.0,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0 and 1 inclusive"
            )
        self._embedder = embedding_service or EmbeddingService()
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        answer: str,
        sources: List[SearchResult],
    ) -> GroundingResult:
        """Score an answer's grounding against its source chunks.

        Args:
            answer: The full LLM-generated answer text.
            sources: The retrieved chunks used as context.

        Returns:
            A GroundingResult with per-sentence and overall scores.
            Returns zeroed-out scores if there are no sources or the
            answer is empty.
        """
        sentences = self._split_sentences(answer)
        if not sentences or not sources:
            return GroundingResult()

        # Embed all answer sentences in one batch and normalise to 2-D.
        sentence_embeddings = _as_2d(self._embedder.embed_texts(sentences))

        # Embed all source chunks in one batch — prefer the
        # pre-computed embeddings stored on chunks where available.
        chunk_embeddings = self._collect_chunk_embeddings(sources)
        if chunk_embeddings.size == 0:
            return GroundingResult()

        # Normalise for cosine similarity.
        sent_norm = _l2_normalise(sentence_embeddings)
        chunk_norm = _l2_normalise(chunk_embeddings)

        # Pairwise cosine: (num_sentences, num_chunks).
        sims = sent_norm @ chunk_norm.T

        per_sentence: List[SentenceGrounding] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for i, sentence in enumerate(sentences):
            row = sims[i]
            best_idx = int(np.argmax(row))
            # Cosine on unit vectors lies in [-1, 1]; clamp to [0, 1]
            # so the score reads as a confidence.
            best_score = float(max(0.0, row[best_idx]))
            best_chunk_id = sources[best_idx].chunk.chunk_id

            per_sentence.append(
                SentenceGrounding(
                    sentence=sentence,
                    confidence=best_score,
                    best_chunk_id=best_chunk_id,
                )
            )

            # Weight each sentence by its character length so
            # longer, content-bearing sentences dominate over
            # short filler like "Yes." or "In summary.".
            weight = max(1, len(sentence))
            weighted_sum += best_score * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight else 0.0

        logger.debug(
            "Grounding: %d sentences scored, overall=%.3f",
            len(sentences), overall,
        )

        return GroundingResult(
            sentences=per_sentence,
            overall_confidence=float(overall),
            method="embedding_cosine",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences, discarding whitespace-only entries."""
        text = text.strip()
        if not text:
            return []
        parts = _SENTENCE_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _collect_chunk_embeddings(
        self,
        sources: List[SearchResult],
    ) -> np.ndarray:
        """Return chunk embeddings as a 2-D array.

        Uses the embedding stored on the chunk if present (set during
        ingestion) and falls back to embedding the content on the fly
        for any chunks that are missing one.  Returns a 0-row array
        if there are no sources.
        """
        if not sources:
            return np.zeros((0, 0), dtype=np.float32)

        missing_indices: List[int] = []
        missing_texts: List[str] = []
        embeddings: List[Optional[np.ndarray]] = []

        for i, result in enumerate(sources):
            emb = getattr(result.chunk, "embedding", None)
            if emb is None:
                embeddings.append(None)
                missing_indices.append(i)
                missing_texts.append(result.chunk.content)
            else:
                embeddings.append(np.asarray(emb, dtype=np.float32))

        if missing_texts:
            filled = _as_2d(self._embedder.embed_texts(missing_texts))
            for idx, vec in zip(missing_indices, filled):
                embeddings[idx] = np.asarray(vec, dtype=np.float32)

        return np.vstack(embeddings)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Coerce an embedding result into a 2-D (n, dim) array.

    ``SentenceTransformer.encode`` returns 1-D for a single input and
    2-D for a batch.  Callers always want 2-D so we can stack or
    matrix-multiply without special-casing.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation.  Zero rows stay zero (no NaN)."""
    matrix = _as_2d(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero — a zero vector stays zero.
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms