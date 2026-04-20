"""
Sentence-level fact verification.

Given an LLM answer and the source chunks it was generated from,
split the answer into individual claim sentences and check how well
each one is semantically supported by at least one source chunk.

Each claim gets a verification status:

    supported   — strong semantic similarity to a source chunk
    partial     — moderate similarity; the chunk is related but
                  does not fully support the claim
    unverified  — no chunk has enough semantic similarity to count
                  as support

The overall verification score is the proportion of claims that are
at least partially supported, weighted so "supported" is worth more
than "partial".

Scoring method
--------------
Uses the shared ``EmbeddingService`` to compute cosine similarity
between each answer sentence and each source chunk.  This replaced
the original term-overlap approach, which consistently reported
artificial 1.000 scores because common domain vocabulary matched
trivially.  Embedding similarity captures paraphrases, synonyms,
and actual semantic entailment, and produces discriminating scores
that meaningfully separate well-supported claims from hallucinations.

Thresholds
----------
The supported / partial / unverified cut-offs are tuned for
``all-MiniLM-L6-v2`` sentence embeddings.  On this model:

    >= 0.60   strongly supported
    0.40..0.60 partially supported
    <  0.40   unverified
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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'`])")

# Sentences shorter than this are treated as filler (e.g. "Yes." or
# "OK.") and skipped entirely — they carry almost no semantic content
# and would dilute the aggregate score with noise.
_MIN_SENTENCE_LENGTH = 10


@dataclass
class ClaimVerification:
    """Verification result for a single answer sentence.

    Attributes:
        claim: The sentence text.
        status: One of "supported", "partial", "unverified".
        score: Cosine similarity (0..1) to the best-matching source
            chunk.
        best_chunk_id: chunk_id of the best-matching source chunk,
            or None if there are no sources.
        best_chunk_preview: Short snippet of the best-matching chunk,
            useful for UI display.
    """

    claim: str
    status: str
    score: float
    best_chunk_id: Optional[str] = None
    best_chunk_preview: Optional[str] = None


# Backwards-compat alias — earlier versions of this module exported
# the per-sentence result as ``SentenceVerdict``.  Other modules
# (and tests) still import that name, so keep it working.
SentenceVerdict = ClaimVerification


@dataclass
class VerificationResult:
    """Aggregate verification result for an answer.

    Attributes:
        claims: Per-sentence verification breakdown.
        supported_count: Number of claims classified as "supported".
        partial_count: Number of claims classified as "partial".
        unverified_count: Number of claims classified as "unverified".
        overall_verification_score: Weighted support score (0..1).
    """

    claims: List[ClaimVerification] = field(default_factory=list)
    supported_count: int = 0
    partial_count: int = 0
    unverified_count: int = 0
    overall_verification_score: float = 0.0

    # Backwards-compat alias — tests and earlier consumers referred
    # to per-claim results as ``verdicts``.
    @property
    def verdicts(self) -> List[ClaimVerification]:
        """Alias for ``claims`` — older name for the same list."""
        return self.claims

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "claims": [
                {
                    "claim": c.claim,
                    "status": c.status,
                    "score": c.score,
                    "best_chunk_id": c.best_chunk_id,
                    "best_chunk_preview": c.best_chunk_preview,
                }
                for c in self.claims
            ],
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "unverified_count": self.unverified_count,
            "overall_verification_score": self.overall_verification_score,
        }


class FactVerifier:
    """Verifies answer claims against source chunks using embeddings.

    Args:
        embedding_service: Optional pre-built embedding service to
            reuse.  If omitted, a new one is created with the
            default ``all-MiniLM-L6-v2`` model.  Pass the same
            service used by the KnowledgeBase to avoid loading the
            model twice.
        supported_threshold: Cosine similarity at or above which a
            claim is considered fully supported.  Defaults to 0.60.
        partial_threshold: Cosine similarity at or above which a
            claim is considered partially supported.  Defaults to
            0.40.  Anything below this is unverified.

    Raises:
        ValueError: If either threshold is outside [0, 1] or
            ``partial_threshold`` is greater than
            ``supported_threshold``.
    """

    DEFAULT_SUPPORTED_THRESHOLD: float = 0.60
    DEFAULT_PARTIAL_THRESHOLD: float = 0.40

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        supported_threshold: Optional[float] = None,
        partial_threshold: Optional[float] = None,
    ):
        supported = (
            supported_threshold
            if supported_threshold is not None
            else self.DEFAULT_SUPPORTED_THRESHOLD
        )
        partial = (
            partial_threshold
            if partial_threshold is not None
            else self.DEFAULT_PARTIAL_THRESHOLD
        )

        if not 0.0 <= partial <= 1.0:
            raise ValueError(
                "partial_threshold must be between 0 and 1 inclusive"
            )
        if not 0.0 <= supported <= 1.0:
            raise ValueError(
                "supported_threshold must be between 0 and 1 inclusive"
            )
        if partial > supported:
            raise ValueError(
                "partial_threshold must be <= supported_threshold"
            )

        self._embedder = embedding_service or EmbeddingService()
        self.supported_threshold = supported
        self.partial_threshold = partial

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: str,
        sources: List[SearchResult],
    ) -> VerificationResult:
        """Verify each sentence of an answer against source chunks.

        Args:
            answer: The full LLM-generated answer text.
            sources: The retrieved chunks used as context.

        Returns:
            A VerificationResult with per-claim breakdowns and
            aggregate counts.  Sentences shorter than the filler
            threshold are skipped entirely.
        """
        sentences = [
            s for s in self._split_sentences(answer)
            if len(s) >= _MIN_SENTENCE_LENGTH
        ]
        if not sentences or not sources:
            return VerificationResult()

        # Embed answer sentences in one batch, coerce to 2-D.
        sentence_embeddings = _as_2d(self._embedder.embed_texts(sentences))

        # Embed chunks, preferring pre-computed embeddings.
        chunk_embeddings = self._collect_chunk_embeddings(sources)
        if chunk_embeddings.size == 0:
            return VerificationResult()

        sent_norm = _l2_normalise(sentence_embeddings)
        chunk_norm = _l2_normalise(chunk_embeddings)
        sims = sent_norm @ chunk_norm.T  # (num_sentences, num_chunks)

        claims: List[ClaimVerification] = []
        supported = partial = unverified = 0
        score_accumulator = 0.0

        for i, sentence in enumerate(sentences):
            row = sims[i]
            best_idx = int(np.argmax(row))
            best_score = float(max(0.0, row[best_idx]))
            best_chunk = sources[best_idx].chunk

            if best_score >= self.supported_threshold:
                status = "supported"
                supported += 1
                score_accumulator += 1.0
            elif best_score >= self.partial_threshold:
                status = "partial"
                partial += 1
                score_accumulator += 0.5
            else:
                status = "unverified"
                unverified += 1
                # contributes 0 to the aggregate

            preview = best_chunk.content.strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "…"

            claims.append(
                ClaimVerification(
                    claim=sentence,
                    status=status,
                    score=best_score,
                    best_chunk_id=best_chunk.chunk_id,
                    best_chunk_preview=preview,
                )
            )

        total = len(sentences)
        overall = score_accumulator / total if total else 0.0

        logger.info(
            "Fact verification: %d supported, %d partial, %d unverified "
            "(score=%.3f)",
            supported, partial, unverified, overall,
        )

        return VerificationResult(
            claims=claims,
            supported_count=supported,
            partial_count=partial,
            unverified_count=unverified,
            overall_verification_score=float(overall),
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
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms