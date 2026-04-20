"""
Answer grounding and confidence scoring.

After the LLM generates an answer, scores each source chunk by how
much it actually contributed to that answer using term overlap.

This gives users transparency into which chunks were genuinely used
vs merely retrieved, and flags answers that may be hallucinated
(low grounding across all chunks).

Design Pattern: Strategy Pattern — GroundingScorer is a standalone
callable; the pipeline delegates to it without coupling.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..ingestion.storage.vector_store import SearchResult


# Common English stopwords to exclude from term matching
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "not", "no", "as", "if", "so", "up", "out", "about",
    "than", "then", "when", "which", "who", "what", "how", "also",
}


@dataclass
class ChunkGroundingScore:
    """Grounding score for a single retrieved chunk.

    Attributes:
        chunk_source: Source document title of the chunk.
        page_number: Page number, if available.
        retrieval_score: Original retrieval similarity score.
        grounding_score: Term overlap score (0.0 to 1.0). Higher
            means the answer draws more heavily on this chunk.
        matched_terms: Significant terms shared between the answer
            and this chunk.
        is_grounded: True if grounding_score meets the threshold.
    """
    chunk_source: str
    page_number: Optional[int]
    retrieval_score: float
    grounding_score: float
    matched_terms: List[str]
    is_grounded: bool


@dataclass
class GroundingResult:
    """Full grounding analysis for a RAG response.

    Attributes:
        chunk_scores: Per-chunk grounding scores, sorted descending.
        overall_confidence: Mean grounding score of grounded chunks,
            or 0.0 if no chunks met the threshold.
        is_well_grounded: True if at least one chunk is grounded.
        answer_terms: Significant terms extracted from the answer.
    """
    chunk_scores: List[ChunkGroundingScore]
    overall_confidence: float
    is_well_grounded: bool
    answer_terms: List[str]


class GroundingScorer:
    """Scores how well an answer is grounded in retrieved chunks.

    Uses term overlap between the generated answer and each source
    chunk to estimate which chunks contributed to the answer. This
    is fast, interpretable, and requires no extra LLM calls.

    Attributes:
        threshold: Minimum grounding_score to mark a chunk as
            grounded (default 0.25).
        min_term_length: Minimum character length for a term to be
            considered significant (default 4).
    """

    def __init__(
        self,
        threshold: float = 0.25,
        min_term_length: int = 4,
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1 exclusive")
        self.threshold = threshold
        self.min_term_length = min_term_length

    def score(
        self,
        answer: str,
        sources: List[SearchResult],
    ) -> GroundingResult:
        """Score each source chunk's contribution to the answer.

        Args:
            answer: The LLM-generated answer text.
            sources: Retrieved chunks passed as context.

        Returns:
            GroundingResult with per-chunk scores and overall confidence.
        """
        answer_terms = self._extract_terms(answer)

        chunk_scores = []
        for result in sources:
            chunk_terms = self._extract_terms(result.chunk.content)
            matched = list(answer_terms & chunk_terms)
            score = self._overlap_score(answer_terms, chunk_terms)

            chunk_scores.append(ChunkGroundingScore(
                chunk_source=result.chunk.source_doc_title,
                page_number=result.chunk.page_number,
                retrieval_score=result.score,
                grounding_score=round(score, 4),
                matched_terms=sorted(matched),
                is_grounded=score >= self.threshold,
            ))

        # Sort by grounding score descending
        chunk_scores.sort(key=lambda c: c.grounding_score, reverse=True)

        grounded = [c for c in chunk_scores if c.is_grounded]
        overall_confidence = (
            round(sum(c.grounding_score for c in grounded) / len(grounded), 4)
            if grounded else 0.0
        )

        return GroundingResult(
            chunk_scores=chunk_scores,
            overall_confidence=overall_confidence,
            is_well_grounded=len(grounded) > 0,
            answer_terms=sorted(answer_terms),
        )

    def _extract_terms(self, text: str) -> Set[str]:
        """Extract significant lowercase terms from text."""
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return {
            t for t in tokens
            if len(t) >= self.min_term_length
            and t not in _STOPWORDS
        }

    def _overlap_score(
        self,
        answer_terms: Set[str],
        chunk_terms: Set[str],
    ) -> float:
        """Recall-oriented overlap score between two term sets.

        Score = |intersection| / |answer_terms| so it reflects
        coverage of the answer, not just general similarity.
        """
        if not answer_terms:
            return 0.0
        overlap = len(answer_terms & chunk_terms)
        return overlap / len(answer_terms)