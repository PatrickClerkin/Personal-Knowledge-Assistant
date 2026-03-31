"""
Answer fact verification at the sentence level.

After the LLM generates an answer, splits it into individual
sentences and checks each one against the retrieved source chunks
using term overlap. Each sentence is labelled:

    supported  — strong overlap with at least one source chunk
    partial    — weak overlap, possibly grounded but uncertain
    unverified — no meaningful overlap with any source chunk

This makes hallucinations visible: sentences the model invented
will appear as "unverified" while grounded sentences are green.

Design Pattern: Extends the Strategy Pattern established by
GroundingScorer — same term extraction logic, applied per sentence.
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional

from ..ingestion.storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Stopwords shared with GroundingScorer
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
class SentenceVerdict:
    """Verification result for a single answer sentence.

    Attributes:
        sentence: The original sentence text.
        status: One of 'supported', 'partial', 'unverified'.
        best_score: Highest term overlap score across all chunks.
        best_source: Source document with the highest overlap.
        best_page: Page number of best source, if available.
    """
    sentence: str
    status: str  # 'supported' | 'partial' | 'unverified'
    best_score: float
    best_source: Optional[str] = None
    best_page: Optional[int] = None


@dataclass
class VerificationResult:
    """Full fact verification result for an answer.

    Attributes:
        verdicts: Per-sentence verification results.
        supported_count: Number of supported sentences.
        partial_count: Number of partially supported sentences.
        unverified_count: Number of unverified sentences.
        overall_verification_score: Fraction of sentences that are
            supported or partial.
    """
    verdicts: List[SentenceVerdict]
    supported_count: int
    partial_count: int
    unverified_count: int
    overall_verification_score: float

    def to_dict(self) -> dict:
        """Serialise for JSON API response."""
        return {
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "unverified_count": self.unverified_count,
            "overall_verification_score": round(
                self.overall_verification_score, 4
            ),
            "verdicts": [
                {
                    "sentence": v.sentence,
                    "status": v.status,
                    "best_score": round(v.best_score, 4),
                    "best_source": v.best_source,
                    "best_page": v.best_page,
                }
                for v in self.verdicts
            ],
        }


class FactVerifier:
    """Verifies answer sentences against retrieved source chunks.

    Splits the generated answer into sentences, then measures term
    overlap between each sentence and every source chunk. The chunk
    with the highest overlap determines the sentence's verdict.

    Attributes:
        supported_threshold: Overlap score required for 'supported'
            (default 0.35).
        partial_threshold: Overlap score required for 'partial'
            (default 0.12).
        min_term_length: Minimum characters for a term to count
            (default 4).
        min_sentence_words: Sentences shorter than this are skipped
            as they carry little semantic content (default 4).
    """

    def __init__(
        self,
        supported_threshold: float = 0.35,
        partial_threshold: float = 0.12,
        min_term_length: int = 4,
        min_sentence_words: int = 4,
    ):
        if not 0.0 < partial_threshold < supported_threshold <= 1.0:
            raise ValueError(
                "Thresholds must satisfy: "
                "0 < partial_threshold < supported_threshold <= 1"
            )
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold
        self.min_term_length = min_term_length
        self.min_sentence_words = min_sentence_words

    def verify(
        self,
        answer: str,
        sources: List[SearchResult],
    ) -> VerificationResult:
        """Verify each sentence of the answer against source chunks.

        Args:
            answer: The LLM-generated answer text.
            sources: Retrieved chunks used as context.

        Returns:
            VerificationResult with per-sentence verdicts.
        """
        if not sources:
            return self._empty_result(answer)

        # Pre-compute chunk term sets once
        chunk_data = []
        for source in sources:
            terms = self._extract_terms(source.chunk.content)
            chunk_data.append({
                "terms": terms,
                "source": source.chunk.source_doc_title,
                "page": source.chunk.page_number,
            })

        sentences = self._split_sentences(answer)
        verdicts = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) < self.min_sentence_words:
                continue

            sent_terms = self._extract_terms(sentence)
            if not sent_terms:
                continue

            verdict = self._verify_sentence(sentence, sent_terms, chunk_data)
            verdicts.append(verdict)

        supported = sum(1 for v in verdicts if v.status == "supported")
        partial = sum(1 for v in verdicts if v.status == "partial")
        unverified = sum(1 for v in verdicts if v.status == "unverified")
        total = len(verdicts)

        overall = (supported + 0.5 * partial) / total if total > 0 else 0.0

        logger.info(
            "Fact verification: %d supported, %d partial, %d unverified "
            "(score=%.3f)",
            supported, partial, unverified, overall,
        )

        return VerificationResult(
            verdicts=verdicts,
            supported_count=supported,
            partial_count=partial,
            unverified_count=unverified,
            overall_verification_score=round(overall, 4),
        )

    def _verify_sentence(
        self,
        sentence: str,
        sent_terms: Set[str],
        chunk_data: List[dict],
    ) -> SentenceVerdict:
        """Find the best-matching chunk and assign a verdict."""
        best_score = 0.0
        best_source = None
        best_page = None

        for cd in chunk_data:
            score = self._overlap_score(sent_terms, cd["terms"])
            if score > best_score:
                best_score = score
                best_source = cd["source"]
                best_page = cd["page"]

        if best_score >= self.supported_threshold:
            status = "supported"
        elif best_score >= self.partial_threshold:
            status = "partial"
        else:
            status = "unverified"

        return SentenceVerdict(
            sentence=sentence,
            status=status,
            best_score=best_score,
            best_source=best_source,
            best_page=best_page,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation boundaries."""
        # Split on . ! ? followed by space or end of string
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = []
        for s in raw:
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences

    def _extract_terms(self, text: str) -> Set[str]:
        """Extract significant lowercase terms, excluding stopwords."""
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return {
            t for t in tokens
            if len(t) >= self.min_term_length
            and t not in _STOPWORDS
        }

    def _overlap_score(
        self,
        sentence_terms: Set[str],
        chunk_terms: Set[str],
    ) -> float:
        """Recall-oriented overlap: what fraction of sentence terms
        appear in the chunk."""
        if not sentence_terms:
            return 0.0
        overlap = len(sentence_terms & chunk_terms)
        return overlap / len(sentence_terms)

    def _empty_result(self, answer: str) -> VerificationResult:
        """Return all-unverified result when no sources available."""
        sentences = self._split_sentences(answer)
        verdicts = [
            SentenceVerdict(
                sentence=s,
                status="unverified",
                best_score=0.0,
            )
            for s in sentences
            if len(s.split()) >= self.min_sentence_words
        ]
        return VerificationResult(
            verdicts=verdicts,
            supported_count=0,
            partial_count=0,
            unverified_count=len(verdicts),
            overall_verification_score=0.0,
        )