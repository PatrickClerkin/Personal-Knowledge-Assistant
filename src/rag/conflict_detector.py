"""
Cross-document conflict detection.

Given a topic, retrieves relevant chunks from the knowledge base,
groups them by source document, then uses the LLM to identify
contradictions and disagreements between documents.

This surfaces cases where Document A says X and Document B says Y
on the same topic — something a simple RAG pipeline would silently
paper over by picking the highest-scoring chunk.

Design Pattern: Pipeline Pattern — retrieve → group → compare →
report, each step feeding cleanly into the next.
"""

import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Dict

from ..ingestion.knowledge_base import KnowledgeBase
from ..rag.llm import LLMProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


_CONFLICT_SYSTEM_PROMPT = """You are an expert fact-checker analysing two document excerpts for contradictions and disagreements.

Compare the two excerpts and identify any factual conflicts, contradictions, or significant disagreements.

Output ONLY a valid JSON object with exactly these fields:
{
  "has_conflict": true or false,
  "conflict_type": "factual" | "definitional" | "numerical" | "procedural" | "none",
  "severity": "high" | "medium" | "low" | "none",
  "description": "Brief description of the conflict in 1-2 sentences",
  "excerpt_a_claim": "The specific claim from Document A that conflicts",
  "excerpt_b_claim": "The specific claim from Document B that conflicts"
}

Rules:
1. Output ONLY the JSON object. No preamble, no markdown fences.
2. Set has_conflict to false if the excerpts are compatible or discuss different aspects.
3. Only flag genuine contradictions, not merely different levels of detail.
4. Keep descriptions concise and factual."""


@dataclass
class Conflict:
    """A detected conflict between two document excerpts.

    Attributes:
        source_a: Title of the first source document.
        source_b: Title of the second source document.
        page_a: Page number in source A, if available.
        page_b: Page number in source B, if available.
        excerpt_a: Relevant excerpt from source A.
        excerpt_b: Relevant excerpt from source B.
        conflict_type: Category of conflict.
        severity: How significant the conflict is.
        description: Human-readable conflict summary.
        claim_a: The specific conflicting claim from source A.
        claim_b: The specific conflicting claim from source B.
    """
    source_a: str
    source_b: str
    page_a: Optional[int]
    page_b: Optional[int]
    excerpt_a: str
    excerpt_b: str
    conflict_type: str
    severity: str
    description: str
    claim_a: str
    claim_b: str

    def to_dict(self) -> dict:
        """Serialise for JSON API response."""
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "page_a": self.page_a,
            "page_b": self.page_b,
            "excerpt_a": self.excerpt_a,
            "excerpt_b": self.excerpt_b,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "description": self.description,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
        }


@dataclass
class ConflictReport:
    """Full conflict detection report for a topic.

    Attributes:
        topic: The topic analysed.
        conflicts: All detected conflicts, sorted by severity.
        documents_analysed: Number of unique documents compared.
        pairs_compared: Number of document pairs evaluated.
        high_count: Number of high-severity conflicts.
        medium_count: Number of medium-severity conflicts.
        low_count: Number of low-severity conflicts.
    """
    topic: str
    conflicts: List[Conflict]
    documents_analysed: int
    pairs_compared: int
    high_count: int
    medium_count: int
    low_count: int

    def to_dict(self) -> dict:
        """Serialise for JSON API response."""
        return {
            "topic": self.topic,
            "documents_analysed": self.documents_analysed,
            "pairs_compared": self.pairs_compared,
            "total_conflicts": len(self.conflicts),
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


class ConflictDetector:
    """Detects contradictions between documents in the knowledge base.

    Retrieves relevant chunks for a topic, groups them by source
    document, then compares each pair of documents using the LLM to
    find factual contradictions or significant disagreements.

    Args:
        knowledge_base: The KnowledgeBase to retrieve from.
        llm_provider: LLM for conflict analysis.
        top_k: Total chunks to retrieve for the topic (default 10).
        max_pairs: Maximum document pairs to compare (default 10).
            Caps cost on large corpora.
        excerpt_length: Characters of each chunk to send to the LLM
            (default 800).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_provider: LLMProvider,
        top_k: int = 10,
        max_pairs: int = 10,
        excerpt_length: int = 800,
    ):
        self.kb = knowledge_base
        self.llm = llm_provider
        self.top_k = top_k
        self.max_pairs = max_pairs
        self.excerpt_length = excerpt_length

    def detect(self, topic: str) -> ConflictReport:
        """Detect conflicts across documents for a given topic.

        Args:
            topic: The subject to search for conflicts on.

        Returns:
            ConflictReport with all detected conflicts.
        """
        logger.info("Detecting conflicts for topic: '%s'", topic)

        # Retrieve relevant chunks
        results = self.kb.search(topic, top_k=self.top_k)
        if not results:
            return self._empty_report(topic)

        # Group chunks by source document
        doc_chunks: Dict[str, dict] = {}
        for result in results:
            source = result.chunk.source_doc_title
            if source not in doc_chunks:
                doc_chunks[source] = {
                    "content": result.chunk.content,
                    "page": result.chunk.page_number,
                    "score": result.score,
                }
            else:
                # Keep the highest-scoring chunk per document
                if result.score > doc_chunks[source]["score"]:
                    doc_chunks[source] = {
                        "content": result.chunk.content,
                        "page": result.chunk.page_number,
                        "score": result.score,
                    }

        sources = list(doc_chunks.keys())
        if len(sources) < 2:
            logger.info(
                "Only %d unique source(s) found — no pairs to compare.",
                len(sources),
            )
            return ConflictReport(
                topic=topic,
                conflicts=[],
                documents_analysed=len(sources),
                pairs_compared=0,
                high_count=0,
                medium_count=0,
                low_count=0,
            )

        # Compare each pair of documents
        pairs = list(combinations(sources, 2))[:self.max_pairs]
        conflicts: List[Conflict] = []

        for source_a, source_b in pairs:
            chunk_a = doc_chunks[source_a]
            chunk_b = doc_chunks[source_b]

            conflict = self._compare_pair(
                source_a=source_a,
                source_b=source_b,
                chunk_a=chunk_a,
                chunk_b=chunk_b,
                topic=topic,
            )

            if conflict:
                conflicts.append(conflict)
                logger.info(
                    "Conflict found: %s vs %s (%s, %s)",
                    source_a[:30], source_b[:30],
                    conflict.conflict_type, conflict.severity,
                )

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        conflicts.sort(key=lambda c: severity_order.get(c.severity, 3))

        high = sum(1 for c in conflicts if c.severity == "high")
        medium = sum(1 for c in conflicts if c.severity == "medium")
        low = sum(1 for c in conflicts if c.severity == "low")

        logger.info(
            "Conflict detection complete: %d conflicts "
            "(%d high, %d medium, %d low) from %d pairs",
            len(conflicts), high, medium, low, len(pairs),
        )

        return ConflictReport(
            topic=topic,
            conflicts=conflicts,
            documents_analysed=len(sources),
            pairs_compared=len(pairs),
            high_count=high,
            medium_count=medium,
            low_count=low,
        )

    def _compare_pair(
        self,
        source_a: str,
        source_b: str,
        chunk_a: dict,
        chunk_b: dict,
        topic: str,
    ) -> Optional[Conflict]:
        """Compare two document chunks for conflicts using the LLM."""
        if not self.llm.is_available():
            return None

        excerpt_a = chunk_a["content"][:self.excerpt_length]
        excerpt_b = chunk_b["content"][:self.excerpt_length]

        prompt = (
            f"Topic being analysed: {topic}\n\n"
            f"Document A ({source_a}):\n{excerpt_a}\n\n"
            f"Document B ({source_b}):\n{excerpt_b}"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_CONFLICT_SYSTEM_PROMPT,
                max_tokens=400,
                temperature=0.1,
            )
            text = response.content.strip()
            text = self._clean_json(text)
            result = json.loads(text)

            if not result.get("has_conflict", False):
                return None

            severity = result.get("severity", "low")
            if severity == "none":
                return None

            return Conflict(
                source_a=source_a,
                source_b=source_b,
                page_a=chunk_a.get("page"),
                page_b=chunk_b.get("page"),
                excerpt_a=excerpt_a,
                excerpt_b=excerpt_b,
                conflict_type=result.get("conflict_type", "factual"),
                severity=severity,
                description=result.get("description", ""),
                claim_a=result.get("excerpt_a_claim", ""),
                claim_b=result.get("excerpt_b_claim", ""),
            )

        except Exception as e:
            logger.warning(
                "Conflict comparison failed for %s vs %s: %s",
                source_a[:30], source_b[:30], e,
            )
            return None

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and extract JSON object."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]
        return text

    def _empty_report(self, topic: str) -> ConflictReport:
        """Return an empty report when no chunks are found."""
        return ConflictReport(
            topic=topic,
            conflicts=[],
            documents_analysed=0,
            pairs_compared=0,
            high_count=0,
            medium_count=0,
            low_count=0,
        )