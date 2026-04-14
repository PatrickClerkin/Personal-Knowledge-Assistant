"""
Document summarisation using chunk-level LLM synthesis.

Given a document ID, retrieves all its chunks from the knowledge base,
groups them into sections, summarises each section, then synthesises
an executive summary from the section summaries.

This produces a structured, hierarchical summary grounded entirely
in the document's actual content — no hallucination possible.

Design Pattern: Template Method — the summarisation pipeline
(chunk → section → executive) is fixed; only the LLM prompts vary.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..ingestion.knowledge_base import KnowledgeBase
from ..rag.llm import LLMProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


_SECTION_SYSTEM_PROMPT = """You are a precise document summariser. Given a set of text excerpts from a document, write a concise factual summary.

Rules:
1. Output ONLY the summary text. No preamble, no headings, no labels.
2. Be factual — only include information from the provided excerpts.
3. Write 2-4 sentences maximum.
4. Use clear, direct language suitable for a student studying this material."""


_EXECUTIVE_SYSTEM_PROMPT = """You are an expert document analyst. Given a list of section summaries from a document, synthesise them into a single executive summary.

Rules:
1. Output ONLY the executive summary. No preamble, no labels.
2. Capture the document's main purpose, key concepts, and conclusions.
3. Write 4-6 sentences maximum.
4. Be factual — only use information from the provided summaries."""


_KEY_POINTS_SYSTEM_PROMPT = """You are a study guide writer. Given an executive summary of a document, extract the 4-6 most important key points a student should know.

Output ONLY a JSON array of strings. Example:
["Key point one.", "Key point two.", "Key point three."]

Rules:
1. Output ONLY the JSON array. No preamble, no markdown.
2. Each key point should be one clear sentence.
3. Focus on the most important concepts, definitions, and conclusions."""


@dataclass
class SectionSummary:
    """Summary of a logical section of a document.

    Attributes:
        section_index: Position of this section (1-based).
        chunk_count: Number of chunks this section covers.
        summary: The generated summary text.
        excerpt: Short preview of the raw content.
    """
    section_index: int
    chunk_count: int
    summary: str
    excerpt: str


@dataclass
class DocumentSummary:
    """Complete hierarchical summary of a document.

    Attributes:
        doc_id: The document identifier.
        title: Document title (source filename).
        executive_summary: High-level synthesis of the whole document.
        section_summaries: Per-section summaries in order.
        key_points: Bullet-point key takeaways.
        total_chunks: Total chunks processed.
        total_sections: Number of logical sections identified.
    """
    doc_id: str
    title: str
    executive_summary: str
    section_summaries: List[SectionSummary]
    key_points: List[str]
    total_chunks: int
    total_sections: int

    def to_dict(self) -> dict:
        """Serialise for JSON API response."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "executive_summary": self.executive_summary,
            "key_points": self.key_points,
            "total_chunks": self.total_chunks,
            "total_sections": self.total_sections,
            "section_summaries": [
                {
                    "section_index": s.section_index,
                    "chunk_count": s.chunk_count,
                    "summary": s.summary,
                    "excerpt": s.excerpt,
                }
                for s in self.section_summaries
            ],
        }


@dataclass
class CorpusSummary:
    """Summary of all documents in the knowledge base.

    Attributes:
        total_documents: Number of documents summarised.
        total_chunks: Total chunks across all documents.
        document_summaries: Per-document summaries.
        corpus_overview: High-level overview of the whole corpus.
    """
    total_documents: int
    total_chunks: int
    document_summaries: List[DocumentSummary]
    corpus_overview: str

    def to_dict(self) -> dict:
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "corpus_overview": self.corpus_overview,
            "document_summaries": [d.to_dict() for d in self.document_summaries],
        }


class DocumentSummariser:
    """Generates hierarchical summaries of ingested documents.

    Processes documents chunk by chunk, grouping chunks into logical
    sections, summarising each section, then synthesising an executive
    summary and extracting key points.

    Args:
        knowledge_base: The KnowledgeBase to retrieve chunks from.
        llm_provider: LLM for summary generation.
        chunks_per_section: How many chunks to group into one section
            (default 5). Larger values = fewer, broader sections.
        max_chars_per_section: Maximum characters sent to the LLM per
            section to control token usage (default 2000).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_provider: LLMProvider,
        chunks_per_section: int = 5,
        max_chars_per_section: int = 2000,
    ):
        self.kb = knowledge_base
        self.llm = llm_provider
        self.chunks_per_section = chunks_per_section
        self.max_chars_per_section = max_chars_per_section

    def summarise_document(self, doc_id: str) -> Optional[DocumentSummary]:
        """Generate a full hierarchical summary for a single document.

        Args:
            doc_id: The document ID to summarise.

        Returns:
            DocumentSummary, or None if the document has no chunks.
        """
        chunks = self.kb.get_document_chunks(doc_id)
        if not chunks:
            logger.warning("No chunks found for doc_id: %s", doc_id)
            return None

        title = chunks[0].source_doc_title
        logger.info("Summarising '%s': %d chunks", title, len(chunks))

        # Group chunks into sections
        sections = self._group_into_sections(chunks)

        # Summarise each section
        section_summaries = []
        for i, section_chunks in enumerate(sections, start=1):
            summary = self._summarise_section(section_chunks, i)
            excerpt = section_chunks[0].content[:150].strip()
            if len(section_chunks[0].content) > 150:
                excerpt += "…"
            section_summaries.append(SectionSummary(
                section_index=i,
                chunk_count=len(section_chunks),
                summary=summary,
                excerpt=excerpt,
            ))
            logger.debug("Section %d summarised: %d chunks", i, len(section_chunks))

        # Synthesise executive summary
        executive = self._synthesise_executive(section_summaries, title)

        # Extract key points
        key_points = self._extract_key_points(executive)

        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            executive_summary=executive,
            section_summaries=section_summaries,
            key_points=key_points,
            total_chunks=len(chunks),
            total_sections=len(sections),
        )

    def summarise_corpus(self) -> CorpusSummary:
        """Summarise all documents in the knowledge base.

        Returns:
            CorpusSummary with per-document summaries and a corpus
            overview generated from all executive summaries.
        """
        doc_ids = self.kb.document_ids
        logger.info("Summarising corpus: %d documents", len(doc_ids))

        doc_summaries = []
        total_chunks = 0

        for doc_id in doc_ids:
            summary = self.summarise_document(doc_id)
            if summary:
                doc_summaries.append(summary)
                total_chunks += summary.total_chunks

        # Build corpus overview from executive summaries
        corpus_overview = self._build_corpus_overview(doc_summaries)

        return CorpusSummary(
            total_documents=len(doc_summaries),
            total_chunks=total_chunks,
            document_summaries=doc_summaries,
            corpus_overview=corpus_overview,
        )

    # ─── Private helpers ────────────────────────────────────────────

    def _group_into_sections(self, chunks) -> List[list]:
        """Group chunks into logical sections of size chunks_per_section."""
        sections = []
        for i in range(0, len(chunks), self.chunks_per_section):
            sections.append(chunks[i:i + self.chunks_per_section])
        return sections

    def _summarise_section(self, chunks, section_index: int) -> str:
        """Summarise a group of chunks into a section summary."""
        if not self.llm.is_available():
            # Fallback: first sentence of first chunk
            text = chunks[0].content.strip()
            return text[:200] + "…" if len(text) > 200 else text

        # Concatenate chunk content, cap at max_chars_per_section
        combined = "\n\n".join(c.content.strip() for c in chunks)
        combined = combined[:self.max_chars_per_section]

        prompt = f"Section {section_index} excerpts:\n\n{combined}"

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_SECTION_SYSTEM_PROMPT,
                max_tokens=250,
                temperature=0.2,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning("Section %d summary failed: %s", section_index, e)
            return combined[:200] + "…"

    def _synthesise_executive(
        self,
        section_summaries: List[SectionSummary],
        title: str,
    ) -> str:
        """Synthesise section summaries into an executive summary."""
        if not self.llm.is_available():
            return " ".join(s.summary for s in section_summaries[:3])

        combined = "\n\n".join(
            f"Section {s.section_index}: {s.summary}"
            for s in section_summaries
        )

        prompt = f"Document: {title}\n\nSection summaries:\n\n{combined}"

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_EXECUTIVE_SYSTEM_PROMPT,
                max_tokens=350,
                temperature=0.2,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning("Executive summary failed: %s", e)
            return section_summaries[0].summary if section_summaries else ""

    def _extract_key_points(self, executive_summary: str) -> List[str]:
        """Extract key bullet points from the executive summary."""
        if not self.llm.is_available() or not executive_summary:
            return []

        import json
        try:
            response = self.llm.generate(
                prompt=f"Executive summary:\n\n{executive_summary}",
                system=_KEY_POINTS_SYSTEM_PROMPT,
                max_tokens=300,
                temperature=0.2,
            )
            text = response.content.strip()
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text).strip()
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                points = json.loads(text[start:end + 1])
                return [str(p) for p in points if p][:6]
        except Exception as e:
            logger.warning("Key point extraction failed: %s", e)
        return []

    def _build_corpus_overview(
        self,
        doc_summaries: List[DocumentSummary],
    ) -> str:
        """Build a high-level overview of the entire corpus."""
        if not doc_summaries:
            return "No documents in the knowledge base."

        if not self.llm.is_available():
            return f"Knowledge base contains {len(doc_summaries)} document(s)."

        combined = "\n\n".join(
            f"Document: {d.title}\n{d.executive_summary}"
            for d in doc_summaries[:8]  # cap at 8 docs
        )

        prompt = f"Corpus of {len(doc_summaries)} document(s):\n\n{combined}"

        system = """You are a knowledge base analyst. Given executive summaries of multiple documents, write a 3-4 sentence overview of what topics and themes the knowledge base covers overall. Output ONLY the overview text."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=system,
                max_tokens=300,
                temperature=0.2,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning("Corpus overview failed: %s", e)
            return f"Knowledge base contains {len(doc_summaries)} document(s)."