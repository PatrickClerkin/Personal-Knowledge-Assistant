"""
Personalised study path generator.

Given a topic, searches the knowledge base for relevant content,
uses the knowledge graph to discover related concepts, then asks
the LLM to produce a dependency-ordered study plan grounded in
the actual ingested documents.

The output is a structured StudyPath — a sequence of StudySections
ordered from foundational to advanced, each with a summary, key
concepts, and source references.

Design Pattern: Facade Pattern — PathGenerator coordinates the KB,
knowledge graph, and LLM behind a single clean interface.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ..ingestion.knowledge_base import KnowledgeBase
from ..rag.llm import LLMProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


_CONCEPT_ORDER_PROMPT = """You are a curriculum designer. Given a topic and a list of related concepts found in a document corpus, order them from most foundational to most advanced for a student studying this topic.

Rules:
1. Output ONLY a JSON array of concept strings in learning order.
2. Remove duplicates and irrelevant concepts.
3. Keep at most 8 concepts.
4. Most fundamental concept first, most advanced last.
5. No explanation, no preamble — just the JSON array.

Example output: ["introduction", "core theory", "advanced applications"]"""


_SECTION_SUMMARY_PROMPT = """You are a study guide writer. Given a concept and relevant excerpts from lecture notes and documents, write a concise study section for a student.

Rules:
1. Write 3-5 sentences maximum.
2. Be factual and grounded in the provided excerpts only.
3. Highlight the most important points a student needs to understand.
4. Do not invent information not present in the excerpts.
5. Output only the summary text, no headers or labels."""


@dataclass
class StudySection:
    """A single section in a personalised study path.

    Attributes:
        concept: The concept or topic this section covers.
        summary: LLM-generated study summary grounded in sources.
        key_terms: Important terms extracted from the content.
        sources: List of source references (title + page).
        chunk_previews: Short previews of the source chunks.
        order: Position in the study path (1 = first to study).
    """
    concept: str
    summary: str
    key_terms: List[str]
    sources: List[str]
    chunk_previews: List[str]
    order: int


@dataclass
class StudyPath:
    """A complete personalised study path for a topic.

    Attributes:
        topic: The original topic requested.
        sections: Ordered list of study sections.
        total_sources: Number of unique source documents used.
        estimated_minutes: Rough reading time estimate.
    """
    topic: str
    sections: List[StudySection]
    total_sources: int
    estimated_minutes: int

    def to_dict(self) -> dict:
        """Serialise to JSON-ready dict for the API."""
        return {
            "topic": self.topic,
            "estimated_minutes": self.estimated_minutes,
            "total_sources": self.total_sources,
            "sections": [
                {
                    "order": s.order,
                    "concept": s.concept,
                    "summary": s.summary,
                    "key_terms": s.key_terms,
                    "sources": s.sources,
                    "chunk_previews": s.chunk_previews,
                }
                for s in self.sections
            ],
        }


class PathGenerator:
    """Generates personalised study paths from the knowledge base.

    Combines semantic search, knowledge graph concept discovery,
    and LLM-powered curriculum ordering to produce study plans
    grounded in the user's actual documents.

    Args:
        knowledge_base: The KnowledgeBase to search.
        llm_provider: LLM for concept ordering and summaries.
        graph: Optional pre-loaded NetworkX graph for concept
            discovery. If None, graph-based expansion is skipped.
        top_k: Chunks to retrieve per concept (default 3).
        max_concepts: Maximum sections in the study path (default 6).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_provider: LLMProvider,
        graph=None,
        top_k: int = 3,
        max_concepts: int = 6,
    ):
        self.kb = knowledge_base
        self.llm = llm_provider
        self.graph = graph
        self.top_k = top_k
        self.max_concepts = max_concepts

    def generate(self, topic: str) -> StudyPath:
        """Generate a personalised study path for a topic.

        Args:
            topic: The subject the user wants to study.

        Returns:
            A StudyPath with ordered sections grounded in documents.
        """
        logger.info("Generating study path for topic: '%s'", topic)

        # Step 1: Find related concepts from KB and knowledge graph
        raw_concepts = self._discover_concepts(topic)
        logger.info("Discovered %d raw concepts", len(raw_concepts))

        # Step 2: Ask LLM to order concepts by learning dependency
        ordered_concepts = self._order_concepts(topic, raw_concepts)
        logger.info("Ordered concepts: %s", ordered_concepts)

        # Step 3: Build a study section for each concept
        sections = []
        all_sources = set()

        for i, concept in enumerate(ordered_concepts[:self.max_concepts], start=1):
            section = self._build_section(concept, i)
            if section:
                sections.append(section)
                all_sources.update(section.sources)

        # Estimate reading time: ~2 minutes per section
        estimated_minutes = max(5, len(sections) * 2)

        return StudyPath(
            topic=topic,
            sections=sections,
            total_sources=len(all_sources),
            estimated_minutes=estimated_minutes,
        )

    def _discover_concepts(self, topic: str) -> List[str]:
        """Find related concepts via KB search and knowledge graph.

        Searches the KB for the topic, extracts key terms from
        results, then expands using graph neighbours if available.
        """
        concepts = set()
        concepts.add(topic.lower())

        # Pull top chunks for the topic
        results = self.kb.search(topic, top_k=8)
        for r in results:
            # Extract multi-word noun phrases as candidate concepts
            words = r.chunk.content.lower().split()
            for word in words:
                clean = word.strip(".,!?;:()")
                if len(clean) > 4 and clean.isalpha():
                    concepts.add(clean)

        # Expand via knowledge graph neighbours
        if self.graph is not None:
            topic_lower = topic.lower()
            # Find nodes that contain the topic word
            matching_nodes = [
                n for n in self.graph.nodes
                if topic_lower in n or n in topic_lower
            ]
            for node in matching_nodes[:3]:
                neighbours = list(self.graph.neighbors(node))
                # Take the most connected neighbours
                neighbours_sorted = sorted(
                    neighbours,
                    key=lambda n: self.graph.degree(n),
                    reverse=True,
                )
                for neighbour in neighbours_sorted[:5]:
                    concepts.add(neighbour)

        return list(concepts)[:20]  # cap before sending to LLM

    def _order_concepts(self, topic: str, concepts: List[str]) -> List[str]:
        """Ask the LLM to order concepts by learning dependency."""
        if not self.llm.is_available():
            # Fallback: put topic first, rest alphabetical
            others = sorted([c for c in concepts if c != topic.lower()])
            return [topic] + others[:self.max_concepts - 1]

        prompt = (
            f"Topic: {topic}\n"
            f"Related concepts found in documents: {json.dumps(concepts)}"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_CONCEPT_ORDER_PROMPT,
                max_tokens=256,
                temperature=0.2,
            )
            text = response.content.strip()
            # Strip markdown code fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            ordered = json.loads(text)
            if isinstance(ordered, list):
                return [str(c) for c in ordered[:self.max_concepts]]
        except Exception as e:
            logger.warning("Concept ordering failed: %s", e)

        # Fallback
        return concepts[:self.max_concepts]

    def _build_section(self, concept: str, order: int) -> Optional[StudySection]:
        """Build a single study section for a concept.

        Retrieves relevant chunks, extracts key terms and sources,
        generates an LLM summary grounded in the content.
        """
        results = self.kb.search(concept, top_k=self.top_k)
        if not results:
            return None

        # Gather source references and chunk previews
        sources = []
        chunk_previews = []
        context_parts = []
        seen_sources = set()

        for r in results:
            source = r.chunk.source_doc_title
            page = r.chunk.page_number
            source_ref = f"{source}" + (f" p.{page}" if page else "")

            if source_ref not in seen_sources:
                sources.append(source_ref)
                seen_sources.add(source_ref)

            preview = r.chunk.content.strip()[:180]
            if len(r.chunk.content) > 180:
                preview += "..."
            chunk_previews.append(preview)
            context_parts.append(r.chunk.content[:600])

        # Extract key terms: words ≥5 chars, not stopwords
        _STOP = {"about", "their", "there", "these", "which", "would",
                  "could", "should", "being", "using", "based", "other"}
        all_text = " ".join(context_parts).lower()
        word_freq: Dict[str, int] = {}
        for word in all_text.split():
            clean = word.strip(".,!?;:()")
            if len(clean) >= 5 and clean.isalpha() and clean not in _STOP:
                word_freq[clean] = word_freq.get(clean, 0) + 1

        key_terms = sorted(word_freq, key=word_freq.get, reverse=True)[:6]

        # Generate summary
        summary = self._generate_summary(concept, context_parts)

        return StudySection(
            concept=concept,
            summary=summary,
            key_terms=key_terms,
            sources=sources,
            chunk_previews=chunk_previews,
            order=order,
        )

    def _generate_summary(self, concept: str, excerpts: List[str]) -> str:
        """Generate a study summary for a concept from source excerpts."""
        if not self.llm.is_available() or not excerpts:
            return f"Study material found for: {concept}."

        context = "\n\n".join(excerpts[:3])
        prompt = f"Concept: {concept}\n\nSource excerpts:\n{context}"

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_SECTION_SUMMARY_PROMPT,
                max_tokens=200,
                temperature=0.3,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning("Summary generation failed for '%s': %s", concept, e)
            return excerpts[0][:300] if excerpts else f"Content on {concept}."