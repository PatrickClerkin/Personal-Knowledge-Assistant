"""
Shared application state and helpers for Flask blueprints.

Centralises lazy-initialised singletons (KnowledgeBase, RAGPipeline,
AnnotationStore) so every blueprint accesses the same instances.
"""

import os
from pathlib import Path

from ...ingestion.knowledge_base import KnowledgeBase
from ...rag.annotations import AnnotationStore
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialised singletons
_kb = None
_rag = None
_annotations = None


def get_kb() -> KnowledgeBase:
    """Get or create the KnowledgeBase singleton."""
    global _kb
    if _kb is None:
        index_path = os.environ.get("KB_INDEX_PATH", "data/index/default")
        strategy = os.environ.get("KB_STRATEGY", "sentence")
        _kb = KnowledgeBase(
            index_path=index_path,
            chunk_strategy=strategy,
        )
        logger.info("KnowledgeBase initialised: %d chunks", _kb.size)
    return _kb


def get_rag():
    """Get or create the RAG pipeline (if API key is set)."""
    global _rag
    if _rag is None and os.environ.get("ANTHROPIC_API_KEY"):
        from ...rag.llm import ClaudeProvider
        from ...rag.pipeline import RAGPipeline
        _rag = RAGPipeline(
            knowledge_base=get_kb(),
            llm_provider=ClaudeProvider(),
            rerank=True,
            hybrid=True,
            top_k=int(os.environ.get("RAG_TOP_K", "8")),
            max_context_chars=int(os.environ.get("RAG_MAX_CONTEXT", "16000")),
            memory_persist_path=Path("data/memory/conversation.json"),
        )
        logger.info(
            "RAG pipeline initialised with reranking, hybrid search, "
            "and persistent memory."
        )
    return _rag


def get_annotations() -> AnnotationStore:
    """Get or create the AnnotationStore singleton."""
    global _annotations
    if _annotations is None:
        _annotations = AnnotationStore()
    return _annotations


def reset_singletons() -> None:
    """Reset all singletons — used by tests."""
    global _kb, _rag, _annotations
    _kb = None
    _rag = None
    _annotations = None


def grounding_label(score: float) -> str:
    """Compute a human-readable grounding label from a score."""
    if score >= 0.40:
        return "strong"
    if score >= 0.15:
        return "partial"
    return "weak"


def grounding_to_list(grounding) -> list:
    """Serialise a GroundingResult's chunk scores to a JSON-safe list."""
    if not grounding:
        return []
    return [
        {
            "source": c.chunk_source,
            "page": c.page_number,
            "grounding_score": c.grounding_score,
            "label": grounding_label(c.grounding_score),
            "is_grounded": c.is_grounded,
            "matched_terms": c.matched_terms,
        }
        for c in grounding.chunk_scores
    ]