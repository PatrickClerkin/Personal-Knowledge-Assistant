"""
Shared application state and helpers for Flask blueprints.

Centralises lazy-initialised singletons (KnowledgeBase, RAGPipeline,
AnnotationStore) so every blueprint accesses the same instances.

Configuration is layered: ``configs/config.yaml`` provides defaults,
environment variables override individual settings for deployment
flexibility, and dataclass defaults from :mod:`src.config` act as a
last-resort fallback.
"""

import os
from pathlib import Path

from ...config import get_config
from ...ingestion.knowledge_base import KnowledgeBase
from ...rag.annotations import AnnotationStore
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialised singletons
_kb = None
_rag = None
_annotations = None


def get_kb() -> KnowledgeBase:
    """Get or create the KnowledgeBase singleton.

    Configuration resolution order:
        1. Environment variable (``KB_INDEX_PATH``, ``KB_STRATEGY``, …)
        2. ``configs/config.yaml``
        3. Dataclass default in :mod:`src.config`
    """
    global _kb
    if _kb is None:
        cfg = get_config()

        index_path = os.environ.get(
            "KB_INDEX_PATH", str(cfg.storage.index_path)
        )
        strategy = os.environ.get("KB_STRATEGY", cfg.chunking.strategy)
        chunk_size = int(
            os.environ.get("KB_CHUNK_SIZE", cfg.chunking.chunk_size)
        )
        chunk_overlap = int(
            os.environ.get("KB_CHUNK_OVERLAP", cfg.chunking.chunk_overlap)
        )
        embedding_model = os.environ.get(
            "KB_EMBEDDING_MODEL", cfg.embedding.model_name
        )

        _kb = KnowledgeBase(
            index_path=index_path,
            embedding_model=embedding_model,
            chunk_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(
            "KnowledgeBase initialised: %d chunks "
            "(strategy=%s, index=%s)",
            _kb.size, strategy, index_path,
        )
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
    """Serialise a GroundingResult to a JSON-safe list of per-sentence scores.

    The grounding module scores each sentence of the answer against
    the retrieved chunks using embedding cosine similarity.  This
    helper flattens that result into a list of dicts consumable by
    the frontend.
    """
    if not grounding or not getattr(grounding, "sentences", None):
        return []
    return [
        {
            "sentence": s.sentence,
            "confidence": round(s.confidence, 3),
            "label": grounding_label(s.confidence),
            "is_grounded": s.confidence >= 0.5,
            "best_chunk_id": s.best_chunk_id,
        }
        for s in grounding.sentences
    ]