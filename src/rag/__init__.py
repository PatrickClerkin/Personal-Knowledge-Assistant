"""Retrieval-Augmented Generation: LLM integration with context grounding."""

from .llm import LLMProvider, ClaudeProvider
from .pipeline import RAGPipeline
from .memory import ConversationMemory, ConversationTurn
from .grounding import GroundingScorer, GroundingResult
from .cache import SemanticCache, CacheEntry
from .fact_verifier import FactVerifier, VerificationResult, SentenceVerdict
from .conflict_detector import ConflictDetector, ConflictReport, Conflict
from .query_history import QueryHistory, QueryRecord
from .annotations import AnnotationStore, Annotation


__all__ = [
    "LLMProvider",
    "ClaudeProvider",
    "RAGPipeline",
    "ConversationMemory",
    "ConversationTurn",
    "GroundingScorer",
    "GroundingResult",
    "SemanticCache",
    "CacheEntry",
    "FactVerifier",
    "VerificationResult",
    "SentenceVerdict",
    "ConflictDetector",
    "ConflictReport",
    "Conflict",
    "QueryHistory",
    "QueryRecord",
    "AnnotationStore",
    "Annotation",
]