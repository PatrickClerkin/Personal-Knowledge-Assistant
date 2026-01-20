"""Document ingestion module for parsing, chunking, embedding, and storing documents."""

from .document import Document, DocumentMetadata, DocumentSection
from .parsers import PDFParser
from .chunking import Chunk, ChunkManager
from .embeddings import EmbeddingService
from .storage import VectorStore, FAISSVectorStore, SearchResult
from .knowledge_base import KnowledgeBase

__all__ = [
    # High-level API
    "KnowledgeBase",
    # Document models
    "Document",
    "DocumentMetadata",
    "DocumentSection",
    # Parsers
    "PDFParser",
    # Chunking
    "Chunk",
    "ChunkManager",
    # Embeddings
    "EmbeddingService",
    # Storage
    "VectorStore",
    "FAISSVectorStore",
    "SearchResult",
]
