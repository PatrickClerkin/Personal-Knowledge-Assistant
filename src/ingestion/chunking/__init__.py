"""
Chunking Module - Multiple strategies for document chunking.

This module provides various chunking strategies ranging from simple
fixed-size chunking to advanced semantic-aware approaches.

Simple Strategies:
- FixedSizeChunker: Basic fixed-size with word boundary awareness
- SentenceChunker: Respects sentence boundaries

Semantic Strategies:
- EmbeddingSimilarityChunker: Detects topic shifts via embedding similarity
- DensityBasedChunker: Uses HDBSCAN clustering for natural groupings
- TopicBasedChunker: BERTopic-based topic segmentation
- RecursiveHierarchicalChunker: Multi-level hierarchical chunks

Unified Interface:
- ChunkManager: Single entry point for all strategies
- SemanticChunker: Unified interface for semantic strategies

Example Usage:
    from src.ingestion.chunking import ChunkManager, Chunk
    
    # Simple chunking
    manager = ChunkManager(strategy="sentence")
    chunks = manager.chunk_document(document)
    
    # Semantic chunking with embedding similarity
    manager = ChunkManager(strategy="embedding_similarity", similarity_threshold=0.4)
    chunks = manager.chunk_document(document)
    
    # Auto-select best semantic method
    manager = ChunkManager(strategy="auto")
    chunks = manager.chunk_document(document)
"""

from .chunk import Chunk
from .base_chunker import BaseChunker
from .fixed_size_chunker import FixedSizeChunker
from .sentence_chunker import SentenceChunker
from .chunk_manager import ChunkManager

# Semantic chunkers - import with error handling for optional dependencies
try:
    from .embedding_similarity_chunker import EmbeddingSimilarityChunker
except ImportError:
    EmbeddingSimilarityChunker = None

try:
    from .density_chunker import DensityBasedChunker
except ImportError:
    DensityBasedChunker = None

try:
    from .topic_chunker import TopicBasedChunker
except ImportError:
    TopicBasedChunker = None

try:
    from .recursive_chunker import RecursiveHierarchicalChunker, MultiResolutionChunker
except ImportError:
    RecursiveHierarchicalChunker = None
    MultiResolutionChunker = None

try:
    from .semantic_chunker import SemanticChunker
except ImportError:
    SemanticChunker = None

__all__ = [
    # Core
    'Chunk',
    'BaseChunker',
    'ChunkManager',
    
    # Simple chunkers
    'FixedSizeChunker',
    'SentenceChunker',
    
    # Semantic chunkers
    'SemanticChunker',
    'EmbeddingSimilarityChunker',
    'DensityBasedChunker',
    'TopicBasedChunker',
    'RecursiveHierarchicalChunker',
    'MultiResolutionChunker',
]
