"""
Chunk Manager - Unified interface for all chunking strategies.

This module provides a single entry point for document chunking,
supporting both simple (fixed, sentence) and semantic (embedding,
clustering, topic, hierarchical) strategies.
"""

from typing import List, Optional, Dict, Any
from .base_chunker import BaseChunker
from .fixed_size_chunker import FixedSizeChunker
from .sentence_chunker import SentenceChunker
from .chunk import Chunk
from ..document import Document


class ChunkManager:
    """
    Manages different chunking strategies with a unified interface.
    
    Supported strategies:
    
    Simple strategies (fast, no ML):
    - "fixed": Fixed-size chunks with word boundary awareness
    - "sentence": Sentence-boundary respecting chunks
    
    Semantic strategies (use embeddings/ML):
    - "semantic": Embedding similarity-based boundaries (default semantic)
    - "embedding_similarity": Same as semantic
    - "density_clustering": HDBSCAN clustering on embeddings
    - "topic_modeling": BERTopic-based topic segmentation
    - "recursive_hierarchical": Multi-level hierarchical chunks
    - "auto": Auto-selects best semantic method for document
    
    Example:
        # Simple chunking
        manager = ChunkManager(strategy="sentence")
        chunks = manager.chunk_document(document)
        
        # Semantic chunking
        manager = ChunkManager(strategy="embedding_similarity", similarity_threshold=0.4)
        chunks = manager.chunk_document(document)
        
        # Compare all methods
        comparison = manager.compare_strategies(document)
    """
    
    # Strategy categories
    SIMPLE_STRATEGIES = ["fixed", "sentence"]
    SEMANTIC_STRATEGIES = [
        "semantic",
        "embedding_similarity", 
        "density_clustering",
        "topic_modeling",
        "recursive_hierarchical",
        "auto"
    ]
    
    def __init__(
        self, 
        strategy: str = "sentence",
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
        embedding_model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize chunking manager.
        
        Args:
            strategy: Chunking strategy name
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            similarity_threshold: For semantic strategies
            embedding_model: Model name for semantic strategies
            **kwargs: Additional strategy-specific parameters
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.kwargs = kwargs
        
        # Create the appropriate chunker
        self.chunker = self._create_chunker()
    
    def _create_chunker(self) -> BaseChunker:
        """Create the appropriate chunker based on strategy."""
        
        # Simple strategies
        if self.strategy == "fixed":
            return FixedSizeChunker(self.chunk_size, self.chunk_overlap)
        
        elif self.strategy == "sentence":
            return SentenceChunker(self.chunk_size, self.chunk_overlap)
        
        # Semantic strategies
        elif self.strategy in ["semantic", "embedding_similarity"]:
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                method="embedding_similarity",
                similarity_threshold=self.similarity_threshold,
                embedding_model=self.embedding_model,
                **self.kwargs
            )
        
        elif self.strategy == "density_clustering":
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                method="density_clustering",
                embedding_model=self.embedding_model,
                **self.kwargs
            )
        
        elif self.strategy == "topic_modeling":
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                method="topic_modeling",
                embedding_model=self.embedding_model,
                **self.kwargs
            )
        
        elif self.strategy == "recursive_hierarchical":
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                method="recursive_hierarchical",
                embedding_model=self.embedding_model,
                **self.kwargs
            )
        
        elif self.strategy == "auto":
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                method="auto",
                similarity_threshold=self.similarity_threshold,
                embedding_model=self.embedding_model,
                **self.kwargs
            )
        
        else:
            available = self.SIMPLE_STRATEGIES + self.SEMANTIC_STRATEGIES
            raise ValueError(
                f"Unknown chunking strategy: {self.strategy}. "
                f"Available: {available}"
            )
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document using the configured strategy."""
        return self.chunker.chunk_document(document)
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def get_analysis(self, document: Document) -> Dict[str, Any]:
        """
        Get analysis information from the chunker.
        
        Only available for semantic strategies.
        """
        if hasattr(self.chunker, 'get_analysis'):
            return self.chunker.get_analysis(document)
        return {"strategy": self.strategy, "analysis": "not available"}
    
    def compare_strategies(
        self,
        document: Document,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple chunking strategies on the same document.
        
        Args:
            document: Document to chunk
            strategies: List of strategies to compare (default: common ones)
            
        Returns:
            Dictionary with comparison results for each strategy
        """
        if strategies is None:
            strategies = ["fixed", "sentence", "embedding_similarity", "density_clustering"]
        
        results = {}
        
        for strategy in strategies:
            try:
                manager = ChunkManager(
                    strategy=strategy,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    similarity_threshold=self.similarity_threshold,
                    embedding_model=self.embedding_model
                )
                
                chunks = manager.chunk_document(document)
                
                # Compute statistics
                sizes = [len(c.content) for c in chunks]
                
                results[strategy] = {
                    "n_chunks": len(chunks),
                    "avg_size": sum(sizes) / len(sizes) if sizes else 0,
                    "min_size": min(sizes) if sizes else 0,
                    "max_size": max(sizes) if sizes else 0,
                    "total_chars": sum(sizes),
                    "coverage_ratio": sum(sizes) / len(document.content) if document.content else 0,
                }
                
                # Add semantic metadata if available
                if chunks and strategy in self.SEMANTIC_STRATEGIES:
                    sample = chunks[0]
                    results[strategy]["has_semantic_metadata"] = any([
                        sample.topic_id is not None,
                        sample.cluster_id is not None,
                        sample.hierarchy_level is not None,
                        sample.boundary_similarity is not None,
                        sample.coherence_score is not None,
                    ])
                
            except Exception as e:
                results[strategy] = {"error": str(e)}
        
        return results
    
    @classmethod
    def available_strategies(cls) -> Dict[str, List[str]]:
        """Get all available chunking strategies grouped by type."""
        return {
            "simple": cls.SIMPLE_STRATEGIES,
            "semantic": cls.SEMANTIC_STRATEGIES
        }
    
    @classmethod
    def recommend_strategy(cls, document: Document) -> str:
        """
        Recommend a chunking strategy based on document characteristics.
        
        Args:
            document: Document to analyze
            
        Returns:
            Recommended strategy name
        """
        text_length = len(document.content)
        num_sections = len(document.sections)
        
        # Very short documents - simple is fine
        if text_length < 1000:
            return "sentence"
        
        # Short to medium with clear structure
        if text_length < 5000 and num_sections > 2:
            return "recursive_hierarchical"
        
        # Medium length - embedding similarity works well
        if text_length < 20000:
            return "embedding_similarity"
        
        # Long documents - density clustering scales better
        return "density_clustering"
