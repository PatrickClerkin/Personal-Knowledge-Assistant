"""
Unified Semantic Chunker

This module provides a unified interface to multiple semantic chunking strategies.
It allows users to select from various approaches based on their needs:

1. embedding_similarity: Uses embedding cosine similarity to detect topic boundaries
2. density_clustering: Uses HDBSCAN to find natural sentence clusters
3. topic_modeling: Uses BERTopic for topic-based segmentation
4. recursive_hierarchical: Creates multi-level hierarchical chunks

All approaches provide genuine semantic awareness, unlike simple page/section splitting.
"""

from typing import List, Optional, Dict, Any
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document


class SemanticChunker(BaseChunker):
    """
    Unified semantic chunker supporting multiple strategies.
    
    This is the main entry point for semantic chunking. It delegates to
    specialized implementations based on the selected method.
    
    Available methods:
    - "embedding_similarity": Best for detecting topic transitions
    - "density_clustering": Best for documents with clear topic clusters
    - "topic_modeling": Best for multi-topic documents needing labels
    - "recursive_hierarchical": Best for structured documents
    - "auto": Automatically selects based on document characteristics
    
    Example:
        chunker = SemanticChunker(method="embedding_similarity")
        chunks = chunker.chunk_document(document)
    """
    
    METHOD_NAME = "semantic"
    
    AVAILABLE_METHODS = [
        "embedding_similarity",
        "density_clustering", 
        "topic_modeling",
        "recursive_hierarchical",
        "auto"
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        method: str = "embedding_similarity",
        similarity_threshold: float = 0.5,
        embedding_model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target maximum chunk size
            chunk_overlap: Overlap between chunks
            method: Semantic chunking method to use
            similarity_threshold: Threshold for embedding_similarity method
            embedding_model: Sentence transformer model name
            **kwargs: Additional arguments passed to the specific chunker
        """
        super().__init__(chunk_size, chunk_overlap)
        
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {self.AVAILABLE_METHODS}"
            )
        
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.kwargs = kwargs
        
        # Lazy-loaded specific chunker
        self._chunker = None
    
    def _get_chunker(self) -> BaseChunker:
        """Get or create the specific chunker implementation."""
        if self._chunker is not None:
            return self._chunker
        
        if self.method == "embedding_similarity":
            from .embedding_similarity_chunker import EmbeddingSimilarityChunker
            self._chunker = EmbeddingSimilarityChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                similarity_threshold=self.similarity_threshold,
                embedding_model=self.embedding_model_name,
                **self.kwargs
            )
        
        elif self.method == "density_clustering":
            from .density_chunker import DensityBasedChunker
            self._chunker = DensityBasedChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_model=self.embedding_model_name,
                **self.kwargs
            )
        
        elif self.method == "topic_modeling":
            from .topic_chunker import TopicBasedChunker
            self._chunker = TopicBasedChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_model=self.embedding_model_name,
                **self.kwargs
            )
        
        elif self.method == "recursive_hierarchical":
            from .recursive_chunker import RecursiveHierarchicalChunker
            self._chunker = RecursiveHierarchicalChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_model=self.embedding_model_name,
                **self.kwargs
            )
        
        elif self.method == "auto":
            # Auto-selection will be done per document
            self._chunker = None
        
        return self._chunker
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document using the selected semantic method.
        """
        if self.method == "auto":
            return self._auto_chunk(document)
        
        chunker = self._get_chunker()
        return chunker.chunk_document(document)
    
    def _auto_chunk(self, document: Document) -> List[Chunk]:
        """
        Automatically select best chunking method based on document characteristics.
        
        Heuristics:
        - Short documents (<2000 chars): embedding_similarity (simple, fast)
        - Well-structured (many sections): recursive_hierarchical
        - Long documents with unclear structure: density_clustering
        - Default: embedding_similarity
        """
        text_length = len(document.content)
        num_sections = len(document.sections)
        
        # Analyze document characteristics
        has_clear_structure = num_sections > 2
        is_short = text_length < 2000
        is_long = text_length > 10000
        
        # Count potential topic indicators (headers, numbered sections, etc.)
        import re
        header_pattern = re.compile(r'(?:^|\n)(?:[A-Z][A-Z\s]+:|(?:\d+\.)+\s+[A-Z]|#{1,3}\s+)', re.MULTILINE)
        header_matches = len(header_pattern.findall(document.content))
        has_many_headers = header_matches > 3
        
        # Select method
        if is_short:
            selected_method = "embedding_similarity"
        elif has_clear_structure or has_many_headers:
            selected_method = "recursive_hierarchical"
        elif is_long:
            selected_method = "density_clustering"
        else:
            selected_method = "embedding_similarity"
        
        # Create appropriate chunker
        if selected_method == "embedding_similarity":
            from .embedding_similarity_chunker import EmbeddingSimilarityChunker
            chunker = EmbeddingSimilarityChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                similarity_threshold=self.similarity_threshold,
                embedding_model=self.embedding_model_name,
            )
        elif selected_method == "recursive_hierarchical":
            from .recursive_chunker import RecursiveHierarchicalChunker
            chunker = RecursiveHierarchicalChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_model=self.embedding_model_name,
            )
        else:  # density_clustering
            from .density_chunker import DensityBasedChunker
            chunker = DensityBasedChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_model=self.embedding_model_name,
            )
        
        chunks = chunker.chunk_document(document)
        
        # Mark chunks with auto-selected method
        for chunk in chunks:
            chunk.chunking_method = f"auto:{selected_method}"
        
        return chunks
    
    def get_analysis(self, document: Document) -> Dict[str, Any]:
        """
        Get analysis/debugging information from the underlying chunker.
        
        Returns method-specific analysis data.
        """
        if self.method == "embedding_similarity":
            chunker = self._get_chunker()
            if hasattr(chunker, 'get_similarity_profile'):
                sentences, similarities = chunker.get_similarity_profile(document)
                return {
                    "method": self.method,
                    "n_sentences": len(sentences),
                    "similarities": similarities.tolist() if len(similarities) > 0 else [],
                    "threshold": self.similarity_threshold,
                    "n_boundaries": sum(1 for s in similarities if s < self.similarity_threshold)
                }
        
        elif self.method == "density_clustering":
            chunker = self._get_chunker()
            if hasattr(chunker, 'get_cluster_analysis'):
                return chunker.get_cluster_analysis(document)
        
        elif self.method == "topic_modeling":
            chunker = self._get_chunker()
            if hasattr(chunker, 'get_topic_analysis'):
                return chunker.get_topic_analysis(document)
        
        elif self.method == "recursive_hierarchical":
            chunker = self._get_chunker()
            if hasattr(chunker, 'get_hierarchy_summary'):
                return chunker.get_hierarchy_summary(document)
        
        return {"method": self.method, "analysis": "not available"}
    
    @classmethod
    def compare_methods(
        cls,
        document: Document,
        methods: List[str] = None,
        chunk_size: int = 512
    ) -> Dict[str, Any]:
        """
        Compare different chunking methods on the same document.
        
        Useful for evaluation and method selection.
        
        Args:
            document: Document to chunk
            methods: List of methods to compare (default: all except auto)
            chunk_size: Chunk size for all methods
            
        Returns:
            Dictionary with comparison results
        """
        if methods is None:
            methods = ["embedding_similarity", "density_clustering", "recursive_hierarchical"]
        
        results = {}
        
        for method in methods:
            try:
                chunker = cls(chunk_size=chunk_size, method=method)
                chunks = chunker.chunk_document(document)
                
                # Compute statistics
                sizes = [len(c.content) for c in chunks]
                
                results[method] = {
                    "n_chunks": len(chunks),
                    "avg_size": sum(sizes) / len(sizes) if sizes else 0,
                    "min_size": min(sizes) if sizes else 0,
                    "max_size": max(sizes) if sizes else 0,
                    "size_std": (
                        (sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes))**0.5
                        if len(sizes) > 1 else 0
                    ),
                    "total_chars": sum(sizes),
                    "coverage": sum(sizes) / len(document.content) if document.content else 0,
                }
                
                # Add method-specific metadata
                if chunks:
                    sample_chunk = chunks[0]
                    results[method]["sample_metadata"] = {
                        "has_topic_id": sample_chunk.topic_id is not None,
                        "has_cluster_id": sample_chunk.cluster_id is not None,
                        "has_hierarchy": sample_chunk.hierarchy_level is not None,
                        "has_coherence": sample_chunk.coherence_score is not None,
                    }
                    
            except Exception as e:
                results[method] = {"error": str(e)}
        
        return results
