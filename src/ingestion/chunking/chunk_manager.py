from typing import List
from .base_chunker import BaseChunker
from .fixed_size_chunker import FixedSizeChunker
from .sentence_chunker import SentenceChunker
from .semantic_chunker import SemanticChunker
from .chunk import Chunk
from ..document import Document

class ChunkManager:
    """Manages different chunking strategies."""
    
    def __init__(
        self, 
        strategy: str = "semantic",
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize chunking manager.
        
        Args:
            strategy: "fixed", "sentence", or "semantic"
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            similarity_threshold: For semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if strategy == "fixed":
            self.chunker = FixedSizeChunker(chunk_size, chunk_overlap)
        elif strategy == "sentence":
            self.chunker = SentenceChunker(chunk_size, chunk_overlap)
        elif strategy == "semantic":
            self.chunker = SemanticChunker(
                chunk_size, 
                chunk_overlap,
                similarity_threshold=similarity_threshold
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
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