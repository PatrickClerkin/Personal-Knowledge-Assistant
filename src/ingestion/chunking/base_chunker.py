from abc import ABC, abstractmethod
from typing import List
from ..document import Document
from .chunk import Chunk

class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index}"
    
    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        source_doc_title: str,
        chunk_index: int,
        start_char: int = 0,
        end_char: int = 0,
        section_id: str = None,
        page_number: int = None,
        total_chunks: int = None
    ) -> Chunk:
        """Helper to create a Chunk object with all metadata."""
        return Chunk(
            chunk_id=self._generate_chunk_id(doc_id, chunk_index),
            doc_id=doc_id,
            content=content,
            source_doc_title=source_doc_title,
            section_id=section_id,
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            total_chunks=total_chunks
        )