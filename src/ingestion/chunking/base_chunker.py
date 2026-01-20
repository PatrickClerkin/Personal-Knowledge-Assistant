from abc import ABC, abstractmethod
from typing import List, Optional
from ..document import Document
from .chunk import Chunk


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    # Class-level identifier for the chunking method
    METHOD_NAME: str = "base"
    
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
        total_chunks: int = None,
        **kwargs  # For additional metadata from semantic chunkers
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
            total_chunks=total_chunks,
            chunking_method=self.METHOD_NAME,
            **kwargs
        )
    
    def _get_page_number(self, char_position: int, document: Document) -> Optional[int]:
        """Determine which page a character position belongs to."""
        for section in document.sections:
            if section.start_char <= char_position < section.end_char:
                return section.page_number
        return None
    
    def _finalize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply final processing to all chunks (set total_chunks, etc.)."""
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = total
            chunk.chunk_index = i
        return chunks
