from typing import List
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document

class FixedSizeChunker(BaseChunker):
    """
    Simple chunker that splits text into fixed-size chunks with overlap.
    
    Pros: Fast, predictable
    Cons: Cuts mid-sentence, no semantic awareness
    Use case: Baseline comparison, very fragmented text
    """
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks with overlap."""
        chunks = []
        text = document.content
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try not to cut in the middle of a word
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                page_number = self._get_page_number(start, document)
                
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    page_number=page_number
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _get_page_number(self, char_position: int, document: Document) -> int:
        """Determine which page a character position belongs to."""
        for section in document.sections:
            if section.start_char <= char_position < section.end_char:
                return section.page_number
        return None