from typing import List
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document

class SemanticChunker(BaseChunker):
    """
    Structure-aware chunker that respects document boundaries (pages/sections).
    
    This is the "smart" chunker for now - it splits by document structure.
    True semantic chunking (with embeddings) will be added in Week 2.
    
    Pros: Fast, respects document structure, better than fixed-size
    Use case: Production use until we add embeddings
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
        embedding_model=None
    ):
        """
        Args:
            chunk_size: Max chunk size (soft limit)
            chunk_overlap: Overlap in characters
            similarity_threshold: Not used yet (for future semantic chunking)
            embedding_model: Not used yet (for future semantic chunking)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document by structure (pages/sections).
        
        This is a smart fallback until we implement true semantic chunking in Week 2.
        """
        return self._chunk_by_structure(document)
    
    def _chunk_by_structure(self, document: Document) -> List[Chunk]:
        """
        Chunk by document structure (sections/pages).
        
        Better than fixed-size because it respects document boundaries.
        Won't cut in the middle of a page or section.
        """
        chunks = []
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        chunk_index = 0
        
        for section in document.sections:
            section_text = section.content.strip()
            
            if not section_text:
                continue
            
            # If section is larger than chunk_size, split it
            if len(section_text) > self.chunk_size:
                start = 0
                while start < len(section_text):
                    end = min(start + self.chunk_size, len(section_text))
                    
                    # Try not to cut words
                    if end < len(section_text):
                        last_space = section_text.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space
                    
                    chunk_content = section_text[start:end].strip()
                    
                    if chunk_content:
                        chunk = self._create_chunk(
                            content=chunk_content,
                            doc_id=doc_id,
                            source_doc_title=doc_title,
                            chunk_index=chunk_index,
                            start_char=section.start_char + start,
                            end_char=section.start_char + end,
                            section_id=section.section_id,
                            page_number=section.page_number
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    start = end - self.chunk_overlap
            else:
                # Section fits in one chunk - keep it whole
                chunk = self._create_chunk(
                    content=section_text,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=chunk_index,
                    start_char=section.start_char,
                    end_char=section.end_char,
                    section_id=section.section_id,
                    page_number=section.page_number
                )
                chunks.append(chunk)
                chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _get_page_number(self, char_position: int, document: Document) -> int:
        """Determine which page a character position belongs to."""
        for section in document.sections:
            if section.start_char <= char_position < section.end_char:
                return section.page_number
        return None