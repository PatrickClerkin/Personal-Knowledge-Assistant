import re
from typing import List
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document

class SentenceChunker(BaseChunker):
    """
    Chunks text by sentences, respecting sentence boundaries.
    
    Pros: Better coherence than fixed-size, respects grammar
    Cons: Sentence detection isn't perfect, no semantic awareness
    Use case: Well-written prose, articles, academic papers
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into chunks at sentence boundaries."""
        chunks = []
        text = document.content
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk).strip()
                
                if chunk_content:
                    page_number = self._get_page_number(start_char, document)
                    
                    chunk = self._create_chunk(
                        content=chunk_content,
                        doc_id=doc_id,
                        source_doc_title=doc_title,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        page_number=page_number
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                start_char = start_char + len(' '.join(current_chunk)) - len(' '.join(overlap_sentences))
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunk_content = ' '.join(current_chunk).strip()
            if chunk_content:
                page_number = self._get_page_number(start_char, document)
                
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                    page_number=page_number
                )
                chunks.append(chunk)
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on chunk_overlap setting."""
        if not sentences:
            return []
        
        overlap_length = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _get_page_number(self, char_position: int, document: Document) -> int:
        """Determine which page a character position belongs to."""
        for section in document.sections:
            if section.start_char <= char_position < section.end_char:
                return section.page_number
        return None