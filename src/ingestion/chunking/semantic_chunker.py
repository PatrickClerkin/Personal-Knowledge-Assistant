from typing import List
import numpy as np
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document

class SemanticChunker(BaseChunker):
    """
    Advanced chunker that splits on semantic boundaries (topic changes).
    
    Pros: Best quality - keeps related content together, splits on topic shifts
    Cons: Slower, requires embedding model (falls back to structure-based)
    Use case: Production use - academic papers, long documents, knowledge bases
    
    How it works:
    1. Split text into sentences
    2. Embed each sentence (convert to vector)
    3. Calculate similarity between consecutive sentences
    4. When similarity drops below threshold → topic changed → split there
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
            similarity_threshold: When cosine similarity drops below this, split
            embedding_model: Model for generating embeddings (None = use fallback)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document on semantic boundaries."""
        
        # If no embedding model provided, fall back to structure-based chunking
        if self.embedding_model is None:
            return self._chunk_by_structure(document)
        
        chunks = []
        text = document.content
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            chunk = self._create_chunk(
                content=text.strip(),
                doc_id=doc_id,
                source_doc_title=doc_title,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
                total_chunks=1
            )
            return [chunk]
        
        embeddings = self._get_embeddings(sentences)
        similarities = self._calculate_similarities(embeddings)
        split_indices = self._find_split_points(similarities, sentences)
        chunks = self._create_chunks_from_splits(
            sentences, split_indices, doc_id, doc_title, document
        )
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences."""
        # TODO: Replace with actual embedding model in Week 2
        # For now, returns random vectors (placeholder)
        embeddings = []
        for sentence in sentences:
            embedding = np.random.rand(384)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _calculate_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Calculate cosine similarity between consecutive sentence embeddings."""
        similarities = []
        
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i + 1]
            
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            similarities.append(similarity)
        
        return similarities
    
    def _find_split_points(
        self, 
        similarities: List[float], 
        sentences: List[str]
    ) -> List[int]:
        """Find indices where we should split (low similarity = topic change)."""
        split_indices = [0]
        current_chunk_size = len(sentences[0])
        
        for i, similarity in enumerate(similarities):
            sentence_length = len(sentences[i + 1])
            
            # Split if similarity drops (topic change) OR chunk too large
            if (similarity < self.similarity_threshold or 
                current_chunk_size + sentence_length > self.chunk_size):
                split_indices.append(i + 1)
                current_chunk_size = sentence_length
            else:
                current_chunk_size += sentence_length
        
        return split_indices
    
    def _create_chunks_from_splits(
        self,
        sentences: List[str],
        split_indices: List[int],
        doc_id: str,
        doc_title: str,
        document: Document
    ) -> List[Chunk]:
        """Create chunks from split points."""
        chunks = []
        
        for i in range(len(split_indices)):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1] if i + 1 < len(split_indices) else len(sentences)
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = ' '.join(chunk_sentences).strip()
            
            if chunk_content:
                char_start = sum(len(s) for s in sentences[:start_idx])
                char_end = char_start + len(chunk_content)
                
                page_number = self._get_page_number(char_start, document)
                
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=len(chunks),
                    start_char=char_start,
                    end_char=char_end,
                    page_number=page_number
                )
                chunks.append(chunk)
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_structure(self, document: Document) -> List[Chunk]:
        """
        Fallback: chunk by document structure (sections/pages).
        
        This is used when no embedding model is available.
        Better than fixed-size because it respects document boundaries.
        """
        chunks = []
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        chunk_index = 0
        
        for section in document.sections:
            section_text = section.content.strip()
            
            if len(section_text) > self.chunk_size:
                # Split large section
                start = 0
                while start < len(section_text):
                    end = min(start + self.chunk_size, len(section_text))
                    
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
                # Section fits in one chunk
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
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _get_page_number(self, char_position: int, document: Document) -> int:
        """Determine which page a character position belongs to."""
        for section in document.sections:
            if section.start_char <= char_position < section.end_char:
                return section.page_number
        return None