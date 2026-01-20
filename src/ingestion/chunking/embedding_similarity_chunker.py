"""
Embedding-based Semantic Chunker

This chunker uses sentence embeddings to detect semantic boundaries.
It splits text when the cosine similarity between consecutive sentences
drops below a threshold, indicating a topic shift.

This is true semantic chunking - it uses the actual meaning of text
to determine boundaries rather than arbitrary character counts.
"""

import re
from typing import List, Optional, Tuple
import numpy as np
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document


class EmbeddingSimilarityChunker(BaseChunker):
    """
    Semantic chunker that detects topic boundaries using embedding similarity.
    
    Algorithm:
    1. Split document into sentences
    2. Generate embeddings for each sentence
    3. Compute similarity between consecutive sentences
    4. Identify boundaries where similarity drops below threshold
    5. Group sentences between boundaries into chunks
    6. Merge small chunks and split large ones while respecting boundaries
    
    This approach genuinely captures semantic structure because it uses
    the actual meaning of sentences (via embeddings) to find natural
    topic boundaries.
    """
    
    METHOD_NAME = "embedding_similarity"
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        embedding_model: Optional[str] = None,
        window_size: int = 3,  # For smoothed similarity
        percentile_threshold: Optional[float] = None,  # Alternative to fixed threshold
    ):
        """
        Initialize the embedding-based semantic chunker.
        
        Args:
            chunk_size: Maximum target chunk size (soft limit)
            chunk_overlap: Overlap between chunks
            similarity_threshold: Fixed similarity threshold for boundaries (0-1)
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
            embedding_model: Model name for embeddings (None = default)
            window_size: Window for computing smoothed similarity
            percentile_threshold: If set, use percentile of similarities as threshold
                                  (e.g., 25 means boundaries at lowest 25% similarities)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.window_size = window_size
        self.percentile_threshold = percentile_threshold
        
        # Lazy load embedding model
        self._embedding_model = None
        self._model_name = embedding_model or "all-MiniLM-L6-v2"
        
        # Sentence splitter
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    @property
    def embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._model_name)
        return self._embedding_model
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document using embedding-based semantic boundaries.
        """
        text = document.content
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        # Step 1: Split into sentences with position tracking
        sentences, positions = self._split_sentences_with_positions(text)
        
        if len(sentences) <= 1:
            # Single sentence or empty - return as single chunk
            return self._create_single_chunk(document)
        
        # Step 2: Generate embeddings for all sentences
        embeddings = self.embedding_model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Step 3: Compute similarities between consecutive sentences
        similarities = self._compute_consecutive_similarities(embeddings)
        
        # Step 4: Identify boundary indices
        boundaries = self._identify_boundaries(similarities)
        
        # Step 5: Create chunks from sentence groups
        chunks = self._create_chunks_from_boundaries(
            sentences=sentences,
            positions=positions,
            boundaries=boundaries,
            similarities=similarities,
            doc_id=doc_id,
            doc_title=doc_title,
            document=document
        )
        
        return self._finalize_chunks(chunks)
    
    def _split_sentences_with_positions(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split text into sentences while tracking character positions.
        
        Returns:
            Tuple of (sentences list, positions list of (start, end) tuples)
        """
        sentences = []
        positions = []
        
        # Find all sentence splits
        parts = self.sentence_pattern.split(text)
        
        current_pos = 0
        for part in parts:
            part = part.strip()
            if part:
                # Find actual position in original text
                start = text.find(part, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(part)
                
                sentences.append(part)
                positions.append((start, end))
                current_pos = end
        
        return sentences, positions
    
    def _compute_consecutive_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between consecutive sentence embeddings.
        
        Uses a smoothed window to reduce noise from individual sentence variations.
        """
        n = len(embeddings)
        if n <= 1:
            return np.array([])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute similarities between consecutive sentences
        raw_similarities = np.array([
            np.dot(normalized[i], normalized[i + 1])
            for i in range(n - 1)
        ])
        
        # Apply smoothing with window
        if self.window_size > 1 and len(raw_similarities) >= self.window_size:
            smoothed = np.convolve(
                raw_similarities,
                np.ones(self.window_size) / self.window_size,
                mode='same'
            )
            return smoothed
        
        return raw_similarities
    
    def _identify_boundaries(self, similarities: np.ndarray) -> List[int]:
        """
        Identify chunk boundaries based on similarity drops.
        
        Returns list of indices where chunks should end (after sentence at index i).
        """
        if len(similarities) == 0:
            return []
        
        # Determine threshold
        if self.percentile_threshold is not None:
            # Dynamic threshold based on document's similarity distribution
            threshold = np.percentile(similarities, self.percentile_threshold)
        else:
            threshold = self.similarity_threshold
        
        # Find boundary points (where similarity drops below threshold)
        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        boundaries: List[int],
        similarities: np.ndarray,
        doc_id: str,
        doc_title: str,
        document: Document
    ) -> List[Chunk]:
        """
        Create chunks by grouping sentences between boundaries.
        """
        chunks = []
        chunk_index = 0
        
        # Add implicit boundaries at start and end
        all_boundaries = [-1] + boundaries + [len(sentences) - 1]
        
        for i in range(len(all_boundaries) - 1):
            start_idx = all_boundaries[i] + 1
            end_idx = all_boundaries[i + 1] + 1
            
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            
            if not chunk_sentences:
                continue
            
            # Combine sentences
            content = ' '.join(chunk_sentences)
            
            # Handle chunks that are too large or too small
            if len(content) > self.chunk_size * 1.5:
                # Split large chunks while trying to respect sub-boundaries
                sub_chunks = self._split_large_chunk(
                    chunk_sentences, positions[start_idx:end_idx],
                    doc_id, doc_title, document, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            elif len(content) < self.min_chunk_size and chunks:
                # Merge small chunk with previous
                prev_chunk = chunks[-1]
                prev_chunk.content = prev_chunk.content + ' ' + content
                prev_chunk.end_char = positions[end_idx - 1][1]
                # Update boundary similarity to lowest in merged region
                if len(similarities) > start_idx - 1 >= 0:
                    prev_sim = prev_chunk.boundary_similarity or 1.0
                    new_sim = similarities[start_idx - 1] if start_idx > 0 else 1.0
                    prev_chunk.boundary_similarity = min(prev_sim, new_sim)
            else:
                # Normal chunk creation
                start_char = positions[start_idx][0]
                end_char = positions[end_idx - 1][1]
                page_number = self._get_page_number(start_char, document)
                
                # Get boundary similarity (lowest similarity at this boundary)
                boundary_sim = None
                if end_idx - 1 < len(similarities):
                    boundary_sim = float(similarities[end_idx - 1])
                
                chunk = self._create_chunk(
                    content=content,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    page_number=page_number,
                    boundary_similarity=boundary_sim
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _split_large_chunk(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        doc_id: str,
        doc_title: str,
        document: Document,
        base_index: int
    ) -> List[Chunk]:
        """
        Split a chunk that exceeds the maximum size.
        Uses simple size-based splitting as fallback.
        """
        chunks = []
        current_sentences = []
        current_length = 0
        chunk_offset = 0
        
        for i, sentence in enumerate(sentences):
            if current_length + len(sentence) > self.chunk_size and current_sentences:
                # Create chunk
                content = ' '.join(current_sentences)
                start_idx = i - len(current_sentences)
                start_char = positions[start_idx][0]
                end_char = positions[i - 1][1]
                
                chunk = self._create_chunk(
                    content=content,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=base_index + chunk_offset,
                    start_char=start_char,
                    end_char=end_char,
                    page_number=self._get_page_number(start_char, document)
                )
                chunks.append(chunk)
                chunk_offset += 1
                
                current_sentences = []
                current_length = 0
            
            current_sentences.append(sentence)
            current_length += len(sentence) + 1  # +1 for space
        
        # Handle remaining sentences
        if current_sentences:
            content = ' '.join(current_sentences)
            start_idx = len(sentences) - len(current_sentences)
            start_char = positions[start_idx][0]
            end_char = positions[-1][1]
            
            chunk = self._create_chunk(
                content=content,
                doc_id=doc_id,
                source_doc_title=doc_title,
                chunk_index=base_index + chunk_offset,
                start_char=start_char,
                end_char=end_char,
                page_number=self._get_page_number(start_char, document)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_single_chunk(self, document: Document) -> List[Chunk]:
        """Create a single chunk for very short documents."""
        content = document.content.strip()
        if not content:
            return []
        
        chunk = self._create_chunk(
            content=content,
            doc_id=document.metadata.doc_id,
            source_doc_title=document.metadata.title,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            page_number=1 if document.sections else None
        )
        chunk.total_chunks = 1
        return [chunk]
    
    def get_similarity_profile(self, document: Document) -> Tuple[List[str], np.ndarray]:
        """
        Get the similarity profile of a document for visualization/analysis.
        
        Returns:
            Tuple of (sentences, similarities array)
        """
        sentences, _ = self._split_sentences_with_positions(document.content)
        
        if len(sentences) <= 1:
            return sentences, np.array([])
        
        embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
        similarities = self._compute_consecutive_similarities(embeddings)
        
        return sentences, similarities
