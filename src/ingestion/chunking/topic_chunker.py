"""
Topic-Based Semantic Chunker using BERTopic

This chunker uses topic modeling to identify semantic boundaries in documents.
It leverages BERTopic (BERT-based topic modeling) to automatically discover
topics within the document and creates chunks based on topic assignments.

Key features:
- Uses modern neural topic modeling
- Can discover any number of topics automatically
- Provides topic labels for each chunk
- Handles documents with multiple interleaved topics
"""

import re
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document


class TopicBasedChunker(BaseChunker):
    """
    Semantic chunker using BERTopic for topic detection.
    
    Algorithm:
    1. Split document into sentences
    2. Use BERTopic to assign topics to sentences
    3. Group consecutive sentences with same topic
    4. Handle topic transitions as chunk boundaries
    5. Apply size constraints while respecting topic boundaries
    
    This is effective for documents covering multiple distinct topics
    and provides interpretable topic labels for each chunk.
    """
    
    METHOD_NAME = "topic_modeling"
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_topic_size: int = 3,
        nr_topics: Optional[int] = None,  # None = auto-detect
        embedding_model: Optional[str] = None,
        diversity: float = 0.5,
        calculate_probabilities: bool = True,
    ):
        """
        Initialize the topic-based chunker.
        
        Args:
            chunk_size: Target maximum chunk size
            chunk_overlap: Overlap between chunks
            min_topic_size: Minimum sentences per topic
            nr_topics: Target number of topics (None = auto)
            embedding_model: Model for embeddings
            diversity: Topic diversity parameter (0-1)
            calculate_probabilities: Include topic probability scores
        """
        super().__init__(chunk_size, chunk_overlap)
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.diversity = diversity
        self.calculate_probabilities = calculate_probabilities
        
        self._embedding_model_name = embedding_model or "all-MiniLM-L6-v2"
        self._topic_model = None
        self._embedding_model = None
        
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model
    
    @property
    def topic_model(self):
        """Lazy load BERTopic model."""
        if self._topic_model is None:
            self._topic_model = self._create_topic_model()
        return self._topic_model
    
    def _create_topic_model(self):
        """Create and configure BERTopic model."""
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Use custom vectorizer for better topic words
            vectorizer = CountVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                min_topic_size=self.min_topic_size,
                nr_topics=self.nr_topics,
                diversity=self.diversity,
                calculate_probabilities=self.calculate_probabilities,
                vectorizer_model=vectorizer,
                verbose=False
            )
            
            return topic_model
            
        except ImportError:
            # Fallback to simple clustering approach
            return None
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document using topic modeling.
        """
        text = document.content
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        # Split into sentences
        sentences, positions = self._split_sentences_with_positions(text)
        
        if len(sentences) < self.min_topic_size:
            return self._create_single_chunk(document)
        
        # Get topic assignments
        try:
            topics, topic_info = self._assign_topics(sentences)
        except Exception as e:
            # Fallback to simple chunking on error
            print(f"Topic modeling failed: {e}, falling back to sentence chunking")
            return self._fallback_chunk(document, sentences, positions)
        
        # Create chunks from topic groups
        chunks = self._create_chunks_from_topics(
            sentences=sentences,
            positions=positions,
            topics=topics,
            topic_info=topic_info,
            doc_id=doc_id,
            doc_title=doc_title,
            document=document
        )
        
        return self._finalize_chunks(chunks)
    
    def _split_sentences_with_positions(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Split text into sentences with position tracking."""
        sentences = []
        positions = []
        
        parts = self.sentence_pattern.split(text)
        current_pos = 0
        
        for part in parts:
            part = part.strip()
            if part:
                start = text.find(part, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(part)
                
                sentences.append(part)
                positions.append((start, end))
                current_pos = end
        
        return sentences, positions
    
    def _assign_topics(self, sentences: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Assign topics to sentences using BERTopic.
        
        Returns:
            Tuple of (topic_ids array, topic_info dict)
        """
        if self.topic_model is None:
            # Fallback: use embedding clustering
            return self._fallback_topic_assignment(sentences)
        
        # Fit topic model
        topics, probs = self.topic_model.fit_transform(sentences)
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Convert to dict format
        info = {
            "topics": topic_info.to_dict('records') if hasattr(topic_info, 'to_dict') else {},
            "probabilities": probs if probs is not None else None
        }
        
        return np.array(topics), info
    
    def _fallback_topic_assignment(self, sentences: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fallback topic assignment using simple clustering.
        """
        embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
        
        try:
            from sklearn.cluster import KMeans
            
            # Estimate number of clusters
            n_clusters = max(2, min(len(sentences) // self.min_topic_size, 10))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            topics = kmeans.fit_predict(embeddings)
            
            return topics, {"method": "kmeans", "n_clusters": n_clusters}
            
        except ImportError:
            # No clustering available - assign sequential topics
            topics = np.zeros(len(sentences), dtype=int)
            return topics, {"method": "none"}
    
    def _create_chunks_from_topics(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        topics: np.ndarray,
        topic_info: Dict[str, Any],
        doc_id: str,
        doc_title: str,
        document: Document
    ) -> List[Chunk]:
        """
        Create chunks by grouping sentences by topic.
        """
        chunks = []
        chunk_index = 0
        
        current_topic = topics[0]
        current_sentences = []
        current_positions = []
        
        for i, (sentence, pos, topic) in enumerate(zip(sentences, positions, topics)):
            # Check for topic change or size limit
            is_topic_change = topic != current_topic
            current_length = sum(len(s) for s in current_sentences) + len(sentence)
            is_size_exceeded = current_length > self.chunk_size and current_sentences
            
            if (is_topic_change or is_size_exceeded) and current_sentences:
                # Create chunk
                chunk = self._create_topic_chunk(
                    sentences=current_sentences,
                    positions=current_positions,
                    topic_id=int(current_topic),
                    doc_id=doc_id,
                    doc_title=doc_title,
                    document=document,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                current_sentences = []
                current_positions = []
            
            current_sentences.append(sentence)
            current_positions.append(pos)
            current_topic = topic
        
        # Handle remaining sentences
        if current_sentences:
            chunk = self._create_topic_chunk(
                sentences=current_sentences,
                positions=current_positions,
                topic_id=int(current_topic),
                doc_id=doc_id,
                doc_title=doc_title,
                document=document,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_topic_chunk(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        topic_id: int,
        doc_id: str,
        doc_title: str,
        document: Document,
        chunk_index: int
    ) -> Chunk:
        """Create a chunk with topic metadata."""
        content = ' '.join(sentences)
        start_char = positions[0][0]
        end_char = positions[-1][1]
        page_number = self._get_page_number(start_char, document)
        
        return self._create_chunk(
            content=content,
            doc_id=doc_id,
            source_doc_title=doc_title,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            page_number=page_number,
            topic_id=topic_id if topic_id != -1 else None
        )
    
    def _create_single_chunk(self, document: Document) -> List[Chunk]:
        """Create single chunk for small documents."""
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
            page_number=1 if document.sections else None,
            topic_id=0
        )
        chunk.total_chunks = 1
        return [chunk]
    
    def _fallback_chunk(
        self,
        document: Document,
        sentences: List[str],
        positions: List[Tuple[int, int]]
    ) -> List[Chunk]:
        """Fallback to simple sentence-based chunking."""
        chunks = []
        chunk_index = 0
        current_sentences = []
        current_positions = []
        current_length = 0
        
        for sentence, pos in zip(sentences, positions):
            if current_length + len(sentence) > self.chunk_size and current_sentences:
                chunk = self._create_topic_chunk(
                    sentences=current_sentences,
                    positions=current_positions,
                    topic_id=-1,
                    doc_id=document.metadata.doc_id,
                    doc_title=document.metadata.title,
                    document=document,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                current_sentences = []
                current_positions = []
                current_length = 0
            
            current_sentences.append(sentence)
            current_positions.append(pos)
            current_length += len(sentence) + 1
        
        if current_sentences:
            chunk = self._create_topic_chunk(
                sentences=current_sentences,
                positions=current_positions,
                topic_id=-1,
                doc_id=document.metadata.doc_id,
                doc_title=document.metadata.title,
                document=document,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_topic_analysis(self, document: Document) -> Dict[str, Any]:
        """
        Get detailed topic analysis for the document.
        """
        sentences, positions = self._split_sentences_with_positions(document.content)
        
        if len(sentences) < self.min_topic_size:
            return {"error": "Too few sentences for topic modeling"}
        
        try:
            topics, topic_info = self._assign_topics(sentences)
            
            # Compute topic distribution
            unique_topics, counts = np.unique(topics, return_counts=True)
            distribution = {int(t): int(c) for t, c in zip(unique_topics, counts)}
            
            return {
                "n_sentences": len(sentences),
                "n_topics": len(unique_topics - set([-1])),
                "topic_distribution": distribution,
                "sentences_per_topic": {
                    int(t): [s for s, st in zip(sentences, topics) if st == t]
                    for t in unique_topics
                },
                "topic_info": topic_info
            }
            
        except Exception as e:
            return {"error": str(e)}
