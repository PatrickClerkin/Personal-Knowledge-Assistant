"""
Density-Based Semantic Chunker using HDBSCAN

This chunker uses HDBSCAN clustering on sentence embeddings to find
natural groupings of semantically similar sentences. Unlike threshold-based
approaches, density clustering automatically discovers the number and
boundaries of topics without requiring a predefined threshold.

Key advantages:
- No need to set a similarity threshold manually
- Can discover varying numbers of topics based on document structure
- Handles noise/outlier sentences gracefully
- More robust to embedding quality variations
"""

import re
from typing import List, Optional, Tuple, Dict
import numpy as np
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document


class DensityBasedChunker(BaseChunker):
    """
    Semantic chunker using HDBSCAN clustering on sentence embeddings.
    
    Algorithm:
    1. Split document into sentences
    2. Generate embeddings for each sentence
    3. Apply HDBSCAN clustering to find semantic clusters
    4. Group consecutive sentences in same cluster
    5. Handle transitions and outliers appropriately
    6. Apply size constraints while respecting cluster boundaries
    
    This is particularly effective for documents with clear topic structure
    and works well without manual threshold tuning.
    """
    
    METHOD_NAME = "density_clustering"
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.0,
        embedding_model: Optional[str] = None,
        use_umap: bool = True,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 10,
        preserve_sequence: bool = True,
    ):
        """
        Initialize the density-based semantic chunker.
        
        Args:
            chunk_size: Target maximum chunk size
            chunk_overlap: Overlap between chunks
            min_cluster_size: Minimum sentences for a cluster (HDBSCAN param)
            min_samples: Minimum samples for core points (HDBSCAN param)
            cluster_selection_epsilon: Distance threshold for cluster selection
            embedding_model: Model name for embeddings
            use_umap: Whether to reduce dimensions before clustering
            umap_n_neighbors: UMAP neighborhood size
            umap_n_components: Target dimensions after UMAP
            preserve_sequence: Try to keep sentence order within chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.use_umap = use_umap
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.preserve_sequence = preserve_sequence
        
        self._embedding_model = None
        self._model_name = embedding_model or "all-MiniLM-L6-v2"
        
        # Sentence splitting pattern
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
        Chunk document using density-based clustering.
        """
        text = document.content
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        # Step 1: Split into sentences with position tracking
        sentences, positions = self._split_sentences_with_positions(text)
        
        if len(sentences) < self.min_cluster_size:
            # Too few sentences for clustering
            return self._create_fallback_chunks(document, sentences, positions)
        
        # Step 2: Generate embeddings
        embeddings = self.embedding_model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Step 3: Optional dimensionality reduction
        if self.use_umap and len(sentences) > self.umap_n_neighbors:
            embeddings = self._reduce_dimensions(embeddings)
        
        # Step 4: Perform HDBSCAN clustering
        cluster_labels = self._cluster_embeddings(embeddings)
        
        # Step 5: Create chunks from clusters
        chunks = self._create_chunks_from_clusters(
            sentences=sentences,
            positions=positions,
            cluster_labels=cluster_labels,
            doc_id=doc_id,
            doc_title=doc_title,
            document=document
        )
        
        return self._finalize_chunks(chunks)
    
    def _split_sentences_with_positions(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Split text into sentences while tracking positions."""
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
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embedding dimensions using UMAP for better clustering.
        
        UMAP preserves local structure better than PCA and helps
        HDBSCAN find more meaningful clusters.
        """
        try:
            import umap
            
            n_samples = len(embeddings)
            n_neighbors = min(self.umap_n_neighbors, n_samples - 1)
            n_components = min(self.umap_n_components, n_samples - 1)
            
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric='cosine',
                random_state=42,
                min_dist=0.0,
            )
            
            return reducer.fit_transform(embeddings)
        except ImportError:
            # Fall back to no reduction if UMAP not available
            return embeddings
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.
        
        Returns:
            Array of cluster labels (-1 for noise/outliers)
        """
        try:
            import hdbscan
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='euclidean',  # UMAP already uses cosine
                cluster_selection_method='eom',  # Excess of Mass
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            return cluster_labels
            
        except ImportError:
            # Fallback: use simple k-means
            from sklearn.cluster import KMeans
            
            n_clusters = max(2, len(embeddings) // self.min_cluster_size)
            n_clusters = min(n_clusters, len(embeddings) // 2)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)
    
    def _create_chunks_from_clusters(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        cluster_labels: np.ndarray,
        doc_id: str,
        doc_title: str,
        document: Document
    ) -> List[Chunk]:
        """
        Create chunks by grouping sentences based on cluster membership.
        
        Handles sequence preservation and cluster transitions.
        """
        chunks = []
        chunk_index = 0
        
        if self.preserve_sequence:
            # Group consecutive sentences with same label
            chunks = self._create_sequential_chunks(
                sentences, positions, cluster_labels,
                doc_id, doc_title, document
            )
        else:
            # Group all sentences by cluster regardless of position
            chunks = self._create_cluster_chunks(
                sentences, positions, cluster_labels,
                doc_id, doc_title, document
            )
        
        return chunks
    
    def _create_sequential_chunks(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        cluster_labels: np.ndarray,
        doc_id: str,
        doc_title: str,
        document: Document
    ) -> List[Chunk]:
        """
        Create chunks preserving sentence order, splitting at cluster transitions.
        """
        chunks = []
        chunk_index = 0
        
        current_group = []
        current_positions = []
        current_cluster = cluster_labels[0] if len(cluster_labels) > 0 else -1
        
        for i, (sentence, pos, label) in enumerate(zip(sentences, positions, cluster_labels)):
            # Check for cluster transition
            is_transition = label != current_cluster
            is_size_exceeded = len(' '.join(current_group)) + len(sentence) > self.chunk_size
            
            # Create new chunk if cluster changes or size exceeded
            if (is_transition or is_size_exceeded) and current_group:
                chunk = self._create_chunk_from_group(
                    sentences=current_group,
                    positions=current_positions,
                    cluster_id=int(current_cluster) if current_cluster != -1 else None,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    document=document,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                current_group = []
                current_positions = []
            
            current_group.append(sentence)
            current_positions.append(pos)
            current_cluster = label
        
        # Handle remaining group
        if current_group:
            chunk = self._create_chunk_from_group(
                sentences=current_group,
                positions=current_positions,
                cluster_id=int(current_cluster) if current_cluster != -1 else None,
                doc_id=doc_id,
                doc_title=doc_title,
                document=document,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_cluster_chunks(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        cluster_labels: np.ndarray,
        doc_id: str,
        doc_title: str,
        document: Document
    ) -> List[Chunk]:
        """
        Create chunks grouping by cluster, not sequence.
        
        Useful when document has interleaved topics.
        """
        # Group sentences by cluster
        cluster_groups: Dict[int, List[Tuple[str, Tuple[int, int]]]] = {}
        
        for sentence, pos, label in zip(sentences, positions, cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append((sentence, pos))
        
        chunks = []
        chunk_index = 0
        
        for cluster_id in sorted(cluster_groups.keys()):
            group = cluster_groups[cluster_id]
            
            # Sort by position for readability
            group.sort(key=lambda x: x[1][0])
            
            current_sentences = []
            current_positions = []
            current_length = 0
            
            for sentence, pos in group:
                if current_length + len(sentence) > self.chunk_size and current_sentences:
                    chunk = self._create_chunk_from_group(
                        sentences=current_sentences,
                        positions=current_positions,
                        cluster_id=cluster_id if cluster_id != -1 else None,
                        doc_id=doc_id,
                        doc_title=doc_title,
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
                chunk = self._create_chunk_from_group(
                    sentences=current_sentences,
                    positions=current_positions,
                    cluster_id=cluster_id if cluster_id != -1 else None,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    document=document,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _create_chunk_from_group(
        self,
        sentences: List[str],
        positions: List[Tuple[int, int]],
        cluster_id: Optional[int],
        doc_id: str,
        doc_title: str,
        document: Document,
        chunk_index: int
    ) -> Chunk:
        """Create a chunk from a group of sentences."""
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
            cluster_id=cluster_id
        )
    
    def _create_fallback_chunks(
        self,
        document: Document,
        sentences: List[str],
        positions: List[Tuple[int, int]]
    ) -> List[Chunk]:
        """Create chunks without clustering for small documents."""
        if not sentences:
            return []
        
        content = ' '.join(sentences)
        chunk = self._create_chunk(
            content=content,
            doc_id=document.metadata.doc_id,
            source_doc_title=document.metadata.title,
            chunk_index=0,
            start_char=positions[0][0] if positions else 0,
            end_char=positions[-1][1] if positions else len(content),
            page_number=1 if document.sections else None
        )
        chunk.total_chunks = 1
        return [chunk]
    
    def get_cluster_analysis(self, document: Document) -> Dict:
        """
        Analyze document clusters for debugging/visualization.
        
        Returns dict with sentences, embeddings, labels, and statistics.
        """
        text = document.content
        sentences, positions = self._split_sentences_with_positions(text)
        
        if len(sentences) < self.min_cluster_size:
            return {"error": "Too few sentences for clustering"}
        
        embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
        
        if self.use_umap and len(sentences) > self.umap_n_neighbors:
            reduced_embeddings = self._reduce_dimensions(embeddings)
        else:
            reduced_embeddings = embeddings
        
        cluster_labels = self._cluster_embeddings(reduced_embeddings)
        
        unique_labels = set(cluster_labels)
        
        return {
            "sentences": sentences,
            "positions": positions,
            "cluster_labels": cluster_labels.tolist(),
            "n_clusters": len(unique_labels - {-1}),
            "n_noise": sum(1 for l in cluster_labels if l == -1),
            "cluster_sizes": {
                int(l): sum(1 for x in cluster_labels if x == l)
                for l in unique_labels
            }
        }
