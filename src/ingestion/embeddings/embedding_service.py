from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from ..chunking.chunk import Chunk


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to 'all-MiniLM-L6-v2' (384 dimensions, fast).
                       Other options:
                       - 'all-mpnet-base-v2': 768 dims, better quality
                       - 'multi-qa-MiniLM-L6-cos-v1': optimized for semantic search
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show a progress bar
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )

    def embed_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = False,
        store_in_chunks: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects to embed
            show_progress: Whether to show a progress bar
            store_in_chunks: If True, stores embeddings in chunk.embedding field

        Returns:
            Numpy array of shape (num_chunks, embedding_dim)
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts, show_progress=show_progress)

        if store_in_chunks:
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def find_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find the most similar embeddings to a query.

        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embeddings to search (num_embeddings, dim)
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]
