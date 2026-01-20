from abc import ABC, abstractmethod
from typing import List, Optional, Any
import numpy as np

from ..chunking.chunk import Chunk


class SearchResult:
    """Represents a single search result."""

    def __init__(
        self,
        chunk: Chunk,
        score: float,
        rank: int
    ):
        self.chunk = chunk
        self.score = score
        self.rank = rank

    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, chunk_id={self.chunk.chunk_id})"


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: Numpy array of shape (num_chunks, embedding_dim)
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_doc_id: Optional document ID to filter results

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Get the number of vectors in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the store."""
        pass
