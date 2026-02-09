"""High-level API for the Personal Knowledge Assistant."""

from pathlib import Path
from typing import List, Optional, Union

from .document import Document
from .parsers.pdf_parser import PDFParser
from .chunking.chunk import Chunk
from .chunking.chunk_manager import ChunkManager
from .embeddings.embedding_service import EmbeddingService
from .storage.faiss_store import FAISSVectorStore
from .storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

class KnowledgeBase:
    """
    High-level API for the Personal Knowledge Assistant.

    Provides a simple interface for:
    - Ingesting documents (PDFs)
    - Semantic search across your knowledge base
    - Managing stored documents

    Example:
        kb = KnowledgeBase()
        kb.ingest("path/to/document.pdf")
        results = kb.search("What is composition in OOP?")
        for result in results:
            print(f"Score: {result.score:.3f}")
            print(f"Content: {result.chunk.content[:200]}...")
    """

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_strategy: str = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the knowledge base.

        Args:
            index_path: Path to save/load the vector index. If None, uses in-memory only.
            embedding_model: Name of the sentence-transformers model to use.
            chunk_strategy: Chunking strategy ("fixed", "sentence", or "semantic").
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        self.index_path = Path(index_path) if index_path else None

        # Initialize components
        self._parser = PDFParser()
        self._chunker = ChunkManager(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedder = EmbeddingService(model_name=embedding_model)
        self._store = FAISSVectorStore(
            embedding_dim=self._embedder.embedding_dimension
        )

        # Load existing index if path provided and exists
        if self.index_path and (Path(str(self.index_path) + ".faiss").exists()):
            self.load()

    def ingest(
        self,
        file_path: Union[str, Path],
        show_progress: bool = True
    ) -> int:
        """
        Ingest a document into the knowledge base.

        Args:
            file_path: Path to the document (currently supports PDF).
            show_progress: Whether to show progress during embedding.

        Returns:
            Number of chunks created from the document.
        """
        file_path = Path(file_path)

        # Parse document
        document = self._parser.parse(file_path)

        # Chunk document
        chunks = self._chunker.chunk_document(document)

        if not chunks:
            return 0

        # Generate embeddings
        embeddings = self._embedder.embed_chunks(
            chunks,
            show_progress=show_progress,
            store_in_chunks=True
        )

        # Store in vector database
        self._store.add(chunks, embeddings)

        # Auto-save if index path is set
        if self.index_path:
            self.save()

        return len(chunks)

    def ingest_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.pdf",
        show_progress: bool = True
    ) -> dict:
        """
        Ingest all matching documents from a directory.

        Args:
            directory: Path to the directory.
            pattern: Glob pattern for files to ingest.
            show_progress: Whether to show progress during embedding.

        Returns:
            Dictionary with ingestion statistics.
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "errors": [],
        }

        for file_path in files:
            try:
                num_chunks = self.ingest(file_path, show_progress=show_progress)
                stats["files_processed"] += 1
                stats["total_chunks"] += num_chunks
            except Exception as e:
                stats["files_failed"] += 1
                stats["errors"].append({"file": str(file_path), "error": str(e)})

        return stats

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search the knowledge base for relevant content.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            filter_doc_id: Optional document ID to restrict search to.

        Returns:
            List of SearchResult objects, sorted by relevance.
        """
        # Generate query embedding
        query_embedding = self._embedder.embed_text(query)

        # Search vector store
        results = self._store.search(
            query_embedding,
            top_k=top_k,
            filter_doc_id=filter_doc_id
        )

        return results

    def delete_document(self, doc_id: str) -> int:
        """
        Delete a document from the knowledge base.

        Args:
            doc_id: ID of the document to delete.

        Returns:
            Number of chunks deleted.
        """
        deleted = self._store.delete_document(doc_id)

        if self.index_path and deleted > 0:
            self.save()

        return deleted

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the knowledge base to disk.

        Args:
            path: Path to save to. If None, uses the index_path from initialization.
        """
        save_path = Path(path) if path else self.index_path
        if not save_path:
            raise ValueError("No save path specified")

        self._store.save(str(save_path))

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load the knowledge base from disk.

        Args:
            path: Path to load from. If None, uses the index_path from initialization.
        """
        load_path = Path(path) if path else self.index_path
        if not load_path:
            raise ValueError("No load path specified")

        self._store.load(str(load_path))

    def clear(self) -> None:
        """Clear all data from the knowledge base."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Get the number of chunks in the knowledge base."""
        return self._store.size

    @property
    def document_ids(self) -> List[str]:
        """Get all document IDs in the knowledge base."""
        return self._store.get_document_ids()

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """Get all chunks belonging to a specific document."""
        return [
            chunk for chunk in self._store.get_all_chunks()
            if chunk.doc_id == doc_id
        ]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"KnowledgeBase(size={self.size}, documents={len(self.document_ids)})"
