"""High-level API for the Personal Knowledge Assistant."""
from pathlib import Path
from typing import List, Optional, Union
from .document import Document
from .document_manager import DocumentManager
from .chunking.chunk import Chunk
from .chunking.chunk_manager import ChunkManager
from .embeddings.embedding_service import EmbeddingService
from .storage.faiss_store import FAISSVectorStore
from .storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBase:
    """High-level API for the Personal Knowledge Assistant.

    Provides a unified interface for ingesting documents of any supported
    format, chunking them into semantic units, generating embeddings,
    and performing similarity search.

    Supports: .pdf, .txt, .md, .docx (via DocumentManager routing).
    """

    def __init__(self, index_path=None, embedding_model="all-MiniLM-L6-v2",
                 chunk_strategy="sentence", chunk_size=512, chunk_overlap=50):
        self.index_path = Path(index_path) if index_path else None
        self._doc_manager = DocumentManager()
        self._chunker = ChunkManager(strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedder = EmbeddingService(model_name=embedding_model)
        self._store = FAISSVectorStore(embedding_dim=self._embedder.embedding_dimension)
        if self.index_path and (Path(str(self.index_path) + ".faiss").exists()):
            self.load()
            logger.info("Loaded existing index from %s (%d chunks)", self.index_path, self.size)

    def ingest(self, file_path, show_progress=True) -> int:
        """Ingest a single document of any supported format.

        Parses the file, chunks the content, generates embeddings,
        and stores everything in the vector index.

        Args:
            file_path: Path to the document (.pdf, .txt, .md, .docx).
            show_progress: Whether to show embedding progress bar.

        Returns:
            Number of chunks created.

        Raises:
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)
        logger.info("Ingesting document: %s", file_path.name)
        document = self._doc_manager.parse_document(file_path)
        chunks = self._chunker.chunk_document(document)
        if not chunks:
            logger.warning("No chunks created from %s", file_path.name)
            return 0
        embeddings = self._embedder.embed_chunks(chunks, show_progress=show_progress, store_in_chunks=True)
        self._store.add(chunks, embeddings)
        logger.info("Ingested %s: %d chunks created", file_path.name, len(chunks))
        if self.index_path: self.save()
        return len(chunks)

    def ingest_directory(self, directory, pattern=None, show_progress=True) -> dict:
        """Ingest all supported documents in a directory.

        Scans the directory recursively for files matching any supported
        format. Uses DocumentManager to automatically route each file
        to the correct parser.

        Args:
            directory: Path to the directory to scan.
            pattern: Optional glob pattern (e.g. '*.pdf'). If None,
                     all supported formats are ingested.
            show_progress: Whether to show embedding progress bar.

        Returns:
            Dict with keys: files_processed, files_failed, total_chunks, errors.
        """
        directory = Path(directory)
        if pattern:
            files = list(directory.rglob(pattern))
        else:
            # Collect all files matching any supported extension
            files = [
                f for f in directory.rglob("*")
                if f.is_file() and self._doc_manager.get_parser(f) is not None
            ]
        stats = {"files_processed": 0, "files_failed": 0, "total_chunks": 0, "errors": []}
        for fp in files:
            try:
                n = self.ingest(fp, show_progress=show_progress)
                stats["files_processed"] += 1
                stats["total_chunks"] += n
            except Exception as e:
                stats["files_failed"] += 1
                stats["errors"].append({"file": str(fp), "error": str(e)})
                logger.error("Failed to ingest %s: %s", fp.name, e)
        logger.info(
            "Directory ingest complete: %d files, %d chunks, %d errors",
            stats["files_processed"], stats["total_chunks"], stats["files_failed"]
        )
        return stats

    @property
    def supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return self._doc_manager.supported_extensions

    def search(self, query: str, top_k: int = 5, filter_doc_id=None) -> List[SearchResult]:
        """Basic semantic search using FAISS."""
        logger.debug("Searching for: '%s' (top_k=%d)", query, top_k)
        query_embedding = self._embedder.embed_text(query)
        results = self._store.search(query_embedding, top_k=top_k, filter_doc_id=filter_doc_id)
        logger.debug("Found %d results", len(results))
        return results

    def advanced_search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        expand_query: Optional[str] = None,
        rerank_candidates: int = 20,
        filter_doc_id=None,
    ) -> List[SearchResult]:
        """Enhanced search with optional reranking and query expansion.

        Supports a two-stage retrieval pipeline:
        1. (Optional) Expand the query into multiple variants.
        2. Retrieve candidates from FAISS for each query variant.
        3. (Optional) Rerank candidates with a cross-encoder.

        Args:
            query: Natural language search query.
            top_k: Number of final results to return.
            rerank: Whether to apply cross-encoder reranking.
            expand_query: Query expansion strategy ('synonym',
                'multi_query', 'hyde', or None to skip).
            rerank_candidates: Number of FAISS candidates to
                retrieve before reranking.
            filter_doc_id: Restrict results to a specific document.

        Returns:
            List of SearchResult objects, ranked by relevance.
        """
        # Stage 1: Query expansion
        queries = [query]
        if expand_query:
            from .retrieval.query_expansion import QueryExpander
            expander = QueryExpander(strategy=expand_query)
            queries = expander.expand(query)
            logger.info("Expanded query into %d variants", len(queries))

        # Stage 2: Retrieve candidates
        retrieve_k = rerank_candidates if rerank else top_k
        seen_ids = set()
        all_results = []

        for q in queries:
            q_embedding = self._embedder.embed_text(q)
            results = self._store.search(
                q_embedding, top_k=retrieve_k, filter_doc_id=filter_doc_id
            )
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    all_results.append(r)

        # Sort by score and trim
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Stage 3: Reranking
        if rerank and all_results:
            from .retrieval.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(top_k=top_k)
            all_results = reranker.rerank(query, all_results, top_k=top_k)
        else:
            all_results = all_results[:top_k]
            for i, r in enumerate(all_results, 1):
                r.rank = i

        return all_results

    def delete_document(self, doc_id: str) -> int:
        deleted = self._store.delete_document(doc_id)
        if self.index_path and deleted > 0: self.save()
        return deleted

    def save(self, path=None):
        save_path = Path(path) if path else self.index_path
        if not save_path: raise ValueError("No save path specified")
        self._store.save(str(save_path))

    def load(self, path=None):
        load_path = Path(path) if path else self.index_path
        if not load_path: raise ValueError("No load path specified")
        self._store.load(str(load_path))

    def clear(self): self._store.clear()
    @property
    def size(self) -> int: return self._store.size
    @property
    def document_ids(self) -> List[str]: return self._store.get_document_ids()
    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        return [c for c in self._store.get_all_chunks() if c.doc_id == doc_id]
    def __len__(self): return self.size
    def __repr__(self): return f"KnowledgeBase(size={self.size}, documents={len(self.document_ids)})"
