"""High-level API for the Personal Knowledge Assistant."""
from pathlib import Path
from typing import List, Optional
from .document import Document
from .document_manager import DocumentManager
from .chunking.chunk import Chunk
from .chunking.chunk_manager import ChunkManager
from .embeddings.embedding_service import EmbeddingService
from .storage.faiss_store import FAISSVectorStore
from .storage.bm25_store import BM25Store
from .storage.document_registry import DocumentRegistry, DocumentRecord
from .storage.vector_store import SearchResult
from .ner_extractor import NERExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBase:
    """High-level API: ingest, search (semantic / hybrid / advanced)."""

    def __init__(self, index_path=None, embedding_model="all-MiniLM-L6-v2",
                 chunk_strategy="sentence", chunk_size=512, chunk_overlap=50):
        self.index_path = Path(index_path) if index_path else None
        self._doc_manager = DocumentManager()
        self._chunker = ChunkManager(strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedder = EmbeddingService(model_name=embedding_model)
        self._store = FAISSVectorStore(embedding_dim=self._embedder.embedding_dimension)
        self._bm25 = BM25Store()
        self._ner = NERExtractor()
        self._registry = DocumentRegistry()
        if self.index_path and (Path(str(self.index_path) + ".faiss").exists()):
            self.load()
            logger.info("Loaded existing index from %s (%d chunks)", self.index_path, self.size)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path, show_progress=True) -> int:
        """
        Ingest a document into the knowledge base.

        Behaviour
        ---------
        - **New document**: parse, chunk, embed, store, register.
        - **Already ingested, file unchanged** (same SHA-256): skip and
          return 0 so callers know nothing happened.
        - **Already ingested, file changed**: delete old chunks, re-ingest
          with fresh embeddings, update the registry record.

        Returns
        -------
        int
            Number of chunks added (0 if skipped as unchanged).
        """
        file_path = Path(file_path)

        # --- hash the file before doing any expensive work ---------------
        file_hash = DocumentRegistry.hash_file(file_path)

        # Derive a deterministic doc_id from the filename stem so it
        # stays consistent across parses, registry, and chunk stores.
        doc_id = file_path.stem

        if self._registry.is_unchanged(doc_id, file_hash):
            logger.info(
                "Skipping '%s' — file is unchanged since last ingest.",
                file_path.name,
            )
            return 0

        # If the document existed before but the file has changed, remove
        # the stale chunks so we don't end up with duplicates.
        if self._registry.get(doc_id) is not None:
            deleted = self.delete_document(doc_id)
            logger.info(
                "Re-ingesting '%s': removed %d stale chunks.",
                file_path.name,
                deleted,
            )

        # --- normal ingest path ------------------------------------------
        document = self._doc_manager.parse_document(file_path)

        # Override the parser-generated doc_id with our deterministic one
        # so chunks, registry, and stores all share the same identifier.
        document.metadata.doc_id = doc_id

        chunks = self._chunker.chunk_document(document)
        if not chunks:
            logger.warning("No chunks created for %s", file_path.name)
            return 0

        self._ner.extract_from_chunks(chunks)
        embeddings = self._embedder.embed_chunks(chunks, show_progress=show_progress, store_in_chunks=True)
        self._store.add(chunks, embeddings)
        self._bm25.add(chunks)

        # --- register the document ---------------------------------------
        self._registry.register(
            doc_id=doc_id,
            filename=file_path.name,
            file_hash=file_hash,
            chunk_count=len(chunks),
        )

        logger.info("Ingested '%s': %d chunks", file_path.name, len(chunks))
        if self.index_path:
            self.save()
        return len(chunks)

    def ingest_directory(self, directory, pattern=None, show_progress=True) -> dict:
        """Ingest all supported documents from a directory."""
        directory = Path(directory)
        files = list(directory.rglob(pattern)) if pattern else [
            f for f in directory.rglob("*")
            if f.is_file() and self._doc_manager.get_parser(f) is not None
        ]
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "errors": [],
        }
        for fp in files:
            try:
                n = self.ingest(fp, show_progress=show_progress)
                if n == 0:
                    stats["files_skipped"] += 1
                else:
                    stats["files_processed"] += 1
                    stats["total_chunks"] += n
            except Exception as e:
                stats["files_failed"] += 1
                stats["errors"].append({"file": str(fp), "error": str(e)})
        return stats

    # ------------------------------------------------------------------
    # Document info
    # ------------------------------------------------------------------

    def get_document_info(self, doc_id: str) -> Optional[DocumentRecord]:
        """
        Return the registry record for *doc_id*, or None if not found.

        The record contains: filename, file_hash, ingested_at, updated_at,
        chunk_count.  Intended for display in CLI / web UI search results.
        """
        return self._registry.get(doc_id)

    def list_documents(self) -> List[DocumentRecord]:
        """Return all registered documents, sorted by most recently updated."""
        return self._registry.all_records

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @property
    def supported_formats(self) -> List[str]:
        """Return all file extensions this knowledge base can ingest."""
        return self._doc_manager.supported_extensions

    def search(self, query: str, top_k: int = 5, filter_doc_id=None) -> List[SearchResult]:
        """Basic semantic search using FAISS."""
        logger.debug("Searching for: '%s'", query[:50])
        query_embedding = self._embedder.embed_text(query)
        results = self._store.search(query_embedding, top_k=top_k, filter_doc_id=filter_doc_id)
        logger.debug("FAISS search returned %d results", len(results))
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        faiss_candidates: int = 20,
        bm25_candidates: int = 20,
        filter_doc_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Hybrid BM25 + FAISS search merged with Reciprocal Rank Fusion."""
        from ..retrieval.hybrid_search import HybridSearcher
        query_embedding = self._embedder.embed_text(query)
        searcher = HybridSearcher(self._store, self._bm25)
        results = searcher.search(
            query=query, query_embedding=query_embedding, top_k=top_k,
            faiss_candidates=faiss_candidates, bm25_candidates=bm25_candidates,
            filter_doc_id=filter_doc_id,
        )
        logger.info("Hybrid search '%s': %d results", query[:50], len(results))
        return results

    def advanced_search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        expand_query: Optional[str] = None,
        rerank_candidates: int = 20,
        filter_doc_id=None,
        hybrid: bool = False,
        entity_boost: bool = True,
        label_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Enhanced search: optional reranking, query expansion, hybrid mode, and entity boosting."""
        queries = [query]
        if expand_query:
            from ..retrieval.query_expansion import QueryExpander
            queries = QueryExpander(strategy=expand_query).expand(query)
            logger.info("Query expanded to %d variants", len(queries))

        retrieve_k = rerank_candidates if rerank else top_k
        seen_ids, all_results = set(), []

        for q in queries:
            if hybrid:
                q_results = self.hybrid_search(q, top_k=retrieve_k, filter_doc_id=filter_doc_id)
            else:
                q_emb = self._embedder.embed_text(q)
                q_results = self._store.search(q_emb, top_k=retrieve_k, filter_doc_id=filter_doc_id)
            for r in q_results:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    all_results.append(r)

        all_results.sort(key=lambda r: r.score, reverse=True)

        if entity_boost or label_filter:
            from ..retrieval.entity_reranker import boost_by_entities
            query_entities = self._ner.extract(query)
            result_pairs = [(r.chunk, r.score) for r in all_results]
            boosted_pairs = boost_by_entities(
                result_pairs, query_entities,
                label_filter=label_filter,
            )
            chunk_to_result = {r.chunk.chunk_id: r for r in all_results}
            all_results = []
            for chunk, score in boosted_pairs:
                original = chunk_to_result[chunk.chunk_id]
                original.score = score
                all_results.append(original)

        if rerank and all_results:
            from ..retrieval.reranker import CrossEncoderReranker
            all_results = CrossEncoderReranker(top_k=top_k).rerank(query, all_results, top_k=top_k)
        else:
            all_results = all_results[:top_k]
            for i, r in enumerate(all_results, 1):
                r.rank = i

        return all_results

    # ------------------------------------------------------------------
    # Delete / persist
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document and remove its registry entry."""
        deleted = self._store.delete_document(doc_id)
        self._bm25.delete_document(doc_id)
        self._registry.remove(doc_id)
        if self.index_path and deleted > 0:
            self.save()
        return deleted

    def save(self, path=None):
        """Save the entire index (FAISS + BM25 + registry) to disk."""
        save_path = Path(path) if path else self.index_path
        if not save_path:
            raise ValueError("No save path specified")
        self._store.save(str(save_path))
        self._bm25.save(str(save_path))
        self._registry.save(str(save_path))

    def load(self, path=None):
        """Load the entire index (FAISS + BM25 + registry) from disk."""
        load_path = Path(path) if path else self.index_path
        if not load_path:
            raise ValueError("No load path specified")
        self._store.load(str(load_path))
        self._bm25.load(str(load_path))
        self._registry.load(str(load_path))

    def clear(self):
        """Remove all data from the index."""
        self._store.clear()
        self._bm25.clear()
        self._registry = DocumentRegistry()

    # ------------------------------------------------------------------
    # Properties / dunder
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Get the number of chunks in the store."""
        return self._store.size

    @property
    def document_ids(self) -> List[str]:
        """Get all unique document IDs in the store."""
        return self._store.get_document_ids()

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """Get all chunks belonging to a specific document."""
        return [c for c in self._store.get_all_chunks() if c.doc_id == doc_id]

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"KnowledgeBase(size={self.size}, documents={len(self.document_ids)})"