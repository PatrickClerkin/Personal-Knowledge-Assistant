"""
BM25 keyword index for the Personal Knowledge Assistant.

Provides a BM25-based keyword search store that mirrors the interface
of FAISSVectorStore, allowing it to be used alongside the semantic FAISS
index for hybrid retrieval.

BM25 (Best Match 25) is a probabilistic keyword ranking function that
excels at exact keyword matches where semantic search may miss.
Combined with FAISS via Reciprocal Rank Fusion (see hybrid_search.py),
it forms a hybrid retrieval pipeline.
"""

import re
import pickle
from pathlib import Path
from typing import List, Optional


def _tokenise(text: str) -> List[str]:
    """Tokenise text for BM25 indexing."""
    tokens = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_'-]*\b", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Store:
    """
    BM25 keyword index over document chunks.

    Maintains a parallel store to FAISSVectorStore: the same chunks are
    indexed by both, allowing scores to be merged at query time via RRF.

    Persistence:
        Saved as a pickle file at ``{path}.bm25.pkl`` alongside the
        FAISS ``.faiss`` and ``.meta.json`` files.
    """

    def __init__(self) -> None:
        self._chunks: List = []
        self._tokenised_corpus: List[List[str]] = []
        self._bm25 = None  # Lazy-built after each add()

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add(self, chunks: List) -> None:
        """Add chunks to the BM25 index."""
        if not chunks:
            return
        new_tokens = [_tokenise(c.content) for c in chunks]
        self._chunks.extend(chunks)
        self._tokenised_corpus.extend(new_tokens)
        self._bm25 = None  # Invalidate cached index

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to doc_id. Returns count deleted."""
        keep_chunks, keep_tokens = [], []
        for chunk, tokens in zip(self._chunks, self._tokenised_corpus):
            if chunk.doc_id != doc_id:
                keep_chunks.append(chunk)
                keep_tokens.append(tokens)

        deleted = len(self._chunks) - len(keep_chunks)
        if deleted:
            self._chunks = keep_chunks
            self._tokenised_corpus = keep_tokens
            self._bm25 = None
        return deleted

    def clear(self) -> None:
        """Remove all chunks from the index."""
        self._chunks = []
        self._tokenised_corpus = []
        self._bm25 = None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    @property
    def _index(self):
        """Lazy-build (or rebuild) the BM25 index."""
        if self._bm25 is None and self._tokenised_corpus:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                raise ImportError(
                    "rank-bm25 is required for hybrid search. "
                    "Install it with: pip install rank-bm25"
                )
            self._bm25 = BM25Okapi(self._tokenised_corpus)
        return self._bm25

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
    ) -> List:
        """
        Keyword search using BM25 scoring.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            filter_doc_id: Restrict results to a single document.

        Returns:
            List of SearchResult objects sorted by BM25 score (highest first).
        """
        from ..storage.vector_store import SearchResult

        if not self._chunks or self._index is None:
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        raw_scores = self._index.get_scores(query_tokens)

        scored = [
            (float(score), chunk)
            for score, chunk in zip(raw_scores, self._chunks)
            if (filter_doc_id is None or chunk.doc_id == filter_doc_id)
            and float(score) > 0.0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, chunk) in enumerate(scored[:top_k], 1):
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))
        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the BM25 store to {path}.bm25.pkl."""
        base = str(Path(path))
        pkl_path = base + ".bm25.pkl"
        data = {
            "chunks": self._chunks,
            "tokenised_corpus": self._tokenised_corpus,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load the BM25 store from {path}.bm25.pkl."""
        base = str(Path(path))
        pkl_path = base + ".bm25.pkl"
        if not Path(pkl_path).exists():
            return  # No BM25 index yet — start fresh

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self._chunks = data["chunks"]
        self._tokenised_corpus = data["tokenised_corpus"]
        self._bm25 = None  # Rebuild on next query

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return len(self._chunks)

    def get_document_ids(self) -> List[str]:
        return list({c.doc_id for c in self._chunks})