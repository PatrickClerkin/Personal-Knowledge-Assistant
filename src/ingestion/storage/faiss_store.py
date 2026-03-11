import json
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from ..chunking.chunk import Chunk
from .vector_store import VectorStore, SearchResult


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for efficient similarity search.

    Uses FAISS IndexFlatIP (inner product) which is equivalent to
    cosine similarity when vectors are normalized.
    """

    def __init__(self, embedding_dim: int = 384):
        if faiss is None:
            raise ImportError(
                "FAISS is not installed. Install it with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._chunks: List[Chunk] = []
        self._id_to_index: Dict[str, int] = {}

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if len(chunks) == 0:
            return

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        start_idx = len(self._chunks)
        self._index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._id_to_index[chunk.chunk_id] = start_idx + i

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        if self._index.ntotal == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        search_k = top_k * 3 if filter_doc_id else top_k
        scores, indices = self._index.search(query, min(search_k, self._index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue

            chunk = self._chunks[idx]

            if filter_doc_id and chunk.doc_id != filter_doc_id:
                continue

            results.append(SearchResult(
                chunk=chunk,
                score=float(score),
                rank=len(results) + 1
            ))

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, doc_id: str) -> int:
        keep_indices = []
        keep_chunks = []

        for i, chunk in enumerate(self._chunks):
            if chunk.doc_id != doc_id:
                keep_indices.append(i)
                keep_chunks.append(chunk)

        deleted_count = len(self._chunks) - len(keep_chunks)

        if deleted_count == 0:
            return 0

        if keep_indices:
            keep_embeddings = np.zeros(
                (len(keep_indices), self.embedding_dim),
                dtype=np.float32
            )
            for new_idx, old_idx in enumerate(keep_indices):
                keep_embeddings[new_idx] = self._index.reconstruct(old_idx)

            self._index.reset()
            self._index.add(keep_embeddings)

            self._chunks = keep_chunks
            self._id_to_index = {
                chunk.chunk_id: i for i, chunk in enumerate(self._chunks)
            }
        else:
            self.clear()

        return deleted_count

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path) + ".faiss")

        chunks_data = []
        for chunk in self._chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "source_doc_title": chunk.source_doc_title,
                "section_id": chunk.section_id,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "metadata": chunk.metadata,  # persist NER entities and other annotations
            })

        metadata = {
            "embedding_dim": self.embedding_dim,
            "chunks": chunks_data
        }

        with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        path = Path(path)

        index_path = str(path) + ".faiss"
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self._index = faiss.read_index(index_path)

        meta_path = str(path) + ".meta.json"
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.embedding_dim = metadata["embedding_dim"]

        self._chunks = []
        self._id_to_index = {}

        for i, chunk_data in enumerate(metadata["chunks"]):
            chunk = Chunk(
                chunk_id=chunk_data["chunk_id"],
                doc_id=chunk_data["doc_id"],
                content=chunk_data["content"],
                source_doc_title=chunk_data["source_doc_title"],
                section_id=chunk_data.get("section_id"),
                page_number=chunk_data.get("page_number"),
                start_char=chunk_data.get("start_char", 0),
                end_char=chunk_data.get("end_char", 0),
                chunk_index=chunk_data.get("chunk_index", 0),
                total_chunks=chunk_data.get("total_chunks"),
                metadata=chunk_data.get("metadata", {}),  # restore NER entities and other annotations
            )
            self._chunks.append(chunk)
            self._id_to_index[chunk.chunk_id] = i

    @property
    def size(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        self._index.reset()
        self._chunks = []
        self._id_to_index = {}

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._id_to_index.get(chunk_id)
        if idx is not None:
            return self._chunks[idx]
        return None

    def get_all_chunks(self) -> List[Chunk]:
        return self._chunks.copy()

    def get_document_ids(self) -> List[str]:
        return list(set(chunk.doc_id for chunk in self._chunks))
