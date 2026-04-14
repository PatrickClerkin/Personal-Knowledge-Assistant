"""
Semantic document similarity using averaged chunk embeddings.

For each document in the knowledge base, computes a single
document-level embedding by averaging the embeddings of all its
chunks. Then uses cosine similarity to compare documents pairwise
or to find documents most similar to a query.

This enables:
- "Find documents similar to this one"
- "Which documents cover the same topics?"
- Corpus clustering for conflict detection prioritisation

Design Pattern: Strategy Pattern — the similarity computation
(cosine) is isolated so it can be swapped without touching the
rest of the pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from .knowledge_base import KnowledgeBase
from .embeddings.embedding_service import EmbeddingService
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimilarityResult:
    """Similarity between two documents.

    Attributes:
        doc_id: The document being compared to the reference.
        title: Human-readable document title.
        similarity: Cosine similarity score (0-1).
        chunk_count: Number of chunks in this document.
    """
    doc_id: str
    title: str
    similarity: float
    chunk_count: int

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "similarity": round(self.similarity, 4),
            "chunk_count": self.chunk_count,
        }


@dataclass
class SimilarityMatrix:
    """Full pairwise similarity matrix for all documents.

    Attributes:
        documents: List of document metadata dicts.
        matrix: 2D list of similarity scores [i][j].
        doc_ids: Ordered list of doc_ids matching matrix rows/cols.
    """
    documents: List[dict]
    matrix: List[List[float]]
    doc_ids: List[str]

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "matrix": self.matrix,
            "doc_ids": self.doc_ids,
        }


class DocumentSimilarity:
    """Computes semantic similarity between documents.

    Averages chunk embeddings per document into a single document
    vector, then uses cosine similarity for comparison.

    Args:
        knowledge_base: The KnowledgeBase to read documents from.
        embedder: EmbeddingService for query embedding. If None,
            creates a new one with the default model.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: Optional[EmbeddingService] = None,
    ):
        self.kb = knowledge_base
        self._embedder = embedder or EmbeddingService()

    def get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Compute a document-level embedding by averaging chunk embeddings.

        If chunks already have embeddings stored (from ingest), uses
        those directly. Otherwise re-embeds from content.

        Args:
            doc_id: Document ID to embed.

        Returns:
            Averaged embedding vector, or None if no chunks found.
        """
        chunks = self.kb.get_document_chunks(doc_id)
        if not chunks:
            return None

        embeddings = []
        for chunk in chunks:
            if hasattr(chunk, "embedding") and chunk.embedding is not None:
                embeddings.append(chunk.embedding)
            else:
                emb = self._embedder.embed_text(chunk.content)
                embeddings.append(emb)

        return np.mean(embeddings, axis=0)

    def find_similar(
        self,
        doc_id: str,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> List[SimilarityResult]:
        """Find documents most similar to a given document.

        Args:
            doc_id: Reference document to compare against.
            top_k: Number of similar documents to return.
            exclude_self: Whether to exclude the reference doc
                from results (default True).

        Returns:
            List of SimilarityResult sorted by similarity descending.
            Empty list if the reference document has no chunks.
        """
        ref_embedding = self.get_document_embedding(doc_id)
        if ref_embedding is None:
            logger.warning("No chunks found for doc_id: %s", doc_id)
            return []

        ref_chunks = self.kb.get_document_chunks(doc_id)
        ref_title = ref_chunks[0].source_doc_title if ref_chunks else doc_id

        results = []
        for other_id in self.kb.document_ids:
            if exclude_self and other_id == doc_id:
                continue

            other_embedding = self.get_document_embedding(other_id)
            if other_embedding is None:
                continue

            similarity = self._cosine_similarity(ref_embedding, other_embedding)
            other_chunks = self.kb.get_document_chunks(other_id)
            other_title = other_chunks[0].source_doc_title if other_chunks else other_id

            results.append(SimilarityResult(
                doc_id=other_id,
                title=other_title,
                similarity=float(similarity),
                chunk_count=len(other_chunks),
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        logger.info(
            "Found %d similar documents for '%s'",
            len(results[:top_k]), ref_title,
        )
        return results[:top_k]

    def find_similar_to_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SimilarityResult]:
        """Find documents most semantically similar to a free-text query.

        Args:
            query: Query string to compare against document embeddings.
            top_k: Number of similar documents to return.

        Returns:
            List of SimilarityResult sorted by similarity descending.
        """
        query_embedding = self._embedder.embed_text(query)
        results = []

        for doc_id in self.kb.document_ids:
            doc_embedding = self.get_document_embedding(doc_id)
            if doc_embedding is None:
                continue

            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            chunks = self.kb.get_document_chunks(doc_id)
            title = chunks[0].source_doc_title if chunks else doc_id

            results.append(SimilarityResult(
                doc_id=doc_id,
                title=title,
                similarity=float(similarity),
                chunk_count=len(chunks),
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def compute_matrix(self) -> SimilarityMatrix:
        """Compute full pairwise similarity matrix for all documents.

        Returns:
            SimilarityMatrix with document metadata and NxN matrix.
        """
        doc_ids = self.kb.document_ids
        logger.info("Computing similarity matrix for %d documents", len(doc_ids))

        # Build document embeddings
        embeddings: Dict[str, np.ndarray] = {}
        doc_meta = []

        for doc_id in doc_ids:
            emb = self.get_document_embedding(doc_id)
            if emb is None:
                continue
            embeddings[doc_id] = emb
            chunks = self.kb.get_document_chunks(doc_id)
            title = chunks[0].source_doc_title if chunks else doc_id
            doc_meta.append({
                "doc_id": doc_id,
                "title": title,
                "chunk_count": len(chunks),
            })

        valid_ids = [d["doc_id"] for d in doc_meta]

        # Build NxN matrix
        n = len(valid_ids)
        matrix = [[0.0] * n for _ in range(n)]

        for i, id_a in enumerate(valid_ids):
            for j, id_b in enumerate(valid_ids):
                if i == j:
                    matrix[i][j] = 1.0
                elif j > i:
                    sim = float(self._cosine_similarity(
                        embeddings[id_a], embeddings[id_b]
                    ))
                    matrix[i][j] = round(sim, 4)
                    matrix[j][i] = round(sim, 4)

        logger.info("Similarity matrix computed: %dx%d", n, n)
        return SimilarityMatrix(
            documents=doc_meta,
            matrix=matrix,
            doc_ids=valid_ids,
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))