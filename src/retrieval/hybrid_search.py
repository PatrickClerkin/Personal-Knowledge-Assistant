"""
Hybrid Search via Reciprocal Rank Fusion (RRF).

Combines BM25 keyword scores and FAISS semantic scores into a single
ranked list. The two approaches are complementary:

- BM25 is strong on exact keyword matches, technical terms, and rare words.
- FAISS handles synonyms, paraphrases, and semantic context.

RRF formula: RRF(d) = sum( 1 / (k + rank_i(d)) ) for each retrieval system i
k=60 is the standard dampening constant from Cormack et al., 2009.
"""

from typing import List, Optional

_RRF_K = 60


def reciprocal_rank_fusion(*ranked_lists, top_k: int = 5, k: int = _RRF_K) -> List:
    """Merge ranked result lists using Reciprocal Rank Fusion."""
    from src.ingestion.storage.vector_store import SearchResult

    rrf_scores: dict = {}
    chunk_map: dict = {}

    for result_list in ranked_lists:
        for result in result_list:
            cid = result.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + result.rank)
            if cid not in chunk_map or result.score > chunk_map[cid].score:
                chunk_map[cid] = result

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    merged = []
    for new_rank, cid in enumerate(sorted_ids[:top_k], 1):
        original = chunk_map[cid]
        merged.append(SearchResult(chunk=original.chunk, score=round(rrf_scores[cid], 6), rank=new_rank))
    return merged


class HybridSearcher:
    """Orchestrates hybrid BM25 + FAISS search with Reciprocal Rank Fusion."""

    def __init__(self, faiss_store, bm25_store, k: int = _RRF_K) -> None:
        self._faiss = faiss_store
        self._bm25 = bm25_store
        self._k = k

    def search(self, query: str, query_embedding, top_k: int = 5,
               faiss_candidates: int = 20, bm25_candidates: int = 20,
               filter_doc_id: Optional[str] = None) -> List:
        """Run hybrid BM25 + FAISS search and merge with RRF."""
        faiss_results = self._faiss.search(query_embedding, top_k=faiss_candidates, filter_doc_id=filter_doc_id)
        bm25_results = self._bm25.search(query, top_k=bm25_candidates, filter_doc_id=filter_doc_id)

        if not faiss_results and not bm25_results:
            return []
        if not bm25_results:
            for i, r in enumerate(faiss_results[:top_k], 1):
                r.rank = i
            return faiss_results[:top_k]
        if not faiss_results:
            return bm25_results[:top_k]

        return reciprocal_rank_fusion(faiss_results, bm25_results, top_k=top_k, k=self._k)