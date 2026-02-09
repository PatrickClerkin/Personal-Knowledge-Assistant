"""
Retrieval evaluation framework with standard IR metrics.

Provides tools for measuring retrieval quality using established
information retrieval metrics. Supports both automated evaluation
against gold-standard relevance judgments and manual inspection.

Metrics implemented:
    - Precision@K: Fraction of retrieved documents that are relevant.
    - Recall@K: Fraction of relevant documents that are retrieved.
    - MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result.
    - nDCG@K (Normalised Discounted Cumulative Gain): Accounts for
      graded relevance and position of results.
    - MAP (Mean Average Precision): Mean of AP across queries.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..ingestion.storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RelevanceJudgment:
    """A query with its relevant document/chunk IDs.

    Attributes:
        query: The search query.
        relevant_ids: Set of chunk_ids or doc_ids considered relevant.
        relevance_scores: Optional graded relevance (id → score).
    """
    query: str
    relevant_ids: Set[str] = field(default_factory=set)
    relevance_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Results from evaluating a single query.

    Attributes:
        query: The evaluated query.
        precision_at_k: Precision at the given K.
        recall_at_k: Recall at the given K.
        reciprocal_rank: 1/rank of first relevant result (0 if none).
        ndcg_at_k: Normalised discounted cumulative gain.
        average_precision: Average precision for this query.
        retrieved_ids: IDs of retrieved chunks in order.
        relevant_retrieved: IDs that were both retrieved and relevant.
    """
    query: str
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg_at_k: float = 0.0
    average_precision: float = 0.0
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_retrieved: List[str] = field(default_factory=list)


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics.

    Compares retrieved results against gold-standard relevance
    judgments to compute precision, recall, MRR, nDCG, and MAP.

    Usage:
        evaluator = RetrievalEvaluator()
        evaluator.add_judgment("What is SOLID?", {"chunk_1", "chunk_5"})
        results = evaluator.evaluate(kb, top_k=5)
    """

    def __init__(self):
        self.judgments: List[RelevanceJudgment] = []

    def add_judgment(
        self,
        query: str,
        relevant_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a relevance judgment for a query.

        Args:
            query: The search query.
            relevant_ids: Set of chunk/doc IDs considered relevant.
            relevance_scores: Optional graded relevance scores.
        """
        self.judgments.append(
            RelevanceJudgment(
                query=query,
                relevant_ids=relevant_ids,
                relevance_scores=relevance_scores or {},
            )
        )

    def evaluate_query(
        self,
        results: List[SearchResult],
        judgment: RelevanceJudgment,
        k: int = 5,
    ) -> EvaluationResult:
        """Evaluate retrieval results for a single query.

        Args:
            results: Search results from the knowledge base.
            judgment: Gold-standard relevance judgment.
            k: Cut-off for metrics.

        Returns:
            EvaluationResult with all computed metrics.
        """
        retrieved_ids = [r.chunk.chunk_id for r in results[:k]]
        relevant = judgment.relevant_ids

        # Precision@K
        relevant_retrieved = [
            rid for rid in retrieved_ids if rid in relevant
        ]
        precision = len(relevant_retrieved) / k if k > 0 else 0.0

        # Recall@K
        recall = (
            len(relevant_retrieved) / len(relevant)
            if relevant else 0.0
        )

        # Reciprocal Rank
        rr = 0.0
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                rr = 1.0 / i
                break

        # nDCG@K
        ndcg = self._compute_ndcg(retrieved_ids, judgment, k)

        # Average Precision
        ap = self._compute_average_precision(retrieved_ids, relevant)

        return EvaluationResult(
            query=judgment.query,
            precision_at_k=precision,
            recall_at_k=recall,
            reciprocal_rank=rr,
            ndcg_at_k=ndcg,
            average_precision=ap,
            retrieved_ids=retrieved_ids,
            relevant_retrieved=relevant_retrieved,
        )

    def evaluate(
        self, knowledge_base, top_k: int = 5
    ) -> Dict[str, object]:
        """Run evaluation across all stored judgments.

        Args:
            knowledge_base: KnowledgeBase instance to search.
            top_k: Number of results to retrieve per query.

        Returns:
            Dict with per-query results and aggregate metrics.
        """
        if not self.judgments:
            logger.warning("No relevance judgments loaded.")
            return {"queries": [], "aggregate": {}}

        query_results = []
        for judgment in self.judgments:
            results = knowledge_base.search(
                judgment.query, top_k=top_k
            )
            eval_result = self.evaluate_query(results, judgment, k=top_k)
            query_results.append(eval_result)

        # Aggregate metrics
        n = len(query_results)
        aggregate = {
            "num_queries": n,
            "mean_precision_at_k": sum(
                r.precision_at_k for r in query_results
            ) / n,
            "mean_recall_at_k": sum(
                r.recall_at_k for r in query_results
            ) / n,
            "mrr": sum(
                r.reciprocal_rank for r in query_results
            ) / n,
            "mean_ndcg_at_k": sum(
                r.ndcg_at_k for r in query_results
            ) / n,
            "map": sum(
                r.average_precision for r in query_results
            ) / n,
        }

        logger.info(
            "Evaluation complete: %d queries, MRR=%.4f, MAP=%.4f",
            n, aggregate["mrr"], aggregate["map"],
        )

        return {"queries": query_results, "aggregate": aggregate}

    def load_judgments(self, path: Path) -> int:
        """Load relevance judgments from a JSON file.

        Expected format:
            [
                {
                    "query": "What is dependency injection?",
                    "relevant_ids": ["chunk_1", "chunk_5"],
                    "relevance_scores": {"chunk_1": 2, "chunk_5": 1}
                }
            ]

        Args:
            path: Path to the JSON file.

        Returns:
            Number of judgments loaded.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for item in data:
            self.add_judgment(
                query=item["query"],
                relevant_ids=set(item.get("relevant_ids", [])),
                relevance_scores=item.get("relevance_scores", {}),
            )
            count += 1

        logger.info("Loaded %d relevance judgments from %s", count, path)
        return count

    def save_judgments(self, path: Path) -> None:
        """Save current judgments to a JSON file."""
        data = [
            {
                "query": j.query,
                "relevant_ids": list(j.relevant_ids),
                "relevance_scores": j.relevance_scores,
            }
            for j in self.judgments
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d judgments to %s", len(data), path)

    @staticmethod
    def _compute_ndcg(
        retrieved_ids: List[str],
        judgment: RelevanceJudgment,
        k: int,
    ) -> float:
        """Compute normalised discounted cumulative gain at K.

        Uses graded relevance scores if available, otherwise
        binary relevance (1 for relevant, 0 for not).
        """
        import math

        def _dcg(scores: List[float]) -> float:
            return sum(
                score / math.log2(i + 2)
                for i, score in enumerate(scores)
            )

        # Actual relevance of retrieved documents
        actual_scores = []
        for rid in retrieved_ids[:k]:
            if rid in judgment.relevance_scores:
                actual_scores.append(judgment.relevance_scores[rid])
            elif rid in judgment.relevant_ids:
                actual_scores.append(1.0)
            else:
                actual_scores.append(0.0)

        dcg = _dcg(actual_scores)

        # Ideal ordering: sort all relevant scores descending
        ideal_scores = sorted(
            [
                judgment.relevance_scores.get(rid, 1.0)
                for rid in judgment.relevant_ids
            ],
            reverse=True,
        )[:k]

        idcg = _dcg(ideal_scores) if ideal_scores else 0.0
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _compute_average_precision(
        retrieved_ids: List[str], relevant: Set[str]
    ) -> float:
        """Compute average precision for a ranked list."""
        if not relevant:
            return 0.0

        hits = 0
        sum_precisions = 0.0
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                hits += 1
                sum_precisions += hits / i

        return sum_precisions / len(relevant) if relevant else 0.0
