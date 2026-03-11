"""
Evaluation suite for the Personal Knowledge Assistant.

Runs a set of labelled test queries against the knowledge base,
computes IR metrics per query, and produces an aggregated report.

Test queries are defined in a JSON file with the format:
    [
        {
            "query": "what is dependency injection?",
            "relevant_doc_ids": ["doc_abc123", "doc_def456"]
        },
        ...
    ]

Usage:
    suite = EvaluationSuite(knowledge_base=kb)
    report = suite.run("data/eval/test_queries.json")
    print(report.summary())
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from .metrics import precision_at_k, mean_reciprocal_rank, ndcg_at_k
from ..ingestion.knowledge_base import KnowledgeBase
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Metrics for a single evaluated query.

    Attributes:
        query: The query string.
        relevant_doc_ids: Ground truth relevant document IDs.
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        precision_at_5: Precision@5 score.
        mrr: Mean Reciprocal Rank score.
        ndcg_at_5: nDCG@5 score.
    """
    query: str
    relevant_doc_ids: List[str]
    retrieved_doc_ids: List[str]
    precision_at_5: float
    mrr: float
    ndcg_at_5: float


@dataclass
class EvaluationReport:
    """Aggregated results from a full evaluation run.

    Attributes:
        query_results: Per-query metric breakdown.
        mean_precision_at_5: Average Precision@5 across all queries.
        mean_mrr: Average MRR across all queries.
        mean_ndcg_at_5: Average nDCG@5 across all queries.
        num_queries: Total number of queries evaluated.
        top_k: Retrieval cutoff used.
    """
    query_results: List[QueryResult]
    mean_precision_at_5: float
    mean_mrr: float
    mean_ndcg_at_5: float
    num_queries: int
    top_k: int

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"Evaluated {self.num_queries} queries @ top-{self.top_k}: "
            f"P@5={self.mean_precision_at_5:.3f}  "
            f"MRR={self.mean_mrr:.3f}  "
            f"nDCG@5={self.mean_ndcg_at_5:.3f}"
        )

    def to_dict(self) -> dict:
        """Serialise to a dict for JSON API responses."""
        return {
            "num_queries": self.num_queries,
            "top_k": self.top_k,
            "mean_precision_at_5": round(self.mean_precision_at_5, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg_at_5": round(self.mean_ndcg_at_5, 4),
            "query_results": [
                {
                    "query": r.query,
                    "precision_at_5": round(r.precision_at_5, 4),
                    "mrr": round(r.mrr, 4),
                    "ndcg_at_5": round(r.ndcg_at_5, 4),
                    "retrieved_doc_ids": r.retrieved_doc_ids,
                    "relevant_doc_ids": r.relevant_doc_ids,
                }
                for r in self.query_results
            ],
        }


class EvaluationSuite:
    """Runs IR evaluation against the knowledge base.

    Args:
        knowledge_base: The KnowledgeBase instance to evaluate.
        top_k: Number of results to retrieve per query (default 5).
    """

    def __init__(self, knowledge_base: KnowledgeBase, top_k: int = 5):
        self.kb = knowledge_base
        self.top_k = top_k

    def run(self, test_queries_path: str) -> EvaluationReport:
        """Run evaluation against a JSON test query file.

        Args:
            test_queries_path: Path to the JSON file containing
                labelled queries with relevant_doc_ids.

        Returns:
            EvaluationReport with per-query and aggregated metrics.

        Raises:
            FileNotFoundError: If the test queries file does not exist.
            ValueError: If the file format is invalid.
        """
        path = Path(test_queries_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Test queries file not found: {test_queries_path}"
            )

        with open(path, encoding="utf-8") as f:
            test_cases = json.load(f)

        if not isinstance(test_cases, list):
            raise ValueError("Test queries file must contain a JSON array")

        query_results = []
        for case in test_cases:
            result = self._evaluate_query(
                query=case["query"],
                relevant_doc_ids=case.get("relevant_doc_ids", []),
            )
            query_results.append(result)
            logger.debug(
                "Query '%s': P@5=%.3f MRR=%.3f nDCG@5=%.3f",
                case["query"][:40],
                result.precision_at_5,
                result.mrr,
                result.ndcg_at_5,
            )

        if not query_results:
            return EvaluationReport(
                query_results=[],
                mean_precision_at_5=0.0,
                mean_mrr=0.0,
                mean_ndcg_at_5=0.0,
                num_queries=0,
                top_k=self.top_k,
            )

        n = len(query_results)
        return EvaluationReport(
            query_results=query_results,
            mean_precision_at_5=sum(r.precision_at_5 for r in query_results) / n,
            mean_mrr=sum(r.mrr for r in query_results) / n,
            mean_ndcg_at_5=sum(r.ndcg_at_5 for r in query_results) / n,
            num_queries=n,
            top_k=self.top_k,
        )

    def _evaluate_query(
        self,
        query: str,
        relevant_doc_ids: List[str],
    ) -> QueryResult:
        """Run a single query and compute its metrics."""
        results = self.kb.search(query, top_k=self.top_k)
        retrieved_doc_ids = [r.chunk.doc_id for r in results]
        relevant_set = set(relevant_doc_ids)

        return QueryResult(
            query=query,
            relevant_doc_ids=relevant_doc_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            precision_at_5=precision_at_k(retrieved_doc_ids, relevant_set, k=5),
            mrr=mean_reciprocal_rank(retrieved_doc_ids, relevant_set),
            ndcg_at_5=ndcg_at_k(retrieved_doc_ids, relevant_set, k=5),
        )