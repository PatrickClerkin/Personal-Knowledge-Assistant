"""
Run RAGAS-style answer evaluation from the command line.

Loads a JSONL test set, runs each question through the full RAG
pipeline, and evaluates the answers using the three LLM-as-judge
metrics defined in :mod:`src.evaluation.answer_eval`:

    - Faithfulness
    - Answer Relevancy
    - Context Precision

Writes a JSON report to ``data/eval/results_<timestamp>.json`` (or a
path given with ``--output``) and prints a table summary.

Usage
-----
    python -m scripts.run_eval
    python -m scripts.run_eval --test-set data/eval/sample_test_set.jsonl
    python -m scripts.run_eval --yes --output data/eval/my_run.json

Each test case must be a JSON object on its own line with at least:
    {"question": "...", "ground_truth": "..."}

By default the runner will show an estimated Claude API call count and
ask for confirmation before starting.  Pass ``--yes`` to skip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# Make `src.*` importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.answer_eval import (  # noqa: E402
    AnswerEvaluator,
    AnswerEvalResult,
)
from src.ingestion.knowledge_base import KnowledgeBase  # noqa: E402
from src.rag.llm import ClaudeProvider  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Rough upper-bound estimate of Claude calls per evaluated question:
#   - 1 RAG generation
#   - 1 claim extraction
#   - 1 claim-batch judgement
#   - 1 reverse-question generation
#   - 1 context-precision batch judgement
#   - occasional extras for HyDE / query rewriting / adaptive retry
_API_CALLS_PER_QUESTION_ESTIMATE = 7


def _load_test_set(path: Path) -> List[dict]:
    """Load a JSONL test set, skipping blank lines."""
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    cases: List[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
            if "question" not in obj or "ground_truth" not in obj:
                raise ValueError(
                    f"Line {lineno} missing 'question' or 'ground_truth'"
                )
            cases.append(obj)
    return cases


def _confirm(prompt: str) -> bool:
    """Simple yes/no prompt with a default of No."""
    try:
        response = input(prompt).strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes"}


def _print_table(results: List[AnswerEvalResult]) -> None:
    """Render per-question scores as a plain ASCII table."""
    header = (
        f"{'#':>3}  {'Faith':>6}  {'Relev':>6}  {'CtxP':>6}  "
        f"{'RAGAS':>6}  Question"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, start=1):
        q = r.question if len(r.question) <= 60 else r.question[:57] + "…"
        print(
            f"{i:>3}  "
            f"{r.faithfulness:>6.3f}  "
            f"{r.answer_relevancy:>6.3f}  "
            f"{r.context_precision:>6.3f}  "
            f"{r.ragas_score:>6.3f}  "
            f"{q}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RAGAS-style answer evaluation against the "
        "full RAG pipeline.",
    )
    parser.add_argument(
        "--test-set",
        default="data/eval/sample_test_set.jsonl",
        help="Path to a JSONL test set "
        "(default: data/eval/sample_test_set.jsonl).",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="KnowledgeBase index path "
        "(default: KB_INDEX_PATH env var or data/index/default).",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Chunking strategy "
        "(default: KB_STRATEGY env var or 'sentence').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON results "
        "(default: data/eval/results_<timestamp>.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K retrieval for the RAG pipeline (default 8).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    test_set_path = Path(args.test_set)
    try:
        cases = _load_test_set(test_set_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    n = len(cases)
    if n == 0:
        print(f"No test cases found in {test_set_path}.")
        return 1

    est_calls = n * _API_CALLS_PER_QUESTION_ESTIMATE
    print(f"Loaded {n} test cases from {test_set_path}")
    print(
        f"Estimated Claude API calls for this run: ~{est_calls} "
        f"(~{_API_CALLS_PER_QUESTION_ESTIMATE} per question)."
    )

    if not args.yes and not _confirm("Proceed? [y/N]: "):
        print("Cancelled.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY is not set. Answer evaluation "
            "requires the Claude API.",
            file=sys.stderr,
        )
        return 2

    # Build the KB / RAG pipeline the same way the web layer does.
    index_path = (
        args.index
        or os.environ.get("KB_INDEX_PATH")
        or "data/index/default"
    )
    strategy = (
        args.strategy
        or os.environ.get("KB_STRATEGY")
        or "sentence"
    )

    print(f"Initialising KnowledgeBase at {index_path}...")
    kb = KnowledgeBase(index_path=index_path, chunk_strategy=strategy)
    print(f"  loaded {kb.size} chunks.")

    llm = ClaudeProvider()
    rag = RAGPipeline(
        knowledge_base=kb,
        llm_provider=llm,
        top_k=args.top_k,
        rerank=True,
        hybrid=True,
    )
    # Share the KB's embedder so we don't load MiniLM twice.
    evaluator = AnswerEvaluator(
        llm_provider=llm,
        embedding_service=kb._embedder,
    )

    results: List[AnswerEvalResult] = []
    start = time.time()
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        ground_truth = case["ground_truth"]
        print(f"\n[{i}/{n}] {question}")

        try:
            rag_response = rag.query(question, top_k=args.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG pipeline failed on %r: %s", question, exc)
            print(f"  SKIPPED — pipeline error: {exc}")
            continue

        contexts = [r.chunk.content for r in rag_response.sources]
        try:
            result = evaluator.evaluate(
                question=question,
                answer=rag_response.answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluator failed on %r: %s", question, exc)
            print(f"  SKIPPED — evaluator error: {exc}")
            continue

        results.append(result)
        print(
            f"  Faith={result.faithfulness:.3f}  "
            f"Relev={result.answer_relevancy:.3f}  "
            f"CtxP={result.context_precision:.3f}  "
            f"RAGAS={result.ragas_score:.3f}"
        )

    elapsed = time.time() - start
    print(f"\nEvaluated {len(results)}/{n} questions in {elapsed:.1f}s.")

    if not results:
        print("No successful evaluations — nothing to report.")
        return 1

    report = evaluator.aggregate(results)

    print("\nPer-question scores:")
    _print_table(results)
    print("\n" + "=" * 64)
    print(report.summary())
    print("=" * 64)

    # Decide where to write the report.
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/eval/results_{ts}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"\nResults written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())