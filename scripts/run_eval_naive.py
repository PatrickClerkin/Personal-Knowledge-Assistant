"""
Naive baseline RAG evaluation — BM25 retrieval + top-K + Claude generation.

Companion to ``scripts/run_eval.py``.  Runs the EXACT SAME RAGAS-style
evaluator (:class:`src.evaluation.answer_eval.AnswerEvaluator`) against
a deliberately stripped-down retrieve-then-generate pipeline so the
resulting numbers can be dropped into a single new row of the
dissertation's Table 5.1 alongside the full-pipeline numbers.

The methodological purpose of this script is to control for everything
except architectural complexity, so that the gap between this baseline
and the full pipeline is interpretable as the contribution of the
architecture itself.

What this naive baseline DOES use
---------------------------------
- The same :class:`KnowledgeBase`, the same chunked corpus, the same
  chunking strategy (whatever was used at ingest time).
- The same BM25 index (``kb._bm25``) that the full pipeline uses as one
  half of its hybrid retrieval — i.e. the same tokeniser, the same
  ``BM25Okapi`` implementation, the same chunk-to-document mapping.
- The same top-K (default 8, matching ``run_eval.py``).
- The same :class:`ClaudeProvider`, the same model, the same default
  system prompt (imported from :mod:`src.rag.pipeline`), and a
  context-format string that mirrors ``RAGPipeline._format_context``.
- The same :class:`AnswerEvaluator` with the same judge prompts and the
  same shared :class:`EmbeddingService`.

What this naive baseline DOES NOT use
-------------------------------------
- FAISS (dense semantic) retrieval.
- Reciprocal Rank Fusion across BM25 + FAISS.
- Cross-encoder reranking.
- HyDE (hypothetical document) query expansion.
- Synonym or multi-query expansion.
- Entity-aware re-ranking.
- Sentence-level grounding scoring.
- Adaptive re-retrieval on low confidence.
- Sentence-level fact verification.
- Semantic answer caching.
- Conversation memory or follow-up query rewriting.

Usage
-----
    python -m scripts.run_eval_naive
    python -m scripts.run_eval_naive --test-set data/eval/domain_test_set.jsonl
    python -m scripts.run_eval_naive --yes --output data/eval/naive_run.json
    python -m scripts.run_eval_naive --top-k 8 --yes

Each test case must be a JSON object on its own line with at least:
    {"question": "...", "ground_truth": "..."}

By default the runner shows an estimated Claude API call count and
asks for confirmation before starting.  Pass ``--yes`` to skip.

To produce the combined Table 5.1 row, run the script once per corpus
(NLP-Papers, ML-Notes) with the matching ``--index`` and ``--test-set``
paths, then average per-metric across the two output JSONs.
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

# Make ``src.*`` importable when this script is run directly,
# matching the convention used by ``scripts/run_eval.py``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.answer_eval import (  # noqa: E402
    AnswerEvaluator,
    AnswerEvalResult,
)
from src.ingestion.knowledge_base import KnowledgeBase  # noqa: E402
from src.ingestion.storage.vector_store import SearchResult  # noqa: E402
from src.rag.llm import ClaudeProvider  # noqa: E402
# Re-use the full pipeline's default system prompt verbatim — same
# generation contract, only the retrieval mechanism differs.
from src.rag.pipeline import _DEFAULT_SYSTEM_PROMPT  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Rough upper-bound estimate of Claude calls per evaluated question for
# the NAIVE baseline only.  Naive baseline never invokes HyDE,
# query-rewrite, reformulation, or grounding-retry, so it runs strictly
# fewer LLM calls per question than the full pipeline.
#
#   - 1 RAG generation
#   - 1 claim extraction (faithfulness)
#   - 1 claim-batch judgement (faithfulness)
#   - 1 reverse-question generation (answer relevancy)
#   - 1 context-precision batch judgement
_API_CALLS_PER_QUESTION_ESTIMATE = 5

# Match the full pipeline's context-window budget so the LLM sees a
# context block of the same maximum size.  Different chunk SET, same
# CONTAINER size — keeps the comparison clean.
_DEFAULT_MAX_CONTEXT_CHARS = 16000


def _load_test_set(path: Path) -> List[dict]:
    """Load a JSONL test set, skipping blank lines.

    Parameters
    ----------
    path
        Path to a JSONL file where each line is a JSON object with at
        least ``question`` and ``ground_truth`` keys.

    Returns
    -------
    list[dict]
        One dict per non-blank line.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If a line is malformed JSON or is missing required keys.
    """
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


def _format_context(
    results: List[SearchResult],
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Format retrieved chunks into a context string for the LLM.

    Mirrors :meth:`src.rag.pipeline.RAGPipeline._format_context` so that
    the LLM sees an identically-shaped context block — only the chunks
    inside it differ.  Holding the format fixed eliminates prompt-shape
    as a confound when comparing the baseline against the full pipeline.

    Parameters
    ----------
    results
        :class:`SearchResult` objects in BM25-rank order.
    max_context_chars
        Hard cap on the size of the assembled context block, matching
        the full pipeline's default.

    Returns
    -------
    str
        Newline-separated context block ready to embed in a prompt.
    """
    context_parts: List[str] = []
    total_chars = 0

    for r in results:
        chunk = r.chunk
        source_info = f"Source: {chunk.source_doc_title}"
        if chunk.page_number:
            source_info += f", Page {chunk.page_number}"
        source_info += f" (relevance: {r.score:.3f})"

        chunk_text = (
            f"--- {source_info} ---\n"
            f"{chunk.content.strip()}\n"
        )

        if total_chars + len(chunk_text) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "\n[truncated]"
                context_parts.append(chunk_text)
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n".join(context_parts)


def _build_prompt(question: str, context: str) -> str:
    """Build the augmented user prompt with retrieved context.

    Identical to :meth:`RAGPipeline._build_prompt` — same wording,
    same delimiters — so the only experimental variable between this
    baseline and the full pipeline is which chunks fill ``context``.
    """
    return (
        f"Context from the knowledge base:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Please answer based on the context above."
    )


def _print_table(results: List[AnswerEvalResult]) -> None:
    """Render per-question scores as a plain ASCII table.

    Mirrors the layout used by ``scripts/run_eval.py`` so the two output
    streams can be eyeballed side-by-side without re-aligning columns.
    """
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
        description="Run RAGAS-style answer evaluation against a NAIVE "
        "BM25-only retrieve-then-generate baseline.",
    )
    parser.add_argument(
        "--test-set",
        default="data/eval/domain_test_set.jsonl",
        help="Path to a JSONL test set "
        "(default: data/eval/domain_test_set.jsonl).",
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
        help="Chunking strategy used at ingest time "
        "(default: KB_STRATEGY env var or 'sentence').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON results "
        "(default: data/eval/naive_results_<timestamp>.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K BM25 retrieval depth (default 8, matches run_eval.py).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    # ---- Load and validate the test set up-front ----------------------
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
        f"Estimated Claude API calls for this naive run: ~{est_calls} "
        f"(~{_API_CALLS_PER_QUESTION_ESTIMATE} per question — fewer "
        f"than the full pipeline because there is no HyDE, no query "
        f"rewriting, no reformulation, and no grounding-retry)."
    )
    print(
        "Retrieval: BM25 only.  "
        "No FAISS, no RRF, no reranker, no expansion, no entity boost."
    )

    if not args.yes and not _confirm("Proceed? [y/N]: "):
        print("Cancelled.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY is not set.  Answer evaluation "
            "requires the Claude API.",
            file=sys.stderr,
        )
        return 2

    # ---- Build the KB and the (lean) generation stack -----------------
    # We deliberately do NOT instantiate RAGPipeline here.  Building
    # the full pipeline would silently turn on cache, HyDE, grounding,
    # fact verification, and adaptive retry — exactly what we are
    # trying to ablate away.  Instead we touch only the BM25 store and
    # the LLM provider.
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

    if kb.size == 0:
        print(
            "Error: knowledge base is empty.  Ingest documents first.",
            file=sys.stderr,
        )
        return 1

    if kb._bm25.size == 0:
        # Defensive: a freshly-loaded FAISS index without a co-located
        # ``.bm25.pkl`` file would leave the BM25 store empty and the
        # baseline would silently retrieve zero chunks for every
        # question.  Fail loudly so the operator can rebuild.
        print(
            "Error: BM25 store is empty for this index.  Re-save the "
            "knowledge base after ingest, or rebuild the index.",
            file=sys.stderr,
        )
        return 1

    llm = ClaudeProvider()

    # Share the KB's embedder so the evaluator does not load a second
    # copy of MiniLM (also avoids a known PyTorch meta-tensor bug on
    # some torch + sentence-transformers combinations).
    evaluator = AnswerEvaluator(
        llm_provider=llm,
        embedding_service=kb._embedder,
    )

    # ---- Per-question loop --------------------------------------------
    results: List[AnswerEvalResult] = []
    start = time.time()
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        ground_truth = case["ground_truth"]
        print(f"\n[{i}/{n}] {question}")

        # Naive retrieval: BM25 only, no fusion, no rerank, no expand.
        try:
            search_results = kb._bm25.search(question, top_k=args.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("BM25 retrieval failed on %r: %s", question, exc)
            print(f"  SKIPPED — retrieval error: {exc}")
            continue

        if not search_results:
            # Mirror the full pipeline's empty-results message so the
            # judge sees a comparable refusal rather than an empty
            # string (which would crash the evaluator's claim
            # extraction with NO_CLAIMS-but-also-no-context).
            print("  WARNING — BM25 returned 0 chunks.")
            answer = (
                "I couldn't find any relevant information in the "
                "knowledge base to answer that question."
            )
            contexts: List[str] = []
        else:
            context = _format_context(search_results)
            prompt = _build_prompt(question, context)
            try:
                # Match the full pipeline's generation defaults so the
                # only experimental variable is the retrieved chunk set.
                llm_response = llm.generate(
                    prompt=prompt,
                    system=_DEFAULT_SYSTEM_PROMPT,
                    max_tokens=1024,
                    temperature=0.3,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Claude generation failed on %r: %s", question, exc
                )
                print(f"  SKIPPED — LLM error: {exc}")
                continue
            answer = llm_response.content
            contexts = [r.chunk.content for r in search_results]

        # Evaluation — identical to run_eval.py.  Same evaluator, same
        # judge prompts, same context list shape.
        try:
            result = evaluator.evaluate(
                question=question,
                answer=answer,
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

    # ---- Aggregate and report -----------------------------------------
    report = evaluator.aggregate(results)

    print("\nPer-question scores (NAIVE baseline):")
    _print_table(results)
    print("\n" + "=" * 64)
    print("NAIVE BASELINE — " + report.summary())
    print("=" * 64)

    # Decide where to write the report.  Default filename includes the
    # word "naive" so it can never be confused with a full-pipeline run.
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/eval/naive_results_{ts}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    # Tag the payload so the dissertation pipeline can tell baseline
    # output apart from full-pipeline output at a glance.
    payload["pipeline_variant"] = "naive_bm25_topk"
    payload["pipeline_components"] = {
        "retrieval": "bm25_only",
        "rerank": False,
        "hybrid": False,
        "expand_query": None,
        "use_hyde": False,
        "ground_answer": False,
        "adaptive_retrieval": False,
        "verify_facts": False,
        "use_cache": False,
        "top_k": args.top_k,
    }
    payload["test_set"] = str(test_set_path)
    payload["index"] = index_path
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())