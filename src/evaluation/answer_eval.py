"""
RAGAS-style answer evaluation.

Implements three LLM-as-judge metrics for evaluating the quality of RAG
pipeline outputs, inspired by the RAGAS methodology (Es et al., 2023)
but implemented from scratch against the project's existing
``LLMProvider`` and ``EmbeddingService`` abstractions.

Metrics
-------
Faithfulness
    Measures whether the answer's factual claims are supported by the
    retrieved context chunks. Catches hallucinations. The judge LLM
    first decomposes the answer into atomic claims, then for each
    claim decides whether the context supports it. Score is the
    fraction of claims that are supported.

Answer Relevancy
    Measures whether the answer actually addresses the question that
    was asked. The judge LLM generates N plausible reverse-questions
    from the answer alone; we embed them and the original question,
    and take the mean cosine similarity.  High similarity means the
    answer is on-topic; low similarity means the model answered a
    different question.

Context Precision
    Measures how useful the retrieved chunks were for answering the
    question, rewarding rankings that place relevant chunks first.
    The judge LLM marks each retrieved chunk as relevant / not
    relevant given the question and ground-truth answer, and we
    aggregate using the rank-weighted RAGAS formula::

        CP = sum_{k=1..K} (precision@k * v_k) / total_relevant

    where v_k is 1 if the chunk at rank k is relevant, 0 otherwise,
    and precision@k = (# relevant in top k) / k.

Design
------
All three metrics share the same LLM provider via the project's
``LLMProvider`` abstraction — no new dependencies. LLM calls are
batched where possible to minimise API spend:

    - Faithfulness: 1 call to extract claims, 1 call to judge them all.
    - Answer Relevancy: 1 call to generate N reverse-questions.
    - Context Precision: 1 call to judge all K contexts at once.

Total: ~4 LLM calls per evaluated question, plus one embedding batch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import List, Optional

import numpy as np

from ..ingestion.embeddings.embedding_service import EmbeddingService
from ..rag.llm import LLMProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------

_CLAIM_EXTRACTION_SYSTEM = """You are an expert at decomposing text into atomic factual claims.

Rules:
1. Output ONE claim per line, nothing else.
2. Each claim must be a single self-contained factual assertion.
3. Strip hedging language ("might", "possibly", "I think").
4. Do NOT number, bullet, or prefix claims in any way.
5. If the text contains no factual claims (refusal, greeting, opinion-only), output the literal string: NO_CLAIMS
6. Do not include preamble, explanation, or closing remarks."""

_CLAIM_EXTRACTION_USER = """Extract the atomic factual claims from this answer:

ANSWER:
{answer}"""


_CLAIM_JUDGEMENT_SYSTEM = """You are a strict fact-checking assistant.

For each claim, decide whether the CONTEXT clearly supports it.
- "supported: true" only if the context directly entails the claim.
- "supported: false" if the context is silent on it, contradicts it,
  or only weakly relates to it.

Output ONLY a JSON array of objects, one per claim, in the same order
as the claims were given. Each object must have the keys:
    "claim": the claim text verbatim
    "supported": true or false

No preamble, no markdown fences, no explanation outside the JSON."""

_CLAIM_JUDGEMENT_USER = """CONTEXT:
{context}

CLAIMS:
{numbered_claims}

Return the JSON array now."""


_REVERSE_QUESTION_SYSTEM = """You are a question-generation assistant.

Given an ANSWER, generate plausible questions that this answer could
be responding to. Each question should be self-contained and the kind
of thing a real user would ask.

Output ONE question per line. No numbering, no bullets, no preamble."""

_REVERSE_QUESTION_USER = """Generate {n} plausible questions for this answer:

ANSWER:
{answer}"""


_CONTEXT_PRECISION_SYSTEM = """You are a relevance-judging assistant.

For each retrieved CONTEXT passage, decide whether it contains
information that directly helps produce the GROUND-TRUTH ANSWER to
the QUESTION. A context is relevant even if it only contributes part
of the answer.

Output ONLY a JSON array of objects, one per context, in the same
order as the contexts were given. Each object must have the keys:
    "index": the 1-based index of the context
    "relevant": true or false

No preamble, no markdown fences, no explanation outside the JSON."""

_CONTEXT_PRECISION_USER = """QUESTION:
{question}

GROUND-TRUTH ANSWER:
{ground_truth}

CONTEXTS:
{numbered_contexts}

Return the JSON array now."""


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass
class ClaimJudgement:
    """Verdict on whether a single extracted claim is supported.

    Attributes:
        claim: The atomic claim extracted from the answer.
        supported: True if the context supports the claim.
    """

    claim: str
    supported: bool

    def to_dict(self) -> dict:
        return {"claim": self.claim, "supported": self.supported}


@dataclass
class ContextJudgement:
    """Verdict on whether a single retrieved chunk is relevant.

    Attributes:
        rank: 1-based rank of the context in the retrieval result.
        content_preview: First ~200 chars of the chunk content.
        relevant: True if the chunk helps answer the question.
    """

    rank: int
    content_preview: str
    relevant: bool

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "content_preview": self.content_preview,
            "relevant": self.relevant,
        }


@dataclass
class AnswerEvalResult:
    """Full evaluation result for a single question.

    Attributes:
        question: The user question.
        answer: The pipeline's generated answer.
        ground_truth: The expected/reference answer.
        faithfulness: Fraction of answer claims supported by context.
        answer_relevancy: Mean cosine similarity between the question
            and LLM-generated reverse-questions from the answer.
        context_precision: Rank-weighted precision of retrieved chunks
            against the ground-truth answer.
        claim_judgements: Per-claim supported/not verdicts.
        context_judgements: Per-chunk relevant/not verdicts.
        reverse_questions: The LLM-generated questions used for
            answer_relevancy.
    """

    question: str
    answer: str
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    claim_judgements: List[ClaimJudgement] = field(default_factory=list)
    context_judgements: List[ContextJudgement] = field(default_factory=list)
    reverse_questions: List[str] = field(default_factory=list)

    @property
    def ragas_score(self) -> float:
        """Harmonic mean of the three metrics (0..1).

        Harmonic mean penalises imbalance more than arithmetic mean —
        a single weak metric drags the overall score down, which is
        the desired behaviour for a composite quality indicator.
        Returns 0 if any metric is 0.
        """
        values = [self.faithfulness, self.answer_relevancy, self.context_precision]
        if any(v <= 0 for v in values):
            return 0.0
        return len(values) / sum(1.0 / v for v in values)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "ragas_score": round(self.ragas_score, 4),
            "claim_judgements": [j.to_dict() for j in self.claim_judgements],
            "context_judgements": [j.to_dict() for j in self.context_judgements],
            "reverse_questions": self.reverse_questions,
        }


@dataclass
class AnswerEvalReport:
    """Aggregated report across a full test set.

    Attributes:
        results: Per-question evaluation results.
        mean_faithfulness: Mean faithfulness across all questions.
        mean_answer_relevancy: Mean answer-relevancy across questions.
        mean_context_precision: Mean context-precision across questions.
        mean_ragas_score: Mean composite score.
        num_queries: Number of questions evaluated.
    """

    results: List[AnswerEvalResult]
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_ragas_score: float
    num_queries: int

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"Answer eval over {self.num_queries} queries: "
            f"Faith={self.mean_faithfulness:.3f}  "
            f"Relev={self.mean_answer_relevancy:.3f}  "
            f"CtxP={self.mean_context_precision:.3f}  "
            f"RAGAS={self.mean_ragas_score:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_answer_relevancy": round(self.mean_answer_relevancy, 4),
            "mean_context_precision": round(self.mean_context_precision, 4),
            "mean_ragas_score": round(self.mean_ragas_score, 4),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------


class AnswerEvaluator:
    """RAGAS-style answer evaluator using LLM-as-judge.

    Args:
        llm_provider: LLM used as the judge.  Reusing your RAG pipeline's
            provider is fine — calls are independent.
        embedding_service: Optional pre-built embedding service.  If
            omitted, a new ``EmbeddingService`` is created.  Passing
            the KB's embedder avoids loading the model twice.
        num_reverse_questions: How many reverse-questions to generate
            per answer for the answer-relevancy metric.  Default 3.
        max_context_chars: Characters of each context chunk sent to
            the judge.  Prevents the prompt from blowing past the
            context window on very large retrieved chunks.  Default
            2000.

    Raises:
        ValueError: If ``num_reverse_questions`` < 1 or
            ``max_context_chars`` <= 0.
    """

    DEFAULT_NUM_REVERSE_QUESTIONS: int = 3
    DEFAULT_MAX_CONTEXT_CHARS: int = 2000

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_service: Optional[EmbeddingService] = None,
        num_reverse_questions: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ):
        n_rev = (
            num_reverse_questions
            if num_reverse_questions is not None
            else self.DEFAULT_NUM_REVERSE_QUESTIONS
        )
        max_ctx = (
            max_context_chars
            if max_context_chars is not None
            else self.DEFAULT_MAX_CONTEXT_CHARS
        )

        if n_rev < 1:
            raise ValueError("num_reverse_questions must be >= 1")
        if max_ctx <= 0:
            raise ValueError("max_context_chars must be positive")

        self.llm = llm_provider
        self.embedder = embedding_service or EmbeddingService()
        self.num_reverse_questions = n_rev
        self.max_context_chars = max_ctx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> AnswerEvalResult:
        """Evaluate a single question/answer/context/ground-truth tuple.

        Args:
            question: The user's question.
            answer: The pipeline's generated answer.
            contexts: The retrieved chunk contents, in rank order.
            ground_truth: The reference answer.

        Returns:
            An ``AnswerEvalResult`` with all three metrics and the
            per-claim / per-context judgements used to derive them.
        """
        logger.debug("Evaluating question: %s", question[:60])

        # Faithfulness
        claim_judgements = self._judge_faithfulness(answer, contexts)
        faithfulness = self._score_faithfulness(claim_judgements)

        # Answer relevancy
        reverse_questions = self._generate_reverse_questions(answer)
        answer_relevancy = self._score_answer_relevancy(
            question, reverse_questions
        )

        # Context precision
        context_judgements = self._judge_context_precision(
            question, contexts, ground_truth
        )
        context_precision = self._score_context_precision(context_judgements)

        return AnswerEvalResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            claim_judgements=claim_judgements,
            context_judgements=context_judgements,
            reverse_questions=reverse_questions,
        )

    def aggregate(
        self, results: List[AnswerEvalResult]
    ) -> AnswerEvalReport:
        """Aggregate per-query results into a full report.

        Args:
            results: A list of ``AnswerEvalResult`` objects.

        Returns:
            An ``AnswerEvalReport`` with per-metric means.  Returns a
            zero-filled report if ``results`` is empty.
        """
        if not results:
            return AnswerEvalReport(
                results=[],
                mean_faithfulness=0.0,
                mean_answer_relevancy=0.0,
                mean_context_precision=0.0,
                mean_ragas_score=0.0,
                num_queries=0,
            )

        return AnswerEvalReport(
            results=results,
            mean_faithfulness=mean(r.faithfulness for r in results),
            mean_answer_relevancy=mean(r.answer_relevancy for r in results),
            mean_context_precision=mean(r.context_precision for r in results),
            mean_ragas_score=mean(r.ragas_score for r in results),
            num_queries=len(results),
        )

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    def _judge_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> List[ClaimJudgement]:
        """Extract atomic claims and judge each against contexts."""
        if not answer.strip():
            return []

        claims = self._extract_claims(answer)
        if not claims:
            return []

        if not contexts:
            # No context to support any claim — all unsupported.
            return [ClaimJudgement(claim=c, supported=False) for c in claims]

        return self._batch_judge_claims(claims, contexts)

    def _extract_claims(self, answer: str) -> List[str]:
        """Ask the LLM to decompose the answer into atomic claims."""
        user = _CLAIM_EXTRACTION_USER.format(answer=answer)
        response = self.llm.generate(
            prompt=user,
            system=_CLAIM_EXTRACTION_SYSTEM,
            max_tokens=1024,
            temperature=0.0,
        )
        raw = response.content.strip()
        if not raw or raw.upper().startswith("NO_CLAIMS"):
            return []

        # Each non-empty line is a claim. Strip any stray bullets the
        # model may have added despite the system instruction.
        claims: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
            if line:
                claims.append(line)
        return claims

    def _batch_judge_claims(
        self,
        claims: List[str],
        contexts: List[str],
    ) -> List[ClaimJudgement]:
        """Judge all claims in a single batched LLM call."""
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        context_block = "\n---\n".join(
            self._truncate(c) for c in contexts
        )
        user = _CLAIM_JUDGEMENT_USER.format(
            context=context_block,
            numbered_claims=numbered,
        )
        response = self.llm.generate(
            prompt=user,
            system=_CLAIM_JUDGEMENT_SYSTEM,
            max_tokens=2048,
            temperature=0.0,
        )
        parsed = _extract_json_array(response.content)
        if parsed is None:
            logger.warning(
                "Faithfulness judge returned unparseable JSON; "
                "defaulting all claims to unsupported."
            )
            return [ClaimJudgement(claim=c, supported=False) for c in claims]

        # Align parsed verdicts back to the original claim list. The
        # model may have dropped, reordered, or hallucinated claims,
        # so we match by position and fall back to False for any gaps.
        judgements: List[ClaimJudgement] = []
        for i, claim in enumerate(claims):
            supported = False
            if i < len(parsed) and isinstance(parsed[i], dict):
                supported = bool(parsed[i].get("supported", False))
            judgements.append(ClaimJudgement(claim=claim, supported=supported))
        return judgements

    @staticmethod
    def _score_faithfulness(judgements: List[ClaimJudgement]) -> float:
        """Fraction of claims supported. No claims → vacuously 1.0."""
        if not judgements:
            # Nothing to hallucinate, nothing to check.
            return 1.0
        supported = sum(1 for j in judgements if j.supported)
        return supported / len(judgements)

    # ------------------------------------------------------------------
    # Answer relevancy
    # ------------------------------------------------------------------

    def _generate_reverse_questions(self, answer: str) -> List[str]:
        """Ask the LLM to reverse-engineer questions from the answer."""
        if not answer.strip():
            return []

        user = _REVERSE_QUESTION_USER.format(
            n=self.num_reverse_questions,
            answer=answer,
        )
        response = self.llm.generate(
            prompt=user,
            system=_REVERSE_QUESTION_SYSTEM,
            max_tokens=512,
            temperature=0.3,
        )
        raw = response.content.strip()
        if not raw:
            return []

        questions: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
            if line:
                questions.append(line)
        return questions[: self.num_reverse_questions]

    def _score_answer_relevancy(
        self,
        question: str,
        reverse_questions: List[str],
    ) -> float:
        """Mean cosine similarity between question and reverse-questions.

        Clamped into [0, 1] — cosine similarity can in theory be
        negative for orthogonal embeddings, but for our use case a
        negative value should be reported as zero relevancy.
        """
        if not reverse_questions:
            return 0.0

        q_vec = self.embedder.embed_text(question)
        r_vecs = self.embedder.embed_texts(reverse_questions)
        if r_vecs.ndim == 1:
            r_vecs = r_vecs.reshape(1, -1)

        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        row_norms = np.linalg.norm(r_vecs, axis=1, keepdims=True) + 1e-12
        r_normed = r_vecs / row_norms

        sims = r_normed @ q_norm
        mean_sim = float(np.mean(sims))
        return max(0.0, min(1.0, mean_sim))

    # ------------------------------------------------------------------
    # Context precision
    # ------------------------------------------------------------------

    def _judge_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
    ) -> List[ContextJudgement]:
        """Judge each retrieved chunk for usefulness in one batched call."""
        if not contexts:
            return []

        numbered = "\n---\n".join(
            f"[{i+1}]\n{self._truncate(c)}"
            for i, c in enumerate(contexts)
        )
        user = _CONTEXT_PRECISION_USER.format(
            question=question,
            ground_truth=ground_truth,
            numbered_contexts=numbered,
        )
        response = self.llm.generate(
            prompt=user,
            system=_CONTEXT_PRECISION_SYSTEM,
            max_tokens=2048,
            temperature=0.0,
        )
        parsed = _extract_json_array(response.content)

        judgements: List[ContextJudgement] = []
        for i, content in enumerate(contexts):
            relevant = False
            if parsed is not None and i < len(parsed):
                entry = parsed[i]
                if isinstance(entry, dict):
                    relevant = bool(entry.get("relevant", False))
            judgements.append(
                ContextJudgement(
                    rank=i + 1,
                    content_preview=content[:200],
                    relevant=relevant,
                )
            )
        return judgements

    @staticmethod
    def _score_context_precision(
        judgements: List[ContextJudgement],
    ) -> float:
        """Rank-weighted precision: rewards relevant chunks appearing first.

        Uses the RAGAS formula::

            CP = sum_{k=1..K} (precision@k * v_k) / total_relevant

        Returns 0 if no context is relevant or if the list is empty.
        """
        K = len(judgements)
        if K == 0:
            return 0.0

        flags = [1 if j.relevant else 0 for j in judgements]
        total_relevant = sum(flags)
        if total_relevant == 0:
            return 0.0

        running_sum = 0.0
        for k in range(1, K + 1):
            if flags[k - 1] == 1:
                precision_at_k = sum(flags[:k]) / k
                running_sum += precision_at_k

        return running_sum / total_relevant

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """Trim a chunk to ``max_context_chars`` for the judge prompt."""
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "…"


# ---------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------


def _extract_json_array(text: str) -> Optional[list]:
    """Pull the first top-level JSON array from an LLM response.

    Tolerates markdown fences, leading preamble, and trailing prose.
    Returns ``None`` if nothing parseable is found.
    """
    if not text:
        return None

    # Strip common markdown code fences first.
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    start = cleaned.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    result = json.loads(candidate)
                    return result if isinstance(result, list) else None
                except json.JSONDecodeError:
                    return None
    return None