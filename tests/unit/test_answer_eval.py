"""Tests for the RAGAS-style answer evaluator."""

from __future__ import annotations

from typing import Iterable, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation.answer_eval import (
    AnswerEvaluator,
    AnswerEvalReport,
    AnswerEvalResult,
    ClaimJudgement,
    ContextJudgement,
    _extract_json_array,
)
from src.rag.llm import LLMResponse


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _ScriptedLLM:
    """LLMProvider stub that replays a fixed sequence of responses."""

    def __init__(self, responses: Iterable[str]):
        self._queue: List[str] = list(responses)
        self.calls: List[dict] = []

    def generate(
        self,
        prompt: str,
        system: str = None,
        history=None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system})
        if not self._queue:
            raise AssertionError(
                "Scripted LLM exhausted — more calls than expected"
            )
        content = self._queue.pop(0)
        return LLMResponse(
            content=content,
            model="test-model",
            usage={"input_tokens": 0, "output_tokens": 0},
        )

    def is_available(self) -> bool:
        return True


def _fake_embedder(
    vectors_for_text: dict,
    default: np.ndarray = None,
):
    """Build a mock EmbeddingService that returns deterministic vectors.

    ``vectors_for_text`` maps raw strings to numpy vectors.  Any text
    not in the map uses ``default`` (a zeros vector of the same shape).
    """
    if default is None and vectors_for_text:
        dim = next(iter(vectors_for_text.values())).shape[0]
        default = np.zeros(dim)

    def embed_text(text: str) -> np.ndarray:
        return vectors_for_text.get(text, default)

    def embed_texts(texts: List[str]) -> np.ndarray:
        return np.array([embed_text(t) for t in texts])

    embedder = MagicMock()
    embedder.embed_text.side_effect = embed_text
    embedder.embed_texts.side_effect = embed_texts
    return embedder


# ---------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_array(self):
        assert _extract_json_array('[{"a": 1}, {"a": 2}]') == [
            {"a": 1}, {"a": 2}
        ]

    def test_array_with_preamble(self):
        text = "Here is the answer:\n[1, 2, 3]\nThanks!"
        assert _extract_json_array(text) == [1, 2, 3]

    def test_markdown_fences(self):
        text = "```json\n[true, false]\n```"
        assert _extract_json_array(text) == [True, False]

    def test_empty_array(self):
        assert _extract_json_array("[]") == []

    def test_no_array_returns_none(self):
        assert _extract_json_array("No JSON here") is None

    def test_malformed_returns_none(self):
        assert _extract_json_array("[{not valid json}") is None

    def test_empty_string_returns_none(self):
        assert _extract_json_array("") is None

    def test_brackets_inside_strings_are_ignored(self):
        text = '[{"note": "contains ] bracket"}]'
        assert _extract_json_array(text) == [
            {"note": "contains ] bracket"}
        ]


# ---------------------------------------------------------------------
# AnswerEvaluator — constructor
# ---------------------------------------------------------------------


class TestConstructor:
    def test_invalid_num_reverse_questions(self):
        llm = _ScriptedLLM([])
        with pytest.raises(ValueError):
            AnswerEvaluator(
                llm_provider=llm,
                embedding_service=_fake_embedder({}),
                num_reverse_questions=0,
            )

    def test_invalid_max_context_chars(self):
        llm = _ScriptedLLM([])
        with pytest.raises(ValueError):
            AnswerEvaluator(
                llm_provider=llm,
                embedding_service=_fake_embedder({}),
                max_context_chars=0,
            )

    def test_defaults_applied(self):
        llm = _ScriptedLLM([])
        evaluator = AnswerEvaluator(
            llm_provider=llm,
            embedding_service=_fake_embedder({}),
        )
        assert evaluator.num_reverse_questions == (
            AnswerEvaluator.DEFAULT_NUM_REVERSE_QUESTIONS
        )
        assert evaluator.max_context_chars == (
            AnswerEvaluator.DEFAULT_MAX_CONTEXT_CHARS
        )


# ---------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------


class TestFaithfulness:
    def test_all_claims_supported(self):
        llm = _ScriptedLLM([
            # Claim extraction
            "Python is a programming language\n"
            "Python is widely used",
            # Claim judgement batch
            '[{"claim": "Python is a programming language", '
            '"supported": true}, '
            '{"claim": "Python is widely used", "supported": true}]',
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="Python is a programming language. It is widely used.",
            contexts=["Python is a general-purpose programming language."],
        )
        assert len(judgements) == 2
        assert all(j.supported for j in judgements)
        assert ev._score_faithfulness(judgements) == 1.0

    def test_partial_support(self):
        llm = _ScriptedLLM([
            "Python is a language\nPython was invented in 1823",
            '[{"claim": "Python is a language", "supported": true},'
            ' {"claim": "Python was invented in 1823", '
            '"supported": false}]',
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="Python is a language. Python was invented in 1823.",
            contexts=["Python is a programming language."],
        )
        assert len(judgements) == 2
        assert judgements[0].supported is True
        assert judgements[1].supported is False
        assert ev._score_faithfulness(judgements) == 0.5

    def test_no_claims_returns_vacuous_one(self):
        llm = _ScriptedLLM(["NO_CLAIMS"])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="I don't know.",
            contexts=["Some context"],
        )
        assert judgements == []
        assert ev._score_faithfulness(judgements) == 1.0

    def test_empty_answer_returns_empty(self):
        llm = _ScriptedLLM([])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="",
            contexts=["Some context"],
        )
        assert judgements == []

    def test_no_contexts_marks_all_unsupported(self):
        llm = _ScriptedLLM(["Claim one\nClaim two"])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="Claim one. Claim two.",
            contexts=[],
        )
        assert len(judgements) == 2
        assert all(not j.supported for j in judgements)

    def test_malformed_judge_output_falls_back_to_unsupported(self):
        llm = _ScriptedLLM([
            "Claim one\nClaim two",
            "I am the judge LLM and I refuse to output JSON today.",
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="Claim one. Claim two.",
            contexts=["context"],
        )
        assert len(judgements) == 2
        assert all(not j.supported for j in judgements)

    def test_claims_with_bullet_prefixes_are_cleaned(self):
        llm = _ScriptedLLM([
            "- Python is a language\n* Python is popular\n1. Python is free",
            '[{"claim": "Python is a language", "supported": true},'
            ' {"claim": "Python is popular", "supported": true},'
            ' {"claim": "Python is free", "supported": true}]',
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_faithfulness(
            answer="Python is a language. Python is popular. Python is free.",
            contexts=["Python overview"],
        )
        assert [j.claim for j in judgements] == [
            "Python is a language",
            "Python is popular",
            "Python is free",
        ]


# ---------------------------------------------------------------------
# Answer relevancy
# ---------------------------------------------------------------------


class TestAnswerRelevancy:
    def test_high_relevancy_on_topic_question(self):
        q = "What is Python?"
        reverse_q1 = "What is the Python programming language?"
        reverse_q2 = "Can you describe Python?"
        reverse_q3 = "Tell me about Python."

        # Engineer embeddings: all four vectors point in near-identical
        # directions so cosine similarity is very high.
        vectors = {
            q: np.array([1.0, 0.0, 0.0]),
            reverse_q1: np.array([0.99, 0.14, 0.0]),
            reverse_q2: np.array([0.98, 0.2, 0.0]),
            reverse_q3: np.array([0.97, 0.24, 0.0]),
        }
        llm = _ScriptedLLM([
            f"{reverse_q1}\n{reverse_q2}\n{reverse_q3}"
        ])
        ev = AnswerEvaluator(
            llm,
            embedding_service=_fake_embedder(vectors),
        )
        rev = ev._generate_reverse_questions("Python is a language...")
        assert rev == [reverse_q1, reverse_q2, reverse_q3]
        score = ev._score_answer_relevancy(q, rev)
        assert score > 0.9

    def test_low_relevancy_off_topic(self):
        q = "What is Python?"
        off1 = "What do pythons eat?"
        off2 = "How long do pythons live?"
        off3 = "Are pythons venomous?"

        # Orthogonal vectors → cosine sim ~ 0.
        vectors = {
            q: np.array([1.0, 0.0, 0.0]),
            off1: np.array([0.0, 1.0, 0.0]),
            off2: np.array([0.0, 1.0, 0.0]),
            off3: np.array([0.0, 1.0, 0.0]),
        }
        llm = _ScriptedLLM([f"{off1}\n{off2}\n{off3}"])
        ev = AnswerEvaluator(
            llm,
            embedding_service=_fake_embedder(vectors),
        )
        rev = ev._generate_reverse_questions("Pythons are snakes...")
        score = ev._score_answer_relevancy(q, rev)
        assert score < 0.2

    def test_empty_answer_returns_zero(self):
        llm = _ScriptedLLM([])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        rev = ev._generate_reverse_questions("")
        assert rev == []
        assert ev._score_answer_relevancy("q", rev) == 0.0

    def test_negative_cosine_clamps_to_zero(self):
        q = "What is Python?"
        rev = "What is not Python?"
        vectors = {
            q: np.array([1.0, 0.0]),
            rev: np.array([-1.0, 0.0]),
        }
        llm = _ScriptedLLM([rev])
        ev = AnswerEvaluator(
            llm,
            embedding_service=_fake_embedder(vectors),
            num_reverse_questions=1,
        )
        generated = ev._generate_reverse_questions("some answer")
        score = ev._score_answer_relevancy(q, generated)
        assert score == 0.0

    def test_generated_questions_capped_at_num_reverse(self):
        llm = _ScriptedLLM(["q1\nq2\nq3\nq4\nq5"])
        ev = AnswerEvaluator(
            llm,
            embedding_service=_fake_embedder({}),
            num_reverse_questions=2,
        )
        rev = ev._generate_reverse_questions("answer")
        assert len(rev) == 2


# ---------------------------------------------------------------------
# Context precision
# ---------------------------------------------------------------------


class TestContextPrecision:
    def test_all_relevant_perfect_score(self):
        llm = _ScriptedLLM([
            '[{"index": 1, "relevant": true}, '
            '{"index": 2, "relevant": true}, '
            '{"index": 3, "relevant": true}]'
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_context_precision(
            question="q",
            contexts=["c1", "c2", "c3"],
            ground_truth="gt",
        )
        assert all(j.relevant for j in judgements)
        assert ev._score_context_precision(judgements) == pytest.approx(1.0)

    def test_irrelevant_first_penalises_score(self):
        # [False, True, True] with K=3 total_relevant=2
        # k=1 not relevant → skip
        # k=2 relevant → precision@2 = 1/2 = 0.5
        # k=3 relevant → precision@3 = 2/3
        # CP = (0.5 + 0.6667) / 2 ≈ 0.5833
        llm = _ScriptedLLM([
            '[{"index": 1, "relevant": false}, '
            '{"index": 2, "relevant": true}, '
            '{"index": 3, "relevant": true}]'
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_context_precision(
            question="q",
            contexts=["c1", "c2", "c3"],
            ground_truth="gt",
        )
        score = ev._score_context_precision(judgements)
        assert score == pytest.approx((0.5 + 2 / 3) / 2, abs=1e-4)

    def test_relevant_first_best_score(self):
        # [True, True, False] total_relevant=2
        # k=1 relevant → precision@1 = 1
        # k=2 relevant → precision@2 = 1
        # k=3 not relevant → skip
        # CP = (1 + 1) / 2 = 1.0
        llm = _ScriptedLLM([
            '[{"index": 1, "relevant": true}, '
            '{"index": 2, "relevant": true}, '
            '{"index": 3, "relevant": false}]'
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_context_precision(
            question="q",
            contexts=["c1", "c2", "c3"],
            ground_truth="gt",
        )
        assert ev._score_context_precision(judgements) == pytest.approx(1.0)

    def test_no_relevant_returns_zero(self):
        llm = _ScriptedLLM([
            '[{"index": 1, "relevant": false}, '
            '{"index": 2, "relevant": false}]'
        ])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_context_precision(
            question="q",
            contexts=["c1", "c2"],
            ground_truth="gt",
        )
        assert ev._score_context_precision(judgements) == 0.0

    def test_empty_contexts_returns_zero(self):
        llm = _ScriptedLLM([])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_context_precision(
            question="q",
            contexts=[],
            ground_truth="gt",
        )
        assert judgements == []
        assert ev._score_context_precision(judgements) == 0.0

    def test_malformed_judge_falls_back_to_all_irrelevant(self):
        llm = _ScriptedLLM(["not JSON at all"])
        ev = AnswerEvaluator(llm, embedding_service=_fake_embedder({}))
        judgements = ev._judge_context_precision(
            question="q",
            contexts=["c1", "c2"],
            ground_truth="gt",
        )
        assert len(judgements) == 2
        assert all(not j.relevant for j in judgements)

    def test_long_contexts_are_truncated_in_prompt(self):
        long_ctx = "x" * 5000
        llm = _ScriptedLLM([
            '[{"index": 1, "relevant": true}]'
        ])
        ev = AnswerEvaluator(
            llm,
            embedding_service=_fake_embedder({}),
            max_context_chars=100,
        )
        _ = ev._judge_context_precision(
            question="q",
            contexts=[long_ctx],
            ground_truth="gt",
        )
        sent_prompt = llm.calls[0]["prompt"]
        # Full 5000-char chunk should not appear verbatim.
        assert long_ctx not in sent_prompt
        # The truncation ellipsis should be present.
        assert "…" in sent_prompt


# ---------------------------------------------------------------------
# End-to-end evaluate()
# ---------------------------------------------------------------------


class TestEvaluateEndToEnd:
    def test_evaluate_returns_full_result(self):
        question = "What is Python?"
        answer = "Python is a programming language. It is popular."

        # Engineer vectors so relevancy is high.
        vectors = {
            question: np.array([1.0, 0.0, 0.0]),
            "What programming language is Python?": np.array(
                [0.98, 0.2, 0.0]
            ),
            "Is Python popular?": np.array([0.97, 0.24, 0.0]),
            "Tell me about Python.": np.array([0.96, 0.28, 0.0]),
        }

        llm = _ScriptedLLM([
            # 1. Claim extraction
            "Python is a programming language\nPython is popular",
            # 2. Claim judgement
            '[{"claim": "Python is a programming language", '
            '"supported": true}, '
            '{"claim": "Python is popular", "supported": true}]',
            # 3. Reverse-question generation
            "What programming language is Python?\n"
            "Is Python popular?\n"
            "Tell me about Python.",
            # 4. Context precision batch
            '[{"index": 1, "relevant": true}, '
            '{"index": 2, "relevant": false}]',
        ])
        ev = AnswerEvaluator(
            llm,
            embedding_service=_fake_embedder(vectors),
        )
        result = ev.evaluate(
            question=question,
            answer=answer,
            contexts=[
                "Python is a widely used programming language.",
                "Bananas grow on trees in tropical climates.",
            ],
            ground_truth="Python is a programming language.",
        )

        assert isinstance(result, AnswerEvalResult)
        assert result.faithfulness == 1.0
        assert result.answer_relevancy > 0.9
        assert result.context_precision == pytest.approx(1.0)
        assert len(result.claim_judgements) == 2
        assert len(result.context_judgements) == 2
        assert len(result.reverse_questions) == 3
        # Exactly 4 LLM calls for the four stages.
        assert len(llm.calls) == 4

    def test_ragas_score_zero_when_any_metric_zero(self):
        result = AnswerEvalResult(
            question="q",
            answer="a",
            ground_truth="gt",
            faithfulness=1.0,
            answer_relevancy=0.9,
            context_precision=0.0,
        )
        assert result.ragas_score == 0.0

    def test_ragas_score_harmonic_mean(self):
        result = AnswerEvalResult(
            question="q",
            answer="a",
            ground_truth="gt",
            faithfulness=0.5,
            answer_relevancy=0.5,
            context_precision=0.5,
        )
        # Harmonic mean of three equal values is just the value itself.
        assert result.ragas_score == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------


class TestAggregate:
    def test_aggregate_empty(self):
        ev = AnswerEvaluator(_ScriptedLLM([]), embedding_service=_fake_embedder({}))
        report = ev.aggregate([])
        assert isinstance(report, AnswerEvalReport)
        assert report.num_queries == 0
        assert report.mean_faithfulness == 0.0
        assert report.mean_answer_relevancy == 0.0
        assert report.mean_context_precision == 0.0
        assert report.mean_ragas_score == 0.0

    def test_aggregate_multiple_results(self):
        r1 = AnswerEvalResult(
            question="q1", answer="a1", ground_truth="gt1",
            faithfulness=1.0, answer_relevancy=0.8, context_precision=0.6,
        )
        r2 = AnswerEvalResult(
            question="q2", answer="a2", ground_truth="gt2",
            faithfulness=0.5, answer_relevancy=0.4, context_precision=0.2,
        )
        ev = AnswerEvaluator(_ScriptedLLM([]), embedding_service=_fake_embedder({}))
        report = ev.aggregate([r1, r2])
        assert report.num_queries == 2
        assert report.mean_faithfulness == pytest.approx(0.75)
        assert report.mean_answer_relevancy == pytest.approx(0.6)
        assert report.mean_context_precision == pytest.approx(0.4)

    def test_aggregate_to_dict_has_expected_keys(self):
        r = AnswerEvalResult(
            question="q", answer="a", ground_truth="gt",
            faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0,
        )
        ev = AnswerEvaluator(_ScriptedLLM([]), embedding_service=_fake_embedder({}))
        data = ev.aggregate([r]).to_dict()
        for key in (
            "num_queries",
            "mean_faithfulness",
            "mean_answer_relevancy",
            "mean_context_precision",
            "mean_ragas_score",
            "results",
        ):
            assert key in data
        assert len(data["results"]) == 1

    def test_summary_format(self):
        ev = AnswerEvaluator(_ScriptedLLM([]), embedding_service=_fake_embedder({}))
        report = ev.aggregate([
            AnswerEvalResult(
                question="q", answer="a", ground_truth="gt",
                faithfulness=0.5, answer_relevancy=0.5,
                context_precision=0.5,
            )
        ])
        summary = report.summary()
        assert "Faith=" in summary
        assert "Relev=" in summary
        assert "CtxP=" in summary
        assert "RAGAS=" in summary


# ---------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------


class TestSerialisation:
    def test_claim_judgement_to_dict(self):
        j = ClaimJudgement(claim="c", supported=True)
        assert j.to_dict() == {"claim": "c", "supported": True}

    def test_context_judgement_to_dict(self):
        j = ContextJudgement(rank=2, content_preview="abc", relevant=False)
        assert j.to_dict() == {
            "rank": 2,
            "content_preview": "abc",
            "relevant": False,
        }

    def test_result_to_dict_has_expected_keys(self):
        result = AnswerEvalResult(
            question="q", answer="a", ground_truth="gt",
            faithfulness=0.5, answer_relevancy=0.5,
            context_precision=0.5,
        )
        data = result.to_dict()
        for key in (
            "question",
            "answer",
            "ground_truth",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "ragas_score",
            "claim_judgements",
            "context_judgements",
            "reverse_questions",
        ):
            assert key in data