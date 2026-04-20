"""
Unit tests for the /api/evaluation/answer endpoint.

Uses Flask's test client and mocks the RAG pipeline + AnswerEvaluator
so no real API calls are made. Test sets are written into tmp_path
inside data/eval/ so the endpoint's path-traversal protection allows
them through.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.web.app import create_app
from src.web.blueprints import shared
from src.evaluation.answer_eval import AnswerEvalResult


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_shared():
    shared.reset_singletons()
    yield
    shared.reset_singletons()


@pytest.fixture
def app():
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def eval_dir(tmp_path, monkeypatch):
    """Create a tmp data/eval/ directory and rebind _EVAL_BASE_DIR to it."""
    d = tmp_path / "data" / "eval"
    d.mkdir(parents=True)
    # Re-bind the resolved base dir so the endpoint accepts our tmp path
    from src.web.blueprints import intelligence
    monkeypatch.setattr(intelligence, "_EVAL_BASE_DIR", d.resolve())
    return d


def _mock_kb():
    kb = MagicMock()
    kb.size = 42
    kb._embedder = MagicMock()
    return kb


def _mock_rag_with_answer():
    """RAG pipeline that returns a plausible response for each query."""
    rag = MagicMock()
    rag.top_k = 8

    def fake_query(question, top_k=None):
        response = MagicMock()
        response.answer = f"The answer to '{question}' is placeholder."
        source = MagicMock()
        source.chunk.content = "Some retrieved context."
        response.sources = [source]
        return response

    rag.query.side_effect = fake_query
    rag.llm = MagicMock()
    return rag


def _mock_eval_result(question: str, score: float = 0.8) -> AnswerEvalResult:
    """Build a realistic AnswerEvalResult without hitting the real evaluator."""
    return AnswerEvalResult(
        question=question,
        answer=f"A: {question}",
        ground_truth=f"GT: {question}",
        faithfulness=score,
        answer_relevancy=score,
        context_precision=score,
    )


# ─── Tests ──────────────────────────────────────────────────────────


class TestAnswerEvaluationEndpoint:

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.get("/api/evaluation/answer")
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.get_json()["error"]

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_path_traversal_returns_400(self, mock_kb, mock_rag, eval_dir, client):
        response = client.get(
            "/api/evaluation/answer?path=../../etc/passwd"
        )
        assert response.status_code == 400
        assert "Invalid path" in response.get_json()["error"]

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_missing_file_returns_404(self, mock_kb, mock_rag, eval_dir, client):
        target = eval_dir / "does_not_exist.jsonl"
        response = client.get(
            f"/api/evaluation/answer?path={target}"
        )
        assert response.status_code == 404

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_invalid_json_returns_400(self, mock_kb, mock_rag, eval_dir, client):
        f = eval_dir / "bad.jsonl"
        f.write_text("this is not json\n", encoding="utf-8")
        response = client.get(f"/api/evaluation/answer?path={f}")
        assert response.status_code == 400
        body = response.get_json()
        assert "Invalid JSON" in body["error"]

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_missing_fields_returns_400(self, mock_kb, mock_rag, eval_dir, client):
        f = eval_dir / "missing_fields.jsonl"
        f.write_text(
            json.dumps({"question": "Q"}) + "\n",   # no ground_truth
            encoding="utf-8",
        )
        response = client.get(f"/api/evaluation/answer?path={f}")
        assert response.status_code == 400
        assert "ground_truth" in response.get_json()["error"]

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_empty_test_set_returns_400(self, mock_kb, mock_rag, eval_dir, client):
        f = eval_dir / "empty.jsonl"
        f.write_text("\n\n   \n", encoding="utf-8")   # only whitespace
        response = client.get(f"/api/evaluation/answer?path={f}")
        assert response.status_code == 400
        assert "empty" in response.get_json()["error"].lower()

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_invalid_top_k_returns_400(self, mock_kb, mock_rag, eval_dir, client):
        f = eval_dir / "one.jsonl"
        f.write_text(
            json.dumps({"question": "Q", "ground_truth": "GT"}) + "\n",
            encoding="utf-8",
        )
        response = client.get(f"/api/evaluation/answer?path={f}&top_k=banana")
        assert response.status_code == 400

        response = client.get(f"/api/evaluation/answer?path={f}&top_k=0")
        assert response.status_code == 400

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_happy_path_returns_expected_structure(
        self, mock_kb, mock_rag, eval_dir, client,
    ):
        f = eval_dir / "good.jsonl"
        f.write_text(
            json.dumps({"question": "Q1", "ground_truth": "GT1"}) + "\n" +
            json.dumps({"question": "Q2", "ground_truth": "GT2"}) + "\n",
            encoding="utf-8",
        )

        # Patch the evaluator so we don't make real Claude calls.
        with patch(
            "src.web.blueprints.intelligence.AnswerEvaluator"
        ) as EvalCls:
            instance = MagicMock()
            instance.evaluate.side_effect = [
                _mock_eval_result("Q1", 0.9),
                _mock_eval_result("Q2", 0.7),
            ]
            # Aggregate needs to return a real report-like object; cheat
            # by letting the real aggregate run on our mock results.
            from src.evaluation.answer_eval import AnswerEvaluator
            real_aggregator = AnswerEvaluator.aggregate
            instance.aggregate.side_effect = (
                lambda results: real_aggregator(instance, results)
            )
            EvalCls.return_value = instance

            response = client.get(f"/api/evaluation/answer?path={f}")

        assert response.status_code == 200
        body = response.get_json()
        for key in (
            "num_queries",
            "mean_faithfulness",
            "mean_answer_relevancy",
            "mean_context_precision",
            "mean_ragas_score",
            "results",
            "errors",
            "test_set",
        ):
            assert key in body, f"Missing key: {key}"
        assert body["num_queries"] == 2
        assert body["test_set"] == "good.jsonl"
        assert body["errors"] == []
        assert len(body["results"]) == 2
        assert body["mean_faithfulness"] == pytest.approx(0.8, abs=1e-6)

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_question_errors_recorded_but_run_continues(
        self, mock_kb, mock_rag, eval_dir, client,
    ):
        f = eval_dir / "mixed.jsonl"
        f.write_text(
            json.dumps({"question": "good", "ground_truth": "gt"}) + "\n" +
            json.dumps({"question": "boom", "ground_truth": "gt"}) + "\n",
            encoding="utf-8",
        )

        with patch(
            "src.web.blueprints.intelligence.AnswerEvaluator"
        ) as EvalCls:
            instance = MagicMock()
            instance.evaluate.side_effect = [
                _mock_eval_result("good", 0.9),
                RuntimeError("API flaked"),
            ]
            from src.evaluation.answer_eval import AnswerEvaluator
            real_aggregator = AnswerEvaluator.aggregate
            instance.aggregate.side_effect = (
                lambda results: real_aggregator(instance, results)
            )
            EvalCls.return_value = instance

            response = client.get(f"/api/evaluation/answer?path={f}")

        assert response.status_code == 200
        body = response.get_json()
        assert body["num_queries"] == 1        # only 'good' succeeded
        assert len(body["errors"]) == 1
        assert body["errors"][0]["question"] == "boom"
        assert "API flaked" in body["errors"][0]["error"]

    @patch.object(shared, "get_rag", return_value=_mock_rag_with_answer())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_all_questions_fail_returns_500(
        self, mock_kb, mock_rag, eval_dir, client,
    ):
        f = eval_dir / "all_broken.jsonl"
        f.write_text(
            json.dumps({"question": "Q", "ground_truth": "GT"}) + "\n",
            encoding="utf-8",
        )

        with patch(
            "src.web.blueprints.intelligence.AnswerEvaluator"
        ) as EvalCls:
            instance = MagicMock()
            instance.evaluate.side_effect = RuntimeError("totally broken")
            EvalCls.return_value = instance

            response = client.get(f"/api/evaluation/answer?path={f}")

        assert response.status_code == 500
        body = response.get_json()
        assert "No test cases" in body["error"]
        assert len(body["errors"]) == 1