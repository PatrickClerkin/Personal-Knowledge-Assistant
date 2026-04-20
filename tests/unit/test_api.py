"""
Unit tests for Flask API endpoints.

Uses Flask's test client to exercise every API route without
needing a running server. LLM-dependent endpoints are tested
with mocked RAG pipelines.

Key design decision: we patch ``get_rag`` and ``get_kb`` as
*functions* rather than setting private variables, because the
real ``get_rag()`` checks ``ANTHROPIC_API_KEY`` and will silently
re-create a live pipeline if the env var is set.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.web.app import create_app
from src.web.blueprints import shared


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_shared():
    """Reset singletons before and after every test."""
    shared.reset_singletons()
    yield
    shared.reset_singletons()


@pytest.fixture
def app():
    """Create a test Flask app."""
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


def _mock_kb():
    """Build a mock KnowledgeBase."""
    kb = MagicMock()
    kb.size = 42
    kb.document_ids = ["doc_a", "doc_b"]
    kb.supported_formats = [".pdf", ".txt", ".md", ".docx"]
    kb.search.return_value = []
    kb.get_document_chunks.return_value = []
    kb.get_document_info.return_value = None
    return kb


def _mock_rag():
    """Build a mock RAG pipeline."""
    rag = MagicMock()
    rag.memory.turns = []
    rag.memory.is_empty.return_value = True
    rag.memory.total_turns = 0
    rag.history.total_queries = 0
    rag.history.get_recent.return_value = []
    rag.history.get_analytics.return_value = {
        "total_queries": 0,
        "cache_hit_rate": 0.0,
        "avg_confidence": 0.0,
        "avg_retrieval_attempts": 1.0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
        "total_tokens": 0,
        "multi_attempt_rate": 0.0,
        "avg_verification_score": None,
        "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
        "top_sources": [],
        "queries_by_day": [],
    }
    rag._cache = None
    return rag


# ─── Health & Info ──────────────────────────────────────────────────

class TestHealthEndpoint:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_returns_200(self, mock, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_returns_ok_status(self, mock, client):
        data = client.get("/api/health").get_json()
        assert data["status"] == "ok"

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_includes_system_info(self, mock, client):
        data = client.get("/api/health").get_json()
        assert "python_version" in data
        assert "index_size" in data
        assert "num_documents" in data

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_index_size_matches_kb(self, mock, client):
        data = client.get("/api/health").get_json()
        assert data["index_size"] == 42


class TestInfoEndpoint:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_returns_200(self, mock, client):
        response = client.get("/api/info")
        assert response.status_code == 200

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_returns_required_fields(self, mock, client):
        data = client.get("/api/info").get_json()
        assert "total_chunks" in data
        assert "num_documents" in data
        assert "supported_formats" in data
        assert "document_ids" in data

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_chunk_count_matches(self, mock, client):
        data = client.get("/api/info").get_json()
        assert data["total_chunks"] == 42


# ─── Search ─────────────────────────────────────────────────────────

class TestSearchEndpoint:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_missing_query_returns_400(self, mock, client):
        response = client.post(
            "/api/search",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_empty_query_returns_400(self, mock, client):
        response = client.post(
            "/api/search",
            json={"query": "  "},
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_valid_query_returns_200(self, mock, client):
        response = client.post(
            "/api/search",
            json={"query": "test query"},
            content_type="application/json",
        )
        assert response.status_code == 200

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_returns_results_list(self, mock, client):
        data = client.post(
            "/api/search",
            json={"query": "test"},
            content_type="application/json",
        ).get_json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_calls_kb_search(self, client):
        kb = _mock_kb()
        with patch.object(shared, "get_kb", return_value=kb):
            client.post(
                "/api/search",
                json={"query": "what is SOLID?"},
                content_type="application/json",
            )
            kb.search.assert_called_once()


# ─── Ingest ─────────────────────────────────────────────────────────

class TestIngestEndpoint:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_no_file_returns_400(self, mock, client):
        response = client.post("/api/ingest")
        assert response.status_code == 400
        assert "No file provided" in response.get_json()["error"]

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_unsupported_extension_returns_400(self, mock, client):
        from io import BytesIO
        data = {"file": (BytesIO(b"fake data"), "malware.exe")}
        response = client.post(
            "/api/ingest",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "Unsupported" in response.get_json()["error"]


# ─── Documents ──────────────────────────────────────────────────────

class TestDocumentsEndpoint:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_list_returns_200(self, mock, client):
        response = client.get("/api/documents")
        assert response.status_code == 200

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_list_returns_documents_list(self, mock, client):
        data = client.get("/api/documents").get_json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_delete_nonexistent_returns_404(self, client):
        kb = _mock_kb()
        kb.delete_document.return_value = 0
        with patch.object(shared, "get_kb", return_value=kb):
            response = client.delete("/api/documents/nonexistent")
            assert response.status_code == 404

    def test_delete_existing_returns_200(self, client):
        kb = _mock_kb()
        kb.delete_document.return_value = 5
        with patch.object(shared, "get_kb", return_value=kb):
            response = client.delete("/api/documents/doc_a")
            assert response.status_code == 200
            data = response.get_json()
            assert data["deleted_chunks"] == 5


# ─── Chat ───────────────────────────────────────────────────────────

class TestChatEndpoint:

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/chat",
            json={"question": "test"},
            content_type="application/json",
        )
        assert response.status_code == 503

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_missing_question_returns_400(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/chat",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_empty_question_returns_400(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/chat",
            json={"question": "   "},
            content_type="application/json",
        )
        assert response.status_code == 400


# ─── Conversation ──────────────────────────────────────────────────

class TestConversationEndpoint:

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_get_conversation_returns_200(self, mock_kb, mock_rag, client):
        response = client.get("/api/conversation")
        assert response.status_code == 200
        data = response.get_json()
        assert "turns" in data
        assert "total" in data

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_get_conversation_no_rag_returns_empty(self, mock_kb, mock_rag, client):
        data = client.get("/api/conversation").get_json()
        assert data["turns"] == []
        assert data["total"] == 0

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_clear_conversation_returns_200(self, mock_kb, mock_rag, client):
        response = client.post("/api/conversation/clear")
        assert response.status_code == 200
        assert response.get_json()["cleared"] is True


# ─── Evaluation ─────────────────────────────────────────────────────

class TestEvaluationEndpoint:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_missing_file_returns_404(self, mock, client):
        response = client.get(
            "/api/evaluation/run?path=data/eval/nonexistent_queries.json"
        )
        assert response.status_code == 404

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_path_traversal_returns_400(self, mock, client):
        response = client.get(
            "/api/evaluation/run?path=../../etc/passwd"
        )
        assert response.status_code == 400
        assert "Invalid path" in response.get_json()["error"]


# ─── Analytics ──────────────────────────────────────────────────────

class TestAnalyticsEndpoint:

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_history_no_rag_returns_empty(self, mock_kb, mock_rag, client):
        data = client.get("/api/history").get_json()
        assert data["records"] == []
        assert data["total"] == 0

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_history_returns_200(self, mock_kb, mock_rag, client):
        response = client.get("/api/history")
        assert response.status_code == 200

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_analytics_returns_200(self, mock_kb, mock_rag, client):
        response = client.get("/api/analytics")
        assert response.status_code == 200


# ─── Annotations ────────────────────────────────────────────────────

class TestAnnotationEndpoints:

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_add_missing_fields_returns_400(self, mock, client):
        response = client.post(
            "/api/annotations",
            json={"chunk_id": "c1"},
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_add_empty_note_returns_400(self, mock, client):
        response = client.post(
            "/api/annotations",
            json={
                "chunk_id": "c1",
                "doc_id": "d1",
                "source_title": "Test",
                "note": "   ",
            },
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_list_returns_200(self, mock, client):
        response = client.get("/api/annotations")
        assert response.status_code == 200
        data = response.get_json()
        assert "annotations" in data
        assert "total" in data

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_delete_nonexistent_returns_404(self, mock, client):
        response = client.delete("/api/annotations/nonexistent-id")
        assert response.status_code == 404

    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_stats_returns_200(self, mock, client):
        response = client.get("/api/annotations/stats")
        assert response.status_code == 200


# ─── Conflicts ──────────────────────────────────────────────────────

class TestConflictEndpoints:

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/conflicts/detect",
            json={"topic": "test"},
            content_type="application/json",
        )
        assert response.status_code == 503

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_missing_topic_returns_400(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/conflicts/detect",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_empty_topic_returns_400(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/conflicts/detect",
            json={"topic": "  "},
            content_type="application/json",
        )
        assert response.status_code == 400


# ─── Similarity ─────────────────────────────────────────────────────

class TestSimilarityEndpoints:

    def test_matrix_returns_200(self, client):
        kb = _mock_kb()
        kb.document_ids = []
        with patch.object(shared, "get_kb", return_value=kb):
            response = client.get("/api/similarity/matrix")
            assert response.status_code == 200


# ─── Study & Quiz ──────────────────────────────────────────────────

class TestStudyEndpoints:

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_study_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/study/generate",
            json={"topic": "test"},
            content_type="application/json",
        )
        assert response.status_code == 503

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_study_missing_topic_returns_400(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/study/generate",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_quiz_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/quiz/generate",
            json={"topic": "test"},
            content_type="application/json",
        )
        assert response.status_code == 503

    @patch.object(shared, "get_rag", return_value=_mock_rag())
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_quiz_missing_topic_returns_400(self, mock_kb, mock_rag, client):
        response = client.post(
            "/api/quiz/generate",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400


# ─── Summarise ──────────────────────────────────────────────────────

class TestSummariseEndpoints:

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_summarise_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.get("/api/summarise/doc_a")
        assert response.status_code == 503

    @patch.object(shared, "get_rag", return_value=None)
    @patch.object(shared, "get_kb", return_value=_mock_kb())
    def test_summarise_all_no_api_key_returns_503(self, mock_kb, mock_rag, client):
        response = client.get("/api/summarise/all")
        assert response.status_code == 503