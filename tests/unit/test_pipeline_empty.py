"""
Unit tests for the RAG pipeline's empty-retrieval behaviour.

If the knowledge base returns zero results on the first retrieval
attempt the pipeline used to crash with AttributeError because
``best_llm_response`` was still None when the code downstream tried
to access ``.content``.  These tests lock in the graceful fallback.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.rag.pipeline import RAGPipeline, RAGResponse


def _make_pipeline(kb_search_returns):
    """Build a pipeline whose KB returns whatever the caller wants."""
    kb = MagicMock()
    kb.search.return_value = kb_search_returns
    kb.document_ids = []

    llm = MagicMock()
    llm.is_available.return_value = True

    tmp_dir = Path(tempfile.mkdtemp())
    pipeline = RAGPipeline(
        knowledge_base=kb,
        llm_provider=llm,
        use_hyde=False,
        adaptive_retrieval=False,
        ground_answer=False,
        verify_facts=False,
        use_cache=False,
        memory_persist_path=tmp_dir / "test_memory.json",
    )
    pipeline._history._records = []
    pipeline._history.persist_path = tmp_dir / "test_history.json"
    return pipeline, kb, llm


class TestEmptyRetrievalResults:

    def test_empty_results_does_not_crash(self):
        """An empty KB must not raise AttributeError."""
        pipeline, kb, llm = _make_pipeline(kb_search_returns=[])
        # This used to throw AttributeError accessing best_llm_response.content.
        response = pipeline.query("anything")
        assert isinstance(response, RAGResponse)

    def test_empty_results_returns_sensible_message(self):
        pipeline, kb, llm = _make_pipeline(kb_search_returns=[])
        response = pipeline.query("anything")
        assert response.sources == []
        assert response.confidence == 0.0
        assert "couldn't find" in response.answer.lower() or \
               "no relevant" in response.answer.lower() or \
               "couldn’t find" in response.answer.lower()

    def test_empty_results_does_not_call_llm(self):
        """No retrieval -> no generation, we save the tokens."""
        pipeline, kb, llm = _make_pipeline(kb_search_returns=[])
        pipeline.query("anything")
        llm.generate.assert_not_called()

    def test_empty_results_are_still_recorded_in_memory(self):
        """The conversation turn should still be logged for context."""
        pipeline, kb, llm = _make_pipeline(kb_search_returns=[])
        assert pipeline.memory.total_turns == 0
        pipeline.query("anything")
        assert pipeline.memory.total_turns == 1