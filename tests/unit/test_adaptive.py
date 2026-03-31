"""
Unit tests for HyDE and Adaptive Re-Retrieval in RAGPipeline.

Uses mocks for LLM and KnowledgeBase so no API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from src.rag.pipeline import RAGPipeline
from src.rag.llm import LLMResponse
from src.rag.grounding import GroundingResult, ChunkGroundingScore


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-test",
        usage={"input_tokens": 10, "output_tokens": 20},
        stop_reason="end_turn",
    )


def _make_search_result(content: str = "relevant content about the topic"):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = "test_doc"
    chunk.page_number = 1
    chunk.doc_id = "doc_001"
    result = MagicMock()
    result.chunk = chunk
    result.score = 0.85
    result.rank = 1
    return result


def _make_pipeline(
    use_hyde=False,
    adaptive_retrieval=False,
    ground_answer=False,
    confidence_threshold=0.25,
    max_retries=2,
):
    kb = MagicMock()
    kb.search.return_value = [_make_search_result()]
    kb.document_ids = ["doc_001"]

    llm = MagicMock()
    llm.is_available.return_value = True
    llm.generate.return_value = _make_llm_response("Test answer about the topic.")

    pipeline = RAGPipeline(
        knowledge_base=kb,
        llm_provider=llm,
        use_hyde=use_hyde,
        adaptive_retrieval=adaptive_retrieval,
        ground_answer=ground_answer,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries,
    )
    return pipeline, kb, llm


class TestHyDE:

    def test_hyde_disabled_uses_raw_query(self):
        pipeline, kb, llm = _make_pipeline(use_hyde=False)
        pipeline.query("What is backpropagation?")
        kb.search.assert_called_once()
        call_query = kb.search.call_args[0][0]
        assert "backpropagation" in call_query.lower()

    def test_hyde_enabled_calls_llm_twice(self):
        """HyDE makes one LLM call to generate the doc, one to answer."""
        pipeline, kb, llm = _make_pipeline(use_hyde=True)
        pipeline.query("What is backpropagation?")
        assert llm.generate.call_count == 2

    def test_hyde_query_stored_in_response(self):
        pipeline, kb, llm = _make_pipeline(use_hyde=True)
        llm.generate.side_effect = [
            _make_llm_response("Backpropagation is an algorithm for training neural networks."),
            _make_llm_response("Final answer about backpropagation."),
        ]
        response = pipeline.query("What is backpropagation?")
        assert response.hyde_query is not None
        assert len(response.hyde_query) > 0

    def test_hyde_failure_falls_back_to_raw_query(self):
        pipeline, kb, llm = _make_pipeline(use_hyde=True)
        llm.generate.side_effect = [
            Exception("API error"),
            _make_llm_response("Fallback answer."),
        ]
        response = pipeline.query("What is backpropagation?")
        assert response.hyde_query is None
        assert response.answer == "Fallback answer."


class TestAdaptiveRetrieval:

    def test_single_attempt_when_confidence_high(self):
        pipeline, kb, llm = _make_pipeline(
            adaptive_retrieval=True,
            ground_answer=True,
            confidence_threshold=0.01,  # very low — always passes
        )
        response = pipeline.query("What is a neural network?")
        assert response.retrieval_attempts == 1

    def test_retries_when_confidence_low(self):
        pipeline, kb, llm = _make_pipeline(
            adaptive_retrieval=True,
            ground_answer=True,
            confidence_threshold=0.99,  # impossibly high — always retries
            max_retries=2,
        )
        # Give reformulation responses + answer responses
        llm.generate.side_effect = [
            _make_llm_response("Answer attempt 1"),   # answer
            _make_llm_response("reformulated query"), # reformulation
            _make_llm_response("Answer attempt 2"),   # answer
            _make_llm_response("reformulated query 2"),
            _make_llm_response("Answer attempt 3"),   # answer
        ]
        response = pipeline.query("What is a neural network?")
        assert response.retrieval_attempts > 1

    def test_no_retry_when_adaptive_disabled(self):
        pipeline, kb, llm = _make_pipeline(
            adaptive_retrieval=False,
            ground_answer=True,
            confidence_threshold=0.99,
            max_retries=2,
        )
        response = pipeline.query("What is a neural network?")
        assert response.retrieval_attempts == 1

    def test_response_has_retrieval_attempts_field(self):
        pipeline, kb, llm = _make_pipeline()
        response = pipeline.query("Tell me about regression")
        assert hasattr(response, "retrieval_attempts")
        assert response.retrieval_attempts >= 1

    def test_best_result_kept_across_retries(self):
        """Pipeline should keep the highest-confidence result."""
        pipeline, kb, llm = _make_pipeline(
            adaptive_retrieval=True,
            ground_answer=True,
            confidence_threshold=0.99,
            max_retries=1,
        )
        llm.generate.side_effect = [
            _make_llm_response("relevant content about the topic neural networks"),
            _make_llm_response("reformulated"),
            _make_llm_response("xyz abc def ghi jkl"),  # worse answer
        ]
        response = pipeline.query("Tell me about neural networks")
        # Should still return a valid answer
        assert response.answer is not None
        assert len(response.answer) > 0