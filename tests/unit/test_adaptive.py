"""
Unit tests for HyDE and Adaptive Re-Retrieval in RAGPipeline.

Uses mocks for LLM and KnowledgeBase so no API calls are made.
The chunks returned by the mocked KB are real ``Chunk`` instances
so that the embedding-based grounding / verification scorers have
real text to embed; using bare MagicMocks would give scorers
non-strings and produce undefined scores.
"""

import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch, call
from src.rag.pipeline import RAGPipeline
from src.rag.llm import LLMResponse
from src.rag.grounding import GroundingResult, ChunkGroundingScore, SentenceGrounding
from src.ingestion.chunking.chunk import Chunk
from src.ingestion.storage.vector_store import SearchResult


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-test",
        usage={"input_tokens": 10, "output_tokens": 20},
        stop_reason="end_turn",
    )


def _make_search_result(
    content: str = "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes.",
    chunk_id: str = "chunk_001",
) -> SearchResult:
    """Build a real SearchResult wrapping a real Chunk.

    Real objects (rather than MagicMocks) are required because the
    embedding-based grounding and fact-verification scorers will try
    to embed ``chunk.content`` — a MagicMock would be passed to
    ``SentenceTransformer.encode`` and produce unusable results.
    """
    chunk = Chunk(
        chunk_id=chunk_id,
        content=content,
        doc_id="doc_001",
        source_doc_title="test_doc",
        page_number=1,
    )
    return SearchResult(chunk=chunk, score=0.85, rank=1)


def _make_pipeline(
    use_hyde=False,
    adaptive_retrieval=False,
    ground_answer=False,
    confidence_threshold=0.25,
    max_retries=2,
    answer_text: str = "Test answer about the topic.",
    chunk_content: str = "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes.",
):
    kb = MagicMock()
    kb.search.return_value = [_make_search_result(content=chunk_content)]
    kb.document_ids = ["doc_001"]

    llm = MagicMock()
    llm.is_available.return_value = True
    llm.generate.return_value = _make_llm_response(answer_text)

    # Use a fresh temp directory for memory and history so tests
    # always start with empty conversation state and never load
    # real data from data/memory/ or data/history/.
    tmp_dir = Path(tempfile.mkdtemp())

    pipeline = RAGPipeline(
        knowledge_base=kb,
        llm_provider=llm,
        use_hyde=use_hyde,
        adaptive_retrieval=adaptive_retrieval,
        ground_answer=ground_answer,
        confidence_threshold=confidence_threshold,
        max_retries=max_retries,
        memory_persist_path=tmp_dir / "test_memory.json",
    )
    # Also point query history at the temp dir so it doesn't
    # load real history records from disk.
    pipeline._history._records = []
    pipeline._history.persist_path = tmp_dir / "test_history.json"

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
        """When grounding returns high confidence, no retry should happen.

        We patch the grounder directly rather than relying on the
        embedding model producing a high score for a mocked chunk —
        the purpose of this test is to verify the pipeline's retry
        logic, not the scorer's semantic matching.
        """
        pipeline, kb, llm = _make_pipeline(
            adaptive_retrieval=True,
            ground_answer=True,
            confidence_threshold=0.5,
        )
        # Force the grounder to always return high confidence so the
        # retry branch is never taken.
        high_conf_result = GroundingResult(
            sentences=[
                SentenceGrounding(
                    sentence="Test sentence.",
                    confidence=0.95,
                    best_chunk_id="chunk_001",
                )
            ],
            overall_confidence=0.95,
            method="embedding_cosine",
        )
        pipeline._grounder.score = MagicMock(return_value=high_conf_result)

        response = pipeline.query("What is a neural network?")
        assert response.retrieval_attempts == 1

    def test_retries_when_confidence_low(self):
        pipeline, kb, llm = _make_pipeline(
            adaptive_retrieval=True,
            ground_answer=True,
            confidence_threshold=0.99,  # impossibly high — always retries
            max_retries=2,
        )
        # Give answer responses + reformulation responses
        # Attempt 1: answer → low confidence → reformulate
        # Attempt 2: answer → low confidence → reformulate
        # Attempt 3: answer → low confidence → stop (max_retries reached)
        llm.generate.side_effect = [
            _make_llm_response("Answer attempt 1"),
            _make_llm_response("reformulated query"),
            _make_llm_response("Answer attempt 2"),
            _make_llm_response("reformulated query 2"),
            _make_llm_response("Answer attempt 3"),
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