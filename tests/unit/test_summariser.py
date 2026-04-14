"""Unit tests for DocumentSummariser."""

import pytest
from unittest.mock import MagicMock
from src.study.summariser import (
    DocumentSummariser, DocumentSummary, SectionSummary, CorpusSummary
)
from src.rag.llm import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-test",
        usage={"input_tokens": 10, "output_tokens": 50},
        stop_reason="end_turn",
    )


def _make_chunk(content: str, source: str = "lecture_1", doc_id: str = "doc_001"):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    chunk.doc_id = doc_id
    return chunk


def _make_summariser(llm_responses=None, chunks=None):
    kb = MagicMock()
    kb.get_document_chunks.return_value = chunks or [
        _make_chunk("Neural networks are computational models inspired by the brain."),
        _make_chunk("They consist of layers of interconnected neurons."),
        _make_chunk("Backpropagation is used to train them by adjusting weights."),
    ]
    kb.document_ids = ["doc_001"]

    llm = MagicMock()
    llm.is_available.return_value = True
    if llm_responses:
        llm.generate.side_effect = [_make_llm_response(r) for r in llm_responses]
    else:
        llm.generate.return_value = _make_llm_response(
            "Neural networks are computational models that learn by adjusting weights."
        )

    return DocumentSummariser(knowledge_base=kb, llm_provider=llm, chunks_per_section=3)


class TestDocumentSummariser:

    def test_returns_document_summary(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        assert isinstance(result, DocumentSummary)

    def test_summary_has_correct_doc_id(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        assert result.doc_id == "doc_001"

    def test_summary_has_title(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        assert result.title == "lecture_1"

    def test_empty_chunks_returns_none(self):
        kb = MagicMock()
        kb.get_document_chunks.return_value = []
        llm = MagicMock()
        llm.is_available.return_value = True
        s = DocumentSummariser(knowledge_base=kb, llm_provider=llm)
        assert s.summarise_document("missing_doc") is None

    def test_has_executive_summary(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        assert isinstance(result.executive_summary, str)
        assert len(result.executive_summary) > 0

    def test_has_section_summaries(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        assert len(result.section_summaries) > 0

    def test_section_summary_has_required_fields(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        sec = result.section_summaries[0]
        assert hasattr(sec, "section_index")
        assert hasattr(sec, "chunk_count")
        assert hasattr(sec, "summary")
        assert hasattr(sec, "excerpt")

    def test_total_chunks_matches(self):
        chunks = [_make_chunk(f"Content {i}") for i in range(6)]
        s = _make_summariser(chunks=chunks)
        result = s.summarise_document("doc_001")
        assert result.total_chunks == 6

    def test_key_points_extracted(self):
        s = _make_summariser(llm_responses=[
            "Section summary.",
            "Executive summary of the document.",
            '["Key point one.", "Key point two.", "Key point three."]',
        ])
        result = s.summarise_document("doc_001")
        assert isinstance(result.key_points, list)

    def test_to_dict_has_required_keys(self):
        s = _make_summariser()
        result = s.summarise_document("doc_001")
        d = result.to_dict()
        assert "doc_id" in d
        assert "title" in d
        assert "executive_summary" in d
        assert "key_points" in d
        assert "section_summaries" in d
        assert "total_chunks" in d
        assert "total_sections" in d

    def test_llm_unavailable_falls_back(self):
        kb = MagicMock()
        kb.get_document_chunks.return_value = [
            _make_chunk("Neural networks are computational models.")
        ]
        llm = MagicMock()
        llm.is_available.return_value = False
        s = DocumentSummariser(knowledge_base=kb, llm_provider=llm)
        result = s.summarise_document("doc_001")
        assert result is not None
        assert len(result.executive_summary) > 0

    def test_corpus_summary_returns_correct_type(self):
        s = _make_summariser()
        result = s.summarise_corpus()
        assert isinstance(result, CorpusSummary)

    def test_corpus_summary_has_required_keys(self):
        s = _make_summariser()
        result = s.summarise_corpus()
        d = result.to_dict()
        assert "total_documents" in d
        assert "total_chunks" in d
        assert "corpus_overview" in d
        assert "document_summaries" in d

    def test_chunks_per_section_grouping(self):
        chunks = [_make_chunk(f"Content {i}") for i in range(10)]
        s = _make_summariser(chunks=chunks)
        s.chunks_per_section = 3
        sections = s._group_into_sections(chunks)
        assert len(sections) == 4  # ceil(10/3)
        assert len(sections[0]) == 3
        assert len(sections[-1]) == 1