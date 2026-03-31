"""Unit tests for PathGenerator and StudyPath."""

import pytest
from unittest.mock import MagicMock, patch
from src.study.path_generator import PathGenerator, StudyPath, StudySection
from src.rag.llm import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-test",
        usage={"input_tokens": 10, "output_tokens": 20},
        stop_reason="end_turn",
    )


def _make_chunk(content: str, source: str = "test_doc", page: int = 1):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    chunk.page_number = page
    return chunk


def _make_search_result(content: str, source: str = "test_doc", page: int = 1):
    result = MagicMock()
    result.chunk = _make_chunk(content, source, page)
    result.score = 0.85
    return result


def _make_generator(llm_responses=None, search_results=None):
    kb = MagicMock()
    kb.search.return_value = search_results or [
        _make_search_result(
            "Neural networks are computational models inspired by the brain. "
            "They consist of layers of neurons that process information.",
            source="lecture_1",
            page=1,
        )
    ]

    llm = MagicMock()
    llm.is_available.return_value = True
    if llm_responses:
        llm.generate.side_effect = [_make_llm_response(r) for r in llm_responses]
    else:
        llm.generate.return_value = _make_llm_response(
            '["introduction", "core concepts", "applications"]'
        )

    return PathGenerator(
        knowledge_base=kb,
        llm_provider=llm,
        graph=None,
        top_k=3,
        max_concepts=4,
    )


class TestPathGenerator:

    def test_generate_returns_study_path(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        assert isinstance(path, StudyPath)

    def test_path_has_correct_topic(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        assert path.topic == "neural networks"

    def test_path_has_sections(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        assert len(path.sections) > 0

    def test_sections_have_required_fields(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        section = path.sections[0]
        assert hasattr(section, "concept")
        assert hasattr(section, "summary")
        assert hasattr(section, "key_terms")
        assert hasattr(section, "sources")
        assert hasattr(section, "chunk_previews")
        assert hasattr(section, "order")

    def test_sections_ordered_sequentially(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        orders = [s.order for s in path.sections]
        assert orders == list(range(1, len(orders) + 1))

    def test_estimated_minutes_positive(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        assert path.estimated_minutes > 0

    def test_to_dict_has_required_keys(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        d = path.to_dict()
        assert "topic" in d
        assert "sections" in d
        assert "estimated_minutes" in d
        assert "total_sources" in d

    def test_llm_unavailable_falls_back_gracefully(self):
        kb = MagicMock()
        kb.search.return_value = [
            _make_search_result("content about machine learning")
        ]
        llm = MagicMock()
        llm.is_available.return_value = False

        gen = PathGenerator(knowledge_base=kb, llm_provider=llm)
        path = gen.generate("machine learning")
        assert isinstance(path, StudyPath)
        assert len(path.sections) > 0

    def test_empty_kb_returns_empty_sections(self):
        kb = MagicMock()
        kb.search.return_value = []
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = _make_llm_response('["intro", "theory"]')

        gen = PathGenerator(knowledge_base=kb, llm_provider=llm)
        path = gen.generate("quantum computing")
        assert path.sections == []

    def test_max_concepts_respected(self):
        gen = _make_generator(
            llm_responses=[
                '["a", "b", "c", "d", "e", "f", "g", "h"]',
                "Summary a.", "Summary b.", "Summary c.", "Summary d.",
            ]
        )
        gen.max_concepts = 3
        path = gen.generate("neural networks")
        assert len(path.sections) <= 3

    def test_key_terms_are_strings(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        for section in path.sections:
            assert all(isinstance(t, str) for t in section.key_terms)

    def test_sources_are_strings(self):
        gen = _make_generator()
        path = gen.generate("neural networks")
        for section in path.sections:
            assert all(isinstance(s, str) for s in section.sources)

    def test_graph_neighbour_expansion(self):
        import networkx as nx
        graph = nx.Graph()
        graph.add_node("neural networks", label="PRODUCT", count=5, sources=[])
        graph.add_node("backpropagation", label="PRODUCT", count=3, sources=[])
        graph.add_edge("neural networks", "backpropagation", weight=2, sources=[])

        kb = MagicMock()
        kb.search.return_value = [
            _make_search_result("neural networks use backpropagation to learn")
        ]
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = _make_llm_response(
            '["neural networks", "backpropagation"]'
        )

        gen = PathGenerator(knowledge_base=kb, llm_provider=llm, graph=graph)
        concepts = gen._discover_concepts("neural networks")
        assert "backpropagation" in concepts