"""Unit tests for GraphBuilder and GraphStore."""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.graph_store import GraphStore


def _make_chunk(content: str, source: str = "test_doc"):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    return chunk


def _make_builder_with_mock_nlp(entities_per_call):
    """Return a GraphBuilder whose NLP is mocked to return given entities."""
    builder = GraphBuilder()
    call_iter = iter(entities_per_call)

    def mock_nlp(text):
        doc = MagicMock()
        ents = []
        for text_val, label in next(call_iter, []):
            ent = MagicMock()
            ent.text = text_val
            ent.label_ = label
            ents.append(ent)
        doc.ents = ents
        return doc

    builder._nlp = mock_nlp
    return builder


class TestGraphBuilder:

    def test_empty_chunks_returns_empty_graph(self):
        builder = _make_builder_with_mock_nlp([])
        graph = builder.build([])
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_single_entity_creates_node(self):
        builder = _make_builder_with_mock_nlp([[("Python", "PRODUCT")]])
        chunk = _make_chunk("Python is a language.")
        graph = builder.build([chunk])
        assert "python" in graph.nodes

    def test_two_entities_in_same_chunk_creates_edge(self):
        builder = _make_builder_with_mock_nlp([
            [("Python", "PRODUCT"), ("Google", "ORG")]
        ])
        chunk = _make_chunk("Google uses Python.")
        graph = builder.build([chunk])
        assert graph.has_edge("python", "google")

    def test_co_occurrence_weight_increases(self):
        builder = _make_builder_with_mock_nlp([
            [("Python", "PRODUCT"), ("Google", "ORG")],
            [("Python", "PRODUCT"), ("Google", "ORG")],
        ])
        chunks = [
            _make_chunk("Google uses Python."),
            _make_chunk("Google also uses Python here."),
        ]
        graph = builder.build(chunks)
        assert graph[" python" if " python" in graph else "python"]["google"]["weight"] == 2 \
               or graph["python"]["google"]["weight"] == 2

    def test_node_count_increments_across_chunks(self):
        builder = _make_builder_with_mock_nlp([
            [("Python", "PRODUCT")],
            [("Python", "PRODUCT")],
        ])
        chunks = [_make_chunk("Python."), _make_chunk("Python again.")]
        graph = builder.build(chunks)
        assert graph.nodes["python"]["count"] == 2

    def test_irrelevant_label_filtered_out(self):
        builder = _make_builder_with_mock_nlp([
            [("2023", "DATE"), ("Python", "PRODUCT")]
        ])
        chunk = _make_chunk("In 2023 Python was popular.")
        graph = builder.build([chunk])
        assert "2023" not in graph.nodes
        assert "python" in graph.nodes

    def test_short_entity_filtered_out(self):
        builder = _make_builder_with_mock_nlp([
            [("AI", "ORG"), ("OpenAI", "ORG")]
        ])
        chunk = _make_chunk("AI and OpenAI.")
        graph = builder.build([chunk])
        assert "ai" not in graph.nodes  # too short
        assert "openai" in graph.nodes

    def test_source_stored_on_node(self):
        builder = _make_builder_with_mock_nlp([
            [("Python", "PRODUCT")]
        ])
        chunk = _make_chunk("Python.", source="lecture_1")
        graph = builder.build([chunk])
        assert "lecture_1" in graph.nodes["python"]["sources"]


class TestGraphStore:

    def test_save_and_load_roundtrip(self, tmp_path):
        import networkx as nx
        store = GraphStore(path=str(tmp_path / "test_graph.json"))
        graph = nx.Graph()
        graph.add_node("python", label="PRODUCT", count=3, sources=["doc1"])
        graph.add_edge("python", "google", weight=2, sources=["doc1"])

        store.save(graph)
        loaded = store.load()

        assert loaded is not None
        assert "python" in loaded.nodes
        assert loaded.has_edge("python", "google")

    def test_load_returns_none_when_no_file(self, tmp_path):
        store = GraphStore(path=str(tmp_path / "missing.json"))
        assert store.load() is None

    def test_exists_returns_false_before_save(self, tmp_path):
        store = GraphStore(path=str(tmp_path / "graph.json"))
        assert store.exists() is False

    def test_exists_returns_true_after_save(self, tmp_path):
        import networkx as nx
        store = GraphStore(path=str(tmp_path / "graph.json"))
        store.save(nx.Graph())
        assert store.exists() is True

    def test_delete_removes_file(self, tmp_path):
        import networkx as nx
        store = GraphStore(path=str(tmp_path / "graph.json"))
        store.save(nx.Graph())
        store.delete()
        assert not store.exists()

    def test_to_dict_structure(self, tmp_path):
        import networkx as nx
        store = GraphStore(path=str(tmp_path / "graph.json"))
        graph = nx.Graph()
        graph.add_node("python", label="PRODUCT", count=1, sources=["doc"])
        graph.add_node("google", label="ORG", count=1, sources=["doc"])
        graph.add_edge("python", "google", weight=1, sources=["doc"])

        d = store.to_dict(graph)
        assert "nodes" in d
        assert "links" in d
        assert d["num_nodes"] == 2
        assert d["num_edges"] == 1
        node_ids = [n["id"] for n in d["nodes"]]
        assert "python" in node_ids
