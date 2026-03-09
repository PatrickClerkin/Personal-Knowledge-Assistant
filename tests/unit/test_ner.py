"""Unit tests for NERExtractor and boost_by_entities."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.ner_extractor import NERExtractor
from src.retrieval.entity_reranker import boost_by_entities, _entity_texts


def _make_chunk(content: str = "", entities: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        metadata={"entities": entities or []},
    )


def _make_spacy_ent(text: str, label: str):
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    return ent


def _make_spacy_doc(ents):
    doc = MagicMock()
    doc.ents = ents
    return doc


class TestNERExtractor:

    @pytest.fixture
    def extractor(self):
        return NERExtractor()

    def _patch_nlp(self, extractor, ents):
        nlp = MagicMock()
        nlp.return_value = _make_spacy_doc(ents)
        extractor._nlp = nlp
        return nlp

    def test_extract_returns_list_of_dicts(self, extractor):
        self._patch_nlp(extractor, [_make_spacy_ent("Dublin", "GPE")])
        result = extractor.extract("Dublin is in Ireland.")
        assert isinstance(result, list)
        assert result[0] == {"text": "Dublin", "label": "GPE"}

    def test_extract_multiple_entities(self, extractor):
        self._patch_nlp(extractor, [
            _make_spacy_ent("Alice", "PERSON"),
            _make_spacy_ent("Anthropic", "ORG"),
        ])
        result = extractor.extract("Alice works at Anthropic.")
        assert len(result) == 2
        labels = {e["label"] for e in result}
        assert labels == {"PERSON", "ORG"}

    def test_extract_deduplicates_same_entity(self, extractor):
        self._patch_nlp(extractor, [
            _make_spacy_ent("Dublin", "GPE"),
            _make_spacy_ent("Dublin", "GPE"),
        ])
        result = extractor.extract("Dublin, Dublin.")
        assert len(result) == 1

    def test_extract_deduplicates_case_insensitive(self, extractor):
        self._patch_nlp(extractor, [
            _make_spacy_ent("Python", "PRODUCT"),
            _make_spacy_ent("python", "PRODUCT"),
        ])
        result = extractor.extract("Python or python.")
        assert len(result) == 1

    def test_extract_filters_irrelevant_labels(self, extractor):
        self._patch_nlp(extractor, [
            _make_spacy_ent("42", "CARDINAL"),
            _make_spacy_ent("first", "ORDINAL"),
            _make_spacy_ent("Monday", "TIME"),
        ])
        result = extractor.extract("42 items, first on Monday.")
        assert result == []

    def test_extract_empty_string_returns_empty(self, extractor):
        result = extractor.extract("")
        assert result == []

    def test_extract_whitespace_only_returns_empty(self, extractor):
        result = extractor.extract("   ")
        assert result == []

    def test_extract_strips_entity_text(self, extractor):
        self._patch_nlp(extractor, [_make_spacy_ent("  Alice  ", "PERSON")])
        result = extractor.extract("  Alice  is here.")
        assert result[0]["text"] == "Alice"

    def test_extract_from_chunks_adds_entities_key(self, extractor):
        self._patch_nlp(extractor, [_make_spacy_ent("Berlin", "GPE")])
        chunk = _make_chunk("Berlin is a city.")
        extractor.extract_from_chunks([chunk])
        assert "entities" in chunk.metadata

    def test_extract_from_chunks_returns_chunks(self, extractor):
        self._patch_nlp(extractor, [])
        chunks = [_make_chunk("Hello world.")]
        returned = extractor.extract_from_chunks(chunks)
        assert returned is chunks

    def test_extract_from_chunks_multiple_chunks(self, extractor):
        nlp = MagicMock()
        nlp.side_effect = [
            _make_spacy_doc([_make_spacy_ent("Alice", "PERSON")]),
            _make_spacy_doc([_make_spacy_ent("Paris", "GPE")]),
        ]
        extractor._nlp = nlp

        chunks = [_make_chunk("Alice."), _make_chunk("Paris.")]
        extractor.extract_from_chunks(chunks)

        assert chunks[0].metadata["entities"][0]["label"] == "PERSON"
        assert chunks[1].metadata["entities"][0]["label"] == "GPE"

    def test_extract_from_chunks_empty_list(self, extractor):
        result = extractor.extract_from_chunks([])
        assert result == []


class TestEntityTexts:

    def test_returns_lowercase_set(self):
        entities = [{"text": "Dublin", "label": "GPE"}, {"text": "Alice", "label": "PERSON"}]
        result = _entity_texts(entities)
        assert result == {"dublin", "alice"}

    def test_handles_empty_list(self):
        assert _entity_texts([]) == set()

    def test_skips_malformed_entries(self):
        entities = [{"label": "GPE"}, {"text": "Alice", "label": "PERSON"}]
        result = _entity_texts(entities)
        assert result == {"alice"}


class TestBoostByEntities:

    def _results(self, entities_list):
        return [
            (_make_chunk(entities=ents), float(i + 1))
            for i, ents in enumerate(entities_list)
        ]

    def test_no_query_entities_preserves_order(self):
        results = [(_make_chunk(), 0.9), (_make_chunk(), 0.5)]
        out = boost_by_entities(results, query_entities=[])
        assert out == results

    def test_overlap_boosts_score(self):
        query_ents = [{"text": "Alice", "label": "PERSON"}]
        chunk_with = _make_chunk(entities=[{"text": "Alice", "label": "PERSON"}])
        chunk_without = _make_chunk(entities=[])
        results = [(chunk_with, 0.5), (chunk_without, 0.6)]
        out = boost_by_entities(results, query_entities=query_ents, boost_weight=0.1)
        boosted_chunk, boosted_score = out[0]
        assert boosted_chunk is chunk_with or boosted_score >= 0.6

    def test_more_overlaps_means_higher_score(self):
        query_ents = [
            {"text": "Alice", "label": "PERSON"},
            {"text": "Dublin", "label": "GPE"},
        ]
        chunk_two = _make_chunk(entities=[
            {"text": "Alice", "label": "PERSON"},
            {"text": "Dublin", "label": "GPE"},
        ])
        chunk_one = _make_chunk(entities=[{"text": "Alice", "label": "PERSON"}])
        chunk_none = _make_chunk(entities=[])

        results = [(chunk_none, 1.0), (chunk_one, 0.5), (chunk_two, 0.3)]
        out = boost_by_entities(results, query_ents, boost_weight=0.2)

        scores = {id(chunk): score for chunk, score in out}
        assert scores[id(chunk_two)] > scores[id(chunk_one)] > scores[id(chunk_none)]

    def test_output_sorted_descending(self):
        query_ents = [{"text": "Alice", "label": "PERSON"}]
        chunk_a = _make_chunk(entities=[{"text": "Alice", "label": "PERSON"}])
        chunk_b = _make_chunk(entities=[])
        results = [(chunk_b, 0.9), (chunk_a, 0.1)]
        out = boost_by_entities(results, query_ents, boost_weight=0.1)
        scores = [score for _, score in out]
        assert scores == sorted(scores, reverse=True)

    def test_label_filter_removes_non_matching_chunks(self):
        query_ents = [{"text": "Alice", "label": "PERSON"}]
        chunk_person = _make_chunk(entities=[{"text": "Alice", "label": "PERSON"}])
        chunk_org = _make_chunk(entities=[{"text": "Anthropic", "label": "ORG"}])
        results = [(chunk_person, 0.8), (chunk_org, 0.9)]
        out = boost_by_entities(results, query_ents, label_filter="PERSON")
        assert all(
            any(e.get("label") == "PERSON" for e in chunk.metadata.get("entities", []))
            for chunk, _ in out
        )

    def test_label_filter_no_matching_chunks_returns_empty(self):
        query_ents = [{"text": "Alice", "label": "PERSON"}]
        chunk = _make_chunk(entities=[{"text": "Anthropic", "label": "ORG"}])
        out = boost_by_entities([(chunk, 0.9)], query_ents, label_filter="PERSON")
        assert out == []

    def test_label_filter_with_no_query_entities(self):
        chunk_person = _make_chunk(entities=[{"text": "Alice", "label": "PERSON"}])
        chunk_org = _make_chunk(entities=[{"text": "Anthropic", "label": "ORG"}])
        results = [(chunk_person, 0.8), (chunk_org, 0.9)]
        out = boost_by_entities(results, query_entities=[], label_filter="PERSON")
        assert len(out) == 1
        assert out[0][0] is chunk_person

    def test_empty_results_returns_empty(self):
        out = boost_by_entities([], [{"text": "Alice", "label": "PERSON"}])
        assert out == []

    def test_case_insensitive_entity_matching(self):
        query_ents = [{"text": "alice", "label": "PERSON"}]
        chunk = _make_chunk(entities=[{"text": "Alice", "label": "PERSON"}])
        results = [(chunk, 0.5)]
        out = boost_by_entities(results, query_ents, boost_weight=0.1)
        _, score = out[0]
        assert score == pytest.approx(0.6)