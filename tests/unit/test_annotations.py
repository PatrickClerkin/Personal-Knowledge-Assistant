"""Unit tests for AnnotationStore."""

import pytest
from src.rag.annotations import AnnotationStore, Annotation


def _make_store(tmp_path) -> AnnotationStore:
    return AnnotationStore(
        persist_path=str(tmp_path / "annotations.json"),
        max_annotations=100,
    )


def _add_sample(store: AnnotationStore, **kwargs) -> Annotation:
    defaults = {
        "chunk_id": "chunk_001",
        "doc_id": "doc_001",
        "source_title": "lecture_notes",
        "note": "This is an important concept.",
        "chunk_preview": "Neural networks are computational models.",
        "page_number": 3,
        "tags": ["important", "ml"],
    }
    defaults.update(kwargs)
    return store.add(**defaults)


class TestAnnotation:

    def test_to_dict_has_required_keys(self, tmp_path):
        store = _make_store(tmp_path)
        a = _add_sample(store)
        d = a.to_dict()
        for key in ["annotation_id", "chunk_id", "doc_id", "source_title",
                    "note", "tags", "timestamp", "chunk_preview"]:
            assert key in d

    def test_from_dict_roundtrip(self, tmp_path):
        store = _make_store(tmp_path)
        a = _add_sample(store)
        d = a.to_dict()
        a2 = Annotation.from_dict(d)
        assert a2.annotation_id == a.annotation_id
        assert a2.note == a.note
        assert a2.tags == a.tags

    def test_timestamp_is_set(self, tmp_path):
        store = _make_store(tmp_path)
        a = _add_sample(store)
        assert a.timestamp is not None
        assert "T" in a.timestamp


class TestAnnotationStore:

    def test_starts_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.total_annotations == 0

    def test_add_increases_count(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store)
        assert store.total_annotations == 1

    def test_add_returns_annotation(self, tmp_path):
        store = _make_store(tmp_path)
        a = _add_sample(store)
        assert isinstance(a, Annotation)
        assert a.annotation_id is not None

    def test_get_all_returns_most_recent_first(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, note="first")
        _add_sample(store, note="second")
        results = store.get_all()
        assert results[0].note == "second"

    def test_get_all_respects_limit(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(10):
            _add_sample(store, note=f"note {i}")
        assert len(store.get_all(limit=3)) == 3

    def test_get_by_doc_filters_correctly(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, doc_id="doc_a")
        _add_sample(store, doc_id="doc_b")
        results = store.get_by_doc("doc_a")
        assert all(a.doc_id == "doc_a" for a in results)
        assert len(results) == 1

    def test_get_by_tag_filters_correctly(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, tags=["important"])
        _add_sample(store, tags=["review"])
        results = store.get_by_tag("important")
        assert all("important" in a.tags for a in results)
        assert len(results) == 1

    def test_get_by_tag_case_insensitive(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, tags=["Important"])
        results = store.get_by_tag("important")
        assert len(results) == 1

    def test_search_matches_note_text(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, note="backpropagation is key")
        _add_sample(store, note="unrelated content")
        results = store.search("backpropagation")
        assert len(results) == 1
        assert "backpropagation" in results[0].note

    def test_search_matches_source_title(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, source_title="deep_learning_notes")
        results = store.search("deep_learning")
        assert len(results) == 1

    def test_search_empty_query_returns_all(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store)
        _add_sample(store)
        results = store.search("")
        assert len(results) == 2

    def test_delete_removes_annotation(self, tmp_path):
        store = _make_store(tmp_path)
        a = _add_sample(store)
        deleted = store.delete(a.annotation_id)
        assert deleted is True
        assert store.total_annotations == 0

    def test_delete_nonexistent_returns_false(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.delete("nonexistent-id") is False

    def test_get_by_chunk(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, chunk_id="chunk_x")
        _add_sample(store, chunk_id="chunk_y")
        results = store.get_by_chunk("chunk_x")
        assert all(a.chunk_id == "chunk_x" for a in results)

    def test_persistence_survives_reload(self, tmp_path):
        path = str(tmp_path / "annotations.json")
        s1 = AnnotationStore(persist_path=path)
        _add_sample(s1, note="persisted note")
        s2 = AnnotationStore(persist_path=path)
        assert s2.total_annotations == 1
        assert s2.get_all()[0].note == "persisted note"

    def test_clear_empties_store(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store)
        store.clear()
        assert store.total_annotations == 0

    def test_max_annotations_evicts_oldest(self, tmp_path):
        store = AnnotationStore(
            persist_path=str(tmp_path / "a.json"),
            max_annotations=3,
        )
        for i in range(5):
            _add_sample(store, note=f"note {i}")
        assert store.total_annotations == 3
        assert store.get_all()[-1].note == "note 2"

    def test_get_stats_returns_required_keys(self, tmp_path):
        store = _make_store(tmp_path)
        _add_sample(store, tags=["ml", "important"])
        stats = store.get_stats()
        assert "total_annotations" in stats
        assert "top_tags" in stats
        assert "top_documents" in stats

    def test_tags_normalised_to_lowercase(self, tmp_path):
        store = _make_store(tmp_path)
        a = _add_sample(store, tags=["ML", "Important"])
        assert "ml" in a.tags
        assert "important" in a.tags

    def test_chunk_preview_truncated_to_200(self, tmp_path):
        store = _make_store(tmp_path)
        long_preview = "x" * 500
        a = store.add(
            chunk_id="c1", doc_id="d1", source_title="t",
            note="note", chunk_preview=long_preview,
        )
        assert len(a.chunk_preview) <= 200