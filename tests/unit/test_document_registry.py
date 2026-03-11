"""Unit tests for DocumentRegistry."""

import json
import time
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from src.ingestion.storage.document_registry import DocumentRecord, DocumentRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return DocumentRegistry()


@pytest.fixture
def sample_record_kwargs():
    return {
        "doc_id": "lecture_notes",
        "filename": "lecture_notes.pdf",
        "file_hash": "abc123",
        "chunk_count": 42,
    }


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_new_document_creates_record(self, registry, sample_record_kwargs):
        record = registry.register(**sample_record_kwargs)
        assert record.doc_id == "lecture_notes"
        assert record.filename == "lecture_notes.pdf"
        assert record.file_hash == "abc123"
        assert record.chunk_count == 42

    def test_ingested_at_and_updated_at_are_equal_on_first_ingest(self, registry, sample_record_kwargs):
        record = registry.register(**sample_record_kwargs)
        assert record.ingested_at == record.updated_at

    def test_re_ingest_preserves_ingested_at(self, registry, sample_record_kwargs):
        first = registry.register(**sample_record_kwargs)
        original_ingested_at = first.ingested_at

        time.sleep(0.01)  # ensure clock advances

        updated = registry.register(
            doc_id="lecture_notes",
            filename="lecture_notes.pdf",
            file_hash="def456",
            chunk_count=50,
        )
        assert updated.ingested_at == original_ingested_at

    def test_re_ingest_updates_file_hash_and_chunk_count(self, registry, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        updated = registry.register(
            doc_id="lecture_notes",
            filename="lecture_notes.pdf",
            file_hash="newHash999",
            chunk_count=99,
        )
        assert updated.file_hash == "newHash999"
        assert updated.chunk_count == 99

    def test_re_ingest_updates_updated_at(self, registry, sample_record_kwargs):
        first = registry.register(**sample_record_kwargs)
        time.sleep(0.01)
        second = registry.register(
            doc_id="lecture_notes",
            filename="lecture_notes.pdf",
            file_hash="newHash",
            chunk_count=5,
        )
        assert second.updated_at > first.updated_at

    def test_multiple_documents_tracked_independently(self, registry):
        registry.register("doc_a", "a.pdf", "hash_a", 10)
        registry.register("doc_b", "b.txt", "hash_b", 20)
        assert registry.get("doc_a").chunk_count == 10
        assert registry.get("doc_b").chunk_count == 20


# ---------------------------------------------------------------------------
# is_unchanged()
# ---------------------------------------------------------------------------

class TestIsUnchanged:
    def test_returns_true_when_hash_matches(self, registry, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        assert registry.is_unchanged("lecture_notes", "abc123") is True

    def test_returns_false_when_hash_differs(self, registry, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        assert registry.is_unchanged("lecture_notes", "differentHash") is False

    def test_returns_false_for_unknown_doc_id(self, registry):
        assert registry.is_unchanged("nonexistent", "anyhash") is False


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_record_for_known_doc(self, registry, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        record = registry.get("lecture_notes")
        assert isinstance(record, DocumentRecord)

    def test_returns_none_for_unknown_doc(self, registry):
        assert registry.get("missing") is None


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

class TestRemove:
    def test_removes_existing_record(self, registry, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        registry.remove("lecture_notes")
        assert registry.get("lecture_notes") is None

    def test_remove_nonexistent_doc_does_not_raise(self, registry):
        registry.remove("ghost_doc")  # should be a no-op


# ---------------------------------------------------------------------------
# all_records
# ---------------------------------------------------------------------------

class TestAllRecords:
    def test_returns_all_registered_documents(self, registry):
        registry.register("a", "a.pdf", "h1", 5)
        registry.register("b", "b.pdf", "h2", 10)
        assert len(registry.all_records) == 2

    def test_sorted_by_most_recently_updated(self, registry):
        registry.register("old_doc", "old.pdf", "h1", 5)
        time.sleep(0.01)
        registry.register("new_doc", "new.pdf", "h2", 10)
        records = registry.all_records
        assert records[0].doc_id == "new_doc"

    def test_empty_registry_returns_empty_list(self, registry):
        assert registry.all_records == []


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_registry_json(self, registry, tmp_path, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        index_path = str(tmp_path / "index")
        registry.save(index_path)
        assert Path(index_path + ".registry.json").exists()

    def test_load_restores_records(self, registry, tmp_path, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        index_path = str(tmp_path / "index")
        registry.save(index_path)

        new_registry = DocumentRegistry()
        new_registry.load(index_path)

        record = new_registry.get("lecture_notes")
        assert record is not None
        assert record.filename == "lecture_notes.pdf"
        assert record.file_hash == "abc123"
        assert record.chunk_count == 42

    def test_load_preserves_ingested_at(self, registry, tmp_path, sample_record_kwargs):
        registry.register(**sample_record_kwargs)
        original_ingested_at = registry.get("lecture_notes").ingested_at
        index_path = str(tmp_path / "index")
        registry.save(index_path)

        new_registry = DocumentRegistry()
        new_registry.load(index_path)
        assert new_registry.get("lecture_notes").ingested_at == original_ingested_at

    def test_load_missing_file_is_noop(self, registry, tmp_path):
        registry.load(str(tmp_path / "nonexistent"))
        assert registry.all_records == []

    def test_roundtrip_multiple_documents(self, registry, tmp_path):
        registry.register("doc_a", "a.pdf", "hash_a", 10)
        registry.register("doc_b", "b.txt", "hash_b", 20)
        index_path = str(tmp_path / "index")
        registry.save(index_path)

        new_registry = DocumentRegistry()
        new_registry.load(index_path)
        assert new_registry.get("doc_a").chunk_count == 10
        assert new_registry.get("doc_b").chunk_count == 20


# ---------------------------------------------------------------------------
# hash_file()
# ---------------------------------------------------------------------------

class TestHashFile:
    def test_returns_sha256_hex_string(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        result = DocumentRegistry.hash_file(f)
        assert isinstance(result, str)
        assert len(result) == 64   # SHA-256 hex digest is always 64 chars

    def test_same_content_produces_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert DocumentRegistry.hash_file(f1) == DocumentRegistry.hash_file(f2)

    def test_different_content_produces_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert DocumentRegistry.hash_file(f1) != DocumentRegistry.hash_file(f2)