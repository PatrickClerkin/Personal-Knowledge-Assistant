"""Registry for tracking ingested documents and their versions."""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DocumentRecord:
    """Metadata record for a single ingested document."""

    doc_id: str
    filename: str
    file_hash: str        # SHA-256 of the raw file bytes
    ingested_at: str      # ISO-8601 UTC — set once on first ingest
    updated_at: str       # ISO-8601 UTC — refreshed on every re-ingest
    chunk_count: int


class DocumentRegistry:
    """
    Persists a mapping of doc_id → DocumentRecord alongside the FAISS index.

    Responsibilities
    ----------------
    - Record when each document was first ingested and last updated.
    - Detect whether a file has changed since last ingest (via SHA-256 hash).
    - Expose metadata so the CLI / web UI can show "last updated" next to results.

    Saved as ``<index_path>.registry.json``.
    """

    def __init__(self) -> None:
        self._records: Dict[str, DocumentRecord] = {}

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def hash_file(file_path) -> str:
        """Return the SHA-256 hex digest of a file's contents."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def register(
        self,
        doc_id: str,
        filename: str,
        file_hash: str,
        chunk_count: int,
    ) -> DocumentRecord:
        """
        Add or update the record for *doc_id*.

        ``ingested_at`` is preserved on updates; only ``updated_at`` and
        ``chunk_count`` (and ``file_hash``) change on re-ingest.
        """
        now = datetime.now(timezone.utc).isoformat()

        if doc_id in self._records:
            existing = self._records[doc_id]
            updated = DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                file_hash=file_hash,
                ingested_at=existing.ingested_at,   # preserve original date
                updated_at=now,
                chunk_count=chunk_count,
            )
        else:
            updated = DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                file_hash=file_hash,
                ingested_at=now,
                updated_at=now,
                chunk_count=chunk_count,
            )

        self._records[doc_id] = updated
        return updated

    def remove(self, doc_id: str) -> None:
        """Remove the record for *doc_id* (called when a document is deleted)."""
        self._records.pop(doc_id, None)

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """Return the record for *doc_id*, or ``None`` if not found."""
        return self._records.get(doc_id)

    def is_unchanged(self, doc_id: str, file_hash: str) -> bool:
        """Return True if *doc_id* is already registered with the same hash."""
        record = self._records.get(doc_id)
        return record is not None and record.file_hash == file_hash

    @property
    def all_records(self) -> List[DocumentRecord]:
        """Return all records, sorted by most recently updated."""
        return sorted(
            self._records.values(),
            key=lambda r: r.updated_at,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the registry to ``<path>.registry.json``."""
        registry_path = Path(path + ".registry.json")
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {doc_id: asdict(record) for doc_id, record in self._records.items()}
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load the registry from ``<path>.registry.json`` (no-op if missing)."""
        registry_path = Path(path + ".registry.json")
        if not registry_path.exists():
            return
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._records = {
            doc_id: DocumentRecord(**record_data)
            for doc_id, record_data in data.items()
        }