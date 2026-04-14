"""
Annotation and notes system for the Personal Knowledge Assistant.

Allows users to attach personal notes to any chunk retrieved from
the knowledge base. Notes are stored persistently on disk, tagged
with the source chunk, document, page, and timestamp.

Notes can be:
    - Created against any chunk_id from search or chat results
    - Retrieved by document, tag, or free-text search
    - Deleted individually
    - Surfaced alongside search results

Design Pattern: Repository Pattern — AnnotationStore is the single
source of truth for all user annotations.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Annotation:
    """A user note attached to a specific chunk.

    Attributes:
        annotation_id: Unique identifier for this annotation.
        chunk_id: The chunk this note is attached to.
        doc_id: The document the chunk belongs to.
        source_title: Human-readable document title.
        page_number: Page number of the chunk, if available.
        chunk_preview: First 200 chars of the chunk content.
        note: The user's note text.
        tags: Optional list of tags for categorisation.
        timestamp: When the annotation was created (ISO UTC).
    """
    annotation_id: str
    chunk_id: str
    doc_id: str
    source_title: str
    page_number: Optional[int]
    chunk_preview: str
    note: str
    tags: List[str]
    timestamp: str

    def to_dict(self) -> dict:
        """Serialise for JSON API response."""
        return {
            "annotation_id": self.annotation_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_title": self.source_title,
            "page_number": self.page_number,
            "chunk_preview": self.chunk_preview,
            "note": self.note,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        """Deserialise from JSON dict."""
        return cls(
            annotation_id=data["annotation_id"],
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source_title=data.get("source_title", ""),
            page_number=data.get("page_number"),
            chunk_preview=data.get("chunk_preview", ""),
            note=data["note"],
            tags=data.get("tags", []),
            timestamp=data["timestamp"],
        )


class AnnotationStore:
    """Persistent store for user annotations on chunks.

    Stores annotations in memory and persists to a JSON file on disk.
    Supports creation, retrieval by document or tag, free-text search,
    and deletion.

    Args:
        persist_path: Path to the JSON file for persistence.
            Defaults to 'data/annotations/annotations.json'.
        max_annotations: Maximum annotations to store.
            Oldest are evicted when limit is reached.
    """

    def __init__(
        self,
        persist_path: str = "data/annotations/annotations.json",
        max_annotations: int = 5000,
    ):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_annotations = max_annotations
        self._annotations: List[Annotation] = []
        self._load()

    def add(
        self,
        chunk_id: str,
        doc_id: str,
        source_title: str,
        note: str,
        chunk_preview: str = "",
        page_number: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Annotation:
        """Create and store a new annotation.

        Args:
            chunk_id: ID of the chunk being annotated.
            doc_id: Document the chunk belongs to.
            source_title: Human-readable document title.
            note: The user's note text.
            chunk_preview: Preview of the chunk content.
            page_number: Page number of the chunk.
            tags: Optional list of tag strings.

        Returns:
            The created Annotation.
        """
        annotation = Annotation(
            annotation_id=str(uuid.uuid4()),
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_title=source_title,
            page_number=page_number,
            chunk_preview=chunk_preview[:200] if chunk_preview else "",
            note=note.strip(),
            tags=[t.strip().lower() for t in (tags or []) if t.strip()],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._annotations.append(annotation)

        # Evict oldest if over limit
        if len(self._annotations) > self.max_annotations:
            self._annotations = self._annotations[-self.max_annotations:]

        self._save()
        logger.info(
            "Annotation added: %s on chunk %s",
            annotation.annotation_id[:8], chunk_id[:8],
        )
        return annotation

    def get_all(self, limit: int = 50) -> List[Annotation]:
        """Return the most recent annotations.

        Args:
            limit: Maximum number of annotations to return.

        Returns:
            List of Annotation, most recent first.
        """
        return list(reversed(self._annotations[-limit:]))

    def get_by_doc(self, doc_id: str) -> List[Annotation]:
        """Return all annotations for a specific document.

        Args:
            doc_id: Document ID to filter by.

        Returns:
            List of Annotation for that document, most recent first.
        """
        results = [a for a in self._annotations if a.doc_id == doc_id]
        return list(reversed(results))

    def get_by_tag(self, tag: str) -> List[Annotation]:
        """Return all annotations with a specific tag.

        Args:
            tag: Tag string to filter by (case-insensitive).

        Returns:
            List of matching Annotation, most recent first.
        """
        tag = tag.strip().lower()
        results = [a for a in self._annotations if tag in a.tags]
        return list(reversed(results))

    def search(self, query: str) -> List[Annotation]:
        """Search annotations by note text or chunk preview.

        Case-insensitive substring search across note text,
        chunk preview, source title, and tags.

        Args:
            query: Search string.

        Returns:
            List of matching Annotation, most recent first.
        """
        q = query.strip().lower()
        if not q:
            return self.get_all()

        results = []
        for a in self._annotations:
            searchable = " ".join([
                a.note.lower(),
                a.chunk_preview.lower(),
                a.source_title.lower(),
                " ".join(a.tags),
            ])
            if q in searchable:
                results.append(a)

        return list(reversed(results))

    def delete(self, annotation_id: str) -> bool:
        """Delete an annotation by ID.

        Args:
            annotation_id: The annotation ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        original_len = len(self._annotations)
        self._annotations = [
            a for a in self._annotations
            if a.annotation_id != annotation_id
        ]
        deleted = len(self._annotations) < original_len
        if deleted:
            self._save()
            logger.info("Annotation deleted: %s", annotation_id[:8])
        return deleted

    def get_by_chunk(self, chunk_id: str) -> List[Annotation]:
        """Return all annotations for a specific chunk.

        Args:
            chunk_id: Chunk ID to filter by.

        Returns:
            List of Annotation for that chunk, most recent first.
        """
        results = [a for a in self._annotations if a.chunk_id == chunk_id]
        return list(reversed(results))

    def get_stats(self) -> dict:
        """Return summary statistics about the annotation store."""
        all_tags = []
        for a in self._annotations:
            all_tags.extend(a.tags)

        from collections import Counter
        tag_counts = Counter(all_tags).most_common(10)
        doc_counts = Counter(a.doc_id for a in self._annotations).most_common(10)

        return {
            "total_annotations": len(self._annotations),
            "top_tags": [{"tag": t, "count": c} for t, c in tag_counts],
            "top_documents": [{"doc_id": d, "count": c} for d, c in doc_counts],
        }

    @property
    def total_annotations(self) -> int:
        """Total number of stored annotations."""
        return len(self._annotations)

    def clear(self) -> None:
        """Delete all annotations and the persist file."""
        self._annotations.clear()
        if self.persist_path.exists():
            self.persist_path.unlink()
        logger.info("All annotations cleared.")

    def _save(self) -> None:
        """Persist annotations to JSON file."""
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump([a.to_dict() for a in self._annotations], f, indent=2)
        except Exception as e:
            logger.warning("Failed to save annotations: %s", e)

    def _load(self) -> None:
        """Load annotations from JSON file if it exists."""
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            self._annotations = [Annotation.from_dict(d) for d in data]
            logger.info(
                "Loaded %d annotations from %s",
                len(self._annotations), self.persist_path,
            )
        except Exception as e:
            logger.warning("Failed to load annotations: %s", e)
            self._annotations = []