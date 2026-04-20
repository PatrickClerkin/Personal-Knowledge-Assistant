"""
Abstract base class for document parsers.

All parsers (PDF, TXT/MD, DOCX) inherit from BaseParser and implement
the parse() method. Common functionality like metadata extraction and
document ID generation lives here.

Design Pattern: Template Method — parse() is abstract; shared steps
like _extract_metadata() and _generate_doc_id() are concrete.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
from datetime import datetime
from ..document import Document, DocumentMetadata


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self):
        self.supported_extensions = []

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse a document from the given file path."""
        pass

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a deterministic document ID from the file path.

        Uses the file's absolute path to produce a stable ID that
        remains consistent across parses of the same file. This
        avoids the issue of timestamp-based IDs changing every time
        a file is re-parsed.

        Note: KnowledgeBase.ingest() overrides this with file_path.stem
        for registry consistency, but the parser-level ID is still
        deterministic for any direct parser usage.
        """
        content = str(file_path.absolute())
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract basic metadata from file."""
        stat = file_path.stat()

        return DocumentMetadata(
            doc_id=self._generate_doc_id(file_path),
            title=file_path.stem,
            source_path=file_path,
            file_type=file_path.suffix.lower().lstrip('.'),
            created_date=datetime.fromtimestamp(stat.st_ctime),
            modified_date=datetime.fromtimestamp(stat.st_mtime),
        )