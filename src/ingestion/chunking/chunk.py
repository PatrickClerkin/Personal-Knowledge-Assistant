from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class Chunk:
    """Represents a chunk of text from a document with its metadata and embedding."""

    chunk_id: str
    doc_id: str
    content: str

    # Source tracking (for citations)
    source_doc_title: str
    section_id: Optional[str] = None
    page_number: Optional[int] = None

    # Position in original document
    start_char: int = 0
    end_char: int = 0

    # Metadata
    chunk_index: int = 0
    total_chunks: Optional[int] = None

    # Embedding vector (populated by EmbeddingService)
    embedding: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.content)

    def __str__(self) -> str:
        return f"Chunk({self.chunk_id}, {len(self)} chars, page {self.page_number})"

    def __repr__(self) -> str:
        return (
            f"Chunk(chunk_id={self.chunk_id!r}, doc_id={self.doc_id!r}, "
            f"len={len(self)}, page={self.page_number})"
        )

    @property
    def has_embedding(self) -> bool:
        """Check if this chunk has an embedding vector."""
        return self.embedding is not None

    def get_citation(self) -> str:
        """Get a citation string for this chunk."""
        citation = f"{self.source_doc_title}"
        if self.page_number:
            citation += f", p. {self.page_number}"
        return citation
