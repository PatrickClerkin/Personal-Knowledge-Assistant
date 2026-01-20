from dataclasses import dataclass, field
from typing import Optional, List
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
    
    # Semantic chunking metadata
    chunking_method: Optional[str] = None  # Which method created this chunk
    topic_id: Optional[int] = None  # For topic-based chunking
    cluster_id: Optional[int] = None  # For density-based chunking
    hierarchy_level: Optional[int] = None  # For recursive chunking
    parent_chunk_id: Optional[str] = None  # For hierarchical relationships
    child_chunk_ids: List[str] = field(default_factory=list)
    
    # Similarity scores (populated during semantic chunking)
    boundary_similarity: Optional[float] = None  # Similarity at chunk boundary
    coherence_score: Optional[float] = None  # Internal coherence measure

    # Embedding vector (populated by EmbeddingService)
    embedding: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.content)

    def __str__(self) -> str:
        method = f", method={self.chunking_method}" if self.chunking_method else ""
        return f"Chunk({self.chunk_id}, {len(self)} chars, page {self.page_number}{method})"

    def __repr__(self) -> str:
        return (
            f"Chunk(chunk_id={self.chunk_id!r}, doc_id={self.doc_id!r}, "
            f"len={len(self)}, page={self.page_number}, method={self.chunking_method})"
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
        if self.section_id:
            citation += f" ({self.section_id})"
        return citation
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary (for serialization)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source_doc_title": self.source_doc_title,
            "section_id": self.section_id,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunking_method": self.chunking_method,
            "topic_id": self.topic_id,
            "cluster_id": self.cluster_id,
            "hierarchy_level": self.hierarchy_level,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "boundary_similarity": self.boundary_similarity,
            "coherence_score": self.coherence_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            content=data["content"],
            source_doc_title=data["source_doc_title"],
            section_id=data.get("section_id"),
            page_number=data.get("page_number"),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks"),
            chunking_method=data.get("chunking_method"),
            topic_id=data.get("topic_id"),
            cluster_id=data.get("cluster_id"),
            hierarchy_level=data.get("hierarchy_level"),
            parent_chunk_id=data.get("parent_chunk_id"),
            child_chunk_ids=data.get("child_chunk_ids", []),
            boundary_similarity=data.get("boundary_similarity"),
            coherence_score=data.get("coherence_score"),
        )
