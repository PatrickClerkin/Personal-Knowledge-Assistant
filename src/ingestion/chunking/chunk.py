from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
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
    
    # Embedding (will be populated later)
    embedding: Optional[list] = None
    
    def __len__(self):
        return len(self.content)
    
    def __str__(self):
        return f"Chunk({self.chunk_id}, {len(self)} chars, from {self.source_doc_title})"