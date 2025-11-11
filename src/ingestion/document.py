from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path

@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    doc_id: str
    title: str
    source_path: Path
    file_type: str
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class Document:
    """Represents a processed document."""
    metadata: DocumentMetadata
    content: str
    sections: List['DocumentSection'] = field(default_factory=list)
    
    def __len__(self):
        return len(self.content)

@dataclass
class DocumentSection:
    """Represents a section within a document."""
    section_id: str
    title: Optional[str]
    content: str
    level: int
    page_number: Optional[int] = None
    start_char: int = 0
    end_char: int = 0