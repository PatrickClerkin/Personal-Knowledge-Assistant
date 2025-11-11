from pathlib import Path
import pymupdf
from typing import List
from .base import BaseParser
from ..document import Document, DocumentSection

class PDFParser(BaseParser):
    """Parser for PDF documents."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']
    
    def parse(self, file_path: Path) -> Document:
        """Parse a PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Open PDF
        pdf_doc = pymupdf.open(file_path)
        
        # Extract text from all pages
        full_text = ""
        sections = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            page_text = page.get_text()
            
            start_char = len(full_text)
            full_text += page_text + "\n"
            end_char = len(full_text)
            
            # Create section for this page
            sections.append(DocumentSection(
                section_id=f"page_{page_num + 1}",
                title=f"Page {page_num + 1}",
                content=page_text,
                level=1,
                page_number=page_num + 1,
                start_char=start_char,
                end_char=end_char
            ))
        
        # Extract metadata
        metadata = self._extract_metadata(file_path)
        
        # Try to get PDF metadata
        pdf_metadata = pdf_doc.metadata
        if pdf_metadata:
            if pdf_metadata.get('author'):
                metadata.author = pdf_metadata['author']
            if pdf_metadata.get('title') and pdf_metadata['title'].strip():
                metadata.title = pdf_metadata['title']
        
        pdf_doc.close()
        
        return Document(
            metadata=metadata,
            content=full_text,
            sections=sections
        )