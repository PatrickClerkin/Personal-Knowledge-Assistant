from pathlib import Path
from typing import List, Optional
from .parsers.base import BaseParser
from .parsers.pdf_parser import PDFParser
from .parsers.text_parser import TextParser
from .parsers.docx_parser import DOCXParser
from .document import Document
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DocumentManager:
    """Manages document parsing with multiple parsers."""
    
    def __init__(self):
        self.parsers: List[BaseParser] = [
            PDFParser(),
            TextParser(),
            DOCXParser(),
        ]
    
    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """Find appropriate parser for file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return all file extensions this manager can handle."""
        extensions = []
        for parser in self.parsers:
            extensions.extend(parser.supported_extensions)
        return extensions
    
    def parse_document(self, file_path: Path) -> Document:
        """Parse a single document."""
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(f"No parser available for {file_path.suffix}")
        return parser.parse(file_path)
    
    def parse_directory(self, directory: Path) -> List[Document]:
        """Parse all supported documents in a directory."""
        documents = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and self.get_parser(file_path):
                try:
                    doc = self.parse_document(file_path)
                    documents.append(doc)
                    logger.info("Parsed: %s", file_path.name)
                except Exception as e:
                    logger.error("Failed to parse %s: %s", file_path.name, e)
        return documents
