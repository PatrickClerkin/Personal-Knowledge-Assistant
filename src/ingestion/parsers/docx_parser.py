"""
Parser for Microsoft Word (.docx) files.

Uses python-docx to extract text content with structural awareness:
headings become document sections, tables are converted to readable text,
and Word metadata (author, title, dates) is preserved.
"""

from pathlib import Path
from typing import List, Optional

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from .base import BaseParser
from ..document import Document, DocumentSection
from ...utils.logger import get_logger

logger = get_logger(__name__)


class DOCXParser(BaseParser):
    """Parser for Microsoft Word (.docx) files."""

    # Mapping from python-docx heading styles to section levels
    _HEADING_LEVELS = {
        "Heading 1": 1,
        "Heading 2": 2,
        "Heading 3": 3,
        "Heading 4": 4,
        "Heading 5": 5,
        "Heading 6": 6,
        "Title": 0,
    }

    def __init__(self):
        super().__init__()
        self.supported_extensions = [".docx"]

    def parse(self, file_path: Path) -> Document:
        """
        Parse a .docx file into a Document.

        Extracts text from paragraphs and tables, creates sections
        from Word heading styles, and preserves document metadata.

        Args:
            file_path: Path to the .docx file.

        Returns:
            Parsed Document with sections and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            PackageNotFoundError: If the file is not a valid .docx.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            docx_doc = DocxDocument(str(file_path))
        except PackageNotFoundError:
            raise ValueError(f"Not a valid .docx file: {file_path}")

        # Build full text and sections
        full_text, sections = self._extract_content(docx_doc)

        # Extract metadata
        metadata = self._extract_metadata(file_path)
        core = docx_doc.core_properties
        if core.author:
            metadata.author = core.author
        if core.title and core.title.strip():
            metadata.title = core.title

        logger.info(
            "Parsed DOCX file: %s (%d sections, %d chars)",
            file_path.name,
            len(sections),
            len(full_text),
        )

        return Document(
            metadata=metadata,
            content=full_text,
            sections=sections,
        )

    def _extract_content(
        self, docx_doc: DocxDocument
    ) -> tuple[str, List[DocumentSection]]:
        """
        Walk through all paragraphs and tables, building full text
        and creating sections from heading styles.
        """
        text_parts: List[str] = []
        sections: List[DocumentSection] = []
        current_pos = 0

        # Track current section being built
        current_heading: Optional[str] = None
        current_level: int = 1
        section_start: int = 0
        section_parts: List[str] = []
        section_count = 0

        for element in docx_doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # It's a paragraph
                from docx.text.paragraph import Paragraph

                para = Paragraph(element, docx_doc)
                para_text = para.text.strip()
                style_name = para.style.name if para.style else ""

                if style_name in self._HEADING_LEVELS and para_text:
                    # Save previous section if exists
                    if section_parts:
                        section_content = "\n\n".join(section_parts)
                        sections.append(
                            DocumentSection(
                                section_id=f"section_{section_count}",
                                title=current_heading,
                                content=section_content,
                                level=current_level,
                                start_char=section_start,
                                end_char=current_pos,
                            )
                        )
                        section_count += 1

                    # Start new section
                    current_heading = para_text
                    current_level = self._HEADING_LEVELS[style_name]
                    section_start = current_pos
                    section_parts = []

                    # Add heading text to full document
                    text_parts.append(para_text)
                    current_pos += len(para_text) + 1  # +1 for newline

                elif para_text:
                    # Regular paragraph
                    text_parts.append(para_text)
                    section_parts.append(para_text)
                    current_pos += len(para_text) + 1

            elif tag == "tbl":
                # It's a table — convert to readable text
                from docx.table import Table

                table = Table(element, docx_doc)
                table_text = self._table_to_text(table)
                if table_text:
                    text_parts.append(table_text)
                    section_parts.append(table_text)
                    current_pos += len(table_text) + 1

        # Save final section
        if section_parts:
            section_content = "\n\n".join(section_parts)
            sections.append(
                DocumentSection(
                    section_id=f"section_{section_count}",
                    title=current_heading,
                    content=section_content,
                    level=current_level,
                    start_char=section_start,
                    end_char=current_pos,
                )
            )

        full_text = "\n".join(text_parts)

        # If no sections were created (no headings), create one from all content
        if not sections and full_text.strip():
            sections.append(
                DocumentSection(
                    section_id="section_0",
                    title=None,
                    content=full_text.strip(),
                    level=1,
                    start_char=0,
                    end_char=len(full_text),
                )
            )

        return full_text, sections

    def _table_to_text(self, table) -> str:
        """
        Convert a Word table to pipe-delimited text.

        Produces a readable representation:
            Header1 | Header2 | Header3
            --------|---------|--------
            Cell1   | Cell2   | Cell3
        """
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
            # Add separator after first row (header)
            if i == 0:
                rows.append(" | ".join(["---"] * len(cells)))

        return "\n".join(rows) if rows else ""
