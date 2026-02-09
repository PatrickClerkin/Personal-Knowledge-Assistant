"""
Parser for plain text and Markdown files.

Handles .txt and .md files. For Markdown files, detects heading structure
(# H1, ## H2, etc.) and creates document sections from them. For plain
text files, splits into sections based on paragraph breaks.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import BaseParser
from ..document import Document, DocumentSection
from ...utils.logger import get_logger

logger = get_logger(__name__)


class TextParser(BaseParser):
    """Parser for plain text (.txt) and Markdown (.md) files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = [".txt", ".md"]

        # Markdown heading pattern: # Heading, ## Heading, etc.
        self._heading_pattern = re.compile(
            r"^(#{1,6})\s+(.+)$", re.MULTILINE
        )
        # Paragraph separator: two or more newlines
        self._paragraph_pattern = re.compile(r"\n\s*\n")

    def parse(self, file_path: Path) -> Document:
        """
        Parse a text or Markdown file.

        For .md files, sections are created from headings.
        For .txt files, sections are created from paragraph breaks.

        Args:
            file_path: Path to the text file.

        Returns:
            Parsed Document with sections.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file encoding cannot be detected.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read with encoding detection fallback
        content = self._read_file(file_path)
        metadata = self._extract_metadata(file_path)

        # Create sections based on file type
        if file_path.suffix.lower() == ".md":
            sections = self._parse_markdown_sections(content)
            logger.info(
                "Parsed Markdown file: %s (%d sections, %d chars)",
                file_path.name,
                len(sections),
                len(content),
            )
        else:
            sections = self._parse_text_sections(content)
            logger.info(
                "Parsed text file: %s (%d sections, %d chars)",
                file_path.name,
                len(sections),
                len(content),
            )

        return Document(
            metadata=metadata,
            content=content,
            sections=sections,
        )

    def _read_file(self, file_path: Path) -> str:
        """
        Read a text file with encoding fallback.

        Tries UTF-8 first, then falls back to latin-1 which accepts
        any byte sequence.
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1,
            f"Could not decode {file_path} with any supported encoding"
        )

    def _parse_markdown_sections(self, content: str) -> List[DocumentSection]:
        """
        Parse Markdown content into sections based on headings.

        Each heading (# H1, ## H2, etc.) starts a new section.
        Content before the first heading becomes an introductory section.
        """
        sections = []
        headings = list(self._heading_pattern.finditer(content))

        if not headings:
            # No headings found — treat as single section
            return self._parse_text_sections(content)

        # Content before the first heading
        if headings[0].start() > 0:
            intro_content = content[: headings[0].start()].strip()
            if intro_content:
                sections.append(
                    DocumentSection(
                        section_id="intro",
                        title="Introduction",
                        content=intro_content,
                        level=0,
                        start_char=0,
                        end_char=headings[0].start(),
                    )
                )

        # Create sections from headings
        for i, match in enumerate(headings):
            heading_level = len(match.group(1))  # Number of # characters
            heading_title = match.group(2).strip()

            # Section content runs from this heading to the next (or end)
            start_char = match.start()
            end_char = (
                headings[i + 1].start() if i + 1 < len(headings) else len(content)
            )
            section_content = content[start_char:end_char].strip()

            sections.append(
                DocumentSection(
                    section_id=f"section_{i + 1}",
                    title=heading_title,
                    content=section_content,
                    level=heading_level,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return sections

    def _parse_text_sections(self, content: str) -> List[DocumentSection]:
        """
        Parse plain text into sections based on paragraph breaks.

        Each paragraph (separated by blank lines) becomes a section.
        Very short paragraphs are merged with the next one.
        """
        paragraphs = self._paragraph_pattern.split(content)
        sections = []
        current_pos = 0

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Find actual position in original content
            start = content.find(para, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(para)

            sections.append(
                DocumentSection(
                    section_id=f"para_{i + 1}",
                    title=None,
                    content=para,
                    level=1,
                    start_char=start,
                    end_char=end,
                )
            )
            current_pos = end

        # If no sections were created (no paragraph breaks), use whole content
        if not sections and content.strip():
            sections.append(
                DocumentSection(
                    section_id="para_1",
                    title=None,
                    content=content.strip(),
                    level=1,
                    start_char=0,
                    end_char=len(content),
                )
            )

        return sections
