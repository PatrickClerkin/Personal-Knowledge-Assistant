"""Tests for the TextParser, DOCXParser, and DocumentManager."""

import pytest
from pathlib import Path

from src.ingestion.parsers.text_parser import TextParser
from src.ingestion.parsers.docx_parser import DOCXParser
from src.ingestion.document_manager import DocumentManager
from src.ingestion.document import Document, DocumentSection


class TestTextParserInit:
    """Test parser initialisation and extension support."""

    def test_supported_extensions(self):
        parser = TextParser()
        assert ".txt" in parser.supported_extensions
        assert ".md" in parser.supported_extensions

    def test_can_parse_txt(self, tmp_path):
        parser = TextParser()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        assert parser.can_parse(txt_file) is True

    def test_can_parse_md(self, tmp_path):
        parser = TextParser()
        md_file = tmp_path / "test.md"
        md_file.write_text("# hello")
        assert parser.can_parse(md_file) is True

    def test_cannot_parse_pdf(self, tmp_path):
        parser = TextParser()
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake")
        assert parser.can_parse(pdf_file) is False


class TestTextParsing:
    """Test parsing of plain .txt files."""

    def test_parse_returns_document(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        assert isinstance(doc, Document)

    def test_parse_extracts_content(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        assert len(doc.content) > 0
        assert "Object-oriented programming" in doc.content

    def test_parse_creates_sections_from_paragraphs(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        # The sample.txt has multiple paragraphs separated by blank lines
        assert len(doc.sections) > 1

    def test_sections_are_document_sections(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        for section in doc.sections:
            assert isinstance(section, DocumentSection)

    def test_sections_have_content(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        for section in doc.sections:
            assert len(section.content.strip()) > 0

    def test_metadata_file_type(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        assert doc.metadata.file_type == "txt"

    def test_metadata_title(self, sample_txt_path):
        parser = TextParser()
        doc = parser.parse(sample_txt_path)
        assert doc.metadata.title == "sample"

    def test_single_paragraph_creates_one_section(self, tmp_path):
        """A file with no paragraph breaks should produce one section."""
        txt_file = tmp_path / "single.txt"
        txt_file.write_text("This is a single paragraph with no blank lines.")
        parser = TextParser()
        doc = parser.parse(txt_file)
        assert len(doc.sections) == 1

    def test_empty_file(self, tmp_path):
        """An empty file should produce zero sections."""
        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("")
        parser = TextParser()
        doc = parser.parse(txt_file)
        assert len(doc.sections) == 0


class TestMarkdownParsing:
    """Test parsing of .md files with heading detection."""

    def test_parse_returns_document(self, sample_md_path):
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        assert isinstance(doc, Document)

    def test_parse_extracts_content(self, sample_md_path):
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        assert "Design Patterns" in doc.content

    def test_parse_detects_headings_as_sections(self, sample_md_path):
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        # sample.md has multiple heading levels
        assert len(doc.sections) > 3

    def test_sections_have_heading_titles(self, sample_md_path):
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        titles = [s.title for s in doc.sections if s.title]
        assert "Creational Patterns" in titles
        assert "Structural Patterns" in titles
        assert "Behavioural Patterns" in titles

    def test_section_levels_match_heading_depth(self, sample_md_path):
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        # Find a known H1 and H2 and H3
        section_map = {s.title: s for s in doc.sections if s.title}
        assert section_map["Design Patterns in Software Engineering"].level == 1
        assert section_map["Creational Patterns"].level == 2
        assert section_map["Singleton Pattern"].level == 3

    def test_metadata_file_type(self, sample_md_path):
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        assert doc.metadata.file_type == "md"

    def test_intro_section_before_first_heading(self, tmp_path):
        """Content before the first heading should become an intro section."""
        md_file = tmp_path / "intro.md"
        md_file.write_text("Some intro text.\n\n# First Heading\n\nContent here.")
        parser = TextParser()
        doc = parser.parse(md_file)
        intro = doc.sections[0]
        assert intro.title == "Introduction"
        assert "intro text" in intro.content

    def test_md_without_headings_falls_back_to_paragraphs(self, tmp_path):
        """Markdown without headings should be parsed like plain text."""
        md_file = tmp_path / "noheadings.md"
        md_file.write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
        parser = TextParser()
        doc = parser.parse(md_file)
        assert len(doc.sections) == 3


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_file_not_found_raises_error(self):
        parser = TextParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.txt"))

    def test_unicode_content(self, tmp_path):
        """Parser should handle Unicode characters correctly."""
        txt_file = tmp_path / "unicode.txt"
        txt_file.write_text("Héllo wörld — «quotes» and émojis 🎉", encoding="utf-8")
        parser = TextParser()
        doc = parser.parse(txt_file)
        assert "Héllo" in doc.content
        assert "🎉" in doc.content

    def test_latin1_fallback(self, tmp_path):
        """Parser should fall back to latin-1 for non-UTF-8 files."""
        txt_file = tmp_path / "latin1.txt"
        txt_file.write_bytes("caf\xe9 cr\xe8me".encode("latin-1"))
        parser = TextParser()
        doc = parser.parse(txt_file)
        assert "café" in doc.content

    def test_section_positions_are_valid(self, sample_md_path):
        """Section start_char and end_char should be within content bounds."""
        parser = TextParser()
        doc = parser.parse(sample_md_path)
        for section in doc.sections:
            assert section.start_char >= 0
            assert section.end_char <= len(doc.content)
            assert section.start_char <= section.end_char


class TestDOCXParserInit:
    """Test DOCX parser initialisation and extension support."""

    def test_supported_extensions(self):
        parser = DOCXParser()
        assert ".docx" in parser.supported_extensions

    def test_can_parse_docx(self, tmp_path):
        parser = DOCXParser()
        docx_file = tmp_path / "test.docx"
        docx_file.write_text("fake")
        assert parser.can_parse(docx_file) is True

    def test_cannot_parse_pdf(self, tmp_path):
        parser = DOCXParser()
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake")
        assert parser.can_parse(pdf_file) is False

    def test_cannot_parse_txt(self, tmp_path):
        parser = DOCXParser()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("fake")
        assert parser.can_parse(txt_file) is False


class TestDOCXParsing:
    """Test parsing of .docx files."""

    def test_parse_returns_document(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        assert isinstance(doc, Document)

    def test_parse_extracts_content(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        assert len(doc.content) > 0
        assert "Layered Architecture" in doc.content

    def test_parse_detects_headings_as_sections(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        assert len(doc.sections) >= 3

    def test_sections_have_titles(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        titles = [s.title for s in doc.sections if s.title]
        assert "Layered Architecture" in titles
        assert "Microservices" in titles

    def test_section_levels_from_heading_styles(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        section_map = {s.title: s for s in doc.sections if s.title}
        assert section_map["Layered Architecture"].level == 1
        assert section_map["Advantages"].level == 2

    def test_metadata_file_type(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        assert doc.metadata.file_type == "docx"

    def test_metadata_author_extracted(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        assert doc.metadata.author == "Test Author"

    def test_metadata_title_extracted(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        assert doc.metadata.title == "Software Architecture Concepts"

    def test_table_content_extracted(self, sample_docx_path):
        parser = DOCXParser()
        doc = parser.parse(sample_docx_path)
        # Table should be converted to pipe-delimited text
        assert "Microservices" in doc.content
        assert "Scalability" in doc.content

    def test_file_not_found_raises_error(self):
        parser = DOCXParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.docx"))


class TestDocumentManager:
    """Test the DocumentManager multi-format routing."""

    def test_supported_extensions(self):
        manager = DocumentManager()
        exts = manager.supported_extensions
        assert ".pdf" in exts
        assert ".txt" in exts
        assert ".md" in exts
        assert ".docx" in exts

    def test_routes_txt_to_text_parser(self, sample_txt_path):
        manager = DocumentManager()
        parser = manager.get_parser(sample_txt_path)
        assert isinstance(parser, TextParser)

    def test_routes_md_to_text_parser(self, sample_md_path):
        manager = DocumentManager()
        parser = manager.get_parser(sample_md_path)
        assert isinstance(parser, TextParser)

    def test_routes_docx_to_docx_parser(self, sample_docx_path):
        manager = DocumentManager()
        parser = manager.get_parser(sample_docx_path)
        assert isinstance(parser, DOCXParser)

    def test_returns_none_for_unsupported(self, tmp_path):
        manager = DocumentManager()
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("hello")
        assert manager.get_parser(unsupported) is None

    def test_parse_txt_document(self, sample_txt_path):
        manager = DocumentManager()
        doc = manager.parse_document(sample_txt_path)
        assert isinstance(doc, Document)
        assert "Object-oriented" in doc.content

    def test_parse_md_document(self, sample_md_path):
        manager = DocumentManager()
        doc = manager.parse_document(sample_md_path)
        assert isinstance(doc, Document)
        assert "Design Patterns" in doc.content

    def test_parse_docx_document(self, sample_docx_path):
        manager = DocumentManager()
        doc = manager.parse_document(sample_docx_path)
        assert isinstance(doc, Document)
        assert "Layered Architecture" in doc.content

    def test_parse_unsupported_raises_error(self, tmp_path):
        manager = DocumentManager()
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("hello")
        with pytest.raises(ValueError, match="No parser available"):
            manager.parse_document(unsupported)

    def test_parse_directory(self, fixtures_dir):
        manager = DocumentManager()
        docs = manager.parse_directory(fixtures_dir)
        # Should parse sample.txt, sample.md, and sample.docx
        assert len(docs) >= 3
        file_types = {doc.metadata.file_type for doc in docs}
        assert "txt" in file_types
        assert "md" in file_types
        assert "docx" in file_types
