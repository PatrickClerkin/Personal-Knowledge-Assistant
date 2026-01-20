from typing import List, Optional
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document


class SemanticChunker(BaseChunker):
    """
    Structure-aware chunker that respects document boundaries (pages/sections).

    This chunker splits documents by their natural structure (pages, sections)
    rather than arbitrary character positions, preserving semantic coherence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
        embedding_model=None
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Maximum chunk size (soft limit)
            chunk_overlap: Overlap in characters between chunks
            similarity_threshold: Reserved for future embedding-based chunking
            embedding_model: Reserved for future embedding-based chunking
        """
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document by structure (pages/sections).

        Respects document boundaries so chunks don't cut across
        pages or sections unnecessarily.
        """
        return self._chunk_by_structure(document)

    def _chunk_by_structure(self, document: Document) -> List[Chunk]:
        """
        Chunk by document structure (sections/pages).

        Keeps sections whole when possible, only splitting when
        a section exceeds the chunk_size limit.
        """
        chunks = []
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        chunk_index = 0

        for section in document.sections:
            section_text = section.content.strip()

            if not section_text:
                continue

            # If section fits in one chunk, keep it whole
            if len(section_text) <= self.chunk_size:
                chunk = self._create_chunk(
                    content=section_text,
                    doc_id=doc_id,
                    source_doc_title=doc_title,
                    chunk_index=chunk_index,
                    start_char=section.start_char,
                    end_char=section.end_char,
                    section_id=section.section_id,
                    page_number=section.page_number
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Section is too large, split it carefully
                start = 0
                while start < len(section_text):
                    end = min(start + self.chunk_size, len(section_text))

                    # Try not to cut words
                    if end < len(section_text):
                        last_space = section_text.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space

                    chunk_content = section_text[start:end].strip()

                    if chunk_content:
                        chunk = self._create_chunk(
                            content=chunk_content,
                            doc_id=doc_id,
                            source_doc_title=doc_title,
                            chunk_index=chunk_index,
                            start_char=section.start_char + start,
                            end_char=section.start_char + end,
                            section_id=section.section_id,
                            page_number=section.page_number
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                    start = end - self.chunk_overlap

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _get_page_number(self, char_position: int, document: Document) -> Optional[int]:
        """Determine which page a character position belongs to."""
        for section in document.sections:
            if section.start_char <= char_position < section.end_char:
                return section.page_number
        return None
