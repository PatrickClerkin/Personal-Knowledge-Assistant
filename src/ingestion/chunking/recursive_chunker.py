"""
Recursive Hierarchical Chunker

This chunker creates a hierarchical tree of chunks at multiple granularity
levels. It uses document structure (sections, paragraphs) combined with
semantic similarity to create a multi-resolution representation.

This is particularly useful for:
- Complex documents with nested structure
- Retrieval at different granularity levels
- Supporting both broad and specific queries
- Maintaining document coherence at multiple scales
"""

import re
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass
from .base_chunker import BaseChunker
from .chunk import Chunk
from ..document import Document, DocumentSection


@dataclass
class HierarchyNode:
    """Represents a node in the chunk hierarchy."""
    content: str
    level: int
    start_char: int
    end_char: int
    children: List['HierarchyNode']
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RecursiveHierarchicalChunker(BaseChunker):
    """
    Creates hierarchical chunks that preserve document structure.
    
    Algorithm:
    1. Parse document structure (sections, paragraphs)
    2. Build hierarchy tree from structure
    3. At each level, split content using semantic boundaries
    4. Create parent-child relationships between chunks
    5. Embed each level for multi-resolution retrieval
    
    The resulting chunks have:
    - hierarchy_level: 0 = document, 1 = section, 2 = paragraph, 3 = sentence
    - parent_chunk_id: Reference to parent chunk
    - child_chunk_ids: References to child chunks
    - coherence_score: How semantically coherent this chunk is
    """
    
    METHOD_NAME = "recursive_hierarchical"
    
    # Hierarchy level names
    LEVEL_DOCUMENT = 0
    LEVEL_SECTION = 1
    LEVEL_PARAGRAPH = 2
    LEVEL_SENTENCE = 3
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_depth: int = 3,
        min_chunk_size: int = 50,
        target_levels: List[int] = None,
        use_semantic_merging: bool = True,
        embedding_model: Optional[str] = None,
        coherence_threshold: float = 0.6,
    ):
        """
        Initialize the recursive hierarchical chunker.
        
        Args:
            chunk_size: Target size for leaf chunks
            chunk_overlap: Overlap between chunks at same level
            max_depth: Maximum hierarchy depth (0=doc, 1=section, 2=para, 3=sent)
            min_chunk_size: Minimum content size for a chunk
            target_levels: Which levels to actually output (None = all)
            use_semantic_merging: Use embeddings to merge similar chunks
            embedding_model: Model for semantic analysis
            coherence_threshold: Minimum coherence to avoid splitting
        """
        super().__init__(chunk_size, chunk_overlap)
        self.max_depth = max_depth
        self.min_chunk_size = min_chunk_size
        self.target_levels = target_levels or [1, 2, 3]  # Default: section + para + sentence
        self.use_semantic_merging = use_semantic_merging
        self.coherence_threshold = coherence_threshold
        
        self._embedding_model = None
        self._model_name = embedding_model or "all-MiniLM-L6-v2"
        
        # Patterns for structure detection
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._model_name)
        return self._embedding_model
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Create hierarchical chunks from document.
        
        Returns flattened list of chunks at requested levels,
        with hierarchy relationships preserved via IDs.
        """
        doc_id = document.metadata.doc_id
        doc_title = document.metadata.title
        
        # Build hierarchy tree
        root = self._build_hierarchy(document)
        
        # Convert tree to flat chunk list with relationships
        chunks = self._flatten_hierarchy(
            node=root,
            doc_id=doc_id,
            doc_title=doc_title,
            document=document,
            parent_id=None
        )
        
        # Filter to requested levels
        if self.target_levels:
            chunks = [c for c in chunks if c.hierarchy_level in self.target_levels]
        
        # Optionally compute coherence scores
        if self.use_semantic_merging:
            self._compute_coherence_scores(chunks)
        
        return self._finalize_chunks(chunks)
    
    def _build_hierarchy(self, document: Document) -> HierarchyNode:
        """
        Build a hierarchy tree from document structure.
        """
        # Root node (entire document)
        root = HierarchyNode(
            content=document.content,
            level=self.LEVEL_DOCUMENT,
            start_char=0,
            end_char=len(document.content),
            children=[]
        )
        
        if self.max_depth < 1:
            return root
        
        # Level 1: Sections (use document sections if available, else split by structure)
        if document.sections:
            section_nodes = self._sections_to_nodes(document.sections)
        else:
            section_nodes = self._split_into_sections(document.content)
        
        root.children = section_nodes
        
        if self.max_depth < 2:
            return root
        
        # Level 2: Paragraphs within sections
        for section_node in root.children:
            paragraph_nodes = self._split_into_paragraphs(
                section_node.content,
                section_node.start_char
            )
            section_node.children = paragraph_nodes
            
            if self.max_depth < 3:
                continue
            
            # Level 3: Sentences within paragraphs
            for para_node in section_node.children:
                sentence_nodes = self._split_into_sentences(
                    para_node.content,
                    para_node.start_char
                )
                para_node.children = sentence_nodes
        
        return root
    
    def _sections_to_nodes(self, sections: List[DocumentSection]) -> List[HierarchyNode]:
        """Convert DocumentSection objects to hierarchy nodes."""
        nodes = []
        for section in sections:
            node = HierarchyNode(
                content=section.content,
                level=self.LEVEL_SECTION,
                start_char=section.start_char,
                end_char=section.end_char,
                children=[],
                section_id=section.section_id,
                page_number=section.page_number
            )
            nodes.append(node)
        return nodes
    
    def _split_into_sections(self, text: str) -> List[HierarchyNode]:
        """
        Split text into section-like chunks based on structure.
        
        Looks for patterns like headers, large whitespace gaps,
        or significant topic shifts.
        """
        # Simple heuristic: split at multiple newlines or header patterns
        header_pattern = re.compile(
            r'(?:^|\n{2,})'
            r'(?:'
            r'(?:[A-Z][A-Za-z\s]+:)|'  # Title case with colon
            r'(?:\d+\.\s+[A-Z])|'       # Numbered sections
            r'(?:[A-Z]{2,}[A-Z\s]+)|'   # All caps
            r'(?:#+ )'                   # Markdown headers
            r')'
        )
        
        nodes = []
        current_start = 0
        
        for match in header_pattern.finditer(text):
            if match.start() > current_start:
                content = text[current_start:match.start()].strip()
                if len(content) >= self.min_chunk_size:
                    nodes.append(HierarchyNode(
                        content=content,
                        level=self.LEVEL_SECTION,
                        start_char=current_start,
                        end_char=match.start(),
                        children=[]
                    ))
            current_start = match.start()
        
        # Handle remaining content
        if current_start < len(text):
            content = text[current_start:].strip()
            if content:
                nodes.append(HierarchyNode(
                    content=content,
                    level=self.LEVEL_SECTION,
                    start_char=current_start,
                    end_char=len(text),
                    children=[]
                ))
        
        # If no sections found, treat whole text as one section
        if not nodes:
            nodes.append(HierarchyNode(
                content=text.strip(),
                level=self.LEVEL_SECTION,
                start_char=0,
                end_char=len(text),
                children=[]
            ))
        
        return nodes
    
    def _split_into_paragraphs(self, text: str, offset: int) -> List[HierarchyNode]:
        """Split text into paragraph nodes."""
        paragraphs = self.paragraph_pattern.split(text)
        nodes = []
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < self.min_chunk_size // 2:
                continue
            
            # Find actual position
            start = text.find(para, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(para)
            
            nodes.append(HierarchyNode(
                content=para,
                level=self.LEVEL_PARAGRAPH,
                start_char=offset + start,
                end_char=offset + end,
                children=[]
            ))
            
            current_pos = end
        
        # If no paragraphs (single block of text), use whole content
        if not nodes and text.strip():
            nodes.append(HierarchyNode(
                content=text.strip(),
                level=self.LEVEL_PARAGRAPH,
                start_char=offset,
                end_char=offset + len(text),
                children=[]
            ))
        
        return nodes
    
    def _split_into_sentences(self, text: str, offset: int) -> List[HierarchyNode]:
        """Split text into sentence nodes."""
        sentences = self.sentence_pattern.split(text)
        nodes = []
        current_pos = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            start = text.find(sent, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(sent)
            
            nodes.append(HierarchyNode(
                content=sent,
                level=self.LEVEL_SENTENCE,
                start_char=offset + start,
                end_char=offset + end,
                children=[]
            ))
            
            current_pos = end
        
        return nodes
    
    def _flatten_hierarchy(
        self,
        node: HierarchyNode,
        doc_id: str,
        doc_title: str,
        document: Document,
        parent_id: Optional[str],
        chunk_list: List[Chunk] = None,
        id_counter: List[int] = None
    ) -> List[Chunk]:
        """
        Convert hierarchy tree to flat chunk list with relationships.
        """
        if chunk_list is None:
            chunk_list = []
        if id_counter is None:
            id_counter = [0]
        
        # Create chunk for this node
        chunk_id = f"{doc_id}_h{node.level}_chunk_{id_counter[0]}"
        id_counter[0] += 1
        
        page_number = self._get_page_number_from_sections(
            node.start_char, document.sections
        ) if document.sections else node.page_number
        
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=node.content,
            source_doc_title=doc_title,
            section_id=node.section_id,
            page_number=page_number,
            start_char=node.start_char,
            end_char=node.end_char,
            chunk_index=len(chunk_list),
            chunking_method=self.METHOD_NAME,
            hierarchy_level=node.level,
            parent_chunk_id=parent_id,
            child_chunk_ids=[]
        )
        
        chunk_list.append(chunk)
        
        # Process children
        for child_node in node.children:
            child_chunks = self._flatten_hierarchy(
                node=child_node,
                doc_id=doc_id,
                doc_title=doc_title,
                document=document,
                parent_id=chunk_id,
                chunk_list=chunk_list,
                id_counter=id_counter
            )
            
            # Track child IDs
            for child_chunk in child_chunks:
                if child_chunk.parent_chunk_id == chunk_id:
                    chunk.child_chunk_ids.append(child_chunk.chunk_id)
        
        return chunk_list
    
    def _get_page_number_from_sections(
        self, char_pos: int, sections: List[DocumentSection]
    ) -> Optional[int]:
        """Find page number from document sections."""
        for section in sections:
            if section.start_char <= char_pos < section.end_char:
                return section.page_number
        return None
    
    def _compute_coherence_scores(self, chunks: List[Chunk]) -> None:
        """
        Compute semantic coherence score for each chunk.
        
        Coherence is measured as the average similarity between
        all sentence pairs within the chunk.
        """
        for chunk in chunks:
            if len(chunk.content) < 100:
                chunk.coherence_score = 1.0
                continue
            
            # Split into sentences
            sentences = self.sentence_pattern.split(chunk.content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                chunk.coherence_score = 1.0
                continue
            
            # Embed sentences
            embeddings = self.embedding_model.encode(
                sentences, convert_to_numpy=True, show_progress_bar=False
            )
            
            # Compute average pairwise similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-10)
            
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(normalized[i], normalized[j])
                    similarities.append(sim)
            
            chunk.coherence_score = float(np.mean(similarities)) if similarities else 1.0
    
    def get_hierarchy_summary(self, document: Document) -> Dict[str, Any]:
        """
        Get a summary of the document hierarchy for analysis.
        """
        root = self._build_hierarchy(document)
        
        def count_nodes(node: HierarchyNode) -> Dict[int, int]:
            counts = {node.level: 1}
            for child in node.children:
                child_counts = count_nodes(child)
                for level, count in child_counts.items():
                    counts[level] = counts.get(level, 0) + count
            return counts
        
        counts = count_nodes(root)
        
        return {
            "total_length": len(document.content),
            "level_counts": {
                "document": counts.get(0, 0),
                "sections": counts.get(1, 0),
                "paragraphs": counts.get(2, 0),
                "sentences": counts.get(3, 0),
            },
            "max_depth": self.max_depth,
            "target_levels": self.target_levels
        }


class MultiResolutionChunker(RecursiveHierarchicalChunker):
    """
    Convenience class that creates chunks at multiple resolutions.
    
    Returns chunks at three granularities:
    - Coarse (section/topic level) for broad context
    - Medium (paragraph level) for typical retrieval
    - Fine (sentence level) for precise matching
    """
    
    METHOD_NAME = "multi_resolution"
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_depth=3,
            target_levels=[1, 2, 3],  # Section, paragraph, sentence
            use_semantic_merging=True,
            embedding_model=embedding_model
        )
    
    def get_coarse_chunks(self, document: Document) -> List[Chunk]:
        """Get only section-level chunks."""
        return [c for c in self.chunk_document(document) 
                if c.hierarchy_level == self.LEVEL_SECTION]
    
    def get_medium_chunks(self, document: Document) -> List[Chunk]:
        """Get only paragraph-level chunks."""
        return [c for c in self.chunk_document(document)
                if c.hierarchy_level == self.LEVEL_PARAGRAPH]
    
    def get_fine_chunks(self, document: Document) -> List[Chunk]:
        """Get only sentence-level chunks."""
        return [c for c in self.chunk_document(document)
                if c.hierarchy_level == self.LEVEL_SENTENCE]
