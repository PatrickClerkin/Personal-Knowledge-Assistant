"""
Knowledge graph construction from ingested document chunks.

Extracts named entities using spaCy and builds co-occurrence
relationships between entities that appear in the same chunk.
The resulting graph shows how concepts connect across the corpus.

Design Pattern: Builder Pattern — GraphBuilder incrementally
constructs the graph from chunks without holding state between
calls.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

import networkx as nx

from ..utils.logger import get_logger

logger = get_logger(__name__)

# spaCy entity labels we care about
_RELEVANT_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "NORP", "FAC",
}

# Minimum character length for an entity to be included
_MIN_ENTITY_LENGTH = 3


@dataclass
class EntityNode:
    """A node in the knowledge graph representing a named entity.

    Attributes:
        name: Normalised entity name (lowercase).
        label: spaCy entity type (e.g. ORG, PERSON).
        count: Number of chunks this entity appears in.
        sources: Set of source document titles.
    """
    name: str
    label: str
    count: int = 1
    sources: Set[str] = field(default_factory=set)


@dataclass
class RelationEdge:
    """An edge between two entities that co-occur in a chunk.

    Attributes:
        source: Name of the first entity.
        target: Name of the second entity.
        weight: Number of chunks they co-occur in.
        sources: Set of source document titles where they co-occur.
    """
    source: str
    target: str
    weight: int = 1
    sources: Set[str] = field(default_factory=set)


class GraphBuilder:
    """Builds a knowledge graph from document chunks using NER.

    Entities are nodes; co-occurrence within the same chunk creates
    or strengthens an edge between them. Edge weight reflects how
    often two entities appear together across the corpus.

    Usage:
        builder = GraphBuilder()
        graph = builder.build(chunks)
    """

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for graph building.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load spaCy model: {e}. "
                    "Run: python -m spacy download en_core_web_sm"
                )
        return self._nlp

    def build(self, chunks) -> nx.Graph:
        """Build a knowledge graph from a list of chunks.

        Args:
            chunks: List of Chunk objects with .content and
                .source_doc_title attributes.

        Returns:
            A NetworkX Graph with entity nodes and co-occurrence edges.
        """
        graph = nx.Graph()
        nodes: Dict[str, EntityNode] = {}
        edges: Dict[Tuple[str, str], RelationEdge] = {}

        for chunk in chunks:
            entities = self._extract_entities(
                chunk.content, chunk.source_doc_title
            )

            # Add/update nodes
            for name, label, source in entities:
                if name in nodes:
                    nodes[name].count += 1
                    nodes[name].sources.add(source)
                else:
                    nodes[name] = EntityNode(
                        name=name, label=label, sources={source}
                    )

            # Add/update edges for all entity pairs in this chunk
            entity_names = list({e[0] for e in entities})
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    a, b = sorted([entity_names[i], entity_names[j]])
                    key = (a, b)
                    source = chunk.source_doc_title
                    if key in edges:
                        edges[key].weight += 1
                        edges[key].sources.add(source)
                    else:
                        edges[key] = RelationEdge(
                            source=a, target=b, sources={source}
                        )

        # Populate graph
        for name, node in nodes.items():
            graph.add_node(
                name,
                label=node.label,
                count=node.count,
                sources=list(node.sources),
            )

        for (a, b), edge in edges.items():
            graph.add_edge(
                a, b,
                weight=edge.weight,
                sources=list(edge.sources),
            )

        logger.info(
            "Graph built: %d nodes, %d edges from %d chunks.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(chunks),
        )
        return graph

    def _extract_entities(
        self,
        text: str,
        source: str,
    ) -> List[Tuple[str, str, str]]:
        """Extract relevant named entities from text.

        Returns:
            List of (normalised_name, label, source) tuples.
        """
        try:
            doc = self.nlp(text[:5000])  # cap for performance
        except Exception as e:
            logger.warning("NER failed for chunk: %s", e)
            return []

        seen = set()
        results = []
        for ent in doc.ents:
            if ent.label_ not in _RELEVANT_LABELS:
                continue
            name = ent.text.strip().lower()
            name = re.sub(r"\s+", " ", name)
            if len(name) < _MIN_ENTITY_LENGTH:
                continue
            if name in seen:
                continue
            seen.add(name)
            results.append((name, ent.label_, source))

        return results