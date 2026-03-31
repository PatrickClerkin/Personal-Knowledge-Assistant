"""
Persistence layer for the knowledge graph.

Serialises a NetworkX graph to JSON so it survives server restarts.
Supports incremental rebuilds by tracking which doc IDs are included.
"""

import json
from pathlib import Path
from typing import Optional, Set

import networkx as nx
from networkx.readwrite import json_graph

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GraphStore:
    """Persists and loads the knowledge graph as JSON.

    Attributes:
        path: File path where the graph JSON is stored.
    """

    def __init__(self, path: str = "data/graph/knowledge_graph.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, graph: nx.Graph) -> None:
        """Serialise and save the graph to disk.

        Args:
            graph: The NetworkX graph to persist.
        """
        data = json_graph.node_link_data(graph)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Graph saved: %d nodes, %d edges → %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self.path,
        )

    def load(self) -> Optional[nx.Graph]:
        """Load the graph from disk.

        Returns:
            The loaded NetworkX graph, or None if no file exists.
        """
        if not self.path.exists():
            logger.debug("No graph file found at %s", self.path)
            return None

        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        graph = json_graph.node_link_graph(data)
        logger.info(
            "Graph loaded: %d nodes, %d edges ← %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self.path,
        )
        return graph

    def exists(self) -> bool:
        """True if a persisted graph file exists."""
        return self.path.exists()

    def delete(self) -> None:
        """Delete the persisted graph file."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Graph file deleted: %s", self.path)

    def to_dict(self, graph: nx.Graph) -> dict:
        """Convert graph to a JSON-serialisable dict for the API.

        Returns nodes with degree/count and edges with weight,
        formatted for D3.js consumption.
        """
        nodes = []
        for node, attrs in graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": attrs.get("label", "UNKNOWN"),
                "count": attrs.get("count", 1),
                "sources": attrs.get("sources", []),
                "degree": graph.degree(node),
            })

        links = []
        for u, v, attrs in graph.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "weight": attrs.get("weight", 1),
                "sources": attrs.get("sources", []),
            })

        return {
            "nodes": nodes,
            "links": links,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }