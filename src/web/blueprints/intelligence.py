"""
Intelligence API blueprint — knowledge graph, conflicts, similarity, evaluation.
"""

from pathlib import Path

from flask import Blueprint, request, jsonify

from . import shared
from ...ingestion.similarity import DocumentSimilarity
from ...evaluation.evaluator import EvaluationSuite
from ...knowledge_graph.graph_builder import GraphBuilder
from ...knowledge_graph.graph_store import GraphStore
from ...rag.conflict_detector import ConflictDetector
from ...utils.logger import get_logger

logger = get_logger(__name__)

intelligence_bp = Blueprint("intelligence", __name__)

# Restrict evaluation query file paths to this directory
_EVAL_BASE_DIR = Path("data/eval").resolve()


@intelligence_bp.route("/api/graph")
def get_graph():
    """Build or return the cached knowledge graph as JSON.

    Query params:
        rebuild: If 'true', forces a full rebuild from the index.
    """
    store = GraphStore()
    rebuild = request.args.get("rebuild", "false").lower() == "true"

    if rebuild or not store.exists():
        kb = shared.get_kb()
        all_chunks = []
        for doc_id in kb.document_ids:
            all_chunks.extend(kb.get_document_chunks(doc_id))

        if not all_chunks:
            return jsonify({
                "nodes": [], "links": [], "num_nodes": 0, "num_edges": 0
            })

        builder = GraphBuilder()
        graph = builder.build(all_chunks)
        store.save(graph)
    else:
        graph = store.load()
        if graph is None:
            return jsonify({
                "nodes": [], "links": [], "num_nodes": 0, "num_edges": 0
            })

    return jsonify(store.to_dict(graph))


@intelligence_bp.route("/api/conflicts/detect", methods=["POST"])
def detect_conflicts():
    """Detect contradictions between documents on a topic.

    Request body:
        {"topic": "neural networks"}

    Returns:
        ConflictReport as JSON with all detected conflicts.
    """
    rag = shared.get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. "
                     "Conflict detection requires an API key."
        }), 503

    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' field"}), 400

    topic = data["topic"].strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400

    try:
        detector = ConflictDetector(
            knowledge_base=shared.get_kb(),
            llm_provider=rag.llm,
        )
        report = detector.detect(topic)
        return jsonify(report.to_dict())
    except Exception as e:
        logger.error("Conflict detection error: %s", e)
        return jsonify({"error": str(e)}), 500


@intelligence_bp.route("/api/similarity/matrix")
def similarity_matrix():
    """Compute full pairwise similarity matrix for all documents."""
    kb = shared.get_kb()
    try:
        ds = DocumentSimilarity(knowledge_base=kb)
        matrix = ds.compute_matrix()
        return jsonify(matrix.to_dict())
    except Exception as e:
        logger.error("Similarity matrix error: %s", e)
        return jsonify({"error": str(e)}), 500


@intelligence_bp.route("/api/similarity/<doc_id>")
def document_similarity(doc_id):
    """Find documents most similar to a given document.

    Query params:
        top_k: Number of results to return (default 5).
    """
    kb = shared.get_kb()
    top_k = int(request.args.get("top_k", 5))

    try:
        ds = DocumentSimilarity(knowledge_base=kb)
        results = ds.find_similar(doc_id, top_k=top_k)
        ref_chunks = kb.get_document_chunks(doc_id)
        ref_title = ref_chunks[0].source_doc_title if ref_chunks else doc_id
        return jsonify({
            "doc_id": doc_id,
            "query_title": ref_title,
            "similar": [r.to_dict() for r in results],
        })
    except Exception as e:
        logger.error("Similarity error: %s", e)
        return jsonify({"error": str(e)}), 500


@intelligence_bp.route("/api/evaluation/run")
def run_evaluation():
    """Run IR evaluation and return metrics as JSON.

    Query params:
        path: Path to test queries JSON file
              (default: data/eval/test_queries.json)

    Path traversal protection: only files within data/eval/ are allowed.
    """
    queries_path = request.args.get("path", "data/eval/test_queries.json")

    # Prevent path traversal — resolve the path and check it stays
    # within the allowed evaluation directory.
    resolved = Path(queries_path).resolve()
    if not str(resolved).startswith(str(_EVAL_BASE_DIR)):
        return jsonify({
            "error": "Invalid path: evaluation files must be in data/eval/"
        }), 400

    kb = shared.get_kb()
    suite = EvaluationSuite(knowledge_base=kb, top_k=5)

    try:
        report = suite.run(queries_path)
        return jsonify(report.to_dict())
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Evaluation error: %s", e)
        return jsonify({"error": str(e)}), 500