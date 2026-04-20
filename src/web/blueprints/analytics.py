"""
Analytics API blueprint — query history, analytics, annotations.
"""

from flask import Blueprint, request, jsonify

from . import shared
from ...utils.logger import get_logger

logger = get_logger(__name__)

analytics_bp = Blueprint("analytics", __name__)


# ─── Query History & Analytics ──────────────────────────────────────

@analytics_bp.route("/api/history")
def get_history():
    """Return recent query history.

    Query params:
        n: Number of recent records to return (default 20).

    Returns:
        List of recent QueryRecord dicts, most recent first.
    """
    rag = shared.get_rag()
    if rag is None:
        return jsonify({"records": [], "total": 0})

    n = int(request.args.get("n", 20))
    records = rag.history.get_recent(n)
    return jsonify({
        "total": rag.history.total_queries,
        "records": [r.to_dict() for r in records],
    })


@analytics_bp.route("/api/analytics")
def get_analytics():
    """Return aggregated query analytics.

    Returns:
        Analytics dict with confidence distribution, cache hit rate,
        token usage, top sources, and queries by day.
    """
    rag = shared.get_rag()
    if rag is None:
        return jsonify({"error": "RAG pipeline not initialised"}), 503

    return jsonify(rag.history.get_analytics())


# ─── Annotations ────────────────────────────────────────────────────

@analytics_bp.route("/api/annotations/stats")
def annotation_stats():
    """Return annotation statistics."""
    return jsonify(shared.get_annotations().get_stats())


@analytics_bp.route("/api/annotations", methods=["POST"])
def add_annotation():
    """Add a note to a chunk.

    Request body:
        {
            "chunk_id": "...",
            "doc_id": "...",
            "source_title": "...",
            "note": "...",
            "chunk_preview": "...",
            "page_number": 3,
            "tags": ["important", "review"]
        }

    Returns:
        The created Annotation as JSON.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    for f in ["chunk_id", "doc_id", "source_title", "note"]:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400

    note = data["note"].strip()
    if not note:
        return jsonify({"error": "Note cannot be empty"}), 400

    try:
        annotation = shared.get_annotations().add(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source_title=data["source_title"],
            note=note,
            chunk_preview=data.get("chunk_preview", ""),
            page_number=data.get("page_number"),
            tags=data.get("tags", []),
        )
        return jsonify(annotation.to_dict()), 201
    except Exception as e:
        logger.error("Annotation error: %s", e)
        return jsonify({"error": str(e)}), 500


@analytics_bp.route("/api/annotations")
def list_annotations():
    """Return recent annotations.

    Query params:
        limit: Number of annotations to return (default 50).
        doc_id: Filter by document ID.
        tag: Filter by tag.
        q: Free-text search query.

    Returns:
        {"annotations": [...], "total": N}
    """
    store = shared.get_annotations()
    doc_id = request.args.get("doc_id")
    tag = request.args.get("tag")
    query = request.args.get("q")
    limit = int(request.args.get("limit", 50))

    if doc_id:
        results = store.get_by_doc(doc_id)
    elif tag:
        results = store.get_by_tag(tag)
    elif query:
        results = store.search(query)
    else:
        results = store.get_all(limit=limit)

    return jsonify({
        "annotations": [a.to_dict() for a in results[:limit]],
        "total": store.total_annotations,
    })


@analytics_bp.route("/api/annotations/<annotation_id>", methods=["DELETE"])
def delete_annotation(annotation_id):
    """Delete an annotation by ID."""
    deleted = shared.get_annotations().delete(annotation_id)
    if deleted:
        return jsonify({"deleted": True, "annotation_id": annotation_id})
    return jsonify({"error": "Annotation not found"}), 404