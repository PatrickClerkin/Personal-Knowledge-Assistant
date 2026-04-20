"""
Core API blueprint — health, info, frontend routes, document management.
"""

from flask import Blueprint, jsonify, render_template

from . import shared

core_bp = Blueprint("core", __name__)


# ─── Frontend Routes ────────────────────────────────────────────────

@core_bp.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@core_bp.route("/dashboard")
def dashboard():
    """Serve the evaluation dashboard page."""
    return render_template("dashboard.html")


@core_bp.route("/graph")
def graph_view():
    """Serve the knowledge graph visualisation page."""
    return render_template("graph.html")


@core_bp.route("/study")
def study_view():
    """Serve the study mode page."""
    return render_template("study.html")


@core_bp.route("/quiz")
def quiz_view():
    """Serve the quiz mode page."""
    return render_template("quiz.html")


@core_bp.route("/conflicts")
def conflicts_view():
    """Serve the conflict detection page."""
    return render_template("conflicts.html")


# ─── API Endpoints ──────────────────────────────────────────────────

@core_bp.route("/api/health")
def health():
    """Health check endpoint with system info."""
    import sys
    kb = shared.get_kb()
    return jsonify({
        "status": "ok",
        "python_version": sys.version.split()[0],
        "index_size": kb.size,
        "num_documents": len(kb.document_ids),
    })


@core_bp.route("/api/info")
def info():
    """Return index statistics."""
    kb = shared.get_kb()
    return jsonify({
        "total_chunks": kb.size,
        "num_documents": len(kb.document_ids),
        "supported_formats": kb.supported_formats,
        "document_ids": kb.document_ids,
    })


@core_bp.route("/api/documents")
def list_documents():
    """List all indexed documents with chunk counts."""
    kb = shared.get_kb()
    documents = []
    for doc_id in kb.document_ids:
        chunks = kb.get_document_chunks(doc_id)
        doc_info = kb.get_document_info(doc_id)
        doc = {
            "doc_id": doc_id,
            "title": chunks[0].source_doc_title if chunks else "Unknown",
            "chunk_count": len(chunks),
        }
        if doc_info:
            doc["ingested_at"] = doc_info.ingested_at
            doc["updated_at"] = doc_info.updated_at
        documents.append(doc)
    return jsonify({"documents": documents})


@core_bp.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete a document from the index."""
    kb = shared.get_kb()
    deleted = kb.delete_document(doc_id)
    if deleted > 0:
        return jsonify({"deleted_chunks": deleted, "doc_id": doc_id})
    return jsonify({"error": "Document not found"}), 404