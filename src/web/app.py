"""
Flask REST API for the Personal Knowledge Assistant.

Provides HTTP endpoints for document ingestion, semantic search,
knowledge base management, and RAG-powered Q&A.

Endpoints:
    GET  /api/health          - Health check
    GET  /api/info            - Index statistics
    POST /api/search          - Semantic search
    POST /api/ingest          - Ingest a document (file upload)
    POST /api/chat            - RAG-powered question answering
    GET  /api/documents       - List indexed documents
    DELETE /api/documents/<id> - Delete a document

Usage:
    python -m src.web.app
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

from ..ingestion.knowledge_base import KnowledgeBase
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Resolve template and static directories relative to this file
_WEB_DIR = Path(__file__).parent
_TEMPLATE_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

app = Flask(
    __name__,
    template_folder=str(_TEMPLATE_DIR),
    static_folder=str(_STATIC_DIR),
)

# Configuration
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB limit

# Lazy-initialised knowledge base
_kb = None
_rag = None


def get_kb() -> KnowledgeBase:
    """Get or create the KnowledgeBase singleton."""
    global _kb
    if _kb is None:
        index_path = os.environ.get("KB_INDEX_PATH", "data/index/default")
        strategy = os.environ.get("KB_STRATEGY", "sentence")
        _kb = KnowledgeBase(
            index_path=index_path,
            chunk_strategy=strategy,
        )
        logger.info("KnowledgeBase initialised: %d chunks", _kb.size)
    return _kb


def get_rag():
    """Get or create the RAG pipeline (if API key is set)."""
    global _rag
    if _rag is None and os.environ.get("ANTHROPIC_API_KEY"):
        from ..rag.llm import ClaudeProvider
        from ..rag.pipeline import RAGPipeline
        _rag = RAGPipeline(
            knowledge_base=get_kb(),
            llm_provider=ClaudeProvider(),
        )
        logger.info("RAG pipeline initialised.")
    return _rag


# ─── Frontend ────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


# ─── API Endpoints ──────────────────────────────────────────────────

@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/api/info")
def info():
    """Return index statistics."""
    kb = get_kb()
    return jsonify({
        "total_chunks": kb.size,
        "num_documents": len(kb.document_ids),
        "supported_formats": kb.supported_formats,
        "document_ids": kb.document_ids,
    })


@app.route("/api/search", methods=["POST"])
def search():
    """Semantic search endpoint.

    Request body:
        {"query": "...", "top_k": 5, "rerank": false, "expand": null}

    Returns:
        {"results": [{"rank": 1, "score": 0.85, "content": "...", ...}]}
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    kb = get_kb()
    query = data["query"]
    top_k = data.get("top_k", 5)
    rerank = data.get("rerank", False)
    expand = data.get("expand", None)

    if rerank or expand:
        results = kb.advanced_search(
            query, top_k=top_k, rerank=rerank, expand_query=expand,
        )
    else:
        results = kb.search(query, top_k=top_k)

    return jsonify({
        "query": query,
        "results": [
            {
                "rank": r.rank,
                "score": round(r.score, 4),
                "content": r.chunk.content,
                "source": r.chunk.source_doc_title,
                "page": r.chunk.page_number,
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
            }
            for r in results
        ],
    })


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """Ingest a document via file upload.

    Accepts multipart/form-data with a 'file' field.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    save_path = UPLOAD_DIR / file.filename
    file.save(str(save_path))

    kb = get_kb()
    try:
        n_chunks = kb.ingest(save_path)
        return jsonify({
            "filename": file.filename,
            "chunks_created": n_chunks,
            "total_chunks": kb.size,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Ingest error: %s", e)
        return jsonify({"error": f"Failed to ingest: {e}"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """RAG-powered question answering.

    Request body:
        {"question": "...", "top_k": 5}

    Returns:
        {"answer": "...", "sources": [...]}
    """
    rag = get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Chat requires an API key."
        }), 503

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        response = rag.query(
            data["question"],
            top_k=data.get("top_k", 5),
        )
        return jsonify({
            "answer": response.answer,
            "sources": [
                {
                    "source": r.chunk.source_doc_title,
                    "page": r.chunk.page_number,
                    "score": round(r.score, 4),
                    "content": r.chunk.content[:200],
                }
                for r in response.sources
            ],
            "usage": response.llm_response.usage,
        })
    except Exception as e:
        logger.error("Chat error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents")
def list_documents():
    """List all indexed documents with chunk counts."""
    kb = get_kb()
    documents = []
    for doc_id in kb.document_ids:
        chunks = kb.get_document_chunks(doc_id)
        documents.append({
            "doc_id": doc_id,
            "title": chunks[0].source_doc_title if chunks else "Unknown",
            "num_chunks": len(chunks),
        })
    return jsonify({"documents": documents})


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete a document from the index."""
    kb = get_kb()
    deleted = kb.delete_document(doc_id)
    if deleted > 0:
        return jsonify({"deleted_chunks": deleted, "doc_id": doc_id})
    return jsonify({"error": "Document not found"}), 404


def create_app() -> Flask:
    """Application factory for the Flask app."""
    return app


if __name__ == "__main__":
    app.run(debug=True, port=5000)
