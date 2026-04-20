"""
Search and ingest API blueprint — semantic search and document upload.
"""

from pathlib import Path

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from . import shared
from ...utils.logger import get_logger

logger = get_logger(__name__)

search_bp = Blueprint("search", __name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _allowed_file(filename: str) -> bool:
    """Check if the file extension is supported."""
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


@search_bp.route("/api/search", methods=["POST"])
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

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    kb = shared.get_kb()
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


@search_bp.route("/api/ingest", methods=["POST"])
def ingest():
    """Ingest a document via file upload.

    Accepts multipart/form-data with a 'file' field.
    Sanitises the filename to prevent path traversal attacks.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Sanitise the filename to prevent path traversal
    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    if not _allowed_file(safe_name):
        return jsonify({
            "error": f"Unsupported file type. Allowed: "
                     f"{', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        }), 400

    # Save uploaded file to the controlled upload directory
    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))

    kb = shared.get_kb()
    try:
        n_chunks = kb.ingest(save_path)
        return jsonify({
            "filename": safe_name,
            "chunks_created": n_chunks,
            "total_chunks": kb.size,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Ingest error: %s", e)
        return jsonify({"error": f"Failed to ingest: {e}"}), 500