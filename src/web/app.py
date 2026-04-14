"""
Flask REST API for the Personal Knowledge Assistant.

Provides HTTP endpoints for document ingestion, semantic search,
knowledge base management, and RAG-powered Q&A.

Endpoints:
    GET  /api/health                  - Health check
    GET  /api/info                    - Index statistics
    POST /api/search                  - Semantic search
    POST /api/ingest                  - Ingest a document (file upload)
    POST /api/chat                    - RAG-powered question answering
    GET  /api/documents               - List indexed documents
    DELETE /api/documents/<id>        - Delete a document
    GET  /dashboard                   - Evaluation dashboard UI
    GET  /api/evaluation/run          - Run IR evaluation and return metrics
    GET  /graph                       - Knowledge graph visualisation UI
    GET  /api/graph                   - Knowledge graph data as JSON
    GET  /study                       - Study mode UI
    POST /api/study/generate          - Generate a personalised study path
    GET  /quiz                        - Quiz mode UI
    POST /api/quiz/generate           - Generate a quiz on a topic
    GET  /conflicts                   - Conflict detection UI
    POST /api/conflicts/detect        - Detect contradictions between documents
    GET  /api/summarise/<doc_id>      - Summarise a single document
    GET  /api/summarise/all           - Summarise all documents in the corpus
    GET  /api/history                 - Return recent query history
    GET  /api/analytics               - Return aggregated query analytics
    GET  /api/similarity/<doc_id>     - Find documents similar to a given doc
    GET  /api/similarity/matrix       - Full pairwise similarity matrix
    POST /api/annotations             - Add a note to a chunk
    GET  /api/annotations             - List/search annotations
    DELETE /api/annotations/<id>      - Delete an annotation
    GET  /api/annotations/stats       - Annotation statistics

Usage:
    python -m src.web.app
"""

import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

from ..ingestion.knowledge_base import KnowledgeBase
from ..ingestion.similarity import DocumentSimilarity
from ..evaluation.evaluator import EvaluationSuite
from ..knowledge_graph.graph_builder import GraphBuilder
from ..knowledge_graph.graph_store import GraphStore
from ..study.path_generator import PathGenerator
from ..study.quiz_generator import QuizGenerator
from ..study.summariser import DocumentSummariser
from ..rag.conflict_detector import ConflictDetector
from ..rag.annotations import AnnotationStore
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

# Lazy-initialised singletons
_kb = None
_rag = None
_annotations = None


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


def get_annotations() -> AnnotationStore:
    """Get or create the AnnotationStore singleton."""
    global _annotations
    if _annotations is None:
        _annotations = AnnotationStore()
    return _annotations


def _grounding_label(score: float) -> str:
    """Compute a human-readable grounding label from a score."""
    if score >= 0.40:
        return "strong"
    if score >= 0.15:
        return "partial"
    return "weak"


# ─── Frontend ────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Serve the evaluation dashboard page."""
    return render_template("dashboard.html")


@app.route("/graph")
def graph_view():
    """Serve the knowledge graph visualisation page."""
    return render_template("graph.html")


@app.route("/study")
def study_view():
    """Serve the study mode page."""
    return render_template("study.html")


@app.route("/quiz")
def quiz_view():
    """Serve the quiz mode page."""
    return render_template("quiz.html")


@app.route("/conflicts")
def conflicts_view():
    """Serve the conflict detection page."""
    return render_template("conflicts.html")


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
        {"answer": "...", "sources": [...], "usage": {...},
         "confidence": 0.85, "cache_hit": false,
         "retrieval_attempts": 1, "grounding": [...],
         "verification": {...}}
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
                    "content": r.chunk.content[:300],
                }
                for r in response.sources
            ],
            "usage": response.llm_response.usage,
            "confidence": response.confidence,
            "cache_hit": response.cache_hit,
            "retrieval_attempts": response.retrieval_attempts,
            "grounding": [
                {
                    "source": c.chunk_source,
                    "page": c.page_number,
                    "grounding_score": c.grounding_score,
                    "label": _grounding_label(c.grounding_score),
                    "is_grounded": c.is_grounded,
                    "matched_terms": c.matched_terms,
                }
                for c in response.grounding.chunk_scores
            ] if response.grounding else [],
            "verification": response.verification.to_dict()
                if response.verification else None,
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


@app.route("/api/evaluation/run")
def run_evaluation():
    """Run IR evaluation and return metrics as JSON.

    Query params:
        path: Path to test queries JSON file
              (default: data/eval/test_queries.json)
    """
    queries_path = request.args.get("path", "data/eval/test_queries.json")
    kb = get_kb()
    suite = EvaluationSuite(knowledge_base=kb, top_k=5)

    try:
        report = suite.run(queries_path)
        return jsonify(report.to_dict())
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error("Evaluation error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph")
def get_graph():
    """Build or return the cached knowledge graph as JSON.

    Query params:
        rebuild: If 'true', forces a full rebuild from the index.
    """
    store = GraphStore()
    rebuild = request.args.get("rebuild", "false").lower() == "true"

    if rebuild or not store.exists():
        kb = get_kb()
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


@app.route("/api/study/generate", methods=["POST"])
def generate_study_path():
    """Generate a personalised study path for a topic.

    Request body:
        {"topic": "neural networks"}

    Returns:
        StudyPath as JSON with ordered sections, summaries, sources.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Study mode requires an API key."
        }), 503

    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' field"}), 400

    topic = data["topic"].strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400

    try:
        store = GraphStore()
        graph = store.load() if store.exists() else None

        generator = PathGenerator(
            knowledge_base=get_kb(),
            llm_provider=rag.llm,
            graph=graph,
        )
        path = generator.generate(topic)
        return jsonify(path.to_dict())

    except Exception as e:
        logger.error("Study path generation error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/quiz/generate", methods=["POST"])
def generate_quiz():
    """Generate a quiz on a topic from knowledge base content.

    Request body:
        {"topic": "neural networks"}

    Returns:
        Quiz as JSON with MCQ and short answer questions.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Quiz mode requires an API key."
        }), 503

    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' field"}), 400

    topic = data["topic"].strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400

    try:
        generator = QuizGenerator(
            knowledge_base=get_kb(),
            llm_provider=rag.llm,
        )
        quiz = generator.generate(topic)
        return jsonify(quiz.to_dict())
    except Exception as e:
        logger.error("Quiz generation error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/conflicts/detect", methods=["POST"])
def detect_conflicts():
    """Detect contradictions between documents on a topic.

    Request body:
        {"topic": "neural networks"}

    Returns:
        ConflictReport as JSON with all detected conflicts.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Conflict detection requires an API key."
        }), 503

    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' field"}), 400

    topic = data["topic"].strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400

    try:
        detector = ConflictDetector(
            knowledge_base=get_kb(),
            llm_provider=rag.llm,
        )
        report = detector.detect(topic)
        return jsonify(report.to_dict())
    except Exception as e:
        logger.error("Conflict detection error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarise/all")
def summarise_all():
    """Summarise all documents in the knowledge base.

    Returns:
        CorpusSummary as JSON with per-document summaries
        and a corpus-level overview.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Summarisation requires an API key."
        }), 503

    try:
        summariser = DocumentSummariser(
            knowledge_base=get_kb(),
            llm_provider=rag.llm,
        )
        corpus = summariser.summarise_corpus()
        return jsonify(corpus.to_dict())
    except Exception as e:
        logger.error("Corpus summarisation error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarise/<doc_id>")
def summarise_document(doc_id):
    """Summarise a single document by doc_id.

    Returns:
        DocumentSummary as JSON with executive summary,
        section summaries, and key points.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Summarisation requires an API key."
        }), 503

    try:
        summariser = DocumentSummariser(
            knowledge_base=get_kb(),
            llm_provider=rag.llm,
        )
        summary = summariser.summarise_document(doc_id)
        if summary is None:
            return jsonify({"error": f"No chunks found for document: {doc_id}"}), 404
        return jsonify(summary.to_dict())
    except Exception as e:
        logger.error("Summarisation error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def get_history():
    """Return recent query history.

    Query params:
        n: Number of recent records to return (default 20).

    Returns:
        List of recent QueryRecord dicts, most recent first.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({"records": [], "total": 0})

    n = int(request.args.get("n", 20))
    records = rag.history.get_recent(n)
    return jsonify({
        "total": rag.history.total_queries,
        "records": [r.to_dict() for r in records],
    })


@app.route("/api/analytics")
def get_analytics():
    """Return aggregated query analytics.

    Returns:
        Analytics dict with confidence distribution, cache hit rate,
        token usage, top sources, and queries by day.
    """
    rag = get_rag()
    if rag is None:
        return jsonify({"error": "RAG pipeline not initialised"}), 503

    return jsonify(rag.history.get_analytics())


@app.route("/api/similarity/matrix")
def similarity_matrix():
    """Compute full pairwise similarity matrix for all documents.

    Returns:
        SimilarityMatrix with document metadata and NxN scores.
    """
    kb = get_kb()
    try:
        ds = DocumentSimilarity(knowledge_base=kb)
        matrix = ds.compute_matrix()
        return jsonify(matrix.to_dict())
    except Exception as e:
        logger.error("Similarity matrix error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/similarity/<doc_id>")
def document_similarity(doc_id):
    """Find documents most similar to a given document.

    Query params:
        top_k: Number of results to return (default 5).

    Returns:
        List of SimilarityResult dicts sorted by similarity descending.
    """
    kb = get_kb()
    top_k = int(request.args.get("top_k", 5))

    try:
        ds = DocumentSimilarity(knowledge_base=kb)
        results = ds.find_similar(doc_id, top_k=top_k)
        ref_chunks = kb.get_document_chunks(doc_id)
        ref_title = ref_chunks[0].source_doc_title if ref_chunks else doc_id
        return jsonify({
            "doc_id": doc_id,
            "title": ref_title,
            "similar": [r.to_dict() for r in results],
        })
    except Exception as e:
        logger.error("Similarity error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/annotations/stats")
def annotation_stats():
    """Return annotation statistics.

    Returns:
        Stats dict with total, top tags, top documents.
    """
    return jsonify(get_annotations().get_stats())


@app.route("/api/annotations", methods=["POST"])
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
        annotation = get_annotations().add(
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


@app.route("/api/annotations")
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
    store = get_annotations()
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


@app.route("/api/annotations/<annotation_id>", methods=["DELETE"])
def delete_annotation(annotation_id):
    """Delete an annotation by ID.

    Returns:
        {"deleted": true} or 404.
    """
    deleted = get_annotations().delete(annotation_id)
    if deleted:
        return jsonify({"deleted": True, "annotation_id": annotation_id})
    return jsonify({"error": "Annotation not found"}), 404


def create_app() -> Flask:
    """Application factory for the Flask app."""
    return app


if __name__ == "__main__":
    app.run(debug=True, port=5000)