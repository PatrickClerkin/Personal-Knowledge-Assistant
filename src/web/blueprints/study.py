"""
Study API blueprint — study paths, quiz generation, document summarisation.
"""

from flask import Blueprint, request, jsonify

from . import shared
from ...knowledge_graph.graph_store import GraphStore
from ...study.path_generator import PathGenerator
from ...study.quiz_generator import QuizGenerator
from ...study.summariser import DocumentSummariser
from ...utils.logger import get_logger

logger = get_logger(__name__)

study_bp = Blueprint("study", __name__)


@study_bp.route("/api/study/generate", methods=["POST"])
def generate_study_path():
    """Generate a personalised study path for a topic.

    Request body:
        {"topic": "neural networks"}

    Returns:
        StudyPath as JSON with ordered sections, summaries, sources.
    """
    rag = shared.get_rag()
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
            knowledge_base=shared.get_kb(),
            llm_provider=rag.llm,
            graph=graph,
        )
        path = generator.generate(topic)
        return jsonify(path.to_dict())

    except Exception as e:
        logger.error("Study path generation error: %s", e)
        return jsonify({"error": str(e)}), 500


@study_bp.route("/api/quiz/generate", methods=["POST"])
def generate_quiz():
    """Generate a quiz on a topic from knowledge base content.

    Request body:
        {"topic": "neural networks"}

    Returns:
        Quiz as JSON with MCQ and short answer questions.
    """
    rag = shared.get_rag()
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
            knowledge_base=shared.get_kb(),
            llm_provider=rag.llm,
        )
        quiz = generator.generate(topic)
        return jsonify(quiz.to_dict())
    except Exception as e:
        logger.error("Quiz generation error: %s", e)
        return jsonify({"error": str(e)}), 500


@study_bp.route("/api/summarise/all")
def summarise_all():
    """Summarise all documents in the knowledge base."""
    rag = shared.get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Summarisation requires an API key."
        }), 503

    try:
        summariser = DocumentSummariser(
            knowledge_base=shared.get_kb(),
            llm_provider=rag.llm,
        )
        corpus = summariser.summarise_corpus()
        return jsonify(corpus.to_dict())
    except Exception as e:
        logger.error("Corpus summarisation error: %s", e)
        return jsonify({"error": str(e)}), 500


@study_bp.route("/api/summarise/<doc_id>")
def summarise_document(doc_id):
    """Summarise a single document by doc_id."""
    rag = shared.get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Summarisation requires an API key."
        }), 503

    try:
        summariser = DocumentSummariser(
            knowledge_base=shared.get_kb(),
            llm_provider=rag.llm,
        )
        summary = summariser.summarise_document(doc_id)
        if summary is None:
            return jsonify({
                "error": f"No chunks found for document: {doc_id}"
            }), 404
        return jsonify(summary.to_dict())
    except Exception as e:
        logger.error("Summarisation error: %s", e)
        return jsonify({"error": str(e)}), 500