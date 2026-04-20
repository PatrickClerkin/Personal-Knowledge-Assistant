"""
Chat API blueprint — RAG-powered Q&A, streaming, conversation memory.
"""

import json

from flask import Blueprint, request, jsonify, Response, stream_with_context

from . import shared
from ...rag.query_history import QueryRecord
from ...utils.logger import get_logger

logger = get_logger(__name__)

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    """RAG-powered question answering (non-streaming).

    Request body:
        {"question": "...", "top_k": 8}

    Returns:
        {"answer": "...", "sources": [...], "usage": {...},
         "confidence": 0.85, "cache_hit": false,
         "retrieval_attempts": 1, "grounding": [...],
         "verification": {...}}
    """
    rag = shared.get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Chat requires an API key."
        }), 503

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        response = rag.query(question, top_k=data.get("top_k", 8))
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
            "grounding": shared.grounding_to_list(response.grounding),
            "verification": response.verification.to_dict()
                if response.verification else None,
        })
    except Exception as e:
        logger.error("Chat error: %s", e)
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming RAG chat via Server-Sent Events.

    Performs retrieval synchronously then streams Claude's response
    token-by-token. Sends a final 'done' event with grounding,
    verification, and usage metadata once generation is complete.

    Request body:
        {"question": "...", "top_k": 8}

    SSE event types:
        {"type": "token",     "text": "..."}
        {"type": "cache_hit", "answer": "...", "confidence": ..., ...}
        {"type": "done",      "confidence": ..., "grounding": [...], ...}
        {"type": "error",     "message": "..."}
    """
    rag = shared.get_rag()
    if rag is None:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Chat requires an API key."
        }), 503

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    top_k = data.get("top_k", 8)

    def generate():
        try:
            # ── Step 0: Semantic cache check ──────────────────────────
            if rag._cache is not None:
                cached = rag._cache.get(question)
                if cached is not None:
                    cached.cache_hit = True
                    payload = {
                        "type": "cache_hit",
                        "answer": cached.answer,
                        "confidence": cached.confidence,
                        "cache_hit": True,
                        "retrieval_attempts": cached.retrieval_attempts,
                        "grounding": shared.grounding_to_list(cached.grounding),
                        "verification": cached.verification.to_dict()
                            if cached.verification else None,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    return

            # ── Step 1: Query rewrite ─────────────────────────────────
            retrieval_query = question
            if not rag.memory.is_empty():
                rewritten = rag._rewrite_query(question)
                if rewritten and rewritten != question:
                    retrieval_query = rewritten

            # ── Step 2: HyDE ──────────────────────────────────────────
            hyde_query = None
            if rag.use_hyde and rag.llm.is_available():
                hyde_query = rag._generate_hypothetical_document(
                    retrieval_query
                )

            # ── Step 3: Retrieve ──────────────────────────────────────
            results = rag._retrieve(hyde_query or retrieval_query, top_k)
            if not results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found. Try ingesting more documents first.'})}\n\n"
                return

            # ── Step 4: Build prompt and stream generation ────────────
            context = rag._format_context(results)
            prompt = rag._build_prompt(question, context)
            history = rag.memory.get_messages()

            full_answer = ""
            input_tokens = 0
            output_tokens = 0

            for chunk in rag.llm.stream_generate(
                prompt=prompt,
                system=rag.system_prompt,
                history=history,
                max_tokens=1024,
            ):
                if chunk["type"] == "token":
                    full_answer += chunk["text"]
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk['text']})}\n\n"
                elif chunk["type"] == "usage":
                    input_tokens = chunk["input_tokens"]
                    output_tokens = chunk["output_tokens"]

            # ── Step 5: Grounding ─────────────────────────────────────
            grounding = None
            confidence = 0.0
            if rag.ground_answer:
                grounding = rag._grounder.score(full_answer, results)
                confidence = grounding.overall_confidence

            # ── Step 6: Fact verification ─────────────────────────────
            verification = None
            if rag._verifier and results:
                verification = rag._verifier.verify(full_answer, results)

            # ── Step 7: Update conversation memory ────────────────────
            rag.memory.add_turn(question=question, answer=full_answer)

            # ── Step 8: Record to query history ───────────────────────
            rag._history.record(QueryRecord(
                query=question,
                answer_preview=full_answer[:200],
                confidence=confidence,
                cache_hit=False,
                retrieval_attempts=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                sources=list(
                    {r.chunk.source_doc_title for r in results}
                ),
                verification_score=(
                    verification.overall_verification_score
                    if verification else None
                ),
            ))

            # ── Step 9: Store in semantic cache ───────────────────────
            if rag._cache is not None:
                from ...rag.pipeline import RAGResponse
                from ...rag.llm import LLMResponse as LLMResp
                cache_resp = RAGResponse(
                    answer=full_answer,
                    sources=results,
                    llm_response=LLMResp(
                        content=full_answer,
                        model=rag.llm.model,
                        usage={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                    ),
                    query=question,
                    grounding=grounding,
                    confidence=confidence,
                    retrieval_attempts=1,
                    cache_hit=False,
                    verification=verification,
                )
                rag._cache.store(question, cache_resp)

            # ── Step 10: Send final metadata event ────────────────────
            done_payload = {
                "type": "done",
                "confidence": confidence,
                "cache_hit": False,
                "retrieval_attempts": 1,
                "grounding": shared.grounding_to_list(grounding),
                "verification": (
                    verification.to_dict() if verification else None
                ),
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "sources": [
                    {
                        "source": r.chunk.source_doc_title,
                        "page": r.chunk.page_number,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ],
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            logger.error("Streaming chat error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@chat_bp.route("/api/conversation")
def get_conversation():
    """Return all current conversation turns for display on page load."""
    rag = shared.get_rag()
    if rag is None:
        return jsonify({"turns": [], "total": 0})

    turns = rag.memory.turns
    return jsonify({
        "turns": [t.to_dict() for t in turns],
        "total": len(turns),
    })


@chat_bp.route("/api/conversation/clear", methods=["POST"])
def clear_conversation():
    """Clear conversation memory and start a fresh chat."""
    rag = shared.get_rag()
    if rag is None:
        return jsonify({"cleared": True})

    rag.clear_history()
    return jsonify({"cleared": True})