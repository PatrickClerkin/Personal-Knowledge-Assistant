"""
Retrieval-Augmented Generation pipeline.

Combines the retrieval system with an LLM to answer questions
grounded in the knowledge base. The pipeline:

1. Optionally rewrites the query using conversation history so
   follow-up questions ("tell me more about that") resolve correctly.
2. Retrieves relevant chunks from the vector store.
3. Formats them as context for the LLM.
4. Generates an answer with source citations.
5. Scores each source chunk for how well it grounds the answer.
6. Maintains a sliding-window conversation memory for follow-ups.

Design Pattern: Mediator Pattern — the pipeline coordinates
between the KnowledgeBase, LLMProvider, and optional
reranker/query expander without them knowing about each other.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLMProvider, Message, LLMResponse
from .memory import ConversationMemory, ConversationTurn
from .grounding import GroundingScorer, GroundingResult
from ..ingestion.knowledge_base import KnowledgeBase
from ..ingestion.storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


_DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context from a document knowledge base.

Rules:
1. Answer ONLY based on the provided context. If the context doesn't contain enough information, say so clearly.
2. Cite your sources by referencing the chunk source and page number when available (e.g., "[Source: document_name, Page 3]").
3. Be concise but thorough. Prefer direct answers over verbose explanations.
4. If the question requires information not in the context, acknowledge the limitation and suggest what additional documents might help.
5. When multiple chunks provide relevant information, synthesise them into a coherent answer."""

_REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant. Your only job is to rewrite a follow-up question into a fully self-contained question that can be understood without any conversation history.

Rules:
1. Output ONLY the rewritten question. No explanation, no preamble.
2. If the question is already self-contained, return it unchanged.
3. Preserve the user's intent exactly — do not add assumptions."""


@dataclass
class RAGResponse:
    """Response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        sources: Retrieved chunks used as context.
        llm_response: Raw LLM response with usage stats.
        query: The original query.
        rewritten_query: The standalone query used for retrieval,
            if different from the original.
        grounding: Per-chunk grounding scores, sorted by support
            strength. None if grounding is disabled.
        confidence: Overall answer confidence score (0–1), derived
            from grounding. None if grounding is disabled.
    """
    answer: str
    sources: List[SearchResult]
    llm_response: LLMResponse
    query: str
    rewritten_query: Optional[str] = None
    grounding: Optional[List[GroundingResult]] = None
    confidence: Optional[float] = None


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Orchestrates retrieval, context formatting, and LLM generation
    to answer questions grounded in the knowledge base.

    Supports:
        - Single-turn and multi-turn conversations with sliding window.
        - Automatic query rewriting for coherent follow-up retrieval.
        - Answer grounding: per-chunk confidence scores post-generation.
        - Optional reranking and query expansion.
        - Optional hybrid BM25+FAISS retrieval.
        - Configurable context window and system prompt.
        - Source citation in responses.

    Usage:
        kb = KnowledgeBase(index_path="data/index/default")
        llm = ClaudeProvider()
        rag = RAGPipeline(knowledge_base=kb, llm_provider=llm)
        response = rag.query("What is dependency injection?")
        print(response.answer)
        print(response.confidence)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_provider: LLMProvider,
        system_prompt: Optional[str] = None,
        top_k: int = 5,
        max_context_chars: int = 8000,
        rerank: bool = False,
        expand_query: Optional[str] = None,
        hybrid: bool = False,
        memory_window: int = 3,
        ground_answer: bool = True,
    ):
        self.kb = knowledge_base
        self.llm = llm_provider
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.rerank = rerank
        self.expand_query = expand_query
        self.hybrid = hybrid
        self.ground_answer = ground_answer
        self.memory = ConversationMemory(window_size=memory_window)
        self._grounder = GroundingScorer()

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_history: bool = True,
    ) -> RAGResponse:
        """Answer a question using retrieval-augmented generation.

        If conversation history exists and ``include_history`` is True,
        the question is first rewritten into a self-contained form for
        accurate retrieval, then answered in the context of the full
        conversation.

        After generation, each source chunk is scored for how well it
        grounds the answer (if ``ground_answer`` is enabled).

        Args:
            question: The user's question.
            top_k: Override the number of chunks to retrieve.
            include_history: Whether to use conversation history.

        Returns:
            RAGResponse with the answer, sources, grounding, and metadata.
        """
        top_k = top_k or self.top_k

        # Rewrite follow-up queries to be self-contained for retrieval
        rewritten_query = None
        retrieval_query = question
        if include_history and not self.memory.is_empty():
            rewritten_query = self._rewrite_query(question)
            if rewritten_query and rewritten_query != question:
                retrieval_query = rewritten_query
                logger.info(
                    "Query rewritten: '%s' -> '%s'",
                    question[:50], rewritten_query[:50],
                )

        # Retrieve relevant context
        if self.hybrid or self.rerank or self.expand_query:
            results = self.kb.advanced_search(
                retrieval_query,
                top_k=top_k,
                rerank=self.rerank,
                expand_query=self.expand_query,
                hybrid=self.hybrid,
            )
        else:
            results = self.kb.search(retrieval_query, top_k=top_k)

        logger.info(
            "Retrieved %d chunks for query: '%s'",
            len(results), retrieval_query[:50],
        )

        # Format context and build prompt
        context = self._format_context(results)
        prompt = self._build_prompt(question, context)

        # Pass sliding-window history to the LLM
        history = self.memory.get_messages() if include_history else None

        # Generate answer
        llm_response = self.llm.generate(
            prompt=prompt,
            system=self.system_prompt,
            history=history,
        )

        # Score answer grounding against source chunks
        grounding = None
        confidence = None
        if self.ground_answer and results:
            grounding = self._grounder.score(llm_response.content, results)
            confidence = self._grounder.overall_confidence(grounding)
            logger.info(
                "Answer confidence: %.3f (top chunk grounding: %.3f)",
                confidence,
                grounding[0].grounding_score if grounding else 0.0,
            )

        # Record this turn in memory
        self.memory.add_turn(
            question=question,
            answer=llm_response.content,
            rewritten_query=rewritten_query,
        )

        logger.info(
            "Generated answer: %d chars, %d input tokens, %d output tokens",
            len(llm_response.content),
            llm_response.usage.get("input_tokens", 0),
            llm_response.usage.get("output_tokens", 0),
        )

        return RAGResponse(
            answer=llm_response.content,
            sources=results,
            llm_response=llm_response,
            query=question,
            rewritten_query=rewritten_query,
            grounding=grounding,
            confidence=confidence,
        )

    def clear_history(self) -> None:
        """Clear conversation memory for a fresh start."""
        self.memory.clear()
        logger.debug("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of turns recorded in memory."""
        return self.memory.total_turns

    def _rewrite_query(self, question: str) -> Optional[str]:
        """Rewrite a follow-up question to be self-contained.

        Uses the last turn of conversation history to give the LLM
        just enough context to resolve references like "it", "that",
        "the previous topic", etc.

        Args:
            question: The user's raw follow-up question.

        Returns:
            A rewritten, self-contained question, or None on failure.
        """
        if not self.llm.is_available():
            return None

        recent = self.memory.turns[-1]
        rewrite_prompt = (
            f"Previous question: {recent.question}\n"
            f"Previous answer summary: {recent.answer[:300]}\n\n"
            f"Follow-up question to rewrite: {question}"
        )

        try:
            response = self.llm.generate(
                prompt=rewrite_prompt,
                system=_REWRITE_SYSTEM_PROMPT,
                max_tokens=128,
                temperature=0.0,
            )
            rewritten = response.content.strip()
            return rewritten if rewritten else None
        except Exception as e:
            logger.warning("Query rewrite failed, using original: %s", e)
            return None

    def _format_context(self, results: List[SearchResult]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        context_parts = []
        total_chars = 0

        for r in results:
            chunk = r.chunk
            source_info = f"Source: {chunk.source_doc_title}"
            if chunk.page_number:
                source_info += f", Page {chunk.page_number}"
            source_info += f" (relevance: {r.score:.3f})"

            chunk_text = (
                f"--- {source_info} ---\n"
                f"{chunk.content.strip()}\n"
            )

            if total_chars + len(chunk_text) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining > 100:
                    chunk_text = chunk_text[:remaining] + "\n[truncated]"
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the augmented prompt with retrieved context."""
        return (
            f"Context from the knowledge base:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Please answer based on the context above."
        )