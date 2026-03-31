"""
Retrieval-Augmented Generation pipeline.

Combines the retrieval system with an LLM to answer questions
grounded in the knowledge base. The pipeline:

1. Checks the semantic cache — returns instantly if a similar query
   was answered recently.
2. Optionally generates a hypothetical document (HyDE) to improve
   retrieval by embedding an ideal answer instead of the raw query.
3. Optionally rewrites follow-up queries for conversational coherence.
4. Retrieves relevant chunks from the vector store.
5. Formats them as context for the LLM.
6. Generates an answer with source citations.
7. Scores answer grounding — if confidence is low, reformulates the
   query and retries retrieval automatically (Adaptive Re-Retrieval).
8. Stores the result in the semantic cache for future queries.
9. Maintains a sliding-window conversation memory for follow-ups.

Design Pattern: Mediator Pattern — the pipeline coordinates
between the KnowledgeBase, LLMProvider, and optional
reranker/query expander without them knowing about each other.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .llm import LLMProvider, Message, LLMResponse
from .memory import ConversationMemory, ConversationTurn
from .grounding import GroundingScorer, GroundingResult
from .cache import SemanticCache
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

_HYDE_SYSTEM_PROMPT = """You are a document generation assistant. Given a question, write a short hypothetical passage (2-4 sentences) that would perfectly answer it, as if it were an excerpt from a relevant document.

Rules:
1. Output ONLY the hypothetical passage. No explanation, no preamble.
2. Write in a factual, document-like style.
3. Stay focused — only include content directly relevant to the question."""

_REFORMULATE_SYSTEM_PROMPT = """You are a search query expert. A retrieval attempt returned low-confidence results. Reformulate the query to find better matches.

Rules:
1. Output ONLY the reformulated query. No explanation, no preamble.
2. Use different keywords and phrasing than the original.
3. Be more specific or try a different angle — do not just rephrase slightly.
4. Keep the reformulated query concise (one sentence maximum)."""


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
        grounding: Grounding analysis of the answer against sources.
            None if grounding is disabled.
        confidence: Overall answer confidence score (0-1), derived
            from grounding. None if grounding is disabled.
        retrieval_attempts: Number of retrieval attempts made.
            Greater than 1 means adaptive re-retrieval was triggered.
        hyde_query: The hypothetical document used for HyDE retrieval,
            if HyDE was enabled. None otherwise.
        cache_hit: True if this response was served from the semantic
            cache without any retrieval or LLM calls.
    """
    answer: str
    sources: List[SearchResult]
    llm_response: LLMResponse
    query: str
    rewritten_query: Optional[str] = None
    grounding: Optional[GroundingResult] = None
    confidence: Optional[float] = None
    retrieval_attempts: int = 1
    hyde_query: Optional[str] = None
    cache_hit: bool = False


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Orchestrates retrieval, context formatting, and LLM generation
    to answer questions grounded in the knowledge base.

    Supports:
        - Semantic query caching: instant responses for similar queries.
        - HyDE: embed a hypothetical answer instead of the raw query.
        - Adaptive Re-Retrieval: retry with reformulated query if
          answer confidence is below threshold.
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
        print(f"Cache hit: {response.cache_hit}")
        print(f"Confidence: {response.confidence}, Attempts: {response.retrieval_attempts}")
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
        use_hyde: bool = True,
        adaptive_retrieval: bool = True,
        confidence_threshold: float = 0.25,
        max_retries: int = 2,
        use_cache: bool = True,
        cache_threshold: float = 0.92,
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
        self.use_hyde = use_hyde
        self.adaptive_retrieval = adaptive_retrieval
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.memory = ConversationMemory(window_size=memory_window)
        self._grounder = GroundingScorer()
        self._cache = SemanticCache(threshold=cache_threshold) if use_cache else None

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_history: bool = True,
    ) -> RAGResponse:
        """Answer a question using retrieval-augmented generation.

        Checks the semantic cache first — if a similar query was
        answered recently, returns that response instantly.

        Otherwise runs HyDE and/or adaptive re-retrieval if enabled.
        The full flow is:
        1. Check semantic cache.
        2. Optionally rewrite follow-up query for retrieval coherence.
        3. Optionally generate a hypothetical document (HyDE).
        4. Retrieve chunks using the best available query form.
        5. Generate answer.
        6. Score grounding — if confidence < threshold, reformulate
           and retry retrieval up to max_retries times.
        7. Store result in cache.
        8. Return the best result found.

        Args:
            question: The user's question.
            top_k: Override the number of chunks to retrieve.
            include_history: Whether to use conversation history.

        Returns:
            RAGResponse with answer, sources, grounding, and metadata.
        """
        top_k = top_k or self.top_k

        # Step 0: Check semantic cache before any retrieval or LLM calls
        if self._cache is not None:
            cached = self._cache.get(question)
            if cached is not None:
                cached.cache_hit = True
                logger.info("Serving answer from semantic cache for: '%s'", question[:50])
                return cached

        # Step 1: Rewrite follow-up queries
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

        # Step 2: HyDE — generate hypothetical document for retrieval
        hyde_query = None
        if self.use_hyde and self.llm.is_available():
            hyde_query = self._generate_hypothetical_document(retrieval_query)
            if hyde_query:
                logger.info("HyDE document generated (%d chars)", len(hyde_query))

        # Step 3 onwards — retrieval + generation + adaptive retry loop
        best_results: List[SearchResult] = []
        best_grounding: Optional[GroundingResult] = None
        best_confidence: float = 0.0
        best_llm_response: Optional[LLMResponse] = None
        attempts = 0
        current_query = hyde_query or retrieval_query

        history = self.memory.get_messages() if include_history else None

        for attempt in range(1 + self.max_retries):
            attempts = attempt + 1

            # Retrieve
            results = self._retrieve(current_query, top_k)
            logger.info(
                "Attempt %d: retrieved %d chunks for query: '%s'",
                attempts, len(results), current_query[:50],
            )

            if not results:
                break

            # Generate answer
            context = self._format_context(results)
            prompt = self._build_prompt(question, context)
            llm_response = self.llm.generate(
                prompt=prompt,
                system=self.system_prompt,
                history=history,
            )

            # Score grounding
            grounding = None
            confidence = 0.0
            if self.ground_answer:
                grounding = self._grounder.score(llm_response.content, results)
                confidence = grounding.overall_confidence
                logger.info(
                    "Attempt %d confidence: %.3f (threshold: %.3f)",
                    attempts, confidence, self.confidence_threshold,
                )

            # Keep best result
            if confidence >= best_confidence:
                best_results = results
                best_grounding = grounding
                best_confidence = confidence
                best_llm_response = llm_response

            # Stop if confidence is acceptable or adaptive retrieval disabled
            if (
                not self.adaptive_retrieval
                or not self.ground_answer
                or confidence >= self.confidence_threshold
                or attempt >= self.max_retries
            ):
                break

            # Reformulate query for next attempt
            reformulated = self._reformulate_query(retrieval_query, attempt + 1)
            if not reformulated or reformulated == current_query:
                logger.info("Reformulation unchanged — stopping early.")
                break

            logger.info(
                "Low confidence (%.3f) — reformulating: '%s'",
                confidence, reformulated[:60],
            )
            current_query = reformulated

        # Record this turn in memory
        self.memory.add_turn(
            question=question,
            answer=best_llm_response.content,
            rewritten_query=rewritten_query,
        )

        logger.info(
            "Final answer: %d chars, confidence=%.3f, attempts=%d, "
            "%d input tokens, %d output tokens",
            len(best_llm_response.content),
            best_confidence,
            attempts,
            best_llm_response.usage.get("input_tokens", 0),
            best_llm_response.usage.get("output_tokens", 0),
        )

        response = RAGResponse(
            answer=best_llm_response.content,
            sources=best_results,
            llm_response=best_llm_response,
            query=question,
            rewritten_query=rewritten_query,
            grounding=best_grounding,
            confidence=best_confidence,
            retrieval_attempts=attempts,
            hyde_query=hyde_query,
            cache_hit=False,
        )

        # Store result in semantic cache for future similar queries
        if self._cache is not None:
            self._cache.store(question, response)

        return response

    def clear_history(self) -> None:
        """Clear conversation memory for a fresh start."""
        self.memory.clear()
        logger.debug("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of turns recorded in memory."""
        return self.memory.total_turns

    @property
    def cache_stats(self) -> Optional[dict]:
        """Return semantic cache statistics, or None if cache disabled."""
        return self._cache.stats if self._cache is not None else None

    # ─── Private helpers ────────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int) -> List[SearchResult]:
        """Run retrieval using the configured strategy."""
        if self.hybrid or self.rerank or self.expand_query:
            return self.kb.advanced_search(
                query,
                top_k=top_k,
                rerank=self.rerank,
                expand_query=self.expand_query,
                hybrid=self.hybrid,
            )
        return self.kb.search(query, top_k=top_k)

    def _generate_hypothetical_document(self, question: str) -> Optional[str]:
        """Generate a hypothetical ideal answer for HyDE retrieval.

        The generated text is used as the embedding query instead of
        the raw question, improving semantic match with stored chunks.

        Args:
            question: The user's question.

        Returns:
            A short hypothetical passage, or None on failure.
        """
        try:
            response = self.llm.generate(
                prompt=f"Question: {question}",
                system=_HYDE_SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.5,
            )
            text = response.content.strip()
            return text if text else None
        except Exception as e:
            logger.warning("HyDE generation failed, using raw query: %s", e)
            return None

    def _reformulate_query(self, question: str, attempt: int) -> Optional[str]:
        """Reformulate a low-confidence query for adaptive re-retrieval.

        Args:
            question: The original retrieval query.
            attempt: Current attempt number (used for logging).

        Returns:
            A reformulated query string, or None on failure.
        """
        try:
            response = self.llm.generate(
                prompt=f"Original query: {question}",
                system=_REFORMULATE_SYSTEM_PROMPT,
                max_tokens=100,
                temperature=0.4,
            )
            reformulated = response.content.strip()
            return reformulated if reformulated else None
        except Exception as e:
            logger.warning(
                "Query reformulation (attempt %d) failed: %s", attempt, e
            )
            return None

    def _rewrite_query(self, question: str) -> Optional[str]:
        """Rewrite a follow-up question to be self-contained."""
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