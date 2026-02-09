"""
Retrieval-Augmented Generation pipeline.

Combines the retrieval system with an LLM to answer questions
grounded in the knowledge base. The pipeline:

1. Retrieves relevant chunks from the vector store.
2. Formats them as context for the LLM.
3. Generates an answer with source citations.
4. Maintains conversation history for follow-up questions.

Design Pattern: Mediator Pattern — the pipeline coordinates
between the KnowledgeBase, LLMProvider, and optional
reranker/query expander without them knowing about each other.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLMProvider, Message, LLMResponse
from ..ingestion.knowledge_base import KnowledgeBase
from ..ingestion.storage.vector_store import SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


# System prompt for grounded question answering
_DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context from a document knowledge base.

Rules:
1. Answer ONLY based on the provided context. If the context doesn't contain enough information, say so clearly.
2. Cite your sources by referencing the chunk source and page number when available (e.g., "[Source: document_name, Page 3]").
3. Be concise but thorough. Prefer direct answers over verbose explanations.
4. If the question requires information not in the context, acknowledge the limitation and suggest what additional documents might help.
5. When multiple chunks provide relevant information, synthesise them into a coherent answer."""


@dataclass
class RAGResponse:
    """Response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        sources: Retrieved chunks used as context.
        llm_response: Raw LLM response with usage stats.
        query: The original query.
    """
    answer: str
    sources: List[SearchResult]
    llm_response: LLMResponse
    query: str


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Orchestrates retrieval, context formatting, and LLM generation
    to answer questions grounded in the knowledge base.

    Supports:
        - Single-turn and multi-turn conversations.
        - Optional reranking and query expansion.
        - Configurable context window and system prompt.
        - Source citation in responses.

    Usage:
        kb = KnowledgeBase(index_path="data/index/default")
        llm = ClaudeProvider()
        rag = RAGPipeline(knowledge_base=kb, llm_provider=llm)
        response = rag.query("What is dependency injection?")
        print(response.answer)
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
    ):
        self.kb = knowledge_base
        self.llm = llm_provider
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.rerank = rerank
        self.expand_query = expand_query
        self._history: List[Message] = []

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_history: bool = True,
    ) -> RAGResponse:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: The user's question.
            top_k: Override the number of chunks to retrieve.
            include_history: Whether to include conversation history.

        Returns:
            RAGResponse with the answer, sources, and metadata.
        """
        top_k = top_k or self.top_k

        # Retrieve relevant context
        if self.rerank or self.expand_query:
            results = self.kb.advanced_search(
                question,
                top_k=top_k,
                rerank=self.rerank,
                expand_query=self.expand_query,
            )
        else:
            results = self.kb.search(question, top_k=top_k)

        logger.info(
            "Retrieved %d chunks for query: '%s'",
            len(results), question[:50],
        )

        # Format context for the LLM
        context = self._format_context(results)

        # Build the augmented prompt
        prompt = self._build_prompt(question, context)

        # Prepare conversation history
        history = self._history if include_history else None

        # Generate answer
        llm_response = self.llm.generate(
            prompt=prompt,
            system=self.system_prompt,
            history=history,
        )

        # Update conversation history
        self._history.append(Message(role="user", content=question))
        self._history.append(
            Message(role="assistant", content=llm_response.content)
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
        )

    def clear_history(self) -> None:
        """Clear conversation history for a fresh start."""
        self._history.clear()
        logger.debug("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of messages in conversation history."""
        return len(self._history)

    def _format_context(self, results: List[SearchResult]) -> str:
        """Format retrieved chunks into a context string for the LLM.

        Includes source metadata and truncates to the maximum
        context length.
        """
        context_parts = []
        total_chars = 0

        for r in results:
            chunk = r.chunk
            # Build source reference
            source_info = f"Source: {chunk.source_doc_title}"
            if chunk.page_number:
                source_info += f", Page {chunk.page_number}"
            source_info += f" (relevance: {r.score:.3f})"

            chunk_text = (
                f"--- {source_info} ---\n"
                f"{chunk.content.strip()}\n"
            )

            # Enforce context length limit
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
