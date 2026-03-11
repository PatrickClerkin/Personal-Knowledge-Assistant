"""
Conversation memory with sliding window for the RAG pipeline.

Stores conversation turns and provides a windowed view of recent
history to pass to the LLM, preventing unbounded context growth.

Design Pattern: Value Object — ConversationTurn is immutable;
ConversationMemory manages the collection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .llm import Message


@dataclass(frozen=True)
class ConversationTurn:
    """A single question/answer exchange.

    Attributes:
        question: The user's original question.
        answer: The assistant's generated answer.
        rewritten_query: The standalone query used for retrieval,
            if query rewriting was applied.
        timestamp: When this turn occurred.
    """
    question: str
    answer: str
    rewritten_query: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMemory:
    """Sliding window conversation memory for multi-turn RAG.

    Stores full conversation history but only exposes the most
    recent ``window_size`` turns to the LLM, preventing token
    explosion over long sessions.

    Attributes:
        window_size: Maximum number of turns to include in the
            active window (each turn = 1 user + 1 assistant message).
    """

    def __init__(self, window_size: int = 3):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self._turns: List[ConversationTurn] = []

    def add_turn(
        self,
        question: str,
        answer: str,
        rewritten_query: Optional[str] = None,
    ) -> None:
        """Record a completed question/answer turn.

        Args:
            question: The user's original question.
            answer: The assistant's answer.
            rewritten_query: Standalone query used for retrieval,
                if different from the original question.
        """
        self._turns.append(ConversationTurn(
            question=question,
            answer=answer,
            rewritten_query=rewritten_query,
        ))

    def get_messages(self) -> List[Message]:
        """Return the windowed history as a flat list of Messages.

        Only includes the most recent ``window_size`` turns, keeping
        token usage bounded regardless of session length.

        Returns:
            Alternating user/assistant Message objects for the LLM.
        """
        recent = self._turns[-self.window_size:]
        messages = []
        for turn in recent:
            messages.append(Message(role="user", content=turn.question))
            messages.append(Message(role="assistant", content=turn.answer))
        return messages

    def is_empty(self) -> bool:
        """True if no turns have been recorded yet."""
        return len(self._turns) == 0

    def clear(self) -> None:
        """Reset memory for a fresh conversation."""
        self._turns.clear()

    @property
    def total_turns(self) -> int:
        """Total number of turns recorded (including outside window)."""
        return len(self._turns)

    @property
    def turns(self) -> List[ConversationTurn]:
        """Read-only view of all turns."""
        return list(self._turns)