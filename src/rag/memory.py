"""
Conversation memory with sliding window for the RAG pipeline.

Stores conversation turns and provides a windowed view of recent
history to pass to the LLM, preventing unbounded context growth.
Turns are persisted to disk so conversations survive restarts.

Design Pattern: Value Object — ConversationTurn is immutable;
ConversationMemory manages the collection.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .llm import Message
from ..utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PERSIST_PATH = Path("data/memory/conversation.json")


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

    def to_dict(self) -> dict:
        """Serialise this turn to a JSON-safe dict."""
        return {
            "question": self.question,
            "answer": self.answer,
            "rewritten_query": self.rewritten_query,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Deserialise a turn from a dict."""
        ts_raw = data.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        return cls(
            question=data["question"],
            answer=data["answer"],
            rewritten_query=data.get("rewritten_query"),
            timestamp=ts,
        )


class ConversationMemory:
    """Sliding window conversation memory for multi-turn RAG.

    Stores full conversation history but only exposes the most
    recent ``window_size`` turns to the LLM, preventing token
    explosion over long sessions.

    Turns are automatically persisted to ``persist_path`` so
    conversations survive Flask restarts and page refreshes.

    Attributes:
        window_size: Maximum number of turns to include in the
            active window (each turn = 1 user + 1 assistant message).
        persist_path: Path to the JSON file used for persistence.
    """

    def __init__(
        self,
        window_size: int = 3,
        persist_path: Optional[Path] = None,
    ):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.persist_path = Path(persist_path) if persist_path else _DEFAULT_PERSIST_PATH
        self._turns: List[ConversationTurn] = []
        self._load()

    # ─── Public API ─────────────────────────────────────────────────

    def add_turn(
        self,
        question: str,
        answer: str,
        rewritten_query: Optional[str] = None,
    ) -> None:
        """Record a completed question/answer turn and persist it.

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
        self._save()

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
        """Reset memory for a fresh conversation and wipe the file."""
        self._turns.clear()
        self._save()
        logger.info("Conversation memory cleared.")

    # ─── Properties ─────────────────────────────────────────────────

    @property
    def total_turns(self) -> int:
        """Total number of turns recorded (including outside window)."""
        return len(self._turns)

    @property
    def turns(self) -> List[ConversationTurn]:
        """Read-only view of all turns."""
        return list(self._turns)

    # ─── Persistence ────────────────────────────────────────────────

    def _save(self) -> None:
        """Write all turns to the persistence file."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"turns": [t.to_dict() for t in self._turns]},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.warning("Failed to persist conversation memory: %s", e)

    def _load(self) -> None:
        """Load turns from the persistence file if it exists."""
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for raw in data.get("turns", []):
                self._turns.append(ConversationTurn.from_dict(raw))
            logger.info(
                "Loaded %d conversation turn(s) from %s",
                len(self._turns), self.persist_path,
            )
        except Exception as e:
            logger.warning(
                "Failed to load conversation memory from %s: %s — starting fresh.",
                self.persist_path, e,
            )
            self._turns = []