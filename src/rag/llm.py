"""
LLM provider abstraction layer.

Defines a common interface for language model providers and implements
the Claude API provider via Anthropic's SDK. The abstraction allows
swapping providers without changing the RAG pipeline logic.

Design Pattern: Strategy Pattern — the RAG pipeline depends on the
LLMProvider interface, not a concrete implementation.
"""

import os
from dotenv import load_dotenv
load_dotenv()
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: Either 'user' or 'assistant'.
        content: The message text.
    """
    role: str
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The generated text.
        model: Model identifier used for generation.
        usage: Token usage statistics.
        stop_reason: Why generation stopped.
    """
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    stop_reason: str = ""


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide a generate() method that takes
    a prompt and optional conversation history.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[Message]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user's current message.
            system: Optional system prompt.
            history: Previous conversation messages.
            max_tokens: Maximum response length.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            LLMResponse with the generated content.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and ready."""
        pass


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider.

    Uses the Anthropic Python SDK to make API calls to Claude.
    Requires the ANTHROPIC_API_KEY environment variable to be set.

    Attributes:
        model: Claude model identifier (default: claude-sonnet-4-20250514).
        api_key: Anthropic API key (from env or parameter).
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-initialise the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Initialised Claude provider: %s", self.model)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: "
                    "pip install anthropic"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if the API key is configured."""
        return self.api_key is not None

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[Message]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate a response using the Claude API.

        Builds a messages array from conversation history and the
        current prompt, then makes an API call to Claude.
        """
        if not self.is_available():
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key."
            )

        # Build messages array
        messages = []
        if history:
            for msg in history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        messages.append({"role": "user", "content": prompt})

        # Build API kwargs
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        logger.debug(
            "Claude API call: %d messages, max_tokens=%d",
            len(messages), max_tokens,
        )

        response = self.client.messages.create(**kwargs)

        # Extract text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason,
        )
