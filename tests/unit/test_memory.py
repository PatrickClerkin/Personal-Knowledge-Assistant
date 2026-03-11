"""Unit tests for ConversationMemory."""

import pytest
from src.rag.memory import ConversationMemory, ConversationTurn


class TestConversationMemory:

    def test_starts_empty(self):
        mem = ConversationMemory(window_size=3)
        assert mem.is_empty()
        assert mem.total_turns == 0
        assert mem.get_messages() == []

    def test_add_turn_updates_state(self):
        mem = ConversationMemory(window_size=3)
        mem.add_turn("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")
        assert not mem.is_empty()
        assert mem.total_turns == 1

    def test_get_messages_returns_alternating_roles(self):
        mem = ConversationMemory(window_size=3)
        mem.add_turn("Question one", "Answer one")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Question one"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Answer one"

    def test_sliding_window_limits_messages(self):
        mem = ConversationMemory(window_size=2)
        for i in range(5):
            mem.add_turn(f"Question {i}", f"Answer {i}")
        assert mem.total_turns == 5
        messages = mem.get_messages()
        # window_size=2 means last 2 turns = 4 messages
        assert len(messages) == 4
        assert messages[0].content == "Question 3"
        assert messages[2].content == "Question 4"

    def test_window_size_one(self):
        mem = ConversationMemory(window_size=1)
        mem.add_turn("First", "First answer")
        mem.add_turn("Second", "Second answer")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Second"

    def test_clear_resets_memory(self):
        mem = ConversationMemory(window_size=3)
        mem.add_turn("Q", "A")
        mem.clear()
        assert mem.is_empty()
        assert mem.total_turns == 0
        assert mem.get_messages() == []

    def test_stores_rewritten_query(self):
        mem = ConversationMemory(window_size=3)
        mem.add_turn("tell me more", "More info...", rewritten_query="Tell me more about RAG chunking strategies")
        assert mem.turns[0].rewritten_query == "Tell me more about RAG chunking strategies"

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            ConversationMemory(window_size=0)

    def test_turns_property_returns_copy(self):
        mem = ConversationMemory(window_size=3)
        mem.add_turn("Q", "A")
        turns = mem.turns
        turns.clear()
        assert mem.total_turns == 1  # original unaffected