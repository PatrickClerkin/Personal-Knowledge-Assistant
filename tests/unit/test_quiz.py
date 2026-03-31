"""Unit tests for QuizGenerator."""

import pytest
from unittest.mock import MagicMock
from src.study.quiz_generator import QuizGenerator, Quiz, MCQQuestion, ShortAnswerQuestion
from src.rag.llm import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="claude-test",
        usage={"input_tokens": 10, "output_tokens": 50},
        stop_reason="end_turn",
    )


def _make_result(content: str, source: str = "lecture_1", page: int = 1):
    chunk = MagicMock()
    chunk.content = content
    chunk.source_doc_title = source
    chunk.page_number = page
    result = MagicMock()
    result.chunk = chunk
    result.score = 0.9
    return result


_SAMPLE_MCQ_JSON = '''[
  {
    "question": "What is backpropagation?",
    "options": ["A) A forward pass", "B) A learning algorithm", "C) A data type", "D) A loss function"],
    "answer": "B",
    "explanation": "Backpropagation is the algorithm used to train neural networks."
  }
]'''

_SAMPLE_SA_JSON = '''[
  {
    "question": "Explain the role of weights in a neural network.",
    "answer": "Weights determine the strength of connections between neurons.",
    "key_points": ["strength of connections", "adjusted during training"]
  }
]'''

_LONG_CONTENT = (
    "Neural networks consist of layers of interconnected neurons. "
    "Weights are parameters that determine the strength of connections. "
    "Backpropagation is the algorithm used to adjust these weights during training. "
    "The loss function measures how far the network output is from the target. "
    "Gradient descent is used to minimise the loss by updating weights iteratively."
)


def _make_generator(mcq_json=None, sa_json=None, search_results=None):
    kb = MagicMock()
    kb.search.return_value = search_results or [
        _make_result(_LONG_CONTENT)
    ]

    llm = MagicMock()
    llm.is_available.return_value = True
    llm.generate.side_effect = [
        _make_llm_response(mcq_json or _SAMPLE_MCQ_JSON),
        _make_llm_response(sa_json or _SAMPLE_SA_JSON),
    ] * 10  # enough for multiple chunks

    return QuizGenerator(
        knowledge_base=kb,
        llm_provider=llm,
        mcq_per_chunk=1,
        sa_per_chunk=1,
        top_k=3,
    )


class TestQuizGenerator:

    def test_generate_returns_quiz(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        assert isinstance(quiz, Quiz)

    def test_quiz_has_correct_topic(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        assert quiz.topic == "neural networks"

    def test_empty_kb_returns_empty_quiz(self):
        kb = MagicMock()
        kb.search.return_value = []
        llm = MagicMock()
        llm.is_available.return_value = True
        gen = QuizGenerator(knowledge_base=kb, llm_provider=llm)
        quiz = gen.generate("neural networks")
        assert quiz.total_questions == 0
        assert quiz.mcq_questions == []
        assert quiz.short_answer_questions == []

    def test_mcq_questions_have_required_fields(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        if quiz.mcq_questions:
            q = quiz.mcq_questions[0]
            assert hasattr(q, "question")
            assert hasattr(q, "options")
            assert hasattr(q, "answer")
            assert hasattr(q, "explanation")
            assert hasattr(q, "source")

    def test_mcq_answer_is_valid_letter(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        for q in quiz.mcq_questions:
            assert q.answer in ["A", "B", "C", "D"]

    def test_mcq_has_four_options(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        for q in quiz.mcq_questions:
            assert len(q.options) == 4

    def test_short_answer_questions_have_required_fields(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        if quiz.short_answer_questions:
            q = quiz.short_answer_questions[0]
            assert hasattr(q, "question")
            assert hasattr(q, "answer")
            assert hasattr(q, "key_points")
            assert hasattr(q, "source")

    def test_source_documents_populated(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        assert len(quiz.source_documents) > 0

    def test_total_questions_matches_lists(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        assert quiz.total_questions == (
            len(quiz.mcq_questions) + len(quiz.short_answer_questions)
        )

    def test_llm_unavailable_returns_empty_quiz(self):
        kb = MagicMock()
        kb.search.return_value = [_make_result(_LONG_CONTENT)]
        llm = MagicMock()
        llm.is_available.return_value = False
        gen = QuizGenerator(knowledge_base=kb, llm_provider=llm)
        quiz = gen.generate("neural networks")
        assert quiz.mcq_questions == []
        assert quiz.short_answer_questions == []

    def test_to_dict_has_required_keys(self):
        gen = _make_generator()
        quiz = gen.generate("neural networks")
        d = quiz.to_dict()
        assert "topic" in d
        assert "mcq_questions" in d
        assert "short_answer_questions" in d
        assert "total_questions" in d
        assert "source_documents" in d

    def test_invalid_mcq_json_handled_gracefully(self):
        kb = MagicMock()
        kb.search.return_value = [_make_result(_LONG_CONTENT)]
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = _make_llm_response("not valid json at all")
        gen = QuizGenerator(knowledge_base=kb, llm_provider=llm)
        quiz = gen.generate("neural networks")
        assert isinstance(quiz, Quiz)

    def test_short_chunk_skipped(self):
        kb = MagicMock()
        kb.search.return_value = [_make_result("Too short.")]
        llm = MagicMock()
        llm.is_available.return_value = True
        gen = QuizGenerator(knowledge_base=kb, llm_provider=llm)
        quiz = gen.generate("neural networks")
        assert quiz.mcq_questions == []