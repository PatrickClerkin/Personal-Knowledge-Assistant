"""
Automatic quiz generator from ingested document content.

Generates multiple choice and short answer questions grounded in
the actual chunks stored in the knowledge base. Each question
includes the source reference so students know exactly where the
answer comes from.

Design Pattern: Factory Pattern — QuizGenerator produces Question
objects from raw document content via LLM prompting.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..ingestion.knowledge_base import KnowledgeBase
from ..rag.llm import LLMProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


_MCQ_SYSTEM_PROMPT = """You are an expert quiz writer. Given an excerpt from a document, generate multiple choice questions that test understanding of the content.

Output ONLY a valid JSON array. Each element must have exactly these fields:
{
  "question": "The question text",
  "options": ["A) option", "B) option", "C) option", "D) option"],
  "answer": "A",
  "explanation": "Brief explanation of why this is correct"
}

Rules:
1. Output ONLY the JSON array, no preamble, no markdown fences.
2. The answer field must be exactly one of: A, B, C, D.
3. Make distractors plausible but clearly wrong on reflection.
4. Questions must be answerable from the excerpt alone.
5. Generate exactly the number of questions requested."""


_SHORT_ANSWER_SYSTEM_PROMPT = """You are an expert quiz writer. Given an excerpt from a document, generate short answer questions that test understanding of the content.

Output ONLY a valid JSON array. Each element must have exactly these fields:
{
  "question": "The question text",
  "answer": "The ideal answer in 1-2 sentences",
  "key_points": ["key point 1", "key point 2"]
}

Rules:
1. Output ONLY the JSON array, no preamble, no markdown fences.
2. Questions should require understanding, not just recall.
3. Answers must be grounded in the excerpt only.
4. Generate exactly the number of questions requested."""


@dataclass
class MCQQuestion:
    """A multiple choice question.

    Attributes:
        question: The question text.
        options: List of 4 answer options (A–D prefixed).
        answer: Correct answer letter (A, B, C, or D).
        explanation: Why the answer is correct.
        source: Source document title.
        page: Source page number if available.
    """
    question: str
    options: List[str]
    answer: str
    explanation: str
    source: str
    page: Optional[int] = None


@dataclass
class ShortAnswerQuestion:
    """A short answer question.

    Attributes:
        question: The question text.
        answer: Model answer.
        key_points: Key points that should appear in a good answer.
        source: Source document title.
        page: Source page number if available.
    """
    question: str
    answer: str
    key_points: List[str]
    source: str
    page: Optional[int] = None


@dataclass
class Quiz:
    """A complete quiz on a topic.

    Attributes:
        topic: The topic this quiz covers.
        mcq_questions: Multiple choice questions.
        short_answer_questions: Short answer questions.
        total_questions: Total number of questions.
        source_documents: Unique sources used.
    """
    topic: str
    mcq_questions: List[MCQQuestion]
    short_answer_questions: List[ShortAnswerQuestion]
    total_questions: int
    source_documents: List[str]

    def to_dict(self) -> dict:
        """Serialise to JSON-ready dict for the API."""
        return {
            "topic": self.topic,
            "total_questions": self.total_questions,
            "source_documents": self.source_documents,
            "mcq_questions": [
                {
                    "question": q.question,
                    "options": q.options,
                    "answer": q.answer,
                    "explanation": q.explanation,
                    "source": q.source,
                    "page": q.page,
                }
                for q in self.mcq_questions
            ],
            "short_answer_questions": [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "key_points": q.key_points,
                    "source": q.source,
                    "page": q.page,
                }
                for q in self.short_answer_questions
            ],
        }


class QuizGenerator:
    """Generates grounded quizzes from knowledge base content.

    Retrieves relevant chunks for a topic, then uses the LLM to
    generate MCQ and short answer questions grounded in that content.

    Args:
        knowledge_base: The KnowledgeBase to retrieve from.
        llm_provider: LLM for question generation.
        mcq_per_chunk: MCQ questions to generate per chunk (default 2).
        sa_per_chunk: Short answer questions per chunk (default 1).
        top_k: Chunks to retrieve for the topic (default 5).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_provider: LLMProvider,
        mcq_per_chunk: int = 2,
        sa_per_chunk: int = 1,
        top_k: int = 5,
    ):
        self.kb = knowledge_base
        self.llm = llm_provider
        self.mcq_per_chunk = mcq_per_chunk
        self.sa_per_chunk = sa_per_chunk
        self.top_k = top_k

    def generate(self, topic: str) -> Quiz:
        """Generate a quiz on the given topic.

        Args:
            topic: The subject to generate questions about.

        Returns:
            A Quiz with MCQ and short answer questions.
        """
        logger.info("Generating quiz for topic: '%s'", topic)

        results = self.kb.search(topic, top_k=self.top_k)
        if not results:
            return Quiz(
                topic=topic,
                mcq_questions=[],
                short_answer_questions=[],
                total_questions=0,
                source_documents=[],
            )

        mcq_questions: List[MCQQuestion] = []
        sa_questions: List[ShortAnswerQuestion] = []
        source_docs = set()

        for result in results:
            chunk = result.chunk
            source = chunk.source_doc_title
            page = chunk.page_number
            content = chunk.content.strip()

            if len(content) < 100:
                continue

            source_docs.add(source)

            # Generate MCQs
            new_mcqs = self._generate_mcq(
                content, source, page, self.mcq_per_chunk
            )
            mcq_questions.extend(new_mcqs)

            # Generate short answer questions
            new_sas = self._generate_short_answer(
                content, source, page, self.sa_per_chunk
            )
            sa_questions.extend(new_sas)

        total = len(mcq_questions) + len(sa_questions)
        logger.info(
            "Generated %d MCQ and %d short answer questions for '%s'",
            len(mcq_questions), len(sa_questions), topic,
        )

        return Quiz(
            topic=topic,
            mcq_questions=mcq_questions,
            short_answer_questions=sa_questions,
            total_questions=total,
            source_documents=list(source_docs),
        )

    def _generate_mcq(
        self,
        content: str,
        source: str,
        page: Optional[int],
        n: int,
    ) -> List[MCQQuestion]:
        """Generate n MCQ questions from a content chunk."""
        if not self.llm.is_available():
            return []

        prompt = (
            f"Generate {n} multiple choice question(s) from this excerpt:\n\n"
            f"{content[:1500]}"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_MCQ_SYSTEM_PROMPT,
                max_tokens=600,
                temperature=0.4,
            )
            text = response.content.strip()
            text = self._clean_json(text)
            items = json.loads(text)

            questions = []
            for item in items:
                if not all(k in item for k in
                           ["question", "options", "answer", "explanation"]):
                    continue
                if len(item["options"]) != 4:
                    continue
                if item["answer"] not in ["A", "B", "C", "D"]:
                    continue
                questions.append(MCQQuestion(
                    question=item["question"],
                    options=item["options"],
                    answer=item["answer"],
                    explanation=item["explanation"],
                    source=source,
                    page=page,
                ))
            return questions

        except Exception as e:
            logger.warning("MCQ generation failed for chunk: %s", e)
            return []

    def _generate_short_answer(
        self,
        content: str,
        source: str,
        page: Optional[int],
        n: int,
    ) -> List[ShortAnswerQuestion]:
        """Generate n short answer questions from a content chunk."""
        if not self.llm.is_available():
            return []

        prompt = (
            f"Generate {n} short answer question(s) from this excerpt:\n\n"
            f"{content[:1500]}"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=_SHORT_ANSWER_SYSTEM_PROMPT,
                max_tokens=400,
                temperature=0.4,
            )
            text = response.content.strip()
            text = self._clean_json(text)
            items = json.loads(text)

            questions = []
            for item in items:
                if not all(k in item for k in
                           ["question", "answer", "key_points"]):
                    continue
                questions.append(ShortAnswerQuestion(
                    question=item["question"],
                    answer=item["answer"],
                    key_points=item.get("key_points", []),
                    source=source,
                    page=page,
                ))
            return questions

        except Exception as e:
            logger.warning("Short answer generation failed for chunk: %s", e)
            return []

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and extract JSON array."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        # Find the JSON array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return text[start:end + 1]
        return text