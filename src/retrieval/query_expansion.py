"""
Query expansion for improving retrieval recall.

Transforms the user's original query into multiple variants or
enriched forms to capture more relevant documents. Implements
three strategies:

1. Synonym Expansion: Adds semantically similar terms using WordNet.
2. Multi-Query: Generates query variations from different angles.
3. HyDE (Hypothetical Document Embeddings): Creates a hypothetical
   answer and uses its embedding for retrieval.
"""

import re
from typing import List, Optional, Set
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QueryExpander:
    """Expands queries to improve retrieval recall.

    Supports multiple expansion strategies that can be used
    individually or combined. Each strategy produces variant
    queries that are merged with the original.

    Attributes:
        strategy: Expansion method ('synonym', 'multi_query', 'hyde').
    """

    STRATEGIES = ["synonym", "multi_query", "hyde"]

    def __init__(self, strategy: str = "synonym"):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {self.STRATEGIES}"
            )
        self.strategy = strategy

    def expand(self, query: str, n_expansions: int = 3) -> List[str]:
        """Expand a query into multiple variants.

        Args:
            query: The original user query.
            n_expansions: Number of expanded queries to generate.

        Returns:
            List of query strings, starting with the original.
        """
        if self.strategy == "synonym":
            return self._synonym_expansion(query, n_expansions)
        elif self.strategy == "multi_query":
            return self._multi_query_expansion(query, n_expansions)
        elif self.strategy == "hyde":
            return self._hyde_expansion(query)
        return [query]

    def _synonym_expansion(
        self, query: str, n_expansions: int = 3
    ) -> List[str]:
        """Expand query using WordNet synonyms.

        Identifies content words in the query and replaces them
        with synonyms to create variant queries.
        """
        try:
            from nltk.corpus import wordnet
            import nltk

            # Ensure WordNet data is available
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
        except ImportError:
            logger.warning("NLTK not available; returning original query.")
            return [query]

        queries = [query]
        words = query.lower().split()
        # Skip common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "and", "or",
            "not", "what", "how", "why", "when", "where", "which", "who",
            "do", "does", "did", "can", "could", "would", "should",
        }
        content_words = [w for w in words if w not in stop_words and len(w) > 2]

        for word in content_words:
            synsets = wordnet.synsets(word)
            synonyms: Set[str] = set()
            for syn in synsets[:3]:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if name != word and name not in query.lower():
                        synonyms.add(name)

            # Create variant queries by replacing the word
            for synonym in list(synonyms)[:n_expansions]:
                variant = re.sub(
                    rf"\b{re.escape(word)}\b",
                    synonym,
                    query,
                    flags=re.IGNORECASE,
                )
                if variant != query:
                    queries.append(variant)

            if len(queries) > n_expansions:
                break

        logger.debug(
            "Synonym expansion: '%s' → %d variants", query, len(queries) - 1
        )
        return queries[:n_expansions + 1]

    def _multi_query_expansion(
        self, query: str, n_expansions: int = 3
    ) -> List[str]:
        """Generate query variations by rephrasing.

        Uses simple heuristic transformations to create queries
        that approach the topic from different angles.
        """
        queries = [query]

        # Strategy 1: Convert question to statement
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        lower = query.lower().strip().rstrip("?")
        first_word = lower.split()[0] if lower.split() else ""

        if first_word in question_words:
            # "What is dependency injection?" → "dependency injection"
            # Remove question word and auxiliary verbs
            cleaned = re.sub(
                r"^(what|how|why|when|where|which|who)\s+(is|are|was|were|does|do|did|can|could)\s+",
                "",
                lower,
                flags=re.IGNORECASE,
            )
            if cleaned and cleaned != lower:
                queries.append(cleaned)

        # Strategy 2: Extract key noun phrases
        # Simple extraction: take words > 3 chars that aren't stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "need",
            "about", "above", "after", "again", "between", "into",
            "through", "during", "before", "with", "from", "that", "this",
            "these", "those", "then", "than", "each", "every", "some",
            "what", "how", "why", "when", "where", "which", "who",
        }
        key_terms = [
            w for w in re.findall(r"\b\w+\b", lower)
            if w not in stop_words and len(w) > 3
        ]
        if key_terms and len(key_terms) > 1:
            queries.append(" ".join(key_terms))

        # Strategy 3: Add context framing
        if len(key_terms) >= 1:
            queries.append(f"{' '.join(key_terms)} explanation overview")

        logger.debug(
            "Multi-query expansion: '%s' → %d variants",
            query, len(queries) - 1,
        )
        return queries[:n_expansions + 1]

    def _hyde_expansion(self, query: str) -> List[str]:
        """Hypothetical Document Embeddings (HyDE).

        Generates a hypothetical answer to the query. The embedding
        of this hypothetical document is often closer to relevant
        real documents than the query embedding alone.

        Note: For full HyDE, an LLM generates the hypothetical document.
        This implementation uses a template-based approach as a fallback
        when no LLM is available.
        """
        # Template-based hypothetical document
        hypothetical = (
            f"The following is a detailed explanation about {query.rstrip('?')}. "
            f"This covers the key concepts, definitions, and practical applications "
            f"related to this topic. The main points include the fundamental "
            f"principles, common approaches, and best practices."
        )

        logger.debug("HyDE expansion for: '%s'", query[:50])
        return [query, hypothetical]
