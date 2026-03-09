"""Named Entity Recognition extractor using spaCy."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_RELEVANT_LABELS = frozenset({
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "NORP",
})


class NERExtractor:
    """Lazy-loading spaCy NER extractor.

    The spaCy model is only loaded on the first call to :meth:`extract`,
    so constructing an instance is cheap and does not block startup.

    Parameters
    ----------
    model:
        Name of the spaCy model to load (default ``en_core_web_sm``).
        If the model is not installed it is downloaded automatically.
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self._model_name = model
        self._nlp = None

    def _get_nlp(self):
        """Return the loaded spaCy Language model, downloading if needed."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self._model_name)
                logger.debug("Loaded spaCy model '%s'", self._model_name)
            except OSError:
                logger.info(
                    "spaCy model '%s' not found — downloading now",
                    self._model_name,
                )
                from spacy.cli import download
                download(self._model_name)
                import spacy
                self._nlp = spacy.load(self._model_name)
        return self._nlp

    def extract(self, text: str) -> List[Dict[str, str]]:
        """Return deduplicated named entities found in *text*.

        Each entity is a ``dict`` with keys ``"text"`` and ``"label"``::

            {"text": "Dublin", "label": "GPE"}

        Only entity types listed in :data:`_RELEVANT_LABELS` are returned.
        Duplicates are collapsed on a case-insensitive (text, label) key.
        """
        if not text or not text.strip():
            return []

        nlp = self._get_nlp()
        doc = nlp(text)

        seen: set[tuple[str, str]] = set()
        entities: List[Dict[str, str]] = []

        for ent in doc.ents:
            if ent.label_ not in _RELEVANT_LABELS:
                continue
            key = (ent.text.strip().lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_})

        return entities

    def extract_from_chunks(self, chunks: List[Any]) -> List[Any]:
        """Annotate *chunks* in-place with an ``"entities"`` metadata key.

        Each chunk's ``metadata`` dict gains an ``"entities"`` key whose
        value is the list returned by :meth:`extract` for that chunk's
        ``content``.  The modified list is also returned for convenience.
        """
        for chunk in chunks:
            chunk.metadata["entities"] = self.extract(chunk.content)
        return chunks