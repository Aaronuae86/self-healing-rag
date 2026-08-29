"""Free lexical groundedness checks against retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class TextResult(Protocol):
    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class GroundednessConfig:
    """Configuration for the transparent lexical-support heuristic."""

    minimum_token_length: int = 3
    minimum_coverage: float = 0.50


@dataclass(frozen=True)
class GroundednessResult:
    """Fraction of answer content tokens supported by retrieved context."""

    coverage: float
    is_grounded: bool
    supported_tokens: tuple[str, ...]
    unsupported_tokens: tuple[str, ...]


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "but",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "only",
    "that",
    "the",
    "their",
    "this",
    "through",
    "using",
    "was",
    "were",
    "which",
    "with",
}


def _content_tokens(text: str, minimum_length: int) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[a-z0-9]+\b", text.lower())
        if len(token) >= minimum_length and token not in STOPWORDS
    }


def check_groundedness(
    answer: str,
    retrieved_documents: Sequence[TextResult],
    config: GroundednessConfig | None = None,
) -> GroundednessResult:
    """Check lexical answer support without an external model or judge service."""

    settings = config or GroundednessConfig()
    answer_tokens = _content_tokens(answer, settings.minimum_token_length)
    context_tokens = _content_tokens(
        " ".join(document.text for document in retrieved_documents),
        settings.minimum_token_length,
    )
    supported = tuple(sorted(answer_tokens & context_tokens))
    unsupported = tuple(sorted(answer_tokens - context_tokens))
    coverage = len(supported) / len(answer_tokens) if answer_tokens else 0.0
    return GroundednessResult(
        coverage=coverage,
        is_grounded=coverage >= settings.minimum_coverage,
        supported_tokens=supported,
        unsupported_tokens=unsupported,
    )
