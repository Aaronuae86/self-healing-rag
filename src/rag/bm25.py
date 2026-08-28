"""A small, explicit BM25 retriever for the Phase 1 corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .baseline import Document


@dataclass(frozen=True)
class BM25Result:
    """A retrieved document together with its BM25 relevance score."""

    document: Document
    score: float

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def title(self) -> str:
        return self.document.title

    @property
    def text(self) -> str:
        return self.document.text


class BM25Retriever:
    """Rank ``Document`` objects with BM25 using simple word tokenization."""

    def __init__(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("At least one document is required to build a BM25 retriever.")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise ImportError("Install rank-bm25 before creating a BM25Retriever.") from error

        self.documents = list(documents)
        self.tokenized_documents = [self.tokenize(document.text) for document in self.documents]
        self.index = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase text and split it into alphanumeric word tokens."""

        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query: str, top_k: int = 3) -> list[BM25Result]:
        """Return the highest-scoring documents for a query."""

        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        scores = self.index.get_scores(self.tokenize(query))
        ranked_positions = sorted(
            range(len(self.documents)), key=lambda position: float(scores[position]), reverse=True
        )[: min(top_k, len(self.documents))]
        return [
            BM25Result(document=self.documents[position], score=float(scores[position]))
            for position in ranked_positions
        ]
