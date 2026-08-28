"""Cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hybrid import HybridResult


@dataclass(frozen=True)
class RerankedResult(HybridResult):
    """A hybrid candidate with an additional cross-encoder score."""

    reranker_score: float = 0.0

    @property
    def rrf_score(self) -> float:
        """Expose the inherited hybrid score with an explicit name."""

        return self.score


class CrossEncoderReranker:
    """Score query-document pairs with a local SentenceTransformers model."""

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ImportError(
                "Install requirements.txt before creating a CrossEncoderReranker."
            ) from error

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Score every query-document pair and return candidates in score order."""

        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 when provided.")
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        if len(scores) != len(candidates):
            raise RuntimeError("The cross-encoder returned an unexpected number of scores.")

        reranked = [
            RerankedResult(
                document=candidate.document,
                score=candidate.score,
                dense_rank=candidate.dense_rank,
                bm25_rank=candidate.bm25_rank,
                reranker_score=float(score),
            )
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(
            key=lambda result: (-result.reranker_score, -result.rrf_score, result.id)
        )
        return reranked if top_k is None else reranked[:top_k]
