"""Retrieval-quality signals collected before answer generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Sequence

from .baseline import RetrievalResult
from .bm25 import BM25Result
from .hybrid import HybridResult, measure_retrieval_overlap
from .reranker import RerankedResult


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Structured retrieval signals for later inspection or routing logic."""

    query: str
    dense_top1_score: float | None
    dense_top1_top2_margin: float | None
    dense_average_top_k_score: float | None
    bm25_top1_score: float | None
    dense_bm25_overlap_ratio: float
    dense_bm25_shared_document_ids: tuple[str, ...]
    hybrid_top_rrf_score: float | None
    cross_encoder_top_score: float | None
    cross_encoder_top1_top2_margin: float | None
    same_document_ranked_highly: bool
    agreed_top_document_ids: tuple[str, ...]
    dense_top_document_id: str | None
    bm25_top_document_id: str | None
    hybrid_top_document_id: str | None
    cross_encoder_top_document_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for display or serialization."""

        return asdict(self)


def _top_document_id(results: Sequence) -> str | None:
    return results[0].id if results else None


def _top_two_margin(scores: Sequence[float]) -> float | None:
    return float(scores[0] - scores[1]) if len(scores) >= 2 else None


def compute_retrieval_diagnostics(
    query: str,
    dense_results: Sequence[RetrievalResult],
    bm25_results: Sequence[BM25Result],
    hybrid_results: Sequence[HybridResult],
    reranked_results: Sequence[RerankedResult],
) -> RetrievalDiagnostics:
    """Compute transparent retrieval signals without classifying success or failure."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    dense_scores = [float(result.score) for result in dense_results]
    reranker_scores = [float(result.reranker_score) for result in reranked_results]
    overlap = measure_retrieval_overlap(dense_results, bm25_results)

    top_document_ids = tuple(
        document_id
        for document_id in (
            _top_document_id(dense_results),
            _top_document_id(bm25_results),
            _top_document_id(hybrid_results),
            _top_document_id(reranked_results),
        )
        if document_id is not None
    )
    top_id_counts = Counter(top_document_ids)
    agreed_top_ids = tuple(
        document_id
        for document_id in dict.fromkeys(top_document_ids)
        if top_id_counts[document_id] >= 2
    )

    return RetrievalDiagnostics(
        query=query,
        dense_top1_score=dense_scores[0] if dense_scores else None,
        dense_top1_top2_margin=_top_two_margin(dense_scores),
        dense_average_top_k_score=float(fmean(dense_scores)) if dense_scores else None,
        bm25_top1_score=float(bm25_results[0].score) if bm25_results else None,
        dense_bm25_overlap_ratio=overlap.overlap_ratio,
        dense_bm25_shared_document_ids=overlap.shared_document_ids,
        hybrid_top_rrf_score=float(hybrid_results[0].score) if hybrid_results else None,
        cross_encoder_top_score=reranker_scores[0] if reranker_scores else None,
        cross_encoder_top1_top2_margin=_top_two_margin(reranker_scores),
        same_document_ranked_highly=bool(agreed_top_ids),
        agreed_top_document_ids=agreed_top_ids,
        dense_top_document_id=_top_document_id(dense_results),
        bm25_top_document_id=_top_document_id(bm25_results),
        hybrid_top_document_id=_top_document_id(hybrid_results),
        cross_encoder_top_document_id=_top_document_id(reranked_results),
    )
