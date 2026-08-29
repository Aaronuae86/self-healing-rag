"""Explicit dense/BM25 fusion with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .baseline import FAISSRetriever, RetrievalResult
from .bm25 import BM25Result, BM25Retriever


class RankedResult(Protocol):
    """Minimum result interface required by overlap measurement."""

    @property
    def id(self) -> str: ...


@dataclass(frozen=True)
class HybridResult(RetrievalResult):
    """A document ranked by RRF with its source ranks retained."""

    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass(frozen=True)
class RetrievalAgreement:
    """Overlap summary for two ranked result lists."""

    dense_document_ids: tuple[str, ...]
    bm25_document_ids: tuple[str, ...]
    shared_document_ids: tuple[str, ...]
    overlap_count: int
    overlap_ratio: float


def measure_retrieval_overlap(
    dense_results: Sequence[RankedResult], bm25_results: Sequence[RankedResult]
) -> RetrievalAgreement:
    """Measure top-k agreement as shared IDs divided by the shorter result list."""

    dense_ids = tuple(result.id for result in dense_results)
    bm25_ids = tuple(result.id for result in bm25_results)
    bm25_id_set = set(bm25_ids)
    shared_ids = tuple(document_id for document_id in dense_ids if document_id in bm25_id_set)
    denominator = min(len(dense_ids), len(bm25_ids))
    overlap_count = len(shared_ids)
    overlap_ratio = overlap_count / denominator if denominator else 0.0
    return RetrievalAgreement(
        dense_document_ids=dense_ids,
        bm25_document_ids=bm25_ids,
        shared_document_ids=shared_ids,
        overlap_count=overlap_count,
        overlap_ratio=overlap_ratio,
    )


class HybridRetriever:
    """Fuse FAISS and BM25 rankings using Reciprocal Rank Fusion (RRF)."""

    RRF_OFFSET = 60

    def __init__(
        self, dense_retriever: FAISSRetriever, bm25_retriever: BM25Retriever
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = 3) -> list[HybridResult]:
        """Retrieve from both indexes and return the top fused documents."""

        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        dense_results = self.dense_retriever.retrieve(query, top_k=top_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        return self.fuse(dense_results, bm25_results, top_k=top_k)

    def fuse(
        self,
        dense_results: Sequence[RetrievalResult],
        bm25_results: Sequence[BM25Result],
        top_k: int,
    ) -> list[HybridResult]:
        """Fuse already-computed dense and BM25 rankings with the same RRF rule."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        fused: dict[str, dict] = {}

        for rank, result in enumerate(dense_results, start=1):
            fused[result.id] = {
                "document": result.document,
                "score": 1.0 / (self.RRF_OFFSET + rank),
                "dense_rank": rank,
                "bm25_rank": None,
            }

        for rank, result in enumerate(bm25_results, start=1):
            entry = fused.setdefault(
                result.id,
                {
                    "document": result.document,
                    "score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                },
            )
            entry["score"] += 1.0 / (self.RRF_OFFSET + rank)
            entry["bm25_rank"] = rank

        results = [HybridResult(**entry) for entry in fused.values()]
        return sorted(results, key=lambda result: (-result.score, result.id))[:top_k]
