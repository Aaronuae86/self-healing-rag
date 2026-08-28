"""Initial heuristic classification of retrieval diagnostics.

These thresholds are intentionally configurable and are not statistically
calibrated. Phase 6 will evaluate and revise them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .diagnostics import RetrievalDiagnostics


class RetrievalFailure(str, Enum):
    """Retrieval states used by the self-healing workflow."""

    HEALTHY = "HEALTHY"
    AMBIGUOUS = "AMBIGUOUS"
    WEAK_RETRIEVAL = "WEAK_RETRIEVAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class FailureDetectorConfig:
    """Centralized, initial heuristic thresholds for retrieval classification."""

    minimum_dense_top1: float = 0.30
    minimum_dense_average: float = 0.20
    minimum_bm25_top1: float = 0.25
    minimum_overlap_ratio: float = 0.20
    minimum_hybrid_top_rrf: float = 0.025
    minimum_cross_encoder_top: float = -2.0
    maximum_weak_cross_encoder_margin: float = 1.0
    maximum_ambiguous_dense_margin: float = 0.01
    maximum_ambiguous_cross_encoder_margin: float = 1.5
    minimum_plausible_cross_encoder_top2: float = -5.0
    insufficient_dense_top1: float = 0.25
    insufficient_dense_average: float = 0.20
    insufficient_bm25_top1: float = 0.0
    insufficient_cross_encoder_top: float = -5.0
    minimum_clear_cross_encoder_margin: float = 1.0
    insufficient_overlap_ratio: float = 0.20
    insufficient_hybrid_top_rrf: float = 0.025
    minimum_weak_signals: int = 3
    minimum_core_insufficient_signals: int = 2
    minimum_insufficient_signals: int = 3


@dataclass(frozen=True)
class FailureDetection:
    """Classification plus the signals that caused it."""

    failure_type: RetrievalFailure
    reasons: tuple[str, ...]


def _below(value: float | None, threshold: float) -> bool:
    return value is None or value < threshold


def _at_or_below(value: float | None, threshold: float) -> bool:
    return value is None or value <= threshold


class RetrievalFailureDetector:
    """Combine multiple diagnostics into an initial heuristic retrieval state."""

    def __init__(self, config: FailureDetectorConfig | None = None) -> None:
        self.config = config or FailureDetectorConfig()

    def classify(self, diagnostics: RetrievalDiagnostics) -> FailureDetection:
        """Classify diagnostics without claiming that the heuristics are calibrated."""

        config = self.config
        insufficient_reasons: list[str] = []
        core_insufficient_signals = 0
        if _below(diagnostics.dense_top1_score, config.insufficient_dense_top1):
            insufficient_reasons.append("very low dense top-1 score")
            core_insufficient_signals += 1
        if _below(
            diagnostics.dense_average_top_k_score,
            config.insufficient_dense_average,
        ):
            insufficient_reasons.append("very low average dense relevance")
            core_insufficient_signals += 1
        if _at_or_below(diagnostics.bm25_top1_score, config.insufficient_bm25_top1):
            insufficient_reasons.append("no meaningful BM25 match")
        cross_top = diagnostics.cross_encoder_top_score
        cross_margin = diagnostics.cross_encoder_top1_top2_margin
        cross_score_is_extremely_poor = _below(
            cross_top, config.insufficient_cross_encoder_top
        )
        cross_result_is_not_clearly_separated = (
            cross_margin is None
            or cross_margin < config.minimum_clear_cross_encoder_margin
        )
        if cross_score_is_extremely_poor and cross_result_is_not_clearly_separated:
            insufficient_reasons.append(
                "extremely poor cross-encoder relevance without a clear winner"
            )
            core_insufficient_signals += 1
        if diagnostics.dense_bm25_overlap_ratio < config.insufficient_overlap_ratio:
            insufficient_reasons.append("weak dense/BM25 overlap")
        if _below(
            diagnostics.hybrid_top_rrf_score,
            config.insufficient_hybrid_top_rrf,
        ):
            insufficient_reasons.append("weak hybrid retrieval consensus")
        if not diagnostics.same_document_ranked_highly:
            insufficient_reasons.append("no top-document agreement")

        if (
            core_insufficient_signals >= config.minimum_core_insufficient_signals
            and len(insufficient_reasons) >= config.minimum_insufficient_signals
        ):
            return FailureDetection(
                RetrievalFailure.INSUFFICIENT_EVIDENCE,
                tuple(insufficient_reasons),
            )

        dense_margin = diagnostics.dense_top1_top2_margin
        dense_candidates_are_close = (
            dense_margin is not None
            and dense_margin <= config.maximum_ambiguous_dense_margin
        )
        cross_top2 = (
            cross_top - cross_margin
            if cross_top is not None and cross_margin is not None
            else None
        )
        cross_candidates_are_close_and_plausible = (
            cross_margin is not None
            and cross_margin <= config.maximum_ambiguous_cross_encoder_margin
            and cross_top2 is not None
            and cross_top2 >= config.minimum_plausible_cross_encoder_top2
        )
        if dense_candidates_are_close and cross_candidates_are_close_and_plausible:
            ambiguous_reasons = [
                "dense top candidates have a very small score margin",
                "cross-encoder top candidates have a relatively small score margin",
                "cross-encoder runner-up remains plausibly relevant",
            ]
            if not diagnostics.same_document_ranked_highly:
                ambiguous_reasons.append(
                    "retrieval stages disagree on the top document"
                )
            return FailureDetection(
                RetrievalFailure.AMBIGUOUS,
                tuple(ambiguous_reasons),
            )

        weak_reasons: list[str] = []
        if _below(diagnostics.dense_top1_score, config.minimum_dense_top1):
            weak_reasons.append("dense top-1 score is weak")
        if _below(diagnostics.dense_average_top_k_score, config.minimum_dense_average):
            weak_reasons.append("average dense score is weak")
        if _below(diagnostics.bm25_top1_score, config.minimum_bm25_top1):
            weak_reasons.append("BM25 top-1 score is weak")
        if diagnostics.dense_bm25_overlap_ratio < config.minimum_overlap_ratio:
            weak_reasons.append("dense/BM25 overlap is weak")
        if _below(diagnostics.hybrid_top_rrf_score, config.minimum_hybrid_top_rrf):
            weak_reasons.append("hybrid RRF consensus is weak")
        if (
            _below(diagnostics.cross_encoder_top_score, config.minimum_cross_encoder_top)
            and (
                cross_margin is None
                or cross_margin <= config.maximum_weak_cross_encoder_margin
            )
        ):
            weak_reasons.append(
                "cross-encoder score is weak and lacks clear separation"
            )
        if not diagnostics.same_document_ranked_highly:
            weak_reasons.append("no top-document agreement")

        if len(weak_reasons) >= config.minimum_weak_signals:
            return FailureDetection(
                RetrievalFailure.WEAK_RETRIEVAL,
                tuple(weak_reasons),
            )

        return FailureDetection(
            RetrievalFailure.HEALTHY,
            ("retrieval signals are sufficiently strong and consistent",),
        )
