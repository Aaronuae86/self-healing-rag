"""Retrieval-only SQuAD 2.0 benchmark for three RAG configurations."""

from __future__ import annotations

import csv
import json
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Sequence

from src.rag import (
    CrossEncoderReranker,
    FAISSRetriever,
    HybridResult,
    HybridRetriever,
    RerankedResult,
    RetrievalFailure,
    RetrievalResult,
    SelfHealingRAGWorkflow,
)

from .squad_dataset import SquadRetrievalDataset, SquadRetrievalQuestion


@dataclass(frozen=True)
class SquadRetrievalBenchmarkConfig:
    """Retrieval depths, objective failure cutoff, and output location."""

    cutoffs: tuple[int, ...] = (1, 3, 5, 10)
    static_candidate_k: int = 20
    initial_failure_k: int = 5
    output_dir: str | Path = "results"


@dataclass(frozen=True)
class RetrievalMetricSummary:
    """Gold-passage ranking metrics for one retrieval configuration."""

    total_questions: int
    recall_at: dict[int, float]
    mrr: float
    mean_gold_rank: float | None
    gold_found_count: int
    evaluated_rank_depth: int

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total_questions,
            **{f"recall@{cutoff}": value for cutoff, value in self.recall_at.items()},
            "mrr": self.mrr,
            "mean_gold_rank": self.mean_gold_rank,
            "gold_found_count": self.gold_found_count,
            "evaluated_rank_depth": self.evaluated_rank_depth,
        }


@dataclass(frozen=True)
class SquadRetrievalRecord:
    """Inspectable rankings and self-healing state for one held-out question."""

    squad_example_id: str
    question: str
    gold_document_id: str
    dense_retrieved_ids: tuple[str, ...]
    dense_retrieved_ranks: tuple[int, ...]
    dense_gold_rank: int | None
    static_strong_retrieved_ids: tuple[str, ...]
    static_strong_retrieved_ranks: tuple[int, ...]
    static_strong_gold_rank: int | None
    self_healing_initial_retrieved_ids: tuple[str, ...]
    self_healing_initial_retrieved_ranks: tuple[int, ...]
    self_healing_initial_gold_rank: int | None
    self_healing_final_retrieved_ids: tuple[str, ...]
    self_healing_final_retrieved_ranks: tuple[int, ...]
    self_healing_final_gold_rank: int | None
    initial_retrieval_failed: bool
    self_healing_detected_initial_failure: bool
    self_healing_initial_failure_type: str
    graph_path: tuple[str, ...]
    retry_count: int
    rewritten_query: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryAtCutoffSummary:
    """Recovery outcomes whose initial and final ranks use one shared cutoff."""

    cutoff: int
    initial_failure_count: int
    initial_failure_rate: float
    recovered_count: int
    recovered_rate: float
    recovery_attempted_count: int
    failed_recovery_count: int
    failed_recovery_rate: float

    def to_dict(self) -> dict:
        cutoff = self.cutoff
        return {
            "cutoff": cutoff,
            f"initial_failure_at_{cutoff}_count": self.initial_failure_count,
            f"initial_failure_at_{cutoff}_rate": self.initial_failure_rate,
            f"recovered_at_{cutoff}_count": self.recovered_count,
            f"recovered_at_{cutoff}_rate": self.recovered_rate,
            "recovery_attempted_count": self.recovery_attempted_count,
            f"failed_recovery_at_{cutoff}_count": self.failed_recovery_count,
            f"failed_recovery_at_{cutoff}_rate": self.failed_recovery_rate,
        }


@dataclass(frozen=True)
class GoldRankMovementSummary:
    """Direction of final gold-rank movement relative to the initial ranking."""

    improved_count: int
    unchanged_count: int
    worsened_count: int
    improved_rate: float
    unchanged_rate: float
    worsened_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SquadRetrievalBenchmarkReport:
    """Aggregate metrics, initial-failure analysis, and saved result paths."""

    records: tuple[SquadRetrievalRecord, ...]
    dense_metrics: RetrievalMetricSummary
    static_strong_metrics: RetrievalMetricSummary
    self_healing_metrics: RetrievalMetricSummary
    recovery_attempt_rate: float
    average_retry_count: float
    unnecessary_modification_rate: float
    primary_recovery_cutoff: int
    recovery_at_cutoff: dict[int, RecoveryAtCutoffSummary]
    gold_rank_movement: GoldRankMovementSummary
    initial_failures_detected: int
    metrics_path: Path
    per_example_path: Path

    @property
    def primary_recovery(self) -> RecoveryAtCutoffSummary:
        return self.recovery_at_cutoff[self.primary_recovery_cutoff]

    def metrics_dict(self) -> dict:
        primary = self.primary_recovery
        cutoff = self.primary_recovery_cutoff
        return {
            "dense_baseline": self.dense_metrics.to_dict(),
            "static_strong_rag": self.static_strong_metrics.to_dict(),
            "self_healing_rag": {
                **self.self_healing_metrics.to_dict(),
                "recovery_attempt_rate": self.recovery_attempt_rate,
                "average_retry_count": self.average_retry_count,
                "unnecessary_modification_rate": self.unnecessary_modification_rate,
            },
            "initial_retrieval_failures": {
                "cutoff": cutoff,
                f"initial_failure_at_{cutoff}_count": primary.initial_failure_count,
                f"initial_failure_at_{cutoff}_rate": primary.initial_failure_rate,
                "detected_by_self_healing": self.initial_failures_detected,
                f"recovered_at_{cutoff}_count": primary.recovered_count,
                f"failed_recovery_at_{cutoff}_count": primary.failed_recovery_count,
            },
            "recovery_by_cutoff": {
                f"recovery_at_{cutoff}": summary.to_dict()
                for cutoff, summary in self.recovery_at_cutoff.items()
            },
            "gold_rank_movement": self.gold_rank_movement.to_dict(),
        }


class CachedDenseRetriever:
    """Reuse batched initial-query embeddings across all benchmark pipelines."""

    def __init__(self, retriever: FAISSRetriever) -> None:
        self.retriever = retriever
        self._cache: dict[str, list[RetrievalResult]] = {}
        self._cache_depth: dict[str, int] = {}

    @property
    def documents(self):
        return self.retriever.documents

    def preload(
        self, queries: Sequence[str], top_k: int
    ) -> list[list[RetrievalResult]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if any(not query.strip() for query in queries):
            raise ValueError("Queries must not be empty.")
        results = self.retriever.retrieve_many(queries, top_k=top_k)
        for query, ranked in zip(queries, results):
            self._cache[query] = ranked
            self._cache_depth[query] = top_k
        return results

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if query in self._cache and self._cache_depth[query] >= top_k:
            return self._cache[query][:top_k]
        ranked = self.retriever.retrieve(query, top_k=top_k)
        self._cache[query] = ranked
        self._cache_depth[query] = top_k
        return ranked

    def retrieve_many(
        self, queries: Sequence[str], top_k: int = 3
    ) -> list[list[RetrievalResult]]:
        return self.preload(queries, top_k=top_k)


class CachedCrossEncoderReranker:
    """Reuse identical query/candidate cross-encoder scores across pipelines."""

    def __init__(self, reranker: CrossEncoderReranker) -> None:
        self.reranker = reranker
        self._cache: dict[tuple, list[RerankedResult]] = {}

    @staticmethod
    def _key(query: str, candidates: Sequence[HybridResult]) -> tuple:
        candidate_key = tuple(
            (
                item.id,
                float(item.score),
                item.dense_rank,
                item.bm25_rank,
            )
            for item in candidates
        )
        return query, candidate_key

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 when provided.")
        if not candidates:
            return []
        key = self._key(query, candidates)
        if key not in self._cache:
            self._cache[key] = self.reranker.rerank(query, candidates, top_k=None)
        ranked = self._cache[key]
        return ranked if top_k is None else ranked[:top_k]

    def rerank_many(
        self,
        queries: Sequence[str],
        candidate_lists: Sequence[Sequence[HybridResult]],
        top_k: int | None = None,
    ) -> list[list[RerankedResult]]:
        """Batch uncached query/candidate pairs without changing ranking semantics."""

        if len(queries) != len(candidate_lists):
            raise ValueError("queries and candidate_lists must be aligned.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 when provided.")
        missing: list[tuple[tuple, str, Sequence[HybridResult]]] = []
        flat_pairs: list[tuple[str, str]] = []
        for query, candidates in zip(queries, candidate_lists):
            if not query.strip():
                raise ValueError("Queries must not be empty.")
            if not candidates:
                continue
            key = self._key(query, candidates)
            if key not in self._cache:
                missing.append((key, query, candidates))
                flat_pairs.extend((query, item.text) for item in candidates)

        if flat_pairs:
            scores = np.asarray(
                self.reranker.model.predict(
                    flat_pairs,
                    batch_size=self.reranker.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            ).reshape(-1)
            if len(scores) != len(flat_pairs):
                raise RuntimeError(
                    "The cross-encoder returned an unexpected number of scores."
                )
            offset = 0
            for key, _query, candidates in missing:
                candidate_scores = scores[offset : offset + len(candidates)]
                offset += len(candidates)
                ranked = [
                    RerankedResult(
                        document=candidate.document,
                        score=candidate.score,
                        dense_rank=candidate.dense_rank,
                        bm25_rank=candidate.bm25_rank,
                        reranker_score=float(score),
                    )
                    for candidate, score in zip(candidates, candidate_scores)
                ]
                ranked.sort(
                    key=lambda item: (-item.reranker_score, -item.rrf_score, item.id)
                )
                self._cache[key] = ranked

        output: list[list[RerankedResult]] = []
        for query, candidates in zip(queries, candidate_lists):
            if not candidates:
                output.append([])
                continue
            ranked = self._cache[self._key(query, candidates)]
            output.append(ranked if top_k is None else ranked[:top_k])
        return output


def gold_document_rank(
    retrieved_document_ids: Sequence[str], gold_document_id: str
) -> int | None:
    """Return the one-based gold rank, or None when it was not retrieved."""

    try:
        return retrieved_document_ids.index(gold_document_id) + 1
    except ValueError:
        return None


def is_initial_failure_at_cutoff(rank: int | None, cutoff: int) -> bool:
    """Return whether the initial gold rank misses the specified cutoff."""

    if cutoff < 1:
        raise ValueError("cutoff must be at least 1.")
    return rank is None or rank > cutoff


def is_recovered_at_cutoff(
    initial_rank: int | None, final_rank: int | None, cutoff: int
) -> bool:
    """Require both the failure and recovery decision to use the same cutoff."""

    return is_initial_failure_at_cutoff(initial_rank, cutoff) and (
        final_rank is not None and final_rank <= cutoff
    )


def is_failed_recovery_at_cutoff(
    initial_rank: int | None,
    final_rank: int | None,
    recovery_attempted: bool,
    cutoff: int,
) -> bool:
    """Return whether an attempted recovery remains outside the same cutoff."""

    return (
        is_initial_failure_at_cutoff(initial_rank, cutoff)
        and recovery_attempted
        and (final_rank is None or final_rank > cutoff)
    )


def gold_rank_movement(
    initial_rank: int | None, final_rank: int | None
) -> str:
    """Classify rank movement, treating an absent gold document as worst rank."""

    if initial_rank is None:
        return "improved" if final_rank is not None else "unchanged"
    if final_rank is None:
        return "worsened"
    if final_rank < initial_rank:
        return "improved"
    if final_rank > initial_rank:
        return "worsened"
    return "unchanged"


def compute_recovery_at_cutoff(
    records: Sequence[SquadRetrievalRecord], cutoff: int
) -> RecoveryAtCutoffSummary:
    """Compute internally consistent recovery metrics at one cutoff."""

    if not records:
        raise ValueError("At least one record is required to compute recovery metrics.")
    initial_failures = [
        record
        for record in records
        if is_initial_failure_at_cutoff(
            record.self_healing_initial_gold_rank, cutoff
        )
    ]
    attempted = [record for record in initial_failures if record.retry_count > 0]
    recovered_count = sum(
        is_recovered_at_cutoff(
            record.self_healing_initial_gold_rank,
            record.self_healing_final_gold_rank,
            cutoff,
        )
        for record in initial_failures
    )
    failed_count = sum(
        is_failed_recovery_at_cutoff(
            record.self_healing_initial_gold_rank,
            record.self_healing_final_gold_rank,
            record.retry_count > 0,
            cutoff,
        )
        for record in initial_failures
    )
    return RecoveryAtCutoffSummary(
        cutoff=cutoff,
        initial_failure_count=len(initial_failures),
        initial_failure_rate=len(initial_failures) / len(records),
        recovered_count=recovered_count,
        recovered_rate=(
            recovered_count / len(initial_failures) if initial_failures else 0.0
        ),
        recovery_attempted_count=len(attempted),
        failed_recovery_count=failed_count,
        failed_recovery_rate=(failed_count / len(attempted) if attempted else 0.0),
    )


def compute_gold_rank_movement(
    records: Sequence[SquadRetrievalRecord],
) -> GoldRankMovementSummary:
    """Count improved, unchanged, and worsened final gold ranks."""

    if not records:
        raise ValueError("At least one record is required to compare gold ranks.")
    movements = [
        gold_rank_movement(
            record.self_healing_initial_gold_rank,
            record.self_healing_final_gold_rank,
        )
        for record in records
    ]
    counts = {label: movements.count(label) for label in ("improved", "unchanged", "worsened")}
    total = len(records)
    return GoldRankMovementSummary(
        improved_count=counts["improved"],
        unchanged_count=counts["unchanged"],
        worsened_count=counts["worsened"],
        improved_rate=counts["improved"] / total,
        unchanged_rate=counts["unchanged"] / total,
        worsened_rate=counts["worsened"] / total,
    )


def compute_retrieval_metrics(
    records: Sequence[SquadRetrievalRecord],
    rank_field: str,
    cutoffs: Sequence[int],
) -> RetrievalMetricSummary:
    """Compute recall and reciprocal rank, counting absent gold documents as zero.

    mean_gold_rank is calculated over found gold documents only; gold_found_count
    makes that denominator explicit.
    """

    if not records:
        raise ValueError("At least one record is required to compute metrics.")
    ranks = [getattr(record, rank_field) for record in records]
    found_ranks = [rank for rank in ranks if rank is not None]
    return RetrievalMetricSummary(
        total_questions=len(records),
        recall_at={
            cutoff: sum(rank is not None and rank <= cutoff for rank in ranks)
            / len(ranks)
            for cutoff in cutoffs
        },
        mrr=float(fmean(0.0 if rank is None else 1.0 / rank for rank in ranks)),
        mean_gold_rank=(float(fmean(found_ranks)) if found_ranks else None),
        gold_found_count=len(found_ranks),
        evaluated_rank_depth=max(cutoffs),
    )


def _build_report_from_records(
    records: Sequence[SquadRetrievalRecord],
    config: SquadRetrievalBenchmarkConfig,
    metrics_path: Path,
    per_example_path: Path,
) -> SquadRetrievalBenchmarkReport:
    if not records:
        raise ValueError("At least one record is required to build a report.")
    cutoffs = config.cutoffs
    primary_cutoff = config.initial_failure_k
    initial_failures = [
        record
        for record in records
        if is_initial_failure_at_cutoff(
            record.self_healing_initial_gold_rank, primary_cutoff
        )
    ]
    objective_initial_successes = [
        record
        for record in records
        if not is_initial_failure_at_cutoff(
            record.self_healing_initial_gold_rank, primary_cutoff
        )
    ]
    recovery_attempts = [record.retry_count > 0 for record in records]
    unnecessary_modifications = [
        record.retry_count > 0 for record in objective_initial_successes
    ]
    return SquadRetrievalBenchmarkReport(
        records=tuple(records),
        dense_metrics=compute_retrieval_metrics(records, "dense_gold_rank", cutoffs),
        static_strong_metrics=compute_retrieval_metrics(
            records, "static_strong_gold_rank", cutoffs
        ),
        self_healing_metrics=compute_retrieval_metrics(
            records, "self_healing_final_gold_rank", cutoffs
        ),
        recovery_attempt_rate=float(fmean(recovery_attempts)),
        average_retry_count=float(fmean(record.retry_count for record in records)),
        unnecessary_modification_rate=(
            float(fmean(unnecessary_modifications))
            if unnecessary_modifications
            else 0.0
        ),
        primary_recovery_cutoff=primary_cutoff,
        recovery_at_cutoff={
            cutoff: compute_recovery_at_cutoff(records, cutoff)
            for cutoff in sorted({primary_cutoff, max(cutoffs)})
        },
        gold_rank_movement=compute_gold_rank_movement(records),
        initial_failures_detected=sum(
            record.self_healing_detected_initial_failure
            for record in initial_failures
        ),
        metrics_path=metrics_path,
        per_example_path=per_example_path,
    )


def _metric_configuration(config: SquadRetrievalBenchmarkConfig) -> dict:
    return {
        "cutoffs": list(config.cutoffs),
        "static_candidate_k": config.static_candidate_k,
        "initial_failure_k": config.initial_failure_k,
        "initial_failure_definition": (
            "initial gold rank is absent or greater than initial_failure_k"
        ),
        "recovered_definition": (
            "initial failure at k and final gold rank is at most the same k"
        ),
        "failed_recovery_definition": (
            "initial failure at k, recovery attempted, and final gold rank is "
            "absent or greater than the same k"
        ),
        "gold_rank_movement_definition": (
            "improved when final rank is smaller, unchanged when equal, and worsened "
            "when larger; an absent gold document is treated as worse than a found one"
        ),
        "unnecessary_modification_definition": (
            "retry_count > 0 among queries whose gold document was present in the "
            "initial top-k"
        ),
        "mean_gold_rank_definition": "mean rank among found gold documents",
        "phase5_thresholds_modified": False,
        "answer_generation_evaluated": False,
    }


class SquadRetrievalBenchmark:
    """Compare dense, static strong, and retrieval-only self-healing RAG."""

    def __init__(
        self,
        dense_retriever: CachedDenseRetriever,
        hybrid_retriever: HybridRetriever,
        reranker: CachedCrossEncoderReranker,
        self_healing_workflow: SelfHealingRAGWorkflow,
        config: SquadRetrievalBenchmarkConfig | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.self_healing_workflow = self_healing_workflow
        self.config = config or SquadRetrievalBenchmarkConfig()
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        config = self.config
        if not config.cutoffs or any(cutoff < 1 for cutoff in config.cutoffs):
            raise ValueError("cutoffs must contain positive integers.")
        if tuple(sorted(set(config.cutoffs))) != config.cutoffs:
            raise ValueError("cutoffs must be unique and sorted.")
        if config.static_candidate_k < max(config.cutoffs):
            raise ValueError("static_candidate_k must cover the largest cutoff.")
        if config.initial_failure_k < 1:
            raise ValueError("initial_failure_k must be at least 1.")
        if config.initial_failure_k > max(config.cutoffs):
            raise ValueError("initial_failure_k must not exceed the largest cutoff.")
        workflow_config = self.self_healing_workflow.config
        if workflow_config.reranked_top_k < max(config.cutoffs):
            raise ValueError("The workflow reranked_top_k must cover every cutoff.")
        if workflow_config.generation_top_k < max(config.cutoffs):
            raise ValueError("The workflow generation_top_k must cover every cutoff.")
        if self.hybrid_retriever.dense_retriever is not self.dense_retriever:
            raise ValueError("Hybrid and benchmark must share the cached dense retriever.")
        if self.self_healing_workflow.dense_retriever is not self.dense_retriever:
            raise ValueError("Self-healing and benchmark must share the dense retriever.")
        if self.self_healing_workflow.hybrid_retriever is not self.hybrid_retriever:
            raise ValueError("Self-healing and benchmark must share the hybrid retriever.")
        if self.self_healing_workflow.reranker is not self.reranker:
            raise ValueError("Static strong and self-healing must share the reranker.")

    def _preload_depth(self) -> int:
        workflow_config = self.self_healing_workflow.config
        return max(
            max(self.config.cutoffs),
            self.config.static_candidate_k,
            workflow_config.initial_retrieval_depth,
            workflow_config.expanded_retrieval_depth,
        )

    def run(
        self,
        dataset: SquadRetrievalDataset,
        progress_callback: (
            Callable[[int, int, SquadRetrievalQuestion], None] | None
        ) = None,
    ) -> SquadRetrievalBenchmarkReport:
        if not dataset.questions:
            raise ValueError("The benchmark dataset contains no questions.")

        queries = [item.question for item in dataset.questions]
        preloaded_dense = self.dense_retriever.preload(
            queries, top_k=self._preload_depth()
        )
        max_cutoff = max(self.config.cutoffs)
        records: list[SquadRetrievalRecord] = []

        for index, (question, dense_full) in enumerate(
            zip(dataset.questions, preloaded_dense), start=1
        ):
            if progress_callback:
                progress_callback(index, len(dataset.questions), question)
            dense_results = dense_full[:max_cutoff]
            dense_ids = tuple(result.id for result in dense_results)

            hybrid_candidates = self.hybrid_retriever.retrieve(
                question.question, top_k=self.config.static_candidate_k
            )
            static_results = self.reranker.rerank(
                question.question, hybrid_candidates, top_k=max_cutoff
            )
            static_ids = tuple(result.id for result in static_results)

            state = self.self_healing_workflow.run_retrieval_only(question.question)
            history = state.get("reranked_results_history", [])
            if not history:
                raise RuntimeError("Self-healing did not retain its initial ranking.")
            initial_ids = tuple(result.id for result in history[0][:max_cutoff])
            final_ids = tuple(
                result.id for result in state["reranked_results"][:max_cutoff]
            )
            initial_failure_type = state["failure_history"][0]
            initial_gold_rank = gold_document_rank(
                initial_ids, question.gold_document_id
            )
            initial_failed = is_initial_failure_at_cutoff(
                initial_gold_rank, self.config.initial_failure_k
            )
            records.append(
                SquadRetrievalRecord(
                    squad_example_id=question.id,
                    question=question.question,
                    gold_document_id=question.gold_document_id,
                    dense_retrieved_ids=dense_ids,
                    dense_retrieved_ranks=tuple(range(1, len(dense_ids) + 1)),
                    dense_gold_rank=gold_document_rank(
                        dense_ids, question.gold_document_id
                    ),
                    static_strong_retrieved_ids=static_ids,
                    static_strong_retrieved_ranks=tuple(
                        range(1, len(static_ids) + 1)
                    ),
                    static_strong_gold_rank=gold_document_rank(
                        static_ids, question.gold_document_id
                    ),
                    self_healing_initial_retrieved_ids=initial_ids,
                    self_healing_initial_retrieved_ranks=tuple(
                        range(1, len(initial_ids) + 1)
                    ),
                    self_healing_initial_gold_rank=initial_gold_rank,
                    self_healing_final_retrieved_ids=final_ids,
                    self_healing_final_retrieved_ranks=tuple(
                        range(1, len(final_ids) + 1)
                    ),
                    self_healing_final_gold_rank=gold_document_rank(
                        final_ids, question.gold_document_id
                    ),
                    initial_retrieval_failed=initial_failed,
                    self_healing_detected_initial_failure=(
                        initial_failure_type != RetrievalFailure.HEALTHY
                    ),
                    self_healing_initial_failure_type=initial_failure_type.value,
                    graph_path=tuple(state.get("path", [])),
                    retry_count=int(state.get("retry_count", 0)),
                    rewritten_query=state.get("rewritten_query"),
                )
            )

        return self._build_and_save_report(dataset, records)

    def _build_and_save_report(
        self,
        dataset: SquadRetrievalDataset,
        records: Sequence[SquadRetrievalRecord],
    ) -> SquadRetrievalBenchmarkReport:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "squad_retrieval_metrics.json"
        per_example_path = output_dir / "squad_retrieval_per_example.csv"
        report = _build_report_from_records(
            records,
            self.config,
            metrics_path,
            per_example_path,
        )
        self._save_metrics(dataset, report)
        self._save_records(report.records, per_example_path)
        return report

    def _save_metrics(
        self,
        dataset: SquadRetrievalDataset,
        report: SquadRetrievalBenchmarkReport,
    ) -> None:
        payload = {
            "benchmark": "SQuAD 2.0 answerable validation retrieval",
            "dataset": {
                "indexed_passages": len(dataset.documents),
                "evaluated_answerable_questions": len(dataset.questions),
                "sample_fingerprint": dataset.fingerprint,
                "sample_manifest": str(dataset.manifest_path),
            },
            "configuration": _metric_configuration(self.config),
            **report.metrics_dict(),
        }
        report.metrics_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _save_records(
        records: Sequence[SquadRetrievalRecord], output_path: Path
    ) -> None:
        fieldnames = list(asdict(records[0]).keys())
        list_fields = {
            "dense_retrieved_ids",
            "dense_retrieved_ranks",
            "static_strong_retrieved_ids",
            "static_strong_retrieved_ranks",
            "self_healing_initial_retrieved_ids",
            "self_healing_initial_retrieved_ranks",
            "self_healing_final_retrieved_ids",
            "self_healing_final_retrieved_ranks",
            "graph_path",
        }
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = asdict(record)
                for field in list_fields:
                    row[field] = json.dumps(row[field], ensure_ascii=False)
                writer.writerow(row)


def _optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def _csv_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _json_tuple(row: dict[str, str], field: str) -> tuple:
    value = row.get(field)
    return tuple(json.loads(value)) if value else ()


def load_squad_retrieval_records(
    per_example_path: str | Path,
) -> list[SquadRetrievalRecord]:
    """Load saved per-example rankings without rerunning retrieval or models."""

    input_path = Path(per_example_path)
    with input_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No per-example rows found in {input_path}.")

    records: list[SquadRetrievalRecord] = []
    for row in rows:
        dense_ids = _json_tuple(row, "dense_retrieved_ids")
        static_ids = _json_tuple(row, "static_strong_retrieved_ids")
        initial_ids = _json_tuple(row, "self_healing_initial_retrieved_ids")
        final_ids = _json_tuple(row, "self_healing_final_retrieved_ids")
        initial_rank = _optional_int(row.get("self_healing_initial_gold_rank"))
        records.append(
            SquadRetrievalRecord(
                squad_example_id=row["squad_example_id"],
                question=row["question"],
                gold_document_id=row["gold_document_id"],
                dense_retrieved_ids=dense_ids,
                dense_retrieved_ranks=(
                    _json_tuple(row, "dense_retrieved_ranks")
                    or tuple(range(1, len(dense_ids) + 1))
                ),
                dense_gold_rank=_optional_int(row.get("dense_gold_rank")),
                static_strong_retrieved_ids=static_ids,
                static_strong_retrieved_ranks=(
                    _json_tuple(row, "static_strong_retrieved_ranks")
                    or tuple(range(1, len(static_ids) + 1))
                ),
                static_strong_gold_rank=_optional_int(
                    row.get("static_strong_gold_rank")
                ),
                self_healing_initial_retrieved_ids=initial_ids,
                self_healing_initial_retrieved_ranks=(
                    _json_tuple(row, "self_healing_initial_retrieved_ranks")
                    or tuple(range(1, len(initial_ids) + 1))
                ),
                self_healing_initial_gold_rank=initial_rank,
                self_healing_final_retrieved_ids=final_ids,
                self_healing_final_retrieved_ranks=(
                    _json_tuple(row, "self_healing_final_retrieved_ranks")
                    or tuple(range(1, len(final_ids) + 1))
                ),
                self_healing_final_gold_rank=_optional_int(
                    row.get("self_healing_final_gold_rank")
                ),
                initial_retrieval_failed=(
                    _csv_bool(row.get("initial_retrieval_failed"))
                    if row.get("initial_retrieval_failed") not in (None, "")
                    else is_initial_failure_at_cutoff(initial_rank, 5)
                ),
                self_healing_detected_initial_failure=_csv_bool(
                    row.get("self_healing_detected_initial_failure")
                ),
                self_healing_initial_failure_type=row.get(
                    "self_healing_initial_failure_type", ""
                ),
                graph_path=_json_tuple(row, "graph_path"),
                retry_count=int(row.get("retry_count") or 0),
                rewritten_query=row.get("rewritten_query") or None,
            )
        )
    return records


def recalculate_squad_retrieval_metrics_from_csv(
    per_example_path: str | Path = "results/squad_retrieval_per_example.csv",
    metrics_path: str | Path = "results/squad_retrieval_metrics.json",
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
    initial_failure_k: int = 5,
) -> SquadRetrievalBenchmarkReport:
    """Recalculate metrics from saved ranks and update JSON without model calls."""

    input_path = Path(per_example_path)
    output_path = Path(metrics_path)
    records = load_squad_retrieval_records(input_path)
    existing_payload = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {
            "benchmark": "SQuAD 2.0 answerable validation retrieval",
            "dataset": {"evaluated_answerable_questions": len(records)},
        }
    )
    existing_config = existing_payload.get("configuration", {})
    config = SquadRetrievalBenchmarkConfig(
        cutoffs=cutoffs,
        static_candidate_k=int(existing_config.get("static_candidate_k", max(cutoffs))),
        initial_failure_k=initial_failure_k,
        output_dir=output_path.parent,
    )
    report = _build_report_from_records(records, config, output_path, input_path)
    existing_payload["configuration"] = _metric_configuration(config)
    existing_payload.update(report.metrics_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(existing_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
