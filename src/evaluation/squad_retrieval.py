"""Retrieval-only SQuAD 2.0 benchmark for three RAG configurations."""

from __future__ import annotations

import csv
import json
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
class SquadRetrievalBenchmarkReport:
    """Aggregate metrics, initial-failure analysis, and saved result paths."""

    records: tuple[SquadRetrievalRecord, ...]
    dense_metrics: RetrievalMetricSummary
    static_strong_metrics: RetrievalMetricSummary
    self_healing_metrics: RetrievalMetricSummary
    recovery_attempt_rate: float
    average_retry_count: float
    unnecessary_modification_rate: float
    initial_failure_count: int
    initial_failure_rate: float
    initial_failures_detected: int
    initial_failures_with_gold_after_healing: int
    metrics_path: Path
    per_example_path: Path

    def metrics_dict(self) -> dict:
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
                "count": self.initial_failure_count,
                "rate": self.initial_failure_rate,
                "detected_by_self_healing": self.initial_failures_detected,
                "gold_present_after_healing": (
                    self.initial_failures_with_gold_after_healing
                ),
            },
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


def gold_document_rank(
    retrieved_document_ids: Sequence[str], gold_document_id: str
) -> int | None:
    """Return the one-based gold rank, or None when it was not retrieved."""

    try:
        return retrieved_document_ids.index(gold_document_id) + 1
    except ValueError:
        return None


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
            initial_failed = (
                initial_gold_rank is None
                or initial_gold_rank > self.config.initial_failure_k
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
        cutoffs = self.config.cutoffs
        initial_failures = [record for record in records if record.initial_retrieval_failed]
        objective_initial_successes = [
            record for record in records if not record.initial_retrieval_failed
        ]
        recovery_attempts = [record.retry_count > 0 for record in records]
        unnecessary_modifications = [
            record.retry_count > 0 for record in objective_initial_successes
        ]
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "squad_retrieval_metrics.json"
        per_example_path = output_dir / "squad_retrieval_per_example.csv"

        report = SquadRetrievalBenchmarkReport(
            records=tuple(records),
            dense_metrics=compute_retrieval_metrics(
                records, "dense_gold_rank", cutoffs
            ),
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
            initial_failure_count=len(initial_failures),
            initial_failure_rate=len(initial_failures) / len(records),
            initial_failures_detected=sum(
                record.self_healing_detected_initial_failure
                for record in initial_failures
            ),
            initial_failures_with_gold_after_healing=sum(
                record.self_healing_final_gold_rank is not None
                for record in initial_failures
            ),
            metrics_path=metrics_path,
            per_example_path=per_example_path,
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
            "configuration": {
                "cutoffs": list(self.config.cutoffs),
                "static_candidate_k": self.config.static_candidate_k,
                "initial_failure_k": self.config.initial_failure_k,
                "initial_failure_definition": (
                    "gold document absent from initial self-healing reranked top-k"
                ),
                "unnecessary_modification_definition": (
                    "retry_count > 0 among queries whose gold document was present "
                    "in the initial top-k"
                ),
                "mean_gold_rank_definition": "mean rank among found gold documents",
                "phase5_thresholds_modified": False,
                "answer_generation_evaluated": False,
            },
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
