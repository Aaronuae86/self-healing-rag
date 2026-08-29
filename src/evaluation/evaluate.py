"""Run reproducible baseline-vs-self-healing evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Callable, Sequence

from src.rag import (
    BaselineRAG,
    RecoveryAction,
    RetrievalFailure,
    SelfHealingRAGWorkflow,
)

from .dataset import EvaluationExample
from .groundedness import GroundednessConfig, check_groundedness


RECOVERY_FAILURE_TYPES = {
    RetrievalFailure.AMBIGUOUS,
    RetrievalFailure.WEAK_RETRIEVAL,
}
ABSTENTION_PHRASES = (
    "do not have enough information",
    "not enough information",
    "insufficient evidence",
    "cannot answer from the provided context",
    "cannot answer based on the provided context",
)


@dataclass(frozen=True)
class EvaluationRecord:
    """Per-example outputs and metric decisions for one system."""

    example_id: str
    system: str
    category: str
    query: str
    expected_failure_type: str
    predicted_failure_type: str
    failure_classification_correct: bool
    should_answer: bool
    abstained: bool
    abstention_correct: bool
    expected_relevant_document_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    retrieval_hit: bool | None
    recovery_expected: bool
    recovery_attempted: bool
    recovery_success: bool | None
    unnecessary_recovery: bool
    retry_count: int
    recovery_action: str
    answer: str
    keyword_coverage: float | None
    keyword_success: bool | None
    groundedness_score: float | None
    groundedness_pass: bool | None
    path: tuple[str, ...]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate metrics for one evaluated system."""

    system: str
    total_examples: int
    failure_classification_accuracy: float
    abstention_accuracy: float
    retrieval_hit_rate: float | None
    recovery_success_rate: float | None
    unnecessary_recovery_rate: float | None
    average_retry_count: float
    answered_percentage: float
    abstained_percentage: float
    average_groundedness_score: float | None
    grounded_answer_rate: float | None
    expected_keyword_hit_rate: float | None
    per_category: dict[str, dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    """Complete baseline and self-healing evaluation output."""

    baseline_records: tuple[EvaluationRecord, ...]
    self_healing_records: tuple[EvaluationRecord, ...]
    baseline_metrics: EvaluationMetrics
    self_healing_metrics: EvaluationMetrics

    def failed_examples(self, system: str) -> list[EvaluationRecord]:
        if system not in {"baseline", "self_healing"}:
            raise ValueError("system must be 'baseline' or 'self_healing'.")
        records = (
            self.baseline_records
            if system == "baseline"
            else self.self_healing_records
        )
        return [record for record in records if record.failure_reasons]


def _is_abstention(answer: str) -> bool:
    normalized = answer.lower()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def _keyword_coverage(answer: str, keywords: Sequence[str]) -> float | None:
    if not keywords:
        return None
    normalized = answer.lower()
    hits = sum(keyword.lower() in normalized for keyword in keywords)
    return hits / len(keywords)


def _retrieval_hit(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str]
) -> bool | None:
    if not relevant_ids:
        return None
    return bool(set(retrieved_ids) & set(relevant_ids))


def _failure_reasons(
    *,
    classification_correct: bool,
    abstention_correct: bool,
    retrieval_hit: bool | None,
    recovery_success: bool | None,
    unnecessary_recovery: bool,
    keyword_success: bool | None,
    groundedness_pass: bool | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not classification_correct:
        reasons.append("failure classification mismatch")
    if not abstention_correct:
        reasons.append("answer/abstention mismatch")
    if retrieval_hit is False:
        reasons.append("relevant document was not retrieved")
    if recovery_success is False:
        reasons.append("expected recovery did not succeed")
    if unnecessary_recovery:
        reasons.append("unnecessary recovery on a healthy query")
    if keyword_success is False:
        reasons.append("expected answer keywords were missing")
    if groundedness_pass is False:
        reasons.append("answer failed lexical groundedness check")
    return tuple(reasons)


def _mean_optional(values: Sequence[float]) -> float | None:
    return float(fmean(values)) if values else None


def _aggregate_values(records: Sequence[EvaluationRecord]) -> dict:
    total = len(records)
    if total == 0:
        raise ValueError("Cannot aggregate an empty evaluation record list.")

    retrieval_values = [record.retrieval_hit for record in records if record.retrieval_hit is not None]
    recovery_values = [
        record.recovery_success
        for record in records
        if record.recovery_success is not None
    ]
    healthy_records = [
        record
        for record in records
        if record.expected_failure_type == RetrievalFailure.HEALTHY.value
    ]
    groundedness_scores = [
        record.groundedness_score
        for record in records
        if record.groundedness_score is not None
    ]
    groundedness_passes = [
        record.groundedness_pass
        for record in records
        if record.groundedness_pass is not None
    ]
    keyword_coverages = [
        record.keyword_coverage
        for record in records
        if record.keyword_coverage is not None
    ]
    abstained_count = sum(record.abstained for record in records)

    return {
        "total_examples": total,
        "failure_classification_accuracy": fmean(
            record.failure_classification_correct for record in records
        ),
        "abstention_accuracy": fmean(record.abstention_correct for record in records),
        "retrieval_hit_rate": _mean_optional(
            [float(value) for value in retrieval_values]
        ),
        "recovery_success_rate": _mean_optional(
            [float(value) for value in recovery_values]
        ),
        "unnecessary_recovery_rate": _mean_optional(
            [float(record.recovery_attempted) for record in healthy_records]
        ),
        "average_retry_count": fmean(record.retry_count for record in records),
        "answered_percentage": 100.0 * (total - abstained_count) / total,
        "abstained_percentage": 100.0 * abstained_count / total,
        "average_groundedness_score": _mean_optional(groundedness_scores),
        "grounded_answer_rate": _mean_optional(
            [float(value) for value in groundedness_passes]
        ),
        "expected_keyword_hit_rate": _mean_optional(keyword_coverages),
    }


def _aggregate_metrics(
    system: str, records: Sequence[EvaluationRecord]
) -> EvaluationMetrics:
    overall = _aggregate_values(records)
    categories = sorted({record.category for record in records})
    per_category = {
        category: _aggregate_values(
            [record for record in records if record.category == category]
        )
        for category in categories
    }
    return EvaluationMetrics(system=system, per_category=per_category, **overall)


class EvaluationRunner:
    """Run the fixed labels through baseline and self-healing RAG."""

    def __init__(
        self,
        baseline_rag: BaselineRAG,
        self_healing_workflow: SelfHealingRAGWorkflow,
        baseline_top_k: int = 3,
        groundedness_config: GroundednessConfig | None = None,
    ) -> None:
        if baseline_top_k < 1:
            raise ValueError("baseline_top_k must be at least 1.")
        self.baseline_rag = baseline_rag
        self.self_healing_workflow = self_healing_workflow
        self.baseline_top_k = baseline_top_k
        self.groundedness_config = groundedness_config or GroundednessConfig()

    def _evaluate_baseline(self, example: EvaluationExample) -> EvaluationRecord:
        result = self.baseline_rag.answer_question(
            example.query, top_k=self.baseline_top_k
        )
        retrieved_ids = tuple(item.id for item in result.retrieved_documents)
        retrieval_hit = _retrieval_hit(
            retrieved_ids, example.expected_relevant_document_ids
        )
        abstained = _is_abstention(result.answer)
        abstention_correct = abstained == (not example.should_answer)
        predicted_failure = RetrievalFailure.HEALTHY
        classification_correct = predicted_failure == example.expected_failure_type
        recovery_expected = example.expected_failure_type in RECOVERY_FAILURE_TYPES
        recovery_success = False if recovery_expected else None
        keyword_coverage = _keyword_coverage(
            result.answer, example.expected_answer_keywords
        )
        keyword_success = (
            keyword_coverage == 1.0 if keyword_coverage is not None else None
        )
        groundedness = (
            check_groundedness(
                result.answer,
                result.retrieved_documents,
                self.groundedness_config,
            )
            if not abstained
            else None
        )
        failure_reasons = _failure_reasons(
            classification_correct=classification_correct,
            abstention_correct=abstention_correct,
            retrieval_hit=retrieval_hit,
            recovery_success=recovery_success,
            unnecessary_recovery=False,
            keyword_success=keyword_success,
            groundedness_pass=(groundedness.is_grounded if groundedness else None),
        )
        return EvaluationRecord(
            example_id=example.id,
            system="baseline",
            category=example.category,
            query=example.query,
            expected_failure_type=example.expected_failure_type.value,
            predicted_failure_type=predicted_failure.value,
            failure_classification_correct=classification_correct,
            should_answer=example.should_answer,
            abstained=abstained,
            abstention_correct=abstention_correct,
            expected_relevant_document_ids=example.expected_relevant_document_ids,
            retrieved_document_ids=retrieved_ids,
            retrieval_hit=retrieval_hit,
            recovery_expected=recovery_expected,
            recovery_attempted=False,
            recovery_success=recovery_success,
            unnecessary_recovery=False,
            retry_count=0,
            recovery_action=RecoveryAction.NONE.value,
            answer=result.answer,
            keyword_coverage=keyword_coverage,
            keyword_success=keyword_success,
            groundedness_score=(groundedness.coverage if groundedness else None),
            groundedness_pass=(groundedness.is_grounded if groundedness else None),
            path=("RETRIEVE", "GENERATE"),
            failure_reasons=failure_reasons,
        )

    def _evaluate_self_healing(
        self, example: EvaluationExample
    ) -> EvaluationRecord:
        state = self.self_healing_workflow.run(example.query)
        retrieved_documents = state.get("retrieved_documents", [])
        retrieved_ids = tuple(item.id for item in retrieved_documents)
        retrieval_hit = _retrieval_hit(
            retrieved_ids, example.expected_relevant_document_ids
        )
        failure_history = state.get("failure_history", [])
        predicted_failure = (
            failure_history[0] if failure_history else state["failure_type"]
        )
        classification_correct = predicted_failure == example.expected_failure_type
        abstained = state["recovery_action"] == RecoveryAction.ABSTAIN
        abstention_correct = abstained == (not example.should_answer)
        retry_count = state.get("retry_count", 0)
        recovery_attempted = retry_count > 0
        recovery_expected = example.expected_failure_type in RECOVERY_FAILURE_TYPES
        expected_action = {
            RetrievalFailure.AMBIGUOUS: RecoveryAction.INCREASE_RETRIEVAL_DEPTH,
            RetrievalFailure.WEAK_RETRIEVAL: RecoveryAction.REWRITE_QUERY,
        }.get(example.expected_failure_type)
        recovery_success = None
        if recovery_expected:
            retrieval_succeeded = retrieval_hit is not False
            recovery_success = (
                recovery_attempted
                and state["recovery_action"] == expected_action
                and retrieval_succeeded
                and abstention_correct
            )
        unnecessary_recovery = (
            example.expected_failure_type == RetrievalFailure.HEALTHY
            and recovery_attempted
        )
        keyword_coverage = _keyword_coverage(
            state["final_answer"], example.expected_answer_keywords
        )
        keyword_success = (
            keyword_coverage == 1.0 if keyword_coverage is not None else None
        )
        groundedness = (
            check_groundedness(
                state["final_answer"],
                retrieved_documents,
                self.groundedness_config,
            )
            if not abstained
            else None
        )
        failure_reasons = _failure_reasons(
            classification_correct=classification_correct,
            abstention_correct=abstention_correct,
            retrieval_hit=retrieval_hit,
            recovery_success=recovery_success,
            unnecessary_recovery=unnecessary_recovery,
            keyword_success=keyword_success,
            groundedness_pass=(groundedness.is_grounded if groundedness else None),
        )
        return EvaluationRecord(
            example_id=example.id,
            system="self_healing",
            category=example.category,
            query=example.query,
            expected_failure_type=example.expected_failure_type.value,
            predicted_failure_type=predicted_failure.value,
            failure_classification_correct=classification_correct,
            should_answer=example.should_answer,
            abstained=abstained,
            abstention_correct=abstention_correct,
            expected_relevant_document_ids=example.expected_relevant_document_ids,
            retrieved_document_ids=retrieved_ids,
            retrieval_hit=retrieval_hit,
            recovery_expected=recovery_expected,
            recovery_attempted=recovery_attempted,
            recovery_success=recovery_success,
            unnecessary_recovery=unnecessary_recovery,
            retry_count=retry_count,
            recovery_action=state["recovery_action"].value,
            answer=state["final_answer"],
            keyword_coverage=keyword_coverage,
            keyword_success=keyword_success,
            groundedness_score=(groundedness.coverage if groundedness else None),
            groundedness_pass=(groundedness.is_grounded if groundedness else None),
            path=tuple(state.get("path", [])),
            failure_reasons=failure_reasons,
        )

    def run(
        self,
        examples: Sequence[EvaluationExample],
        progress_callback: Callable[[int, int, EvaluationExample], None] | None = None,
    ) -> EvaluationReport:
        """Evaluate every example with both systems in fixed dataset order."""

        if not examples:
            raise ValueError("At least one evaluation example is required.")
        baseline_records: list[EvaluationRecord] = []
        self_healing_records: list[EvaluationRecord] = []
        total = len(examples)
        for index, example in enumerate(examples, start=1):
            if progress_callback:
                progress_callback(index, total, example)
            baseline_records.append(self._evaluate_baseline(example))
            self_healing_records.append(self._evaluate_self_healing(example))

        return EvaluationReport(
            baseline_records=tuple(baseline_records),
            self_healing_records=tuple(self_healing_records),
            baseline_metrics=_aggregate_metrics("baseline", baseline_records),
            self_healing_metrics=_aggregate_metrics(
                "self_healing", self_healing_records
            ),
        )
