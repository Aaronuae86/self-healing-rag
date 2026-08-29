"""Train-only calibration and objective failure-detection metrics."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence

from src.rag import (
    FailureDetectorConfig,
    RetrievalDiagnostics,
    RetrievalFailure,
    RetrievalFailureDetector,
)


class ObjectiveFailureLabel(str, Enum):
    """Labels derived from corpus membership and observed gold rank."""

    HEALTHY = "HEALTHY"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    NATURAL_UNANSWERABLE = "NATURAL_UNANSWERABLE"


@dataclass(frozen=True)
class DiagnosticCalibrationExample:
    """One train-only diagnostic observation and objective outcome."""

    example_id: str
    objective_label: ObjectiveFailureLabel
    diagnostics: RetrievalDiagnostics


@dataclass(frozen=True)
class BinaryFailureMetrics:
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FailureModeMetrics:
    per_class: dict[str, dict[str, float | int]]
    macro_f1: float
    confusion_matrix: dict[str, dict[str, int]]
    raw_phase5_prediction_confusion: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DetectorEvaluation:
    binary: BinaryFailureMetrics
    failure_mode: FailureModeMetrics
    insufficient_evidence_recall: float
    natural_unanswerable_prediction_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DetectorCalibrationResult:
    """Original and selected configs with train-only selection evidence."""

    label: str
    seed: int
    candidate_count: int
    selection_score: float
    original_config: FailureDetectorConfig
    selected_config: FailureDetectorConfig
    original_train_metrics: DetectorEvaluation
    selected_train_metrics: DetectorEvaluation
    diagnostic_distributions: dict[str, dict[str, dict[str, float | int | None]]]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "seed": self.seed,
            "candidate_count": self.candidate_count,
            "selection_score": self.selection_score,
            "original_config": asdict(self.original_config),
            "selected_config": asdict(self.selected_config),
            "original_train_metrics": self.original_train_metrics.to_dict(),
            "selected_train_metrics": self.selected_train_metrics.to_dict(),
            "diagnostic_distributions": self.diagnostic_distributions,
        }


SUPERVISED_LABELS = (
    ObjectiveFailureLabel.HEALTHY,
    ObjectiveFailureLabel.RETRIEVAL_FAILURE,
    ObjectiveFailureLabel.MISSING_EVIDENCE,
)


def objective_label_from_rank(
    gold_rank: int | None,
    cutoff: int = 5,
    gold_intentionally_removed: bool = False,
) -> ObjectiveFailureLabel:
    if gold_intentionally_removed:
        return ObjectiveFailureLabel.MISSING_EVIDENCE
    if gold_rank is not None and gold_rank <= cutoff:
        return ObjectiveFailureLabel.HEALTHY
    return ObjectiveFailureLabel.RETRIEVAL_FAILURE


def prediction_to_objective_mode(
    prediction: RetrievalFailure,
) -> ObjectiveFailureLabel:
    if prediction == RetrievalFailure.HEALTHY:
        return ObjectiveFailureLabel.HEALTHY
    if prediction == RetrievalFailure.INSUFFICIENT_EVIDENCE:
        return ObjectiveFailureLabel.MISSING_EVIDENCE
    return ObjectiveFailureLabel.RETRIEVAL_FAILURE


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_binary_failure_metrics(
    actual: Sequence[ObjectiveFailureLabel],
    predicted: Sequence[RetrievalFailure],
) -> BinaryFailureMetrics:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and aligned.")
    pairs = [
        (truth != ObjectiveFailureLabel.HEALTHY, guess != RetrievalFailure.HEALTHY)
        for truth, guess in zip(actual, predicted)
        if truth != ObjectiveFailureLabel.NATURAL_UNANSWERABLE
    ]
    true_positive = sum(truth and guess for truth, guess in pairs)
    false_positive = sum(not truth and guess for truth, guess in pairs)
    true_negative = sum(not truth and not guess for truth, guess in pairs)
    false_negative = sum(truth and not guess for truth, guess in pairs)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return BinaryFailureMetrics(
        precision=precision,
        recall=recall,
        f1=_safe_ratio(2 * precision * recall, precision + recall),
        false_positive_rate=_safe_ratio(false_positive, false_positive + true_negative),
        false_negative_rate=_safe_ratio(false_negative, false_negative + true_positive),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def compute_failure_mode_metrics(
    actual: Sequence[ObjectiveFailureLabel],
    predicted: Sequence[RetrievalFailure],
) -> FailureModeMetrics:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and aligned.")
    supervised_pairs = [
        (truth, prediction_to_objective_mode(guess))
        for truth, guess in zip(actual, predicted)
        if truth in SUPERVISED_LABELS
    ]
    matrix = {
        truth.value: {guess.value: 0 for guess in SUPERVISED_LABELS}
        for truth in SUPERVISED_LABELS
    }
    raw_matrix = {
        truth.value: {guess.value: 0 for guess in RetrievalFailure}
        for truth in SUPERVISED_LABELS
    }
    for truth, guess in zip(actual, predicted):
        if truth in SUPERVISED_LABELS:
            raw_matrix[truth.value][guess.value] += 1
    for truth, guess in supervised_pairs:
        matrix[truth.value][guess.value] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in SUPERVISED_LABELS:
        name = label.value
        true_positive = matrix[name][name]
        false_positive = sum(matrix[other.value][name] for other in SUPERVISED_LABELS if other != label)
        false_negative = sum(matrix[name][other.value] for other in SUPERVISED_LABELS if other != label)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[name].values()),
        }
    return FailureModeMetrics(
        per_class=per_class,
        macro_f1=float(fmean(f1_values)),
        confusion_matrix=matrix,
        raw_phase5_prediction_confusion=raw_matrix,
    )


def evaluate_detector(
    examples: Sequence[DiagnosticCalibrationExample],
    detector: RetrievalFailureDetector,
) -> DetectorEvaluation:
    if not examples:
        raise ValueError("At least one diagnostic example is required.")
    actual = [example.objective_label for example in examples]
    predicted = [detector.classify(example.diagnostics).failure_type for example in examples]
    missing_predictions = [
        guess
        for truth, guess in zip(actual, predicted)
        if truth == ObjectiveFailureLabel.MISSING_EVIDENCE
    ]
    natural_predictions: dict[str, int] = {}
    for truth, guess in zip(actual, predicted):
        if truth == ObjectiveFailureLabel.NATURAL_UNANSWERABLE:
            natural_predictions[guess.value] = natural_predictions.get(guess.value, 0) + 1
    return DetectorEvaluation(
        binary=compute_binary_failure_metrics(actual, predicted),
        failure_mode=compute_failure_mode_metrics(actual, predicted),
        insufficient_evidence_recall=_safe_ratio(
            sum(item == RetrievalFailure.INSUFFICIENT_EVIDENCE for item in missing_predictions),
            len(missing_predictions),
        ),
        natural_unanswerable_prediction_counts=natural_predictions,
    )


DIAGNOSTIC_SIGNALS = (
    "dense_top1_score",
    "dense_top1_top2_margin",
    "dense_average_top_k_score",
    "bm25_top1_score",
    "dense_bm25_overlap_ratio",
    "hybrid_top_rrf_score",
    "cross_encoder_top_score",
    "cross_encoder_top1_top2_margin",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_diagnostic_distributions(
    examples: Sequence[DiagnosticCalibrationExample],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    output: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for label in SUPERVISED_LABELS:
        group = [item for item in examples if item.objective_label == label]
        output[label.value] = {}
        for signal in DIAGNOSTIC_SIGNALS:
            values = [
                float(value)
                for item in group
                if (value := getattr(item.diagnostics, signal)) is not None
            ]
            output[label.value][signal] = {
                "count": len(values),
                "mean": float(fmean(values)) if values else None,
                "std": float(pstdev(values)) if len(values) > 1 else 0.0 if values else None,
                "min": min(values) if values else None,
                "p25": _percentile(values, 0.25) if values else None,
                "p50": _percentile(values, 0.50) if values else None,
                "p75": _percentile(values, 0.75) if values else None,
                "max": max(values) if values else None,
            }
    return output


def _signal_values(
    examples: Iterable[DiagnosticCalibrationExample], signal: str
) -> list[float]:
    if signal == "cross_encoder_second_score":
        return [
            float(top - margin)
            for item in examples
            if (top := item.diagnostics.cross_encoder_top_score) is not None
            and (margin := item.diagnostics.cross_encoder_top1_top2_margin)
            is not None
        ]
    return [
        float(value)
        for item in examples
        if (value := getattr(item.diagnostics, signal)) is not None
    ]


def _candidate_values(values: Sequence[float], fallback: float) -> tuple[float, ...]:
    if not values:
        return (fallback,)
    return tuple(
        dict.fromkeys(
            round(_percentile(values, fraction), 8)
            for fraction in (0.15, 0.30, 0.45, 0.60, 0.75)
        )
    )


def _selection_score(metrics: DetectorEvaluation) -> float:
    binary = metrics.binary
    specificity = 1.0 - binary.false_positive_rate
    return (
        0.35 * binary.recall
        + 0.25 * specificity
        + 0.25 * metrics.insufficient_evidence_recall
        + 0.15 * metrics.failure_mode.macro_f1
    )


def calibrate_failure_detector(
    examples: Sequence[DiagnosticCalibrationExample],
    original_config: FailureDetectorConfig | None = None,
    seed: int = 42,
    candidate_count: int = 256,
) -> DetectorCalibrationResult:
    """Search train-derived quantile combinations without validation feedback."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1.")
    supervised = [
        item
        for item in examples
        if item.objective_label != ObjectiveFailureLabel.NATURAL_UNANSWERABLE
    ]
    if not supervised:
        raise ValueError("Calibration requires supervised train examples.")
    base = original_config or FailureDetectorConfig()
    missing = [item for item in supervised if item.objective_label == ObjectiveFailureLabel.MISSING_EVIDENCE]
    rng = random.Random(seed)

    all_candidates = {
        signal: _candidate_values(_signal_values(supervised, signal), getattr(base, field))
        for signal, field in {
            "dense_top1_score": "minimum_dense_top1",
            "dense_average_top_k_score": "minimum_dense_average",
            "bm25_top1_score": "minimum_bm25_top1",
            "dense_bm25_overlap_ratio": "minimum_overlap_ratio",
            "hybrid_top_rrf_score": "minimum_hybrid_top_rrf",
            "cross_encoder_top_score": "minimum_cross_encoder_top",
            "cross_encoder_top1_top2_margin": "maximum_weak_cross_encoder_margin",
        }.items()
    }
    ambiguity_candidates = {
        "dense_margin": _candidate_values(
            _signal_values(supervised, "dense_top1_top2_margin"),
            base.maximum_ambiguous_dense_margin,
        ),
        "cross_margin": _candidate_values(
            _signal_values(supervised, "cross_encoder_top1_top2_margin"),
            base.maximum_ambiguous_cross_encoder_margin,
        ),
        "cross_second": _candidate_values(
            _signal_values(supervised, "cross_encoder_second_score"),
            base.minimum_plausible_cross_encoder_top2,
        ),
    }
    missing_candidates = {
        signal: _candidate_values(_signal_values(missing, signal), getattr(base, field))
        for signal, field in {
            "dense_top1_score": "insufficient_dense_top1",
            "dense_average_top_k_score": "insufficient_dense_average",
            "bm25_top1_score": "insufficient_bm25_top1",
            "dense_bm25_overlap_ratio": "insufficient_overlap_ratio",
            "hybrid_top_rrf_score": "insufficient_hybrid_top_rrf",
            "cross_encoder_top_score": "insufficient_cross_encoder_top",
            "cross_encoder_top1_top2_margin": "minimum_clear_cross_encoder_margin",
        }.items()
    }

    configs = [base]
    seen = {tuple(asdict(base).items())}
    attempts = 0
    maximum_attempts = candidate_count * 50
    while len(configs) < candidate_count + 1 and attempts < maximum_attempts:
        attempts += 1
        candidate = replace(
            base,
            minimum_dense_top1=rng.choice(all_candidates["dense_top1_score"]),
            minimum_dense_average=rng.choice(all_candidates["dense_average_top_k_score"]),
            minimum_bm25_top1=rng.choice(all_candidates["bm25_top1_score"]),
            minimum_overlap_ratio=rng.choice(all_candidates["dense_bm25_overlap_ratio"]),
            minimum_hybrid_top_rrf=rng.choice(all_candidates["hybrid_top_rrf_score"]),
            minimum_cross_encoder_top=rng.choice(all_candidates["cross_encoder_top_score"]),
            maximum_weak_cross_encoder_margin=rng.choice(all_candidates["cross_encoder_top1_top2_margin"]),
            maximum_ambiguous_dense_margin=rng.choice(
                ambiguity_candidates["dense_margin"]
            ),
            maximum_ambiguous_cross_encoder_margin=rng.choice(
                ambiguity_candidates["cross_margin"]
            ),
            minimum_plausible_cross_encoder_top2=rng.choice(
                ambiguity_candidates["cross_second"]
            ),
            minimum_weak_signals=rng.choice((2, 3, 4, 5)),
            insufficient_dense_top1=rng.choice(missing_candidates["dense_top1_score"]),
            insufficient_dense_average=rng.choice(missing_candidates["dense_average_top_k_score"]),
            insufficient_bm25_top1=rng.choice(missing_candidates["bm25_top1_score"]),
            insufficient_overlap_ratio=rng.choice(missing_candidates["dense_bm25_overlap_ratio"]),
            insufficient_hybrid_top_rrf=rng.choice(missing_candidates["hybrid_top_rrf_score"]),
            insufficient_cross_encoder_top=rng.choice(missing_candidates["cross_encoder_top_score"]),
            minimum_clear_cross_encoder_margin=rng.choice(
                missing_candidates["cross_encoder_top1_top2_margin"]
            ),
            minimum_core_insufficient_signals=rng.choice((1, 2, 3)),
            minimum_insufficient_signals=rng.choice((2, 3, 4, 5)),
        )
        key = tuple(asdict(candidate).items())
        if key not in seen:
            seen.add(key)
            configs.append(candidate)

    original_metrics = evaluate_detector(supervised, RetrievalFailureDetector(base))
    best_config = base
    best_metrics = original_metrics
    best_score = _selection_score(best_metrics)
    for config in configs[1:]:
        metrics = evaluate_detector(supervised, RetrievalFailureDetector(config))
        score = _selection_score(metrics)
        tie_breaker = (metrics.binary.f1, metrics.failure_mode.macro_f1)
        best_tie_breaker = (best_metrics.binary.f1, best_metrics.failure_mode.macro_f1)
        if score > best_score or (score == best_score and tie_breaker > best_tie_breaker):
            best_config = config
            best_metrics = metrics
            best_score = score

    return DetectorCalibrationResult(
        label="TRAIN-CALIBRATED",
        seed=seed,
        candidate_count=len(configs),
        selection_score=best_score,
        original_config=base,
        selected_config=best_config,
        original_train_metrics=original_metrics,
        selected_train_metrics=best_metrics,
        diagnostic_distributions=summarize_diagnostic_distributions(supervised),
    )


def save_detector_calibration(
    result: DetectorCalibrationResult,
    path: str | Path = "results/squad_detector_calibration.json",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
