"""Lightweight, reproducible evaluation for baseline and self-healing RAG."""

from .dataset import EvaluationExample, load_evaluation_set
from .evaluate import (
    EvaluationMetrics,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRunner,
)
from .groundedness import (
    GroundednessConfig,
    GroundednessResult,
    check_groundedness,
)

__all__ = [
    "EvaluationExample",
    "EvaluationMetrics",
    "EvaluationRecord",
    "EvaluationReport",
    "EvaluationRunner",
    "GroundednessConfig",
    "GroundednessResult",
    "check_groundedness",
    "load_evaluation_set",
]
