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
from .squad_dataset import (
    SquadRetrievalDataset,
    SquadRetrievalQuestion,
    SquadSamplingConfig,
    prepare_squad_retrieval_dataset,
    sampling_config_to_dict,
    stable_context_id,
)
from .squad_retrieval import (
    CachedCrossEncoderReranker,
    CachedDenseRetriever,
    RetrievalMetricSummary,
    SquadRetrievalBenchmark,
    SquadRetrievalBenchmarkConfig,
    SquadRetrievalBenchmarkReport,
    SquadRetrievalRecord,
    compute_retrieval_metrics,
    gold_document_rank,
)

__all__ = [
    "EvaluationExample",
    "EvaluationMetrics",
    "EvaluationRecord",
    "EvaluationReport",
    "EvaluationRunner",
    "GroundednessConfig",
    "GroundednessResult",
    "CachedCrossEncoderReranker",
    "CachedDenseRetriever",
    "RetrievalMetricSummary",
    "SquadRetrievalBenchmark",
    "SquadRetrievalBenchmarkConfig",
    "SquadRetrievalBenchmarkReport",
    "SquadRetrievalDataset",
    "SquadRetrievalQuestion",
    "SquadRetrievalRecord",
    "SquadSamplingConfig",
    "check_groundedness",
    "compute_retrieval_metrics",
    "gold_document_rank",
    "load_evaluation_set",
    "prepare_squad_retrieval_dataset",
    "sampling_config_to_dict",
    "stable_context_id",
]
