"""Baseline retrieval-augmented generation components."""

from .baseline import (
    BaselineRAG,
    Document,
    FAISSRetriever,
    LocalQwenGenerator,
    RAGAnswer,
    RAGConfig,
    RetrievalResult,
    load_documents,
)
from .bm25 import BM25Result, BM25Retriever
from .hybrid import (
    HybridResult,
    HybridRetriever,
    RetrievalAgreement,
    measure_retrieval_overlap,
)
from .reranker import CrossEncoderReranker, RerankedResult
from .diagnostics import RetrievalDiagnostics, compute_retrieval_diagnostics
from .failure_detection import (
    FailureDetection,
    FailureDetectorConfig,
    RetrievalFailure,
    RetrievalFailureDetector,
)
from .self_healing import (
    LocalQwenQueryRewriter,
    RecoveryAction,
    SelfHealingRAGWorkflow,
    SelfHealingState,
    SelfHealingWorkflowConfig,
)

__all__ = [
    "BaselineRAG",
    "BM25Result",
    "BM25Retriever",
    "CrossEncoderReranker",
    "Document",
    "FAISSRetriever",
    "FailureDetection",
    "FailureDetectorConfig",
    "HybridResult",
    "HybridRetriever",
    "LocalQwenGenerator",
    "LocalQwenQueryRewriter",
    "RAGAnswer",
    "RAGConfig",
    "RetrievalResult",
    "RetrievalAgreement",
    "RetrievalDiagnostics",
    "RetrievalFailure",
    "RetrievalFailureDetector",
    "RerankedResult",
    "RecoveryAction",
    "SelfHealingRAGWorkflow",
    "SelfHealingState",
    "SelfHealingWorkflowConfig",
    "compute_retrieval_diagnostics",
    "load_documents",
    "measure_retrieval_overlap",
]
