"""Baseline retrieval-augmented generation components."""

from .baseline import (
    BaselineRAG,
    Document,
    FAISSRetriever,
    RAGAnswer,
    RAGConfig,
    RetrievalResult,
    load_documents,
)
from .bm25 import BM25Result, BM25Retriever
from .diagnostics import RetrievalDiagnostics, compute_retrieval_diagnostics
from .hybrid import (
    HybridResult,
    HybridRetriever,
    RetrievalAgreement,
    measure_retrieval_overlap,
)
from .reranker import CrossEncoderReranker, RerankedResult

__all__ = [
    "BaselineRAG",
    "BM25Result",
    "BM25Retriever",
    "CrossEncoderReranker",
    "Document",
    "FAISSRetriever",
    "HybridResult",
    "HybridRetriever",
    "RAGAnswer",
    "RAGConfig",
    "RetrievalResult",
    "RetrievalAgreement",
    "RetrievalDiagnostics",
    "RerankedResult",
    "compute_retrieval_diagnostics",
    "load_documents",
    "measure_retrieval_overlap",
]
