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
from .hybrid import (
    HybridResult,
    HybridRetriever,
    RetrievalAgreement,
    measure_retrieval_overlap,
)

__all__ = [
    "BaselineRAG",
    "BM25Result",
    "BM25Retriever",
    "Document",
    "FAISSRetriever",
    "HybridResult",
    "HybridRetriever",
    "RAGAnswer",
    "RAGConfig",
    "RetrievalResult",
    "RetrievalAgreement",
    "load_documents",
    "measure_retrieval_overlap",
]
