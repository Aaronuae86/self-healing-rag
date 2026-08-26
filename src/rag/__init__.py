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

__all__ = [
    "BaselineRAG",
    "Document",
    "FAISSRetriever",
    "RAGAnswer",
    "RAGConfig",
    "RetrievalResult",
    "load_documents",
]
