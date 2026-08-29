"""Bounded LangGraph workflow for heuristic self-healing retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from .baseline import FAISSRetriever, LocalQwenGenerator, RetrievalResult
from .bm25 import BM25Result, BM25Retriever
from .diagnostics import RetrievalDiagnostics, compute_retrieval_diagnostics
from .failure_detection import RetrievalFailure, RetrievalFailureDetector
from .hybrid import HybridResult, HybridRetriever
from .reranker import CrossEncoderReranker, RerankedResult


class RecoveryAction(str, Enum):
    """Actions recorded in the final workflow state."""

    NONE = "NONE"
    INCREASE_RETRIEVAL_DEPTH = "INCREASE_RETRIEVAL_DEPTH"
    REWRITE_QUERY = "REWRITE_QUERY"
    PROCEED_TO_GENERATION = "PROCEED_TO_GENERATION"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class SelfHealingWorkflowConfig:
    """Centralized retrieval depths, retry bound, and output settings."""

    initial_retrieval_depth: int = 5
    expanded_retrieval_depth: int = 10
    reranked_top_k: int = 5
    generation_top_k: int = 3
    max_retries: int = 1
    rewrite_max_new_tokens: int = 64
    abstention_message: str = (
        "I do not have enough information in the provided corpus to answer that question."
    )


class SelfHealingState(TypedDict, total=False):
    """State retained as a query moves through the LangGraph workflow."""

    original_query: str
    active_query: str
    rewritten_query: str | None
    diagnostics: RetrievalDiagnostics
    diagnostics_history: list[RetrievalDiagnostics]
    failure_type: RetrievalFailure
    failure_history: list[RetrievalFailure]
    classification_reasons: tuple[str, ...]
    retry_count: int
    retrieval_depth: int
    dense_results: list[RetrievalResult]
    bm25_results: list[BM25Result]
    hybrid_results: list[HybridResult]
    reranked_results: list[RerankedResult]
    reranked_results_history: list[list[RerankedResult]]
    retrieved_documents: list[RerankedResult]
    recovery_action: RecoveryAction
    final_answer: str
    path: list[str]


class LocalQwenQueryRewriter:
    """Rewrite a search query with the already-loaded local Qwen model."""

    SYSTEM_INSTRUCTION = (
        "Rewrite the user's question as one concise, standalone search query. "
        "Preserve the original meaning and named entities. Do not answer the question. "
        "Return only the rewritten query."
    )

    def __init__(self, generator: LocalQwenGenerator, max_new_tokens: int = 64) -> None:
        self.generator = generator
        self.max_new_tokens = max_new_tokens

    def rewrite(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": query},
        ]
        output = self.generator.generate_messages(
            messages, max_new_tokens=self.max_new_tokens
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        rewritten = lines[0] if lines else query
        for prefix in ("rewritten query:", "search query:", "query:"):
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix) :].strip()
                break
        return rewritten.strip(" \"'") or query


class SelfHealingRAGWorkflow:
    """LangGraph workflow for retrieval, diagnosis, recovery, and generation."""

    def __init__(
        self,
        dense_retriever: FAISSRetriever,
        bm25_retriever: BM25Retriever,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: LocalQwenGenerator,
        failure_detector: RetrievalFailureDetector | None = None,
        config: SelfHealingWorkflowConfig | None = None,
        query_rewriter: LocalQwenQueryRewriter | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.failure_detector = failure_detector or RetrievalFailureDetector()
        self.config = config or SelfHealingWorkflowConfig()
        self._validate_config()
        self.query_rewriter = query_rewriter or LocalQwenQueryRewriter(
            generator, max_new_tokens=self.config.rewrite_max_new_tokens
        )
        self.graph = self._build_graph()
        self.retrieval_graph = self._build_retrieval_graph()

    def _validate_config(self) -> None:
        config = self.config
        if config.initial_retrieval_depth < 1:
            raise ValueError("initial_retrieval_depth must be at least 1.")
        if config.expanded_retrieval_depth < config.initial_retrieval_depth:
            raise ValueError(
                "expanded_retrieval_depth must be at least initial_retrieval_depth."
            )
        if config.reranked_top_k < 1 or config.generation_top_k < 1:
            raise ValueError("reranked_top_k and generation_top_k must be at least 1.")
        if config.max_retries < 0:
            raise ValueError("max_retries must not be negative.")

    @staticmethod
    def _append_path(state: SelfHealingState, step: str) -> list[str]:
        return [*state.get("path", []), step]

    def _retrieve_node(self, state: SelfHealingState) -> dict:
        query = state["active_query"]
        depth = state.get("retrieval_depth", self.config.initial_retrieval_depth)
        dense_results = self.dense_retriever.retrieve(query, top_k=depth)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=depth)
        return {
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "hybrid_results": self.hybrid_retriever.fuse(
                dense_results, bm25_results, top_k=depth
            ),
            "path": self._append_path(state, "RETRIEVE"),
        }

    def _rerank_node(self, state: SelfHealingState) -> dict:
        reranked = self.reranker.rerank(
            state["active_query"],
            state["hybrid_results"],
            top_k=self.config.reranked_top_k,
        )
        return {
            "reranked_results": reranked,
            "reranked_results_history": [
                *state.get("reranked_results_history", []),
                reranked,
            ],
            "retrieved_documents": reranked[: self.config.generation_top_k],
            "path": self._append_path(state, "RERANK"),
        }

    def _diagnostics_node(self, state: SelfHealingState) -> dict:
        diagnostics = compute_retrieval_diagnostics(
            query=state["active_query"],
            dense_results=state["dense_results"],
            bm25_results=state["bm25_results"],
            hybrid_results=state["hybrid_results"],
            reranked_results=state["reranked_results"],
        )
        return {
            "diagnostics": diagnostics,
            "diagnostics_history": [
                *state.get("diagnostics_history", []),
                diagnostics,
            ],
            "path": self._append_path(state, "DIAGNOSTICS"),
        }

    def _classify_node(self, state: SelfHealingState) -> dict:
        detection = self.failure_detector.classify(state["diagnostics"])
        return {
            "failure_type": detection.failure_type,
            "failure_history": [
                *state.get("failure_history", []),
                detection.failure_type,
            ],
            "classification_reasons": detection.reasons,
            "path": self._append_path(state, detection.failure_type.value),
        }

    def _route_after_classification(self, state: SelfHealingState) -> str:
        failure_type = state["failure_type"]
        if failure_type == RetrievalFailure.INSUFFICIENT_EVIDENCE:
            return "abstain"
        if failure_type == RetrievalFailure.HEALTHY:
            return "generate"
        if state.get("retry_count", 0) >= self.config.max_retries:
            return "generate"
        if failure_type == RetrievalFailure.AMBIGUOUS:
            return "expanded_retrieval"
        return "rewrite_query"

    def _expanded_retrieval_node(self, state: SelfHealingState) -> dict:
        return {
            "retrieval_depth": self.config.expanded_retrieval_depth,
            "retry_count": state.get("retry_count", 0) + 1,
            "recovery_action": RecoveryAction.INCREASE_RETRIEVAL_DEPTH,
            "path": self._append_path(state, "EXPAND_RETRIEVAL"),
        }

    def _rewrite_query_node(self, state: SelfHealingState) -> dict:
        rewritten_query = self.query_rewriter.rewrite(state["original_query"])
        return {
            "active_query": rewritten_query,
            "rewritten_query": rewritten_query,
            "retry_count": state.get("retry_count", 0) + 1,
            "recovery_action": RecoveryAction.REWRITE_QUERY,
            "path": self._append_path(state, "REWRITE_QUERY"),
        }

    def _generate_node(self, state: SelfHealingState) -> dict:
        answer = self.generator.generate(
            state["original_query"], state["retrieved_documents"]
        )
        recovery_action = state.get("recovery_action", RecoveryAction.NONE)
        if recovery_action == RecoveryAction.NONE:
            recovery_action = RecoveryAction.PROCEED_TO_GENERATION
        return {
            "final_answer": answer,
            "recovery_action": recovery_action,
            "path": self._append_path(state, "GENERATE"),
        }

    def _abstain_node(self, state: SelfHealingState) -> dict:
        return {
            "final_answer": self.config.abstention_message,
            "recovery_action": RecoveryAction.ABSTAIN,
            "path": self._append_path(state, "ABSTAIN"),
        }

    def _finalize_retrieval_node(self, state: SelfHealingState) -> dict:
        """Finish a retrieval-only run without invoking answer generation."""

        return {"path": self._append_path(state, "FINALIZE_RETRIEVAL")}

    def _retrieval_abstain_node(self, state: SelfHealingState) -> dict:
        """Record an abstention route without producing an answer string."""

        return {
            "recovery_action": RecoveryAction.ABSTAIN,
            "path": self._append_path(state, "ABSTAIN"),
        }

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise ImportError(
                "Install requirements.txt before creating SelfHealingRAGWorkflow."
            ) from error

        builder = StateGraph(SelfHealingState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("rerank", self._rerank_node)
        builder.add_node("diagnostics", self._diagnostics_node)
        builder.add_node("classify", self._classify_node)
        builder.add_node("expanded_retrieval", self._expanded_retrieval_node)
        builder.add_node("rewrite_query", self._rewrite_query_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("abstain", self._abstain_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "diagnostics")
        builder.add_edge("diagnostics", "classify")
        builder.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "generate": "generate",
                "expanded_retrieval": "expanded_retrieval",
                "rewrite_query": "rewrite_query",
                "abstain": "abstain",
            },
        )
        builder.add_edge("expanded_retrieval", "retrieve")
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("generate", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def _build_retrieval_graph(self):
        """Compile the same healing path with retrieval-only terminal nodes."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise ImportError(
                "Install requirements.txt before creating SelfHealingRAGWorkflow."
            ) from error

        builder = StateGraph(SelfHealingState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("rerank", self._rerank_node)
        builder.add_node("diagnostics", self._diagnostics_node)
        builder.add_node("classify", self._classify_node)
        builder.add_node("expanded_retrieval", self._expanded_retrieval_node)
        builder.add_node("rewrite_query", self._rewrite_query_node)
        builder.add_node("finalize_retrieval", self._finalize_retrieval_node)
        builder.add_node("abstain", self._retrieval_abstain_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "diagnostics")
        builder.add_edge("diagnostics", "classify")
        builder.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "generate": "finalize_retrieval",
                "expanded_retrieval": "expanded_retrieval",
                "rewrite_query": "rewrite_query",
                "abstain": "abstain",
            },
        )
        builder.add_edge("expanded_retrieval", "retrieve")
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("finalize_retrieval", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def _initial_state(self, query: str) -> SelfHealingState:
        return {
            "original_query": query,
            "active_query": query,
            "rewritten_query": None,
            "retry_count": 0,
            "retrieval_depth": self.config.initial_retrieval_depth,
            "recovery_action": RecoveryAction.NONE,
            "diagnostics_history": [],
            "failure_history": [],
            "reranked_results_history": [],
            "path": [],
        }

    def run(self, query: str) -> SelfHealingState:
        """Run one query through the compiled, bounded self-healing graph."""

        if not query.strip():
            raise ValueError("Query must not be empty.")
        return self.graph.invoke(self._initial_state(query))

    def run_retrieval_only(self, query: str) -> SelfHealingState:
        """Run retrieval, diagnostics, and recovery without generating an answer."""

        if not query.strip():
            raise ValueError("Query must not be empty.")
        return self.retrieval_graph.invoke(self._initial_state(query))
