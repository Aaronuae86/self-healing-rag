"""Local FastAPI adapter for the existing self-healing RAG workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, Request
from pydantic import BaseModel, StringConstraints

from .rag import (
    BM25Retriever,
    CrossEncoderReranker,
    FAISSRetriever,
    HybridRetriever,
    LocalQwenGenerator,
    RAGConfig,
    RecoveryAction,
    RetrievalFailure,
    RetrievalFailureDetector,
    SelfHealingRAGWorkflow,
    SelfHealingState,
    SelfHealingWorkflowConfig,
    load_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "phase1_corpus.json"
INDEX_DIR = PROJECT_ROOT / "data" / "phase1_faiss_index"

NonEmptyQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
WorkflowFactory = Callable[[], SelfHealingRAGWorkflow]


class QueryRequest(BaseModel):
    """One non-empty natural-language question."""

    query: NonEmptyQuery


class QueryResponse(BaseModel):
    """The answer and control metadata retained by the LangGraph state."""

    answer: str
    failure_type: RetrievalFailure
    abstained: bool
    recovery_action: RecoveryAction
    retry_count: int
    graph_path: list[str]
    rewritten_query: str | None = None
    classification_reasons: list[str]


class HealthResponse(BaseModel):
    status: str


def build_self_healing_workflow() -> SelfHealingRAGWorkflow:
    """Construct the same local/open-source stack used by the Phase 5 notebook."""

    rag_config = RAGConfig(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        generation_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=220,
    )
    documents = load_documents(CORPUS_PATH)

    dense_retriever = FAISSRetriever(
        embedding_model_name=rag_config.embedding_model_name,
        device=rag_config.device,
        batch_size=rag_config.embedding_batch_size,
    )
    index_files = (INDEX_DIR / "documents.faiss", INDEX_DIR / "metadata.json")
    if all(path.exists() for path in index_files):
        dense_retriever.load(INDEX_DIR)
    else:
        dense_retriever.build(documents)
        dense_retriever.save(INDEX_DIR)

    bm25_retriever = BM25Retriever(documents)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
    reranker = CrossEncoderReranker(device=rag_config.device)
    generator = LocalQwenGenerator(rag_config)

    return SelfHealingRAGWorkflow(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        failure_detector=RetrievalFailureDetector(),
        config=SelfHealingWorkflowConfig(
            initial_retrieval_depth=5,
            expanded_retrieval_depth=10,
            reranked_top_k=5,
            generation_top_k=3,
            max_retries=1,
        ),
    )


def _query_response(state: SelfHealingState) -> QueryResponse:
    """Expose only final workflow fields that are present in SelfHealingState."""

    recovery_action = state["recovery_action"]
    return QueryResponse(
        answer=state["final_answer"],
        failure_type=state["failure_type"],
        abstained=recovery_action == RecoveryAction.ABSTAIN,
        recovery_action=recovery_action,
        retry_count=state["retry_count"],
        graph_path=state["path"],
        rewritten_query=state.get("rewritten_query"),
        classification_reasons=list(state.get("classification_reasons", ())),
    )


def create_app(workflow_factory: WorkflowFactory = build_self_healing_workflow) -> FastAPI:
    """Create an app whose workflow is initialized once during application startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.workflow = workflow_factory()
        app.state.workflow_lock = Lock()
        yield

    api = FastAPI(
        title="Self-Healing RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        workflow: SelfHealingRAGWorkflow = request.app.state.workflow
        workflow_lock: Lock = request.app.state.workflow_lock
        with workflow_lock:
            state = workflow.run(payload.query)
        return _query_response(state)

    return api


app = create_app()
