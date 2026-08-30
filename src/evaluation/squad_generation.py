"""End-to-end SQuAD generation evaluation using frozen Chunk 2 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from src.rag import (
    BM25Retriever,
    Document,
    FailureDetectorConfig,
    HybridRetriever,
    LocalQwenGenerator,
    LocalQwenQueryRewriter,
    RecoveryAction,
    RerankedResult,
    RetrievalFailureDetector,
    RetrievalResult,
    SelfHealingRAGWorkflow,
    SelfHealingWorkflowConfig,
)

from .generation_metrics import (
    NLIAnswerResult,
    NLIConfig,
    PersistentGenerationCache,
    detect_safe_refusal,
    generation_cache_key,
    squad_exact_match,
    squad_token_f1,
    summarize_latency,
    unsupported_answer_flag,
)
from .groundedness import check_groundedness
from .semantic_groundedness import LocalNLIGroundednessEvaluator, NLIRequest
from .squad_dataset import stable_context_id
from .squad_retrieval import gold_document_rank, gold_rank_movement
from .squad_stress import ExcludingRetriever, RetrievalExclusionController


GENERATION_MANIFEST_VERSION = 1
SYSTEM_DENSE = "DENSE_BASELINE"
SYSTEM_STATIC = "STATIC_STRONG"
SYSTEM_SELF_HEALING = "SELF_HEALING"
SYSTEMS = (SYSTEM_DENSE, SYSTEM_STATIC, SYSTEM_SELF_HEALING)

TRACK_CLEAN = "CLEAN_ANSWERABLE"
TRACK_HARD = "HARD_DISTRACTOR"
TRACK_PARAPHRASE = "PARAPHRASE"
TRACK_MISSING = "CONTROLLED_MISSING_EVIDENCE"
TRACK_NATURAL_UNANSWERABLE = "NATURAL_UNANSWERABLE"
ANSWERABLE_TRACKS = {TRACK_CLEAN, TRACK_HARD, TRACK_PARAPHRASE}


@dataclass(frozen=True)
class Chunk2ArtifactPaths:
    calibration: Path = Path("results/squad_detector_calibration.json")
    stress_manifest: Path = Path("results/squad_stress_manifest.json")
    stress_per_example: Path = Path("results/squad_stress_per_example.csv")
    stress_metrics: Path = Path("results/squad_stress_metrics.json")


@dataclass(frozen=True)
class Chunk2Artifacts:
    paths: Chunk2ArtifactPaths
    calibration: dict
    stress_manifest: dict
    stress_rows: tuple[dict[str, str], ...]
    stress_metrics: dict
    sha256: dict[str, str]
    frozen_detector_config: FailureDetectorConfig


@dataclass(frozen=True)
class GenerationSubsetConfig:
    clean_answerable_count: int = 150
    hard_distractor_count: int = 50
    paraphrase_count: int = 50
    controlled_missing_count: int = 75
    natural_unanswerable_count: int = 75
    seed: int = 42
    dataset_name: str = "rajpurkar/squad_v2"
    dataset_revision: str | None = None
    cache_dir: str | Path | None = None


@dataclass(frozen=True)
class GenerationExample:
    id: str
    track: str
    question: str
    original_question: str
    gold_document_id: str | None
    source_document_id: str
    reference_answers: tuple[str, ...]
    excluded_document_ids: tuple[str, ...]
    distractor_ids: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return self.track in ANSWERABLE_TRACKS


@dataclass(frozen=True)
class GenerationDataset:
    documents: tuple[Document, ...]
    examples: tuple[GenerationExample, ...]
    manifest_path: Path
    category_counts: dict[str, int]
    fingerprint: str


@dataclass
class GenerationEvaluationRecord:
    example_id: str
    track: str
    question: str
    system: str
    gold_document_id: str | None
    retrieved_document_ids: tuple[str, ...]
    context_document_ids: tuple[str, ...]
    retrieved_contexts: tuple[dict[str, str], ...]
    initial_gold_rank: int | None
    final_gold_rank: int | None
    graph_path: tuple[str, ...]
    retry_count: int
    recovery_action: str
    controller_abstained: bool
    generated_answer: str
    safe_refusal: bool
    safe_refusal_phrase: str | None
    substantive_claim_count: int
    reference_answers: tuple[str, ...]
    exact_match: float | None
    token_f1: float | None
    lexical_groundedness: float | None
    lexical_grounded: bool | None
    nli_claim_count: int = 0
    nli_entailed_count: int = 0
    nli_neutral_count: int = 0
    nli_contradicted_count: int = 0
    nli_entailed_percentage: float | None = None
    nli_neutral_percentage: float | None = None
    nli_contradicted_percentage: float | None = None
    nli_fully_grounded: bool | None = None
    unsupported_answer: bool = False
    generation_cache_hit: bool = False
    latency: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunk2_artifacts(
    paths: Chunk2ArtifactPaths | None = None,
) -> Chunk2Artifacts:
    """Load all frozen inputs or fail without silently recalibrating."""

    required = paths or Chunk2ArtifactPaths()
    path_map = asdict(required)
    missing = [str(path) for path in path_map.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Chunk 3 requires the completed Chunk 2 artifacts. Missing:\n- "
            + "\n- ".join(missing)
            + "\nRun notebooks/06c_squad_stress_benchmark.ipynb on Colab T4 first."
        )
    calibration = json.loads(required.calibration.read_text(encoding="utf-8"))
    if calibration.get("label") != "TRAIN-CALIBRATED":
        raise ValueError("The detector artifact is not labeled TRAIN-CALIBRATED.")
    selected = calibration.get("selected_config")
    if not isinstance(selected, dict):
        raise ValueError("The detector artifact has no frozen selected_config.")
    with required.stress_per_example.open(encoding="utf-8", newline="") as handle:
        stress_rows = tuple(csv.DictReader(handle))
    if not stress_rows:
        raise ValueError("The Chunk 2 per-example artifact is empty.")
    return Chunk2Artifacts(
        paths=required,
        calibration=calibration,
        stress_manifest=json.loads(required.stress_manifest.read_text(encoding="utf-8")),
        stress_rows=stress_rows,
        stress_metrics=json.loads(required.stress_metrics.read_text(encoding="utf-8")),
        sha256={name: _sha256(Path(path)) for name, path in path_map.items()},
        frozen_detector_config=FailureDetectorConfig(**selected),
    )


def _answer_texts(record: dict[str, Any]) -> tuple[str, ...]:
    answers = record.get("answers") or {}
    return tuple(str(item).strip() for item in answers.get("text", []) if str(item).strip())


def _sample_available(ids: Sequence[str], count: int, seed: int) -> list[str]:
    values = sorted(dict.fromkeys(str(item) for item in ids))
    random.Random(seed).shuffle(values)
    return values[: min(count, len(values))]


def _subset_config_dict(config: GenerationSubsetConfig) -> dict[str, Any]:
    output = asdict(config)
    if output["cache_dir"] is not None:
        output["cache_dir"] = str(output["cache_dir"])
    return output


def prepare_generation_subset(
    artifacts: Chunk2Artifacts,
    documents: Sequence[Document],
    config: GenerationSubsetConfig | None = None,
    manifest_path: str | Path = "results/squad_generation_manifest.json",
    validation_records: Sequence[dict[str, Any]] | None = None,
) -> GenerationDataset:
    """Select fixed held-out IDs without consulting Chunk 2 outcome columns."""

    settings = config or GenerationSubsetConfig()
    counts = (
        settings.clean_answerable_count,
        settings.hard_distractor_count,
        settings.paraphrase_count,
        settings.controlled_missing_count,
        settings.natural_unanswerable_count,
    )
    if any(value < 1 for value in counts):
        raise ValueError("Every generation subset target must be at least 1.")
    if validation_records is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError("Install requirements.txt to load SQuAD 2.0.") from error
        kwargs: dict[str, str] = {}
        if settings.cache_dir is not None:
            kwargs["cache_dir"] = str(settings.cache_dir)
        if settings.dataset_revision is not None:
            kwargs["revision"] = settings.dataset_revision
        validation_records = [
            dict(item)
            for item in load_dataset(settings.dataset_name, **kwargs)["validation"]
        ]
    records = [dict(item) for item in validation_records]
    by_id = {str(item["id"]): item for item in records}
    document_ids = {item.id for item in documents}
    calibrated_rows = [
        row
        for row in artifacts.stress_rows
        if row.get("detector_variant") == "TRAIN_CALIBRATED"
    ]
    available_by_track: dict[str, list[str]] = defaultdict(list)
    for row in calibrated_rows:
        available_by_track[str(row["track"])].append(str(row["example_id"]))

    output_path = Path(manifest_path)
    subset_config = _subset_config_dict(settings)
    manifest: dict[str, Any] | None = None
    if output_path.exists():
        candidate = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            candidate.get("manifest_version") == GENERATION_MANIFEST_VERSION
            and candidate.get("subset_config") == subset_config
            and candidate.get("chunk2_artifact_sha256") == artifacts.sha256
        ):
            manifest = candidate

    if manifest is None:
        stress_ids = {
            str(row["example_id"])
            for row in calibrated_rows
        }
        clean_candidates = [
            str(item["id"])
            for item in records
            if _answer_texts(item)
            and stable_context_id(str(item["context"])) in document_ids
            and str(item["id"]) not in stress_ids
        ]
        selected = {
            TRACK_CLEAN: _sample_available(
                clean_candidates, settings.clean_answerable_count, settings.seed
            ),
            TRACK_HARD: _sample_available(
                available_by_track["HARD_DISTRACTOR"],
                settings.hard_distractor_count,
                settings.seed + 1,
            ),
            TRACK_PARAPHRASE: _sample_available(
                available_by_track["PARAPHRASE_TRANSFORMED"],
                settings.paraphrase_count,
                settings.seed + 2,
            ),
            TRACK_MISSING: _sample_available(
                available_by_track["CONTROLLED_MISSING_EVIDENCE"],
                settings.controlled_missing_count,
                settings.seed + 3,
            ),
            TRACK_NATURAL_UNANSWERABLE: _sample_available(
                available_by_track["NATURAL_UNANSWERABLE"],
                settings.natural_unanswerable_count,
                settings.seed + 4,
            ),
        }
        if len(selected[TRACK_CLEAN]) < settings.clean_answerable_count:
            raise RuntimeError(
                "The frozen corpus does not contain enough unused held-out clean "
                "answerable questions for the requested generation subset."
            )
        manifest = {
            "manifest_version": GENERATION_MANIFEST_VERSION,
            "seed": settings.seed,
            "subset_config": subset_config,
            "chunk2_artifact_sha256": artifacts.sha256,
            "selected_ids": selected,
            "available_chunk2_track_counts": {
                key: len(set(value)) for key, value in available_by_track.items()
            },
            "selection_policy": (
                "Fixed seeded IDs; Chunk 2 rows are filtered only by detector variant, "
                "track, and ID presence, never by outcomes or scores."
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    paraphrases = artifacts.stress_manifest.get("paraphrases", {})
    hard_distractors = artifacts.stress_manifest.get("hard_distractors", {})
    examples: list[GenerationExample] = []
    for track, ids in manifest["selected_ids"].items():
        for example_id in ids:
            if example_id not in by_id:
                raise ValueError(f"Selected validation ID is unavailable: {example_id}")
            record = by_id[example_id]
            source_id = stable_context_id(str(record["context"]))
            references = _answer_texts(record)
            if track == TRACK_PARAPHRASE:
                question = paraphrases.get(example_id)
                if not question:
                    raise ValueError(f"Missing frozen paraphrase for {example_id}.")
            else:
                question = str(record["question"]).strip()
            gold_id = source_id if track != TRACK_NATURAL_UNANSWERABLE else None
            excluded = (source_id,) if track == TRACK_MISSING else ()
            examples.append(
                GenerationExample(
                    id=str(example_id),
                    track=str(track),
                    question=str(question),
                    original_question=str(record["question"]).strip(),
                    gold_document_id=gold_id,
                    source_document_id=source_id,
                    reference_answers=references,
                    excluded_document_ids=excluded,
                    distractor_ids=tuple(hard_distractors.get(example_id, ())),
                )
            )
    category_counts = dict(Counter(item.track for item in examples))
    fingerprint_payload = json.dumps(
        manifest["selected_ids"], sort_keys=True, separators=(",", ":")
    )
    return GenerationDataset(
        documents=tuple(documents),
        examples=tuple(examples),
        manifest_path=output_path,
        category_counts=category_counts,
        fingerprint=hashlib.sha256(fingerprint_payload.encode()).hexdigest()[:16],
    )


class LatencyRecorder:
    def __init__(self) -> None:
        self.values: dict[str, float] = defaultdict(float)

    def reset(self) -> None:
        self.values.clear()

    def add(self, name: str, seconds: float) -> None:
        self.values[name] += float(seconds)

    def snapshot(self) -> dict[str, float]:
        return dict(self.values)


class TimedDenseRetriever:
    """Time query encoding separately from exact FAISS index search."""

    def __init__(self, retriever, recorder: LatencyRecorder) -> None:
        self.retriever = retriever
        self.recorder = recorder

    @property
    def documents(self):
        return self.retriever.documents

    def retrieve(self, query: str, top_k: int = 3):
        if self.retriever.index is None:
            raise RuntimeError("Build or load the dense index before retrieval.")
        started = time.perf_counter()
        vectors = self.retriever._encode([query])
        self.recorder.add("query_encoding_latency_seconds", time.perf_counter() - started)
        started = time.perf_counter()
        scores, positions = self.retriever.index.search(
            vectors, min(top_k, len(self.retriever.documents))
        )
        results = [
            RetrievalResult(self.retriever.documents[position], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
        self.recorder.add("dense_retrieval_latency_seconds", time.perf_counter() - started)
        return results


class TimedBM25Retriever:
    def __init__(self, retriever, recorder: LatencyRecorder) -> None:
        self.retriever = retriever
        self.recorder = recorder

    @property
    def documents(self):
        return self.retriever.documents

    def retrieve(self, query: str, top_k: int = 3):
        started = time.perf_counter()
        output = self.retriever.retrieve(query, top_k=top_k)
        self.recorder.add("bm25_retrieval_latency_seconds", time.perf_counter() - started)
        return output


class TimedHybridRetriever:
    def __init__(self, retriever, recorder: LatencyRecorder) -> None:
        self.retriever = retriever
        self.recorder = recorder

    def fuse(self, dense_results, bm25_results, top_k: int):
        started = time.perf_counter()
        output = self.retriever.fuse(dense_results, bm25_results, top_k=top_k)
        self.recorder.add("hybrid_fusion_latency_seconds", time.perf_counter() - started)
        return output


class TimedReranker:
    def __init__(self, reranker, recorder: LatencyRecorder) -> None:
        self.reranker = reranker
        self.recorder = recorder

    def rerank(self, query, candidates, top_k=None):
        started = time.perf_counter()
        output = self.reranker.rerank(query, candidates, top_k=top_k)
        self.recorder.add("reranking_latency_seconds", time.perf_counter() - started)
        return output


class TimedFailureDetector:
    def __init__(self, detector, recorder: LatencyRecorder) -> None:
        self.detector = detector
        self.recorder = recorder

    def classify(self, diagnostics):
        started = time.perf_counter()
        output = self.detector.classify(diagnostics)
        self.recorder.add("classification_latency_seconds", time.perf_counter() - started)
        return output


class TimedQueryRewriter:
    def __init__(self, rewriter, recorder: LatencyRecorder) -> None:
        self.rewriter = rewriter
        self.recorder = recorder

    def rewrite(self, query: str) -> str:
        started = time.perf_counter()
        output = self.rewriter.rewrite(query)
        self.recorder.add("recovery_rewrite_latency_seconds", time.perf_counter() - started)
        return output


class TimedSelfHealingWorkflow(SelfHealingRAGWorkflow):
    def __init__(self, *args, latency_recorder: LatencyRecorder, **kwargs) -> None:
        self.latency_recorder = latency_recorder
        super().__init__(*args, **kwargs)

    def _diagnostics_node(self, state):
        started = time.perf_counter()
        output = super()._diagnostics_node(state)
        self.latency_recorder.add(
            "diagnostics_latency_seconds", time.perf_counter() - started
        )
        return output


class SquadGenerationEvaluator:
    """Run Dense, Static Strong, and frozen-detector Self-Healing systems."""

    def __init__(
        self,
        dense_retriever,
        bm25_retriever: BM25Retriever,
        reranker,
        generator: LocalQwenGenerator,
        frozen_detector_config: FailureDetectorConfig,
        generation_cache_path: str | Path = "results/squad_generation_cache.jsonl",
        nli_evaluator: LocalNLIGroundednessEvaluator | None = None,
        workflow_config: SelfHealingWorkflowConfig | None = None,
        dense_top_k: int = 5,
        static_candidate_k: int = 20,
        final_top_k: int = 5,
        generation_top_k: int = 3,
        output_dir: str | Path = "results",
    ) -> None:
        self.generator = generator
        self.nli_evaluator = nli_evaluator
        self.dense_top_k = dense_top_k
        self.static_candidate_k = static_candidate_k
        self.final_top_k = final_top_k
        self.generation_top_k = generation_top_k
        self.output_dir = Path(output_dir)
        self.cache = PersistentGenerationCache(generation_cache_path)
        self.recorder = LatencyRecorder()
        timed_dense = TimedDenseRetriever(dense_retriever, self.recorder)
        timed_bm25 = TimedBM25Retriever(bm25_retriever, self.recorder)
        self.dense = ExcludingRetriever(timed_dense)
        self.bm25 = ExcludingRetriever(timed_bm25)
        self.exclusions = RetrievalExclusionController(self.dense, self.bm25)
        self.hybrid = TimedHybridRetriever(
            HybridRetriever(self.dense, self.bm25), self.recorder
        )
        self.reranker = TimedReranker(reranker, self.recorder)
        detector = TimedFailureDetector(
            RetrievalFailureDetector(frozen_detector_config), self.recorder
        )
        rewriter = TimedQueryRewriter(
            LocalQwenQueryRewriter(generator), self.recorder
        )
        settings = workflow_config or SelfHealingWorkflowConfig(
            initial_retrieval_depth=20,
            expanded_retrieval_depth=40,
            reranked_top_k=10,
            generation_top_k=generation_top_k,
            max_retries=1,
        )
        self.workflow = TimedSelfHealingWorkflow(
            self.dense,
            self.bm25,
            self.hybrid,
            self.reranker,
            generator,
            failure_detector=detector,
            config=settings,
            query_rewriter=rewriter,
            latency_recorder=self.recorder,
        )

    @staticmethod
    def _contexts(results: Sequence) -> tuple[dict[str, str], ...]:
        return tuple(
            {"id": item.id, "title": item.title, "text": item.text}
            for item in results
        )

    def _generate(self, question: str, evidence: Sequence) -> tuple[str, bool, float, float]:
        contexts = self._contexts(evidence)
        config = {
            "model": self.generator.config.generation_model_name,
            "max_new_tokens": self.generator.config.max_new_tokens,
            "do_sample": self.generator.config.do_sample,
            "temperature": self.generator.config.temperature,
        }
        key = generation_cache_key(
            system_instruction=self.generator.SYSTEM_INSTRUCTION,
            question=question,
            contexts=contexts,
            generation_config=config,
        )
        lookup_started = time.perf_counter()
        cached = self.cache.get(key)
        if cached is not None:
            observed = time.perf_counter() - lookup_started
            return (
                str(cached["answer"]),
                True,
                float(cached["generation_latency_seconds"]),
                observed,
            )
        started = time.perf_counter()
        answer = self.generator.generate(question, evidence)
        generation_seconds = time.perf_counter() - started
        self.cache.put(
            {
                "key": key,
                "answer": answer,
                "generation_latency_seconds": generation_seconds,
                "question": question,
                "context_ids": [item["id"] for item in contexts],
                "generation_config": config,
            }
        )
        return answer, False, generation_seconds, generation_seconds

    def _finalize_record(
        self,
        *,
        example: GenerationExample,
        system: str,
        ranked_results: Sequence,
        evidence: Sequence,
        initial_rank: int | None,
        final_rank: int | None,
        graph_path: Sequence[str] = (),
        retry_count: int = 0,
        recovery_action: str = "NONE",
        controller_abstained: bool = False,
        total_started: float,
    ) -> GenerationEvaluationRecord:
        cache_hit = False
        if controller_abstained:
            answer = self.workflow.config.abstention_message
            generation_seconds = 0.0
        else:
            answer, cache_hit, generation_seconds, _observed_generation = self._generate(
                example.question, evidence
            )
        decision = detect_safe_refusal(answer)
        lexical = None
        if not decision.is_safe_refusal and decision.substantive_claim_count:
            lexical = check_groundedness(answer, evidence)
        exact_match = (
            squad_exact_match(answer, example.reference_answers)
            if example.answerable
            else None
        )
        token_f1 = (
            squad_token_f1(answer, example.reference_answers)
            if example.answerable
            else None
        )
        self.recorder.add("generation_latency_seconds", generation_seconds)
        latency = self.recorder.snapshot()
        latency["total_latency_seconds"] = sum(latency.values())
        latency["observed_total_latency_seconds"] = time.perf_counter() - total_started
        unsupported = (
            unsupported_answer_flag(
                controller_abstained=controller_abstained,
                safe_refusal=decision.is_safe_refusal,
                substantive_claim_count=decision.substantive_claim_count,
            )
            if example.track == TRACK_MISSING
            else False
        )
        return GenerationEvaluationRecord(
            example_id=example.id,
            track=example.track,
            question=example.question,
            system=system,
            gold_document_id=example.gold_document_id,
            retrieved_document_ids=tuple(item.id for item in ranked_results),
            context_document_ids=tuple(item.id for item in evidence),
            retrieved_contexts=self._contexts(evidence),
            initial_gold_rank=initial_rank,
            final_gold_rank=final_rank,
            graph_path=tuple(str(item) for item in graph_path),
            retry_count=retry_count,
            recovery_action=recovery_action,
            controller_abstained=controller_abstained,
            generated_answer=answer,
            safe_refusal=decision.is_safe_refusal,
            safe_refusal_phrase=decision.matched_phrase,
            substantive_claim_count=decision.substantive_claim_count,
            reference_answers=example.reference_answers,
            exact_match=exact_match,
            token_f1=token_f1,
            lexical_groundedness=(lexical.coverage if lexical else None),
            lexical_grounded=(lexical.is_grounded if lexical else None),
            unsupported_answer=unsupported,
            generation_cache_hit=cache_hit,
            latency=latency,
        )

    def _dense(self, example: GenerationExample) -> GenerationEvaluationRecord:
        self.recorder.reset()
        started = time.perf_counter()
        ranked = self.dense.retrieve(example.question, top_k=self.dense_top_k)
        evidence = ranked[: self.generation_top_k]
        rank = (
            gold_document_rank([item.id for item in ranked], example.gold_document_id)
            if example.gold_document_id
            else None
        )
        return self._finalize_record(
            example=example,
            system=SYSTEM_DENSE,
            ranked_results=ranked,
            evidence=evidence,
            initial_rank=rank,
            final_rank=rank,
            graph_path=("DENSE_RETRIEVE", "GENERATE"),
            total_started=started,
        )

    def _static(self, example: GenerationExample) -> GenerationEvaluationRecord:
        self.recorder.reset()
        started = time.perf_counter()
        dense = self.dense.retrieve(example.question, top_k=self.static_candidate_k)
        bm25 = self.bm25.retrieve(example.question, top_k=self.static_candidate_k)
        hybrid = self.hybrid.fuse(dense, bm25, top_k=self.static_candidate_k)
        ranked = self.reranker.rerank(
            example.question, hybrid, top_k=self.final_top_k
        )
        evidence = ranked[: self.generation_top_k]
        rank = (
            gold_document_rank([item.id for item in ranked], example.gold_document_id)
            if example.gold_document_id
            else None
        )
        return self._finalize_record(
            example=example,
            system=SYSTEM_STATIC,
            ranked_results=ranked,
            evidence=evidence,
            initial_rank=rank,
            final_rank=rank,
            graph_path=("DENSE_BM25", "RRF", "RERANK", "GENERATE"),
            total_started=started,
        )

    def _self_healing(self, example: GenerationExample) -> GenerationEvaluationRecord:
        self.recorder.reset()
        started = time.perf_counter()
        state = self.workflow.run_retrieval_only(example.question)
        history = state.get("reranked_results_history", [])
        initial = history[0] if history else state.get("reranked_results", [])
        ranked = state.get("reranked_results", [])[: self.final_top_k]
        evidence = state.get("retrieved_documents", [])[: self.generation_top_k]
        initial_rank = (
            gold_document_rank(
                [item.id for item in initial[: self.final_top_k]],
                example.gold_document_id,
            )
            if example.gold_document_id
            else None
        )
        final_rank = (
            gold_document_rank([item.id for item in ranked], example.gold_document_id)
            if example.gold_document_id
            else None
        )
        action = state.get("recovery_action", RecoveryAction.NONE)
        action_value = action.value if hasattr(action, "value") else str(action)
        controller_abstained = action_value == RecoveryAction.ABSTAIN.value
        return self._finalize_record(
            example=example,
            system=SYSTEM_SELF_HEALING,
            ranked_results=ranked,
            evidence=evidence,
            initial_rank=initial_rank,
            final_rank=final_rank,
            graph_path=state.get("path", ()),
            retry_count=int(state.get("retry_count", 0)),
            recovery_action=action_value,
            controller_abstained=controller_abstained,
            total_started=started,
        )

    def run(
        self,
        dataset: GenerationDataset,
        chunk2_artifacts: Chunk2Artifacts,
        nli_config: NLIConfig | None = None,
        progress_callback: Callable[[int, int, GenerationExample, str], None] | None = None,
    ) -> tuple[list[GenerationEvaluationRecord], dict[str, Any]]:
        records: list[GenerationEvaluationRecord] = []
        total = len(dataset.examples) * len(SYSTEMS)
        progress = 0
        for example in dataset.examples:
            self.exclusions.set_excluded(example.excluded_document_ids)
            for system, runner in (
                (SYSTEM_DENSE, self._dense),
                (SYSTEM_STATIC, self._static),
                (SYSTEM_SELF_HEALING, self._self_healing),
            ):
                progress += 1
                if progress_callback:
                    progress_callback(progress, total, example, system)
                records.append(runner(example))
        self.exclusions.clear()

        nli_started = time.perf_counter()
        nli_requests: list[NLIRequest] = []
        nli_record_indices: list[int] = []
        for index, record in enumerate(records):
            if record.safe_refusal or record.substantive_claim_count == 0:
                continue
            nli_record_indices.append(index)
            nli_requests.append(
                NLIRequest(record.generated_answer, record.retrieved_contexts)
            )
        evaluator = self.nli_evaluator or LocalNLIGroundednessEvaluator(nli_config)
        nli_results = evaluator.evaluate_many(nli_requests)
        if len(nli_results) != len(nli_record_indices):
            raise RuntimeError("NLI result count does not match generated answer count.")
        claim_rows: list[dict[str, Any]] = []
        for record_index, result in zip(nli_record_indices, nli_results):
            record = records[record_index]
            record.nli_claim_count = result.claim_count
            record.nli_entailed_count = result.entailed_count
            record.nli_neutral_count = result.neutral_count
            record.nli_contradicted_count = result.contradicted_count
            record.nli_entailed_percentage = result.entailed_percentage
            record.nli_neutral_percentage = result.neutral_percentage
            record.nli_contradicted_percentage = result.contradicted_percentage
            record.nli_fully_grounded = result.fully_grounded
            for claim_index, claim in enumerate(result.claims):
                claim_rows.append(
                    {
                        "example_id": record.example_id,
                        "track": record.track,
                        "system": record.system,
                        "claim_index": claim_index,
                        **asdict(claim),
                    }
                )
        nli_elapsed = time.perf_counter() - nli_started
        metrics = summarize_generation_metrics(records)
        payload = {
            "benchmark": "SQuAD 2.0 end-to-end generation evaluation",
            "scientific_integrity": {
                "detector_calibration_split": "TRAIN only",
                "validation_usage": "held-out evaluation only",
                "detector_recalibrated_in_chunk3": False,
                "paid_services_used": False,
            },
            "dataset": {
                "total_observations": len(dataset.examples),
                "category_counts": dataset.category_counts,
                "fingerprint": dataset.fingerprint,
                "manifest": str(dataset.manifest_path),
            },
            "chunk2_artifact_sha256": chunk2_artifacts.sha256,
            "frozen_detector_config": asdict(chunk2_artifacts.frozen_detector_config),
            "generation": {
                "model": self.generator.config.generation_model_name,
                "system_instruction": self.generator.SYSTEM_INSTRUCTION,
                "do_sample": self.generator.config.do_sample,
                "max_new_tokens": self.generator.config.max_new_tokens,
            },
            "nli": {
                "config": asdict(nli_config or NLIConfig()),
                "evaluation_latency_seconds": nli_elapsed,
                "aggregation": (
                    "Each meaningful claim is entailed if any top evidence passage "
                    "meets the entailment threshold; otherwise contradiction threshold, "
                    "otherwise neutral."
                ),
            },
            "metrics": metrics,
        }
        self._save(records, claim_rows, payload)
        return records, payload

    def _save(self, records, claim_rows, payload) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "squad_generation_metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        row_dicts = [item.to_dict() for item in records]
        json_fields = {
            "retrieved_document_ids",
            "context_document_ids",
            "retrieved_contexts",
            "graph_path",
            "reference_answers",
            "latency",
        }
        with (self.output_dir / "squad_generation_per_example.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0]))
            writer.writeheader()
            for row in row_dicts:
                for field_name in json_fields:
                    row[field_name] = json.dumps(row[field_name], ensure_ascii=False)
                writer.writerow(row)
        with (self.output_dir / "squad_nli_per_claim.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = (
                "example_id",
                "track",
                "system",
                "claim_index",
                "claim",
                "label",
                "entailment_probability",
                "neutral_probability",
                "contradiction_probability",
                "best_passage_id",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(claim_rows)
        (self.output_dir / "squad_generation_summary.md").write_text(
            render_generation_summary(payload), encoding="utf-8"
        )


def _mean(values: Sequence[float]) -> float | None:
    return float(fmean(values)) if values else None


def _group_metrics(records: Sequence[GenerationEvaluationRecord]) -> dict[str, Any]:
    answerable = [item for item in records if item.track in ANSWERABLE_TRACKS]
    gold_retrieved = [item for item in answerable if item.final_gold_rank is not None and item.final_gold_rank <= 5]
    claims = sum(item.nli_claim_count for item in records)
    latency = summarize_latency(item.latency for item in records)
    return {
        "observations": len(records),
        "recall_at_5": _mean(
            [float(item.final_gold_rank is not None and item.final_gold_rank <= 5) for item in answerable]
        ),
        "answer_em": _mean([float(item.exact_match) for item in answerable if item.exact_match is not None]),
        "answer_f1": _mean([float(item.token_f1) for item in answerable if item.token_f1 is not None]),
        "answer_em_given_gold_retrieved_at_5": _mean([float(item.exact_match) for item in gold_retrieved if item.exact_match is not None]),
        "answer_f1_given_gold_retrieved_at_5": _mean([float(item.token_f1) for item in gold_retrieved if item.token_f1 is not None]),
        "lexical_heuristic_mean_coverage": _mean([float(item.lexical_groundedness) for item in records if item.lexical_groundedness is not None]),
        "lexical_heuristic_grounded_rate": _mean([float(item.lexical_grounded) for item in records if item.lexical_grounded is not None]),
        "nli_claim_count": claims,
        "nli_entailed_percentage": (sum(item.nli_entailed_count for item in records) / claims if claims else None),
        "nli_neutral_percentage": (sum(item.nli_neutral_count for item in records) / claims if claims else None),
        "nli_contradicted_percentage": (sum(item.nli_contradicted_count for item in records) / claims if claims else None),
        "nli_fully_grounded_rate": _mean([float(item.nli_fully_grounded) for item in records if item.nli_fully_grounded is not None]),
        "safe_refusal_rate": _mean([float(item.safe_refusal) for item in records]),
        "unsupported_answer_rate": _mean([float(item.unsupported_answer) for item in records]),
        "controller_abstention_rate": _mean([float(item.controller_abstained) for item in records]),
        "mean_retry_count": _mean([float(item.retry_count) for item in records]),
        "generation_cache_hit_rate": _mean([float(item.generation_cache_hit) for item in records]),
        "latency": latency,
    }


def summarize_generation_metrics(
    records: Sequence[GenerationEvaluationRecord],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty generation benchmark.")
    by_system: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [item for item in records if item.system == system]
        by_system[system] = {
            "overall": _group_metrics(system_rows),
            "per_track": {
                track: _group_metrics([item for item in system_rows if item.track == track])
                for track in sorted({item.track for item in system_rows})
            },
        }

    self_rows = [item for item in records if item.system == SYSTEM_SELF_HEALING]
    controller_scope = [
        item
        for item in self_rows
        if item.track != TRACK_NATURAL_UNANSWERABLE
    ]
    true_positive = sum(
        item.track == TRACK_MISSING and item.controller_abstained
        for item in controller_scope
    )
    false_positive = sum(
        item.track != TRACK_MISSING and item.controller_abstained
        for item in controller_scope
    )
    false_negative = sum(
        item.track == TRACK_MISSING and not item.controller_abstained
        for item in controller_scope
    )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    controller = {
        "definition": (
            "Positive class is controlled missing evidence; answerable tracks are "
            "negatives; natural unanswerable is reported separately and excluded."
        ),
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

    static_by_key = {
        (item.example_id, item.track): item
        for item in records
        if item.system == SYSTEM_STATIC
    }
    recovered = [item for item in self_rows if item.retry_count > 0]
    answer_changes = Counter()
    for item in recovered:
        static = static_by_key.get((item.example_id, item.track))
        if static is None or item.token_f1 is None or static.token_f1 is None:
            continue
        if item.token_f1 > static.token_f1:
            answer_changes["improved"] += 1
        elif item.token_f1 < static.token_f1:
            answer_changes["worsened"] += 1
        else:
            answer_changes["unchanged"] += 1
    recovery = {
        "attempted_count": len(recovered),
        "attempt_rate": len(recovered) / len(self_rows) if self_rows else 0.0,
        "gold_entered_top5_count": sum(
            (item.initial_gold_rank is None or item.initial_gold_rank > 5)
            and item.final_gold_rank is not None
            and item.final_gold_rank <= 5
            for item in recovered
            if item.gold_document_id is not None
        ),
        "gold_rank_improved_count": sum(
            gold_rank_movement(item.initial_gold_rank, item.final_gold_rank) == "improved"
            for item in recovered
            if item.gold_document_id is not None
        ),
        "answer_f1_improved_count": answer_changes["improved"],
        "answer_f1_worsened_count": answer_changes["worsened"],
        "answer_f1_unchanged_count": answer_changes["unchanged"],
        "rewrite_count": sum("REWRITE_QUERY" in item.graph_path for item in self_rows),
        "expanded_retrieval_count": sum("EXPAND_RETRIEVAL" in item.graph_path for item in self_rows),
        "controller_abstained_count": sum(item.controller_abstained for item in self_rows),
        "unnecessary_recovery_rate": _mean(
            [
                float(item.retry_count > 0)
                for item in self_rows
                if item.track in ANSWERABLE_TRACKS
                and item.initial_gold_rank is not None
                and item.initial_gold_rank <= 5
            ]
        ),
        "average_retry_count": _mean([float(item.retry_count) for item in self_rows]),
    }
    static_mean = by_system[SYSTEM_STATIC]["overall"]["latency"].get(
        "total_latency_seconds", {}
    ).get("mean")
    self_mean = by_system[SYSTEM_SELF_HEALING]["overall"]["latency"].get(
        "total_latency_seconds", {}
    ).get("mean")
    overhead = (
        (self_mean - static_mean) / static_mean
        if static_mean not in (None, 0.0) and self_mean is not None
        else None
    )
    comparison = {
        system: {
            "recall_at_5": by_system[system]["overall"]["recall_at_5"],
            "answer_em": by_system[system]["overall"]["answer_em"],
            "answer_f1": by_system[system]["overall"]["answer_f1"],
            "nli_fully_grounded_rate": by_system[system]["overall"]["nli_fully_grounded_rate"],
            "safe_refusal_rate": by_system[system]["per_track"].get(TRACK_MISSING, {}).get("safe_refusal_rate"),
            "unsupported_answer_rate": by_system[system]["per_track"].get(TRACK_MISSING, {}).get("unsupported_answer_rate"),
            "mean_latency_seconds": by_system[system]["overall"]["latency"].get("total_latency_seconds", {}).get("mean"),
            "median_latency_seconds": by_system[system]["overall"]["latency"].get("total_latency_seconds", {}).get("median"),
            "controller_abstention_rate": (
                by_system[system]["overall"]["controller_abstention_rate"]
                if system == SYSTEM_SELF_HEALING
                else None
            ),
            "recovery_attempt_rate": recovery["attempt_rate"] if system == SYSTEM_SELF_HEALING else None,
        }
        for system in SYSTEMS
    }
    return {
        "by_system": by_system,
        "controller_abstention": controller,
        "self_healing_recovery": recovery,
        "self_healing_latency_overhead_vs_static_strong": overhead,
        "three_way_comparison": comparison,
    }


def representative_error_cases(
    records: Sequence[GenerationEvaluationRecord], limit: int = 2
) -> dict[str, list[dict[str, Any]]]:
    by_key = defaultdict(dict)
    for item in records:
        by_key[(item.example_id, item.track)][item.system] = item

    def brief(item: GenerationEvaluationRecord) -> dict[str, Any]:
        return {
            "example_id": item.example_id,
            "track": item.track,
            "question": item.question,
            "system": item.system,
            "initial_gold_rank": item.initial_gold_rank,
            "final_gold_rank": item.final_gold_rank,
            "retry_count": item.retry_count,
            "recovery_action": item.recovery_action,
            "controller_abstained": item.controller_abstained,
            "safe_refusal": item.safe_refusal,
            "unsupported_answer": item.unsupported_answer,
            "em": item.exact_match,
            "f1": item.token_f1,
            "answer": item.generated_answer[:400],
            "context_ids": item.context_document_ids,
        }

    categories: dict[str, list[GenerationEvaluationRecord]] = defaultdict(list)
    for systems in by_key.values():
        dense = systems.get(SYSTEM_DENSE)
        static = systems.get(SYSTEM_STATIC)
        healing = systems.get(SYSTEM_SELF_HEALING)
        if dense and static and (dense.final_gold_rank is None or dense.final_gold_rank > 5) and static.final_gold_rank and static.final_gold_rank <= 5:
            categories["dense_failure_fixed_by_static"].append(static)
        if static and healing and (static.final_gold_rank is None or static.final_gold_rank > 5) and healing.final_gold_rank and healing.final_gold_rank <= 5:
            categories["static_failure_fixed_by_self_healing"].append(healing)
        if healing and healing.retry_count > 0:
            if (healing.initial_gold_rank is None or healing.initial_gold_rank > 5) and healing.final_gold_rank and healing.final_gold_rank <= 5:
                categories["successful_self_healing_recovery"].append(healing)
            elif healing.gold_document_id is not None:
                categories["failed_self_healing_recovery"].append(healing)
        if healing and healing.track == TRACK_MISSING and healing.controller_abstained:
            categories["correct_controller_abstention"].append(healing)
        if healing and healing.track == TRACK_MISSING and not healing.controller_abstained:
            categories["missed_missing_evidence"].append(healing)
        for item in systems.values():
            if item.unsupported_answer:
                categories["unsupported_generation"].append(item)
            if item.safe_refusal and not item.controller_abstained:
                categories["safe_generated_refusal"].append(item)
            if item.nli_contradicted_count or item.nli_neutral_count:
                categories["nli_detected_unsupported_claim"].append(item)
            if item.final_gold_rank and item.final_gold_rank <= 5 and item.exact_match == 0.0:
                categories["gold_retrieved_but_answer_incorrect"].append(item)
    requested = (
        "dense_failure_fixed_by_static",
        "static_failure_fixed_by_self_healing",
        "successful_self_healing_recovery",
        "failed_self_healing_recovery",
        "correct_controller_abstention",
        "missed_missing_evidence",
        "unsupported_generation",
        "safe_generated_refusal",
        "nli_detected_unsupported_claim",
        "gold_retrieved_but_answer_incorrect",
    )
    return {
        name: [brief(item) for item in categories.get(name, [])[:limit]]
        for name in requested
    }


def load_generation_records(
    path: str | Path = "results/squad_generation_per_example.csv",
) -> list[GenerationEvaluationRecord]:
    """Reload lightweight saved rows without rerunning retrieval or generation."""

    def optional_float(value: str) -> float | None:
        return None if value in ("", "None") else float(value)

    def optional_int(value: str) -> int | None:
        return None if value in ("", "None") else int(value)

    def boolean(value: str) -> bool:
        return value.strip().lower() == "true"

    records: list[GenerationEvaluationRecord] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                GenerationEvaluationRecord(
                    example_id=row["example_id"],
                    track=row["track"],
                    question=row["question"],
                    system=row["system"],
                    gold_document_id=row["gold_document_id"] or None,
                    retrieved_document_ids=tuple(json.loads(row["retrieved_document_ids"])),
                    context_document_ids=tuple(json.loads(row["context_document_ids"])),
                    retrieved_contexts=tuple(json.loads(row["retrieved_contexts"])),
                    initial_gold_rank=optional_int(row["initial_gold_rank"]),
                    final_gold_rank=optional_int(row["final_gold_rank"]),
                    graph_path=tuple(json.loads(row["graph_path"])),
                    retry_count=int(row["retry_count"]),
                    recovery_action=row["recovery_action"],
                    controller_abstained=boolean(row["controller_abstained"]),
                    generated_answer=row["generated_answer"],
                    safe_refusal=boolean(row["safe_refusal"]),
                    safe_refusal_phrase=row["safe_refusal_phrase"] or None,
                    substantive_claim_count=int(row["substantive_claim_count"]),
                    reference_answers=tuple(json.loads(row["reference_answers"])),
                    exact_match=optional_float(row["exact_match"]),
                    token_f1=optional_float(row["token_f1"]),
                    lexical_groundedness=optional_float(row["lexical_groundedness"]),
                    lexical_grounded=(
                        None
                        if row["lexical_grounded"] in ("", "None")
                        else boolean(row["lexical_grounded"])
                    ),
                    nli_claim_count=int(row["nli_claim_count"]),
                    nli_entailed_count=int(row["nli_entailed_count"]),
                    nli_neutral_count=int(row["nli_neutral_count"]),
                    nli_contradicted_count=int(row["nli_contradicted_count"]),
                    nli_entailed_percentage=optional_float(row["nli_entailed_percentage"]),
                    nli_neutral_percentage=optional_float(row["nli_neutral_percentage"]),
                    nli_contradicted_percentage=optional_float(row["nli_contradicted_percentage"]),
                    nli_fully_grounded=(
                        None
                        if row["nli_fully_grounded"] in ("", "None")
                        else boolean(row["nli_fully_grounded"])
                    ),
                    unsupported_answer=boolean(row["unsupported_answer"]),
                    generation_cache_hit=boolean(row["generation_cache_hit"]),
                    latency=json.loads(row["latency"]),
                )
            )
    return records


def render_generation_summary(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    comparison = metrics["three_way_comparison"]
    lines = [
        "# SQuAD 2.0 generation evaluation",
        "",
        "Detector calibration used SQuAD TRAIN only. Validation was used only for held-out evaluation. No paid API or service was used.",
        "",
        "## Dataset",
        "",
        f"Total observations: {payload['dataset']['total_observations']}",
        "",
        "```json",
        json.dumps(payload["dataset"]["category_counts"], indent=2),
        "```",
        "",
        "## Three-way comparison",
        "",
        "| System | Recall@5 | EM | F1 | NLI fully grounded | Missing safe refusal | Missing unsupported | Mean latency (s) | Median latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        item = comparison[system]
        def fmt(value):
            return "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {system} | {fmt(item['recall_at_5'])} | {fmt(item['answer_em'])} | "
            f"{fmt(item['answer_f1'])} | {fmt(item['nli_fully_grounded_rate'])} | "
            f"{fmt(item['safe_refusal_rate'])} | {fmt(item['unsupported_answer_rate'])} | "
            f"{fmt(item['mean_latency_seconds'])} | {fmt(item['median_latency_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## Controller abstention",
            "",
            "```json",
            json.dumps(metrics["controller_abstention"], indent=2),
            "```",
            "",
            "## Self-healing recovery",
            "",
            "```json",
            json.dumps(metrics["self_healing_recovery"], indent=2),
            "```",
            "",
            "Lexical groundedness is explicitly a LEXICAL HEURISTIC. NLI results are reported separately and are not combined into one score.",
        ]
    )
    return "\n".join(lines) + "\n"
