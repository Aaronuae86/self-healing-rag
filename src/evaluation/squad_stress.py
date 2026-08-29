"""Controlled SQuAD 2.0 stress tracks and retrieval-only evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from src.rag import (
    Document,
    HybridRetriever,
    RetrievalDiagnostics,
    RetrievalFailure,
    RetrievalFailureDetector,
    SelfHealingRAGWorkflow,
    compute_retrieval_diagnostics,
)

from .detector_calibration import (
    DiagnosticCalibrationExample,
    ObjectiveFailureLabel,
    compute_binary_failure_metrics,
    compute_failure_mode_metrics,
    objective_label_from_rank,
)
from .squad_dataset import stable_context_id
from .squad_retrieval import (
    CachedCrossEncoderReranker,
    CachedDenseRetriever,
    gold_document_rank,
    gold_rank_movement,
    is_failed_recovery_at_cutoff,
    is_initial_failure_at_cutoff,
    is_recovered_at_cutoff,
)


STRESS_MANIFEST_VERSION = 1


class StressTrack(str, Enum):
    TRAIN_CALIBRATION = "TRAIN_CALIBRATION"
    HARD_DISTRACTOR = "HARD_DISTRACTOR"
    PARAPHRASE_ORIGINAL = "PARAPHRASE_ORIGINAL"
    PARAPHRASE_TRANSFORMED = "PARAPHRASE_TRANSFORMED"
    CONTROLLED_MISSING_EVIDENCE = "CONTROLLED_MISSING_EVIDENCE"
    NATURAL_UNANSWERABLE = "NATURAL_UNANSWERABLE"


@dataclass(frozen=True)
class SquadStressSamplingConfig:
    corpus_size: int = 5_000
    calibration_question_count: int = 1_000
    calibration_missing_count: int = 250
    hard_distractor_count: int = 100
    paraphrase_pair_count: int = 50
    controlled_missing_count: int = 100
    natural_unanswerable_count: int = 100
    hard_distractors_per_query: int = 20
    seed: int = 42
    dataset_name: str = "rajpurkar/squad_v2"
    dataset_revision: str | None = None
    cache_dir: str | Path | None = None


@dataclass(frozen=True)
class StressQuestion:
    id: str
    track: StressTrack
    original_query: str
    transformed_query: str | None
    gold_document_id: str | None
    source_document_id: str
    title: str

    @property
    def active_query(self) -> str:
        return self.transformed_query or self.original_query


@dataclass(frozen=True)
class SquadStressDataset:
    documents: tuple[Document, ...]
    calibration_questions: tuple[StressQuestion, ...]
    calibration_missing_ids: tuple[str, ...]
    validation_questions: tuple[StressQuestion, ...]
    manifest_path: Path
    fingerprint: str


@dataclass(frozen=True)
class InitialRetrievalSnapshot:
    diagnostics: RetrievalDiagnostics
    reranked_results: tuple
    gold_rank: int | None


@dataclass(frozen=True)
class StressEvaluationRecord:
    example_id: str
    track: str
    detector_variant: str
    original_query: str
    transformed_query: str | None
    active_query: str
    gold_document_id: str | None
    distractor_ids: tuple[str, ...]
    gold_intentionally_removed: bool
    initial_gold_rank: int | None
    final_gold_rank: int | None
    objective_failure_label: str
    predicted_failure_type: str
    dense_top1_score: float | None
    dense_top1_top2_margin: float | None
    dense_average_top_k_score: float | None
    bm25_top1_score: float | None
    dense_bm25_overlap_ratio: float
    hybrid_top_rrf_score: float | None
    cross_encoder_top_score: float | None
    cross_encoder_top1_top2_margin: float | None
    graph_path: tuple[str, ...]
    retry_count: int
    rewritten_query: str | None
    recovered_at_5: bool
    failed_recovery_at_5: bool
    rank_movement: str
    paired_original_gold_rank: int | None
    paraphrase_rank_movement: str
    paraphrase_induced_failure_at_5: bool
    initial_retrieved_ids: tuple[str, ...]
    final_retrieved_ids: tuple[str, ...]


def _answer_texts(record: dict[str, Any]) -> tuple[str, ...]:
    answers = record.get("answers") or {}
    return tuple(str(item).strip() for item in answers.get("text", []) if str(item).strip())


def _document(record: dict[str, Any]) -> Document:
    text = str(record["context"]).strip()
    return Document(
        id=stable_context_id(text),
        title=str(record.get("title") or "SQuAD passage").strip(),
        text=text,
    )


def _config_dict(config: SquadStressSamplingConfig) -> dict:
    values = asdict(config)
    if values["cache_dir"] is not None:
        values["cache_dir"] = str(values["cache_dir"])
    return values


def _validate_sampling_config(config: SquadStressSamplingConfig) -> None:
    count_fields = (
        config.corpus_size,
        config.calibration_question_count,
        config.hard_distractor_count,
        config.paraphrase_pair_count,
        config.controlled_missing_count,
        config.natural_unanswerable_count,
    )
    if any(value < 1 for value in count_fields):
        raise ValueError("Corpus and track counts must be at least 1.")
    if not 0 <= config.calibration_missing_count <= config.calibration_question_count:
        raise ValueError("calibration_missing_count must fit inside calibration questions.")


def _manifest_matches(manifest: dict, config: SquadStressSamplingConfig) -> bool:
    return (
        manifest.get("manifest_version") == STRESS_MANIFEST_VERSION
        and manifest.get("sampling_config") == _config_dict(config)
    )


def _sample_ids(records: Sequence[dict], count: int, seed: int) -> list[str]:
    shuffled = [str(record["id"]) for record in records]
    random.Random(seed).shuffle(shuffled)
    if count > len(shuffled):
        raise ValueError(f"Requested {count} examples but only {len(shuffled)} are available.")
    return shuffled[:count]


def prepare_squad_stress_dataset(
    config: SquadStressSamplingConfig | None = None,
    manifest_path: str | Path = "results/squad_stress_manifest.json",
) -> SquadStressDataset:
    """Build disjoint train calibration and validation stress samples."""

    settings = config or SquadStressSamplingConfig()
    _validate_sampling_config(settings)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install requirements.txt to load SQuAD 2.0.") from error
    load_kwargs: dict[str, str] = {}
    if settings.cache_dir is not None:
        load_kwargs["cache_dir"] = str(settings.cache_dir)
    if settings.dataset_revision is not None:
        load_kwargs["revision"] = settings.dataset_revision
    dataset = load_dataset(settings.dataset_name, **load_kwargs)
    train = [dict(item) for item in dataset["train"]]
    validation = [dict(item) for item in dataset["validation"]]
    train_by_id = {str(item["id"]): item for item in train}
    validation_by_id = {str(item["id"]): item for item in validation}
    train_answerable = [item for item in train if _answer_texts(item)]
    validation_answerable = [item for item in validation if _answer_texts(item)]
    validation_unanswerable = [item for item in validation if not _answer_texts(item)]

    output_path = Path(manifest_path)
    manifest = None
    if output_path.exists():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if _manifest_matches(loaded, settings):
            manifest = loaded

    context_pool: dict[str, Document] = {}
    if manifest is None:
        calibration_ids = _sample_ids(
            train_answerable, settings.calibration_question_count, settings.seed
        )
        validation_ids = _sample_ids(
            validation_answerable,
            settings.hard_distractor_count
            + settings.paraphrase_pair_count
            + settings.controlled_missing_count,
            settings.seed + 1,
        )
        hard_end = settings.hard_distractor_count
        paraphrase_end = hard_end + settings.paraphrase_pair_count
        hard_ids = validation_ids[:hard_end]
        paraphrase_ids = validation_ids[hard_end:paraphrase_end]
        missing_ids = validation_ids[paraphrase_end:]
        natural_ids = _sample_ids(
            validation_unanswerable,
            settings.natural_unanswerable_count,
            settings.seed + 2,
        )

        selected_records = [
            *(train_by_id[item] for item in calibration_ids),
            *(validation_by_id[item] for item in validation_ids),
            *(validation_by_id[item] for item in natural_ids),
        ]
        required_document_ids = list(
            dict.fromkeys(_document(record).id for record in selected_records)
        )
        for record in chain(validation, train):
            document = _document(record)
            context_pool.setdefault(document.id, document)
        if len(required_document_ids) > settings.corpus_size:
            raise ValueError("corpus_size cannot hold all selected gold/source contexts.")
        filler_ids = [item for item in context_pool if item not in set(required_document_ids)]
        random.Random(settings.seed + 3).shuffle(filler_ids)
        document_ids = [
            *required_document_ids,
            *filler_ids[: settings.corpus_size - len(required_document_ids)],
        ]
        if len(document_ids) != settings.corpus_size:
            raise ValueError("Not enough unique SQuAD contexts for the configured corpus.")
        manifest = {
            "manifest_version": STRESS_MANIFEST_VERSION,
            "sampling_config": _config_dict(settings),
            "document_ids": document_ids,
            "calibration_question_ids": calibration_ids,
            "calibration_missing_ids": calibration_ids[: settings.calibration_missing_count],
            "validation_tracks": {
                StressTrack.HARD_DISTRACTOR.value: hard_ids,
                StressTrack.PARAPHRASE_ORIGINAL.value: paraphrase_ids,
                StressTrack.CONTROLLED_MISSING_EVIDENCE.value: missing_ids,
                StressTrack.NATURAL_UNANSWERABLE.value: natural_ids,
            },
            "hard_distractors": {},
            "paraphrases": {},
            "train_fingerprint": getattr(dataset["train"], "_fingerprint", None),
            "validation_fingerprint": getattr(dataset["validation"], "_fingerprint", None),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not context_pool:
        for record in chain(validation, train):
            document = _document(record)
            context_pool.setdefault(document.id, document)
    documents = tuple(context_pool[item] for item in manifest["document_ids"])

    def make_question(record: dict, track: StressTrack) -> StressQuestion:
        source_id = _document(record).id
        answerable = bool(_answer_texts(record))
        return StressQuestion(
            id=str(record["id"]),
            track=track,
            original_query=str(record["question"]).strip(),
            transformed_query=(
                manifest.get("paraphrases", {}).get(str(record["id"]))
                if track == StressTrack.PARAPHRASE_TRANSFORMED
                else None
            ),
            gold_document_id=source_id if answerable else None,
            source_document_id=source_id,
            title=str(record.get("title") or "").strip(),
        )

    calibration_questions = tuple(
        make_question(train_by_id[item], StressTrack.TRAIN_CALIBRATION)
        for item in manifest["calibration_question_ids"]
    )
    tracks = manifest["validation_tracks"]
    validation_questions: list[StressQuestion] = []
    validation_questions.extend(
        make_question(validation_by_id[item], StressTrack.HARD_DISTRACTOR)
        for item in tracks[StressTrack.HARD_DISTRACTOR.value]
    )
    for item in tracks[StressTrack.PARAPHRASE_ORIGINAL.value]:
        validation_questions.append(
            make_question(validation_by_id[item], StressTrack.PARAPHRASE_ORIGINAL)
        )
        validation_questions.append(
            make_question(validation_by_id[item], StressTrack.PARAPHRASE_TRANSFORMED)
        )
    validation_questions.extend(
        make_question(validation_by_id[item], StressTrack.CONTROLLED_MISSING_EVIDENCE)
        for item in tracks[StressTrack.CONTROLLED_MISSING_EVIDENCE.value]
    )
    validation_questions.extend(
        make_question(validation_by_id[item], StressTrack.NATURAL_UNANSWERABLE)
        for item in tracks[StressTrack.NATURAL_UNANSWERABLE.value]
    )
    fingerprint_payload = "\n".join(manifest["document_ids"] + manifest["calibration_question_ids"])
    return SquadStressDataset(
        documents=documents,
        calibration_questions=calibration_questions,
        calibration_missing_ids=tuple(manifest["calibration_missing_ids"]),
        validation_questions=tuple(validation_questions),
        manifest_path=output_path,
        fingerprint=hashlib.sha256(fingerprint_payload.encode()).hexdigest()[:16],
    )


class ExcludingRetriever:
    """Filter exact stable document IDs while retaining the wrapped result type."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self.excluded_document_ids: set[str] = set()

    @property
    def documents(self):
        return self.retriever.documents

    def retrieve(self, query: str, top_k: int = 3):
        fetch_k = min(len(self.documents), top_k + len(self.excluded_document_ids))
        results = self.retriever.retrieve(query, top_k=fetch_k)
        return [
            item for item in results if item.id not in self.excluded_document_ids
        ][:top_k]


class RetrievalExclusionController:
    def __init__(self, *retrievers: ExcludingRetriever) -> None:
        self.retrievers = retrievers

    def set_excluded(self, document_ids: Sequence[str]) -> None:
        excluded = set(document_ids)
        for retriever in self.retrievers:
            retriever.excluded_document_ids = excluded

    def clear(self) -> None:
        self.set_excluded(())


class LocalQwenParaphraser:
    """Deterministic local paraphrasing without answer text in the prompt."""

    INSTRUCTION = (
        "Rewrite the question as one natural paraphrase. Preserve its information "
        "need and named entities, do not answer it, do not add clues, and output only "
        "the rewritten question."
    )

    def __init__(self, generator, max_new_tokens: int = 64) -> None:
        self.generator = generator
        self.max_new_tokens = max_new_tokens

    def paraphrase(self, question: str) -> str:
        text = self.generator.generate_messages(
            [
                {"role": "system", "content": self.INSTRUCTION},
                {"role": "user", "content": question},
            ],
            max_new_tokens=self.max_new_tokens,
        )
        candidate = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return candidate.strip(" \"'") or question


class CachedQueryRewriter:
    """Reuse deterministic local query rewrites across detector variants."""

    def __init__(self, rewriter) -> None:
        self.rewriter = rewriter
        self._cache: dict[str, str] = {}

    def rewrite(self, query: str) -> str:
        if query not in self._cache:
            self._cache[query] = self.rewriter.rewrite(query)
        return self._cache[query]


def prepare_paraphrases(
    dataset: SquadStressDataset,
    paraphraser: LocalQwenParaphraser,
    progress_callback: Callable[[int, int, StressQuestion], None] | None = None,
) -> dict[str, str]:
    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    paraphrases = dict(manifest.get("paraphrases", {}))
    questions = [
        item
        for item in dataset.validation_questions
        if item.track == StressTrack.PARAPHRASE_ORIGINAL
    ]
    for index, item in enumerate(questions, start=1):
        if progress_callback:
            progress_callback(index, len(questions), item)
        paraphrases.setdefault(item.id, paraphraser.paraphrase(item.original_query))
    manifest["paraphrases"] = paraphrases
    dataset.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paraphrases


def select_hard_distractors(
    dataset: SquadStressDataset,
    dense_retriever: CachedDenseRetriever,
    per_question: int = 20,
    search_depth: int = 100,
) -> dict[str, tuple[str, ...]]:
    questions = [
        item
        for item in dataset.validation_questions
        if item.track == StressTrack.HARD_DISTRACTOR
    ]
    ranked_lists = dense_retriever.preload(
        [item.original_query for item in questions], top_k=search_depth
    )
    document_by_id = {item.id: item for item in dataset.documents}
    selected: dict[str, tuple[str, ...]] = {}
    for question, ranked in zip(questions, ranked_lists):
        non_gold = [item for item in ranked if item.id != question.gold_document_id]
        same_topic = [
            item.id
            for item in non_gold
            if document_by_id[item.id].title == question.title
        ]
        nearest = [item.id for item in non_gold]
        selected[question.id] = tuple(dict.fromkeys([*same_topic, *nearest]))[:per_question]
    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    manifest["hard_distractors"] = {
        key: list(value) for key, value in selected.items()
    }
    dataset.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return selected


def collect_initial_snapshot(
    query: str,
    gold_document_id: str | None,
    dense_retriever,
    bm25_retriever,
    hybrid_retriever: HybridRetriever,
    reranker: CachedCrossEncoderReranker,
    candidate_k: int = 20,
    final_k: int = 10,
) -> InitialRetrievalSnapshot:
    dense = dense_retriever.retrieve(query, top_k=candidate_k)
    bm25 = bm25_retriever.retrieve(query, top_k=candidate_k)
    hybrid = hybrid_retriever.fuse(dense, bm25, top_k=candidate_k)
    reranked = reranker.rerank(query, hybrid, top_k=final_k)
    diagnostics = compute_retrieval_diagnostics(query, dense, bm25, hybrid, reranked)
    ids = tuple(item.id for item in reranked)
    return InitialRetrievalSnapshot(
        diagnostics=diagnostics,
        reranked_results=tuple(reranked),
        gold_rank=(
            gold_document_rank(ids, gold_document_id)
            if gold_document_id is not None
            else None
        ),
    )


def _diagnostic_row(
    example_id: str,
    split_variant: str,
    query: str,
    gold_document_id: str,
    snapshot: InitialRetrievalSnapshot,
    objective_label: ObjectiveFailureLabel,
    prediction: RetrievalFailure,
) -> dict:
    diagnostics = snapshot.diagnostics
    return {
        "example_id": example_id,
        "split_variant": split_variant,
        "query": query,
        "gold_document_id": gold_document_id,
        "objective_failure_label": objective_label.value,
        "initial_gold_rank": snapshot.gold_rank,
        "gold_in_top1": snapshot.gold_rank is not None and snapshot.gold_rank <= 1,
        "gold_in_top3": snapshot.gold_rank is not None and snapshot.gold_rank <= 3,
        "gold_in_top5": snapshot.gold_rank is not None and snapshot.gold_rank <= 5,
        "gold_in_top10": snapshot.gold_rank is not None and snapshot.gold_rank <= 10,
        "predicted_phase5_failure_type": prediction.value,
        **diagnostics.to_dict(),
    }


def collect_calibration_diagnostics(
    dataset: SquadStressDataset,
    dense_retriever,
    bm25_retriever,
    hybrid_retriever: HybridRetriever,
    reranker: CachedCrossEncoderReranker,
    exclusion_controller: RetrievalExclusionController,
    original_detector: RetrievalFailureDetector,
    output_path: str | Path = "results/squad_calibration_diagnostics.csv",
    cutoff: int = 5,
    progress_callback: Callable[[int, int, StressQuestion], None] | None = None,
) -> tuple[list[DiagnosticCalibrationExample], list[dict]]:
    rows: list[dict] = []
    examples: list[DiagnosticCalibrationExample] = []
    missing_ids = set(dataset.calibration_missing_ids)
    total = len(dataset.calibration_questions) + len(missing_ids)
    progress_index = 0
    for question in dataset.calibration_questions:
        progress_index += 1
        if progress_callback:
            progress_callback(progress_index, total, question)
        exclusion_controller.clear()
        snapshot = collect_initial_snapshot(
            question.original_query,
            question.gold_document_id,
            dense_retriever,
            bm25_retriever,
            hybrid_retriever,
            reranker,
        )
        label = objective_label_from_rank(snapshot.gold_rank, cutoff=cutoff)
        prediction = original_detector.classify(snapshot.diagnostics).failure_type
        examples.append(DiagnosticCalibrationExample(question.id, label, snapshot.diagnostics))
        rows.append(
            _diagnostic_row(
                question.id,
                "TRAIN_RETRIEVAL",
                question.original_query,
                question.gold_document_id or "",
                snapshot,
                label,
                prediction,
            )
        )
        if question.id in missing_ids:
            progress_index += 1
            exclusion_controller.set_excluded((question.gold_document_id,))
            missing_snapshot = collect_initial_snapshot(
                question.original_query,
                question.gold_document_id,
                dense_retriever,
                bm25_retriever,
                hybrid_retriever,
                reranker,
            )
            missing_label = ObjectiveFailureLabel.MISSING_EVIDENCE
            missing_prediction = original_detector.classify(
                missing_snapshot.diagnostics
            ).failure_type
            examples.append(
                DiagnosticCalibrationExample(
                    f"{question.id}:missing",
                    missing_label,
                    missing_snapshot.diagnostics,
                )
            )
            rows.append(
                _diagnostic_row(
                    question.id,
                    "TRAIN_CONTROLLED_MISSING",
                    question.original_query,
                    question.gold_document_id or "",
                    missing_snapshot,
                    missing_label,
                    missing_prediction,
                )
            )
    exclusion_controller.clear()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return examples, rows


class SquadStressBenchmark:
    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        hybrid_retriever: HybridRetriever,
        reranker: CachedCrossEncoderReranker,
        exclusion_controller: RetrievalExclusionController,
        original_detector: RetrievalFailureDetector,
        calibrated_detector: RetrievalFailureDetector,
        original_workflow: SelfHealingRAGWorkflow,
        calibrated_workflow: SelfHealingRAGWorkflow,
        cutoff: int = 5,
        output_dir: str | Path = "results",
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.exclusion_controller = exclusion_controller
        self.detectors = {
            "ORIGINAL_PHASE5": original_detector,
            "TRAIN_CALIBRATED": calibrated_detector,
        }
        self.workflows = {
            "ORIGINAL_PHASE5": original_workflow,
            "TRAIN_CALIBRATED": calibrated_workflow,
        }
        self.cutoff = cutoff
        self.output_dir = Path(output_dir)

    def run(
        self,
        dataset: SquadStressDataset,
        hard_distractors: dict[str, Sequence[str]],
        paraphrases: dict[str, str],
        calibration_summary: dict,
        progress_callback: Callable[[int, int, StressQuestion], None] | None = None,
    ) -> tuple[list[StressEvaluationRecord], dict]:
        questions = [
            StressQuestion(
                id=item.id,
                track=item.track,
                original_query=item.original_query,
                transformed_query=(
                    paraphrases.get(item.id)
                    if item.track == StressTrack.PARAPHRASE_TRANSFORMED
                    else item.transformed_query
                ),
                gold_document_id=item.gold_document_id,
                source_document_id=item.source_document_id,
                title=item.title,
            )
            for item in dataset.validation_questions
        ]
        original_paraphrase_ranks: dict[str, int | None] = {}
        records: list[StressEvaluationRecord] = []
        for index, question in enumerate(questions, start=1):
            if progress_callback:
                progress_callback(index, len(questions), question)
            gold_removed = question.track == StressTrack.CONTROLLED_MISSING_EVIDENCE
            self.exclusion_controller.set_excluded(
                (question.gold_document_id,) if gold_removed else ()
            )
            snapshot = collect_initial_snapshot(
                question.active_query,
                question.gold_document_id,
                self.dense_retriever,
                self.bm25_retriever,
                self.hybrid_retriever,
                self.reranker,
            )
            if question.track == StressTrack.NATURAL_UNANSWERABLE:
                objective = ObjectiveFailureLabel.NATURAL_UNANSWERABLE
            else:
                objective = objective_label_from_rank(
                    snapshot.gold_rank,
                    cutoff=self.cutoff,
                    gold_intentionally_removed=gold_removed,
                )
            if question.track == StressTrack.PARAPHRASE_ORIGINAL:
                original_paraphrase_ranks[question.id] = snapshot.gold_rank
            original_rank = original_paraphrase_ranks.get(question.id)
            induced = (
                question.track == StressTrack.PARAPHRASE_TRANSFORMED
                and original_rank is not None
                and original_rank <= self.cutoff
                and is_initial_failure_at_cutoff(snapshot.gold_rank, self.cutoff)
            )
            for variant, detector in self.detectors.items():
                prediction = detector.classify(snapshot.diagnostics).failure_type
                state = self.workflows[variant].run_retrieval_only(question.active_query)
                initial_history = state.get("reranked_results_history", [])
                initial_results = initial_history[0] if initial_history else snapshot.reranked_results
                final_results = state["reranked_results"]
                initial_ids = tuple(item.id for item in initial_results[:10])
                final_ids = tuple(item.id for item in final_results[:10])
                initial_rank = (
                    gold_document_rank(initial_ids, question.gold_document_id)
                    if question.gold_document_id is not None
                    else None
                )
                final_rank = (
                    gold_document_rank(final_ids, question.gold_document_id)
                    if question.gold_document_id is not None
                    else None
                )
                diagnostics = snapshot.diagnostics
                records.append(
                    StressEvaluationRecord(
                        example_id=question.id,
                        track=question.track.value,
                        detector_variant=variant,
                        original_query=question.original_query,
                        transformed_query=question.transformed_query,
                        active_query=question.active_query,
                        gold_document_id=question.gold_document_id,
                        distractor_ids=tuple(hard_distractors.get(question.id, ())),
                        gold_intentionally_removed=gold_removed,
                        initial_gold_rank=initial_rank,
                        final_gold_rank=final_rank,
                        objective_failure_label=objective.value,
                        predicted_failure_type=prediction.value,
                        dense_top1_score=diagnostics.dense_top1_score,
                        dense_top1_top2_margin=diagnostics.dense_top1_top2_margin,
                        dense_average_top_k_score=diagnostics.dense_average_top_k_score,
                        bm25_top1_score=diagnostics.bm25_top1_score,
                        dense_bm25_overlap_ratio=diagnostics.dense_bm25_overlap_ratio,
                        hybrid_top_rrf_score=diagnostics.hybrid_top_rrf_score,
                        cross_encoder_top_score=diagnostics.cross_encoder_top_score,
                        cross_encoder_top1_top2_margin=diagnostics.cross_encoder_top1_top2_margin,
                        graph_path=tuple(state.get("path", [])),
                        retry_count=int(state.get("retry_count", 0)),
                        rewritten_query=state.get("rewritten_query"),
                        recovered_at_5=(
                            question.gold_document_id is not None
                            and is_recovered_at_cutoff(initial_rank, final_rank, self.cutoff)
                        ),
                        failed_recovery_at_5=(
                            question.gold_document_id is not None
                            and is_failed_recovery_at_cutoff(
                                initial_rank,
                                final_rank,
                                int(state.get("retry_count", 0)) > 0,
                                self.cutoff,
                            )
                        ),
                        rank_movement=(
                            gold_rank_movement(initial_rank, final_rank)
                            if question.gold_document_id is not None
                            else "NOT_APPLICABLE"
                        ),
                        paired_original_gold_rank=(
                            original_rank
                            if question.track == StressTrack.PARAPHRASE_TRANSFORMED
                            else None
                        ),
                        paraphrase_rank_movement=(
                            gold_rank_movement(original_rank, snapshot.gold_rank)
                            if question.track == StressTrack.PARAPHRASE_TRANSFORMED
                            else "NOT_APPLICABLE"
                        ),
                        paraphrase_induced_failure_at_5=induced,
                        initial_retrieved_ids=initial_ids,
                        final_retrieved_ids=final_ids,
                    )
                )
        self.exclusion_controller.clear()
        metrics, confusion_rows = summarize_stress_metrics(records)
        payload = {
            "benchmark": "SQuAD 2.0 controlled stress tracks",
            "cutoff": self.cutoff,
            "train_development_calibration": calibration_summary,
            "held_out_validation_stress": metrics,
        }
        self._save(records, payload, confusion_rows)
        return records, payload

    def _save(self, records, payload, confusion_rows) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "squad_stress_metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        row_dicts = [asdict(item) for item in records]
        list_fields = {"distractor_ids", "graph_path", "initial_retrieved_ids", "final_retrieved_ids"}
        with (self.output_dir / "squad_stress_per_example.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0]))
            writer.writeheader()
            for row in row_dicts:
                for field in list_fields:
                    row[field] = json.dumps(row[field], ensure_ascii=False)
                writer.writerow(row)
        with (self.output_dir / "squad_failure_confusion.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("detector_variant", "track", "actual", "predicted", "count"),
            )
            writer.writeheader()
            writer.writerows(confusion_rows)


def summarize_stress_metrics(
    records: Sequence[StressEvaluationRecord],
) -> tuple[dict, list[dict]]:
    output: dict[str, dict] = {}
    confusion_rows: list[dict] = []
    variants = sorted({item.detector_variant for item in records})
    tracks = sorted({item.track for item in records})
    for variant in variants:
        output[variant] = {}
        for track in tracks:
            group = [
                item
                for item in records
                if item.detector_variant == variant and item.track == track
            ]
            if not group:
                continue
            actual = [ObjectiveFailureLabel(item.objective_failure_label) for item in group]
            predicted = [RetrievalFailure(item.predicted_failure_type) for item in group]
            supervised = all(item != ObjectiveFailureLabel.NATURAL_UNANSWERABLE for item in actual)
            binary = compute_binary_failure_metrics(actual, predicted) if supervised else None
            mode = compute_failure_mode_metrics(actual, predicted) if supervised else None
            genuine_failures = [
                item
                for item in group
                if item.objective_failure_label
                in {
                    ObjectiveFailureLabel.RETRIEVAL_FAILURE.value,
                    ObjectiveFailureLabel.MISSING_EVIDENCE.value,
                }
            ]
            movements = Counter(
                item.rank_movement
                for item in group
                if item.rank_movement != "NOT_APPLICABLE"
            )
            healthy = [
                item
                for item in group
                if item.objective_failure_label == ObjectiveFailureLabel.HEALTHY.value
            ]
            prediction_counts = Counter(item.predicted_failure_type for item in group)
            paraphrase_movements = Counter(
                item.paraphrase_rank_movement
                for item in group
                if item.paraphrase_rank_movement != "NOT_APPLICABLE"
            )
            output[variant][track] = {
                "example_count": len(group),
                "supervised_failure_metrics_available": supervised,
                "initial_failure_rate": (
                    len(genuine_failures) / len(group) if supervised else None
                ),
                "binary_failure_detection": binary.to_dict() if binary else None,
                "failure_mode": mode.to_dict() if mode else None,
                "genuine_initial_failures": len(genuine_failures) if supervised else None,
                "genuine_failures_detected": (
                    sum(item.predicted_failure_type != RetrievalFailure.HEALTHY.value for item in genuine_failures)
                    if supervised
                    else None
                ),
                "recovery_attempts": sum(item.retry_count > 0 for item in group),
                "recovered_at_5": sum(item.recovered_at_5 for item in group),
                "failed_recovery_at_5": sum(item.failed_recovery_at_5 for item in group),
                "gold_rank_movement": {
                    "improved": movements["improved"],
                    "unchanged": movements["unchanged"],
                    "worsened": movements["worsened"],
                },
                "false_positive_healing_rate": (
                    sum(item.retry_count > 0 for item in healthy) / len(healthy)
                    if healthy
                    else 0.0
                ),
                "prediction_counts": dict(prediction_counts),
                "paraphrase_induced_failures_at_5": sum(
                    item.paraphrase_induced_failure_at_5 for item in group
                ),
                "paraphrase_gold_rank_movement": {
                    "improved": paraphrase_movements["improved"],
                    "unchanged": paraphrase_movements["unchanged"],
                    "worsened": paraphrase_movements["worsened"],
                },
            }
            if mode:
                for truth, row in mode.raw_phase5_prediction_confusion.items():
                    for guess, count in row.items():
                        confusion_rows.append(
                            {
                                "detector_variant": variant,
                                "track": track,
                                "actual": truth,
                                "predicted": guess,
                                "count": count,
                            }
                        )
    return output, confusion_rows
