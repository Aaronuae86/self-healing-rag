"""Reproducible SQuAD 2.0 sampling for retrieval-only evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

from src.rag import Document


MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SquadSamplingConfig:
    """Configurable held-out question and mixed-passage corpus sample."""

    corpus_size: int = 5_000
    question_count: int = 2_000
    seed: int = 42
    dataset_name: str = "rajpurkar/squad_v2"
    dataset_revision: str | None = None
    cache_dir: str | Path | None = None


@dataclass(frozen=True)
class SquadRetrievalQuestion:
    """One answerable validation question and its gold passage identifier."""

    id: str
    question: str
    gold_document_id: str
    answer_texts: tuple[str, ...]


@dataclass(frozen=True)
class SquadRetrievalDataset:
    """Documents, held-out questions, and reproducibility metadata."""

    documents: tuple[Document, ...]
    questions: tuple[SquadRetrievalQuestion, ...]
    unanswerable_validation_ids: tuple[str, ...]
    manifest_path: Path
    fingerprint: str


def _canonical_context(context: str) -> str:
    return context.strip()


def stable_context_id(context: str) -> str:
    """Assign the same compact ID to identical context text across splits."""

    digest = hashlib.sha256(_canonical_context(context).encode("utf-8")).hexdigest()
    return f"squad2-context-{digest[:20]}"


def _answer_texts(record: dict[str, Any]) -> tuple[str, ...]:
    answers = record.get("answers") or {}
    texts = answers.get("text", []) if isinstance(answers, dict) else []
    return tuple(dict.fromkeys(str(text).strip() for text in texts if str(text).strip()))


def _document_from_record(record: dict[str, Any]) -> Document:
    context = _canonical_context(str(record["context"]))
    title = str(record.get("title") or "SQuAD passage").strip()
    return Document(id=stable_context_id(context), title=title, text=context)


def _dataset_load_kwargs(config: SquadSamplingConfig) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if config.cache_dir is not None:
        kwargs["cache_dir"] = str(config.cache_dir)
    if config.dataset_revision is not None:
        kwargs["revision"] = config.dataset_revision
    return kwargs


def _validate_config(config: SquadSamplingConfig) -> None:
    if config.corpus_size < 1:
        raise ValueError("corpus_size must be at least 1.")
    if config.question_count < 1:
        raise ValueError("question_count must be at least 1.")
    if not config.dataset_name.strip():
        raise ValueError("dataset_name must not be empty.")


def _build_context_pool(
    validation_records: Iterable[dict[str, Any]],
    train_records: Iterable[dict[str, Any]],
) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    canonical_by_id: dict[str, str] = {}
    for record in chain(validation_records, train_records):
        document = _document_from_record(record)
        canonical = _canonical_context(document.text)
        previous = canonical_by_id.get(document.id)
        if previous is not None and previous != canonical:
            raise RuntimeError(f"Stable document ID collision: {document.id}")
        canonical_by_id[document.id] = canonical
        documents.setdefault(document.id, document)
    return documents


def _manifest_matches(manifest: dict[str, Any], config: SquadSamplingConfig) -> bool:
    expected = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "seed": config.seed,
        "corpus_size": config.corpus_size,
        "question_count": config.question_count,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def _fingerprint(document_ids: Iterable[str], question_ids: Iterable[str]) -> str:
    payload = "\n".join([*document_ids, "--questions--", *question_ids])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prepare_squad_retrieval_dataset(
    config: SquadSamplingConfig | None = None,
    manifest_path: str | Path = "results/squad_retrieval_sample_manifest.json",
) -> SquadRetrievalDataset:
    """Load SQuAD 2.0 and build or replay a deterministic retrieval sample.

    Evaluated questions always come from the answerable validation subset. Their
    gold contexts are inserted first; remaining corpus slots are deterministic
    distractors from validation and train. Train questions are never scored.
    """

    settings = config or SquadSamplingConfig()
    _validate_config(settings)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install requirements.txt to load SQuAD 2.0.") from error

    dataset = load_dataset(settings.dataset_name, **_dataset_load_kwargs(settings))
    validation_records = [dict(record) for record in dataset["validation"]]
    train_records = dataset["train"]
    validation_by_id = {str(record["id"]): record for record in validation_records}
    context_pool = _build_context_pool(validation_records, train_records)
    unanswerable_ids = tuple(
        str(record["id"])
        for record in validation_records
        if not _answer_texts(record)
    )

    output_path = Path(manifest_path)
    manifest: dict[str, Any] | None = None
    if output_path.exists():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if _manifest_matches(loaded, settings):
            manifest = loaded

    if manifest is None:
        answerable = [record for record in validation_records if _answer_texts(record)]
        random.Random(settings.seed).shuffle(answerable)
        selected_records: list[dict[str, Any]] = []
        gold_document_ids: list[str] = []
        gold_document_set: set[str] = set()
        for record in answerable:
            document_id = stable_context_id(str(record["context"]))
            if (
                document_id not in gold_document_set
                and len(gold_document_set) >= settings.corpus_size
            ):
                continue
            selected_records.append(record)
            if document_id not in gold_document_set:
                gold_document_ids.append(document_id)
                gold_document_set.add(document_id)
            if len(selected_records) == settings.question_count:
                break

        if len(selected_records) != settings.question_count:
            raise ValueError(
                "The requested question_count cannot fit while retaining every gold "
                "context inside the configured corpus_size. Increase corpus_size or "
                "reduce question_count."
            )

        filler_ids = [
            document_id
            for document_id in context_pool
            if document_id not in gold_document_set
        ]
        random.Random(settings.seed + 1).shuffle(filler_ids)
        required_fillers = settings.corpus_size - len(gold_document_ids)
        if required_fillers > len(filler_ids):
            raise ValueError(
                f"corpus_size={settings.corpus_size} exceeds the available unique "
                f"SQuAD contexts ({len(context_pool)})."
            )
        document_ids = [*gold_document_ids, *filler_ids[:required_fillers]]
        question_ids = [str(record["id"]) for record in selected_records]
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "dataset_name": settings.dataset_name,
            "dataset_revision": settings.dataset_revision,
            "seed": settings.seed,
            "corpus_size": settings.corpus_size,
            "question_count": settings.question_count,
            "document_ids": document_ids,
            "question_ids": question_ids,
            "unanswerable_validation_ids": list(unanswerable_ids),
            "validation_fingerprint": getattr(dataset["validation"], "_fingerprint", None),
            "train_fingerprint": getattr(dataset["train"], "_fingerprint", None),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    document_ids = [str(item) for item in manifest["document_ids"]]
    question_ids = [str(item) for item in manifest["question_ids"]]
    missing_documents = [item for item in document_ids if item not in context_pool]
    missing_questions = [item for item in question_ids if item not in validation_by_id]
    if missing_documents or missing_questions:
        raise ValueError(
            "The saved manifest does not match the loaded dataset version. "
            f"Missing documents={missing_documents[:3]}, questions={missing_questions[:3]}."
        )

    documents = tuple(context_pool[document_id] for document_id in document_ids)
    document_id_set = set(document_ids)
    questions: list[SquadRetrievalQuestion] = []
    for example_id in question_ids:
        record = validation_by_id[example_id]
        answer_texts = _answer_texts(record)
        gold_document_id = stable_context_id(str(record["context"]))
        if not answer_texts:
            raise ValueError(f"Manifest question {example_id} is not answerable.")
        if gold_document_id not in document_id_set:
            raise ValueError(
                f"Gold context {gold_document_id} for {example_id} is absent from corpus."
            )
        questions.append(
            SquadRetrievalQuestion(
                id=example_id,
                question=str(record["question"]).strip(),
                gold_document_id=gold_document_id,
                answer_texts=answer_texts,
            )
        )

    return SquadRetrievalDataset(
        documents=documents,
        questions=tuple(questions),
        unanswerable_validation_ids=tuple(
            str(item) for item in manifest["unanswerable_validation_ids"]
        ),
        manifest_path=output_path,
        fingerprint=_fingerprint(document_ids, question_ids),
    )


def sampling_config_to_dict(config: SquadSamplingConfig) -> dict[str, Any]:
    """Serialize sampling settings without leaking Path objects into JSON."""

    values = asdict(config)
    if values["cache_dir"] is not None:
        values["cache_dir"] = str(values["cache_dir"])
    return values
