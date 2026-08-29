"""Schema and validation for the fixed Phase 6 evaluation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.rag import RetrievalFailure


EVALUATION_CATEGORIES = {
    "healthy_supported",
    "paraphrased",
    "ambiguous",
    "weak_retrieval",
    "missing_evidence",
}


@dataclass(frozen=True)
class EvaluationExample:
    """One labeled query and its expected retrieval/generation behavior."""

    id: str
    category: str
    query: str
    expected_failure_type: RetrievalFailure
    expected_relevant_document_ids: tuple[str, ...]
    should_answer: bool
    expected_answer_keywords: tuple[str, ...]


def load_evaluation_set(path: str | Path) -> list[EvaluationExample]:
    """Load and validate the Phase 6 JSON evaluation set."""

    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("The evaluation set must be a non-empty JSON list.")

    examples: list[EvaluationExample] = []
    seen_ids: set[str] = set()
    required_fields = {
        "id",
        "category",
        "query",
        "expected_failure_type",
        "expected_relevant_document_ids",
        "should_answer",
        "expected_answer_keywords",
    }
    for position, record in enumerate(records):
        if not isinstance(record, dict) or required_fields - record.keys():
            raise ValueError(f"Evaluation entry {position} is missing required fields.")
        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation id: {record['id']}")
        if record["category"] not in EVALUATION_CATEGORIES:
            raise ValueError(f"Unknown evaluation category: {record['category']}")
        if not isinstance(record["should_answer"], bool):
            raise ValueError(f"should_answer must be Boolean for {record['id']}.")
        if not isinstance(record["expected_relevant_document_ids"], list):
            raise ValueError(
                f"expected_relevant_document_ids must be a list for {record['id']}."
            )
        if not isinstance(record["expected_answer_keywords"], list):
            raise ValueError(
                f"expected_answer_keywords must be a list for {record['id']}."
            )

        example = EvaluationExample(
            id=str(record["id"]),
            category=str(record["category"]),
            query=str(record["query"]),
            expected_failure_type=RetrievalFailure(record["expected_failure_type"]),
            expected_relevant_document_ids=tuple(
                str(item) for item in record["expected_relevant_document_ids"]
            ),
            should_answer=record["should_answer"],
            expected_answer_keywords=tuple(
                str(item) for item in record["expected_answer_keywords"]
            ),
        )
        if not example.id.strip() or not example.query.strip():
            raise ValueError(f"Evaluation entry {position} has an empty id or query.")
        if not example.should_answer and example.expected_relevant_document_ids:
            raise ValueError(
                f"Abstention example {example.id} must not declare relevant documents."
            )
        seen_ids.add(example.id)
        examples.append(example)
    return examples
