"""Transparent answer, safety, NLI, cache, and latency utilities for Chunk 3."""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Sequence


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")
WORD = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

SAFE_REFUSAL_PHRASES = (
    "insufficient information",
    "insufficient evidence",
    "not enough information",
    "do not have enough information",
    "cannot determine from the provided context",
    "can't determine from the provided context",
    "cannot determine from provided context",
    "context does not contain the answer",
    "provided context does not contain the answer",
    "cannot answer from the provided context",
    "cannot answer based on the provided context",
)


def normalize_squad_answer(text: str) -> str:
    """Apply the official SQuAD lowercase/punctuation/article/space normalization."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = ARTICLES.sub(" ", without_punctuation)
    return WHITESPACE.sub(" ", without_articles).strip()


def squad_exact_match(prediction: str, references: Sequence[str]) -> float:
    """Maximum normalized exact match across accepted SQuAD references."""

    if not references:
        raise ValueError("Answerable SQuAD scoring requires at least one reference.")
    normalized_prediction = normalize_squad_answer(prediction)
    return float(
        any(normalized_prediction == normalize_squad_answer(item) for item in references)
    )


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_squad_answer(prediction).split()
    reference_tokens = normalize_squad_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_token_f1(prediction: str, references: Sequence[str]) -> float:
    """Maximum official-style token F1 across accepted SQuAD references."""

    if not references:
        raise ValueError("Answerable SQuAD scoring requires at least one reference.")
    return max(_token_f1(prediction, item) for item in references)


@dataclass(frozen=True)
class SafeRefusalDecision:
    is_safe_refusal: bool
    matched_phrase: str | None
    substantive_claim_count: int


def split_meaningful_claims(text: str) -> tuple[str, ...]:
    """Split an answer into auditable sentence-like factual claims."""

    claims: list[str] = []
    for sentence in SENTENCE_BOUNDARY.split(text.strip()):
        candidate = sentence.strip(" \t\r\n-•")
        if len(WORD.findall(candidate)) >= 3:
            claims.append(candidate)
    return tuple(claims)


def detect_safe_refusal(answer: str) -> SafeRefusalDecision:
    """Detect an explicit refusal while rejecting answers that append factual claims."""

    normalized = WHITESPACE.sub(" ", answer.lower()).strip()
    matched = next(
        (phrase for phrase in SAFE_REFUSAL_PHRASES if phrase in normalized), None
    )
    claims = split_meaningful_claims(answer)
    substantive = tuple(
        claim
        for claim in claims
        if not any(phrase in claim.lower() for phrase in SAFE_REFUSAL_PHRASES)
    )
    if matched is not None:
        remainder = normalized.replace(matched, " ")
        remainder_tokens = {
            token
            for token in WORD.findall(remainder)
            if token
            not in {
                "i",
                "am",
                "sorry",
                "there",
                "is",
                "are",
                "available",
                "do",
                "does",
                "not",
                "have",
                "enough",
                "in",
                "from",
                "based",
                "on",
                "contain",
                "determine",
                "cannot",
                "the",
                "a",
                "an",
                "to",
                "answer",
                "question",
                "that",
                "this",
                "provided",
                "context",
                "corpus",
                "evidence",
                "information",
            }
        }
        if remainder_tokens:
            substantive = (*substantive, remainder.strip())
    return SafeRefusalDecision(
        is_safe_refusal=matched is not None and not substantive,
        matched_phrase=matched,
        substantive_claim_count=len(substantive if matched else claims),
    )


def unsupported_answer_flag(
    *, controller_abstained: bool, safe_refusal: bool, substantive_claim_count: int
) -> bool:
    """Conservative controlled-missing definition required by the benchmark."""

    return (
        not controller_abstained
        and not safe_refusal
        and substantive_claim_count > 0
    )


@dataclass(frozen=True)
class NLIConfig:
    model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"
    batch_size: int = 32
    max_length: int = 384
    max_passages: int = 3
    entailment_threshold: float = 0.70
    contradiction_threshold: float = 0.70
    contradiction_label: int = 0
    entailment_label: int = 1
    neutral_label: int = 2
    device: str | None = None


@dataclass(frozen=True)
class NLIClaimResult:
    claim: str
    label: str
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float
    best_passage_id: str | None


@dataclass(frozen=True)
class NLIAnswerResult:
    claim_count: int
    entailed_count: int
    neutral_count: int
    contradicted_count: int
    entailed_percentage: float | None
    neutral_percentage: float | None
    contradicted_percentage: float | None
    fully_grounded: bool | None
    claims: tuple[NLIClaimResult, ...]


def aggregate_nli_claim(
    claim: str,
    passage_ids: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    config: NLIConfig | None = None,
) -> NLIClaimResult:
    """Aggregate passage-level NLI: one strong entailment is sufficient."""

    settings = config or NLIConfig()
    if len(passage_ids) != len(probabilities) or not probabilities:
        raise ValueError("Passage IDs and NLI probabilities must be non-empty and aligned.")
    triples = [tuple(float(value) for value in row) for row in probabilities]
    if any(len(row) < 3 for row in triples):
        raise ValueError("Each NLI probability row must contain three labels.")
    entailments = [row[settings.entailment_label] for row in triples]
    contradictions = [row[settings.contradiction_label] for row in triples]
    neutrals = [row[settings.neutral_label] for row in triples]
    best_entailment = max(entailments)
    best_entailment_index = entailments.index(best_entailment)
    best_contradiction = max(contradictions)
    if best_entailment >= settings.entailment_threshold:
        label = "ENTAILED"
        best_index = best_entailment_index
    elif best_contradiction >= settings.contradiction_threshold:
        label = "CONTRADICTED"
        best_index = contradictions.index(best_contradiction)
    else:
        label = "NEUTRAL"
        best_index = neutrals.index(max(neutrals))
    selected = triples[best_index]
    return NLIClaimResult(
        claim=claim,
        label=label,
        entailment_probability=selected[settings.entailment_label],
        neutral_probability=selected[settings.neutral_label],
        contradiction_probability=selected[settings.contradiction_label],
        best_passage_id=passage_ids[best_index],
    )


def aggregate_nli_answer(claims: Sequence[NLIClaimResult]) -> NLIAnswerResult:
    """Keep entailment, neutral, and contradiction rates separate."""

    total = len(claims)
    counts = Counter(item.label for item in claims)
    return NLIAnswerResult(
        claim_count=total,
        entailed_count=counts["ENTAILED"],
        neutral_count=counts["NEUTRAL"],
        contradicted_count=counts["CONTRADICTED"],
        entailed_percentage=(counts["ENTAILED"] / total if total else None),
        neutral_percentage=(counts["NEUTRAL"] / total if total else None),
        contradicted_percentage=(counts["CONTRADICTED"] / total if total else None),
        fully_grounded=(counts["ENTAILED"] == total if total else None),
        claims=tuple(claims),
    )


def generation_cache_key(
    *,
    system_instruction: str,
    question: str,
    contexts: Sequence[dict[str, str]],
    generation_config: dict[str, Any],
) -> str:
    """Hash every semantic generation input, independent of system label."""

    payload = {
        "system_instruction": system_instruction,
        "question": question,
        "contexts": [
            {
                "id": str(item["id"]),
                "title": str(item.get("title", "")),
                "text": str(item["text"]),
            }
            for item in contexts
        ],
        "generation_config": generation_config,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PersistentGenerationCache:
    """Append-only JSONL cache that survives interrupted Colab sessions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid generation cache JSON on line {line_number}."
                        ) from error
                    self.entries[str(entry["key"])] = entry

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def put(self, entry: dict[str, Any]) -> None:
        key = str(entry["key"])
        if key in self.entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
        self.entries[key] = entry


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latency(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    """Return mean/median for every component and p95 for total latency."""

    materialized = list(rows)
    if not materialized:
        return {}
    fields = sorted(set().union(*(row.keys() for row in materialized)))
    output: dict[str, dict[str, float | None]] = {}
    for field in fields:
        values = [float(row[field]) for row in materialized if row.get(field) is not None]
        output[field] = {
            "mean": float(fmean(values)) if values else None,
            "median": float(median(values)) if values else None,
            "p95": percentile(values, 0.95) if field == "total_latency_seconds" else None,
        }
    return output


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
