"""Local batched NLI groundedness evaluation for generated answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .generation_metrics import (
    NLIAnswerResult,
    NLIClaimResult,
    NLIConfig,
    aggregate_nli_answer,
    aggregate_nli_claim,
    split_meaningful_claims,
)


@dataclass(frozen=True)
class NLIRequest:
    answer: str
    passages: tuple[dict[str, str], ...]


class LocalNLIGroundednessEvaluator:
    """Evaluate claims against passages with a free Hugging Face NLI model."""

    def __init__(self, config: NLIConfig | None = None) -> None:
        self.config = config or NLIConfig()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Install requirements.txt before creating the NLI evaluator."
            ) from error
        self.torch = torch
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name
        ).to(self.device)
        self.model.eval()
        named_labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
            if not str(label).lower().startswith("label_")
        }
        expected = {
            self.config.contradiction_label: "contradiction",
            self.config.entailment_label: "entailment",
            self.config.neutral_label: "neutral",
        }
        for index, label in expected.items():
            if index in named_labels and named_labels[index] != label:
                raise ValueError(
                    "Configured NLI label mapping disagrees with the model: "
                    f"index {index} is {named_labels[index]!r}, expected {label!r}."
                )

    def _predict_probabilities(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[list[float]]:
        probabilities: list[list[float]] = []
        torch = self.torch
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _hypothesis in batch],
                [hypothesis for _premise, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                batch_probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            probabilities.extend(batch_probabilities)
        return probabilities

    def evaluate_many(
        self, requests: Sequence[NLIRequest]
    ) -> list[NLIAnswerResult]:
        """Batch all claim/passage pairs and transparently reconstruct answers."""

        claim_metadata: list[tuple[int, str, tuple[str, ...], int, int]] = []
        pairs: list[tuple[str, str]] = []
        claims_by_request: list[tuple[str, ...]] = []
        for request_index, request in enumerate(requests):
            claims = split_meaningful_claims(request.answer)
            claims_by_request.append(claims)
            passages = request.passages[: self.config.max_passages]
            passage_ids = tuple(str(item["id"]) for item in passages)
            for claim in claims:
                start = len(pairs)
                pairs.extend((str(item["text"]), claim) for item in passages)
                claim_metadata.append(
                    (request_index, claim, passage_ids, start, len(pairs))
                )

        probabilities = self._predict_probabilities(pairs) if pairs else []
        if len(probabilities) != len(pairs):
            raise RuntimeError("The NLI model returned an unexpected number of rows.")
        by_request: list[list[NLIClaimResult]] = [[] for _ in requests]
        for request_index, claim, passage_ids, start, end in claim_metadata:
            if not passage_ids:
                by_request[request_index].append(
                    NLIClaimResult(
                        claim=claim,
                        label="NEUTRAL",
                        entailment_probability=0.0,
                        neutral_probability=1.0,
                        contradiction_probability=0.0,
                        best_passage_id=None,
                    )
                )
                continue
            by_request[request_index].append(
                aggregate_nli_claim(
                    claim,
                    passage_ids,
                    probabilities[start:end],
                    self.config,
                )
            )
        return [aggregate_nli_answer(items) for items in by_request]
