"""Explicit, reusable baseline RAG components.

The implementation intentionally exposes the core flow:
document text -> embedding -> normalized vector -> FAISS inner-product index.
It has no Colab-specific imports and does not use hosted inference services.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Document:
    """A source passage in the controlled corpus."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document together with its cosine-similarity score."""

    document: Document
    score: float

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def title(self) -> str:
        return self.document.title

    @property
    def text(self) -> str:
        return self.document.text


@dataclass
class RAGConfig:
    """All model and generation settings for the Phase 1 baseline."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    top_k: int = 3
    max_new_tokens: int = 220
    temperature: float = 0.2
    do_sample: bool = False
    embedding_batch_size: int = 32
    device: str | None = None


@dataclass(frozen=True)
class RAGAnswer:
    """Answer plus retrieval metadata retained for later diagnostics."""

    question: str
    answer: str
    retrieved_documents: list[RetrievalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_documents": [
                {"id": item.id, "title": item.title, "text": item.text, "score": item.score}
                for item in self.retrieved_documents
            ],
        }


def load_documents(corpus_path: str | Path) -> list[Document]:
    """Load a JSON list of documents and validate the Phase 1 corpus schema."""

    path = Path(corpus_path)
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list) or not records:
        raise ValueError("The corpus must be a non-empty JSON list.")

    documents: list[Document] = []
    document_ids: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict) or set(("id", "title", "text")) - record.keys():
            raise ValueError(f"Corpus entry {position} must contain id, title, and text.")
        document = Document(
            id=str(record["id"]), title=str(record["title"]), text=str(record["text"])
        )
        if not all((document.id.strip(), document.title.strip(), document.text.strip())):
            raise ValueError(f"Corpus entry {position} contains an empty required field.")
        if document.id in document_ids:
            raise ValueError(f"Duplicate document id: {document.id}")
        document_ids.add(document.id)
        documents.append(document)
    return documents


def _resolve_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class FAISSRetriever:
    """SentenceTransformer embeddings and a transparent FAISS cosine index."""

    def __init__(
        self,
        embedding_model_name: str,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.documents: list[Document] = []
        self.index = None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("Install requirements.txt before creating a FAISSRetriever.") from error
        self.model = SentenceTransformer(self.embedding_model_name, device=self.device)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode and L2-normalize vectors so inner product is cosine similarity."""

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def build(self, documents: Sequence[Document]) -> None:
        """Embed documents and add normalized vectors to an IndexFlatIP index."""

        if not documents:
            raise ValueError("At least one document is required to build an index.")
        try:
            import faiss
        except ImportError as error:
            raise ImportError("Install faiss-cpu before building the retrieval index.") from error

        self.documents = list(documents)
        document_vectors = self._encode([document.text for document in self.documents])
        self.index = faiss.IndexFlatIP(document_vectors.shape[1])
        self.index.add(document_vectors)

    def save(self, directory: str | Path) -> None:
        """Persist FAISS vectors and their position-to-document mapping."""

        if self.index is None:
            raise RuntimeError("Build or load an index before saving it.")
        try:
            import faiss
        except ImportError as error:
            raise ImportError("Install faiss-cpu before saving the retrieval index.") from error

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "documents.faiss"))
        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "documents": [asdict(document) for document in self.documents],
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        """Load a persisted FAISS index and its exact document mapping."""

        try:
            import faiss
        except ImportError as error:
            raise ImportError("Install faiss-cpu before loading the retrieval index.") from error

        index_dir = Path(directory)
        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        saved_model = metadata.get("embedding_model_name")
        if saved_model != self.embedding_model_name:
            raise ValueError(
                "Index embedding model does not match the configured model: "
                f"{saved_model!r} != {self.embedding_model_name!r}."
            )
        self.documents = [Document(**record) for record in metadata["documents"]]
        self.index = faiss.read_index(str(index_dir / "documents.faiss"))
        if self.index.ntotal != len(self.documents):
            raise ValueError("The FAISS index size does not match its document metadata.")

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return top-k documents in descending cosine-similarity order."""

        if self.index is None:
            raise RuntimeError("Build or load an index before calling retrieve.")
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self._encode([query])
        scores, positions = self.index.search(query_vector, min(top_k, len(self.documents)))
        return [
            RetrievalResult(document=self.documents[position], score=float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]


class LocalQwenGenerator:
    """Run an instruct model locally through Hugging Face Transformers."""

    SYSTEM_INSTRUCTION = (
        "You answer questions using only the retrieved context. "
        "Do not invent facts or use outside knowledge. "
        "If the context does not contain enough evidence, say: "
        "'I do not have enough information in the provided context to answer that.'"
    )

    def __init__(self, config: RAGConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("Install requirements.txt before creating LocalQwenGenerator.") from error

        self.config = config
        self.device = _resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.generation_model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(config.generation_model_name, **model_kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def format_context(results: Sequence[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[Source {rank}: {result.title} | id={result.id}]\n{result.text}"
            for rank, result in enumerate(results, start=1)
        )

    def generate(self, question: str, results: Sequence[RetrievalResult]) -> str:
        """Generate one answer grounded exclusively in the supplied retrieval results."""

        context = self.format_context(results)
        user_prompt = f"Retrieved context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
        else:
            encoded = self.tokenizer("\n\n".join(message["content"] for message in messages), return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        input_device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(input_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
        generated = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        new_tokens = generated[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class BaselineRAG:
    """Connect explicit FAISS retrieval to a local grounded-generation model."""

    def __init__(self, retriever: FAISSRetriever, generator: LocalQwenGenerator, config: RAGConfig) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config

    def answer_question(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Retrieve context, generate a grounded answer, and retain all metadata."""

        results = self.retriever.retrieve(
            question, top_k=self.config.top_k if top_k is None else top_k
        )
        answer = self.generator.generate(question, results)
        return RAGAnswer(question=question, answer=answer, retrieved_documents=results)
