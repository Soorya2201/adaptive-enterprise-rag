"""Embedding adapter; model downloads and inference happen only when requested."""

from collections.abc import Sequence
from typing import Protocol

from .model_store import EMBEDDING_MODEL, model_snapshot


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_INSTRUCTION = (
    "Instruct: Given an enterprise support question, retrieve passages that "
    "contain the facts needed to answer the question.\nQuery: "
)


class Embedder(Protocol):
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """Qwen documents are encoded plainly; only questions get an instruction."""

    def __init__(
        self, batch_size: int = 8, *, model_name: str = EMBEDDING_MODEL,
        revision: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model_name not in {EMBEDDING_MODEL, MINILM_MODEL}:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        self.model_name = model_name
        self.revision = revision
        self.query_instruction = QUERY_INSTRUCTION if model_name == EMBEDDING_MODEL else ""
        self.batch_size = batch_size
        self._model = None
        self.device = "unloaded"

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings need sentence-transformers. "
                    "Install with: python -m pip install -e '.[local-embeddings]'"
                ) from exc
            import torch

            path = model_snapshot(self.model_name, revision=self.revision)
            self.revision = path.name
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(
                str(path), device=self.device, local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"dtype": torch.float16 if self.device == "mps" else torch.float32},
            )
            # Our passages are small. Bound memory and reject oversized inputs
            # rather than silently using the model's much larger native window.
            if self.model_name == EMBEDDING_MODEL:
                self._model.max_seq_length = 2048
        return self._model

    def warmup(self) -> None:
        self.embed_query("Prepare the enterprise support retrieval model.")

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Enter a question to search")
        return self.embed_texts([self.query_instruction + question])[0]

    @property
    def tokenizer(self):
        return self._load_model().tokenizer

    @property
    def max_input_tokens(self) -> int:
        return self._load_model().max_seq_length

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every embedding input must be non-empty text")
        model = self._load_model()
        max_tokens = model.max_seq_length
        for index, text in enumerate(texts):
            token_count = len(model.tokenizer.encode(text, add_special_tokens=True))
            if token_count > max_tokens:
                raise ValueError(
                    f"Input {index} has {token_count} model tokens, above the {max_tokens} "
                    "token limit. Rebuild with a smaller --max-words value."
                )
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


class MiniLMEmbedder(LocalEmbedder):
    """Compatibility for existing saved indexes, not the midterm default."""

    def __init__(self, batch_size: int = 32) -> None:
        super().__init__(batch_size, model_name=MINILM_MODEL)
