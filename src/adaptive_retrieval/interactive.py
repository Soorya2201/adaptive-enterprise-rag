"""Persistent retrieval session and an interactive terminal interface."""

import shutil
import textwrap
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from .data import load_documents
from .embedding import Embedder, LocalEmbedder
from .model_store import EMBEDDING_MODEL
from .models import Chunk, Document
from .vector_index import file_sha256, load_index, search_index


class SearchSession:
    """Keep the corpus, vector index, and encoder available across questions."""

    def __init__(
        self, index_path: Path, corpus_path: Path, *, embedder: Embedder | None = None
    ) -> None:
        index_path, corpus_path = Path(index_path), Path(corpus_path)
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Index not found: {index_path}. Run 'chunk' and 'embed' before searching."
            )
        self.metadata, self.rows = load_index(index_path)
        self.embedder = embedder if embedder is not None else LocalEmbedder(
            model_name=self.metadata["model"], revision=self.metadata.get("model_revision"),
        )
        if self.metadata["model"] != self.embedder.model_name:
            raise ValueError(f"Index uses a different embedding model: {self.metadata['model']}")
        if self.metadata["model"] == EMBEDDING_MODEL:
            if self.metadata.get("query_instruction") != self.embedder.query_instruction:
                raise ValueError("Index uses a different query instruction; rebuild with 'embed'")
        if not self.rows:
            raise ValueError("The index has no passages. Run 'chunk' and 'embed' again.")
        self.documents: dict[str, Document] = {}
        if corpus_path.is_file():
            if file_sha256(corpus_path) != self.metadata["corpus_sha256"]:
                raise ValueError("The corpus changed since indexing; run 'chunk' and 'embed' again")
            self.documents = {document.id: document for document in load_documents(corpus_path)}
        self.last_timings: dict[str, float] = {}

    def search(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        question = question.strip()
        if not question:
            raise ValueError("Enter a question to search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = perf_counter()
        if hasattr(self.embedder, "embed_query"):
            query_vector = self.embedder.embed_query(question)
        else:
            query_vector = self.embedder.embed_texts([question])[0]
        encoded = perf_counter()
        results = search_index(self.rows, query_vector, top_k)
        self.last_timings = {
            "embedding_ms": (encoded - started) * 1000,
            "vector_search_ms": (perf_counter() - encoded) * 1000,
        }
        return results


def _print_text(text: str, output: Callable[[str], None], width: int) -> None:
    for line in text.splitlines():
        output(textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False))


HELP = (
    "Type a question to search the documents.\n"
    "/doc N   Read the full source document for result N from the latest search\n"
    "/top N   Set the number of passages to display (1-20)\n"
    "/help    Show these commands\n"
    "/quit    Exit (also: exit, quit, Ctrl-D, or Ctrl-C)"
)


def run_interactive(
    session: SearchSession,
    top_k: int = 3,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> None:
    """Ask repeatedly and display verbatim passages with their source identity."""
    if not 1 <= top_k <= 20:
        raise ValueError("Interactive --top-k must be between 1 and 20")
    read = input_fn if input_fn is not None else input
    output = output_fn if output_fn is not None else print
    width = min(110, max(40, shutil.get_terminal_size().columns))
    document_count = len({chunk.document_id for chunk, _ in session.rows})
    output(f"Enterprise retrieval | {document_count} documents | {len(session.rows)} indexed passages")
    output("Passages are source text; relevance scores do not verify their claims.")
    output(f"{session.metadata['model']} loads on your first question and is reused.")
    output(HELP)
    latest: list[tuple[Chunk, float]] = []

    try:
        while True:
            text = read("\nQuestion> ").strip()
            if not text:
                continue
            if text.lower() in {"/quit", "/exit", "quit", "exit"}:
                break
            if text.lower() == "/help":
                output(HELP)
                continue
            if text.startswith("/"):
                parts = text.split()
                command = parts[0].lower()
                if command not in {"/doc", "/top"}:
                    output("Unknown command. Type /help for available commands.")
                    continue
                if len(parts) != 2 or not parts[1].isdigit():
                    output(f"Usage: {command} N (for example, {command} 1)")
                    continue
                number = int(parts[1])
                if command == "/top":
                    if not 1 <= number <= 20:
                        output("Choose a passage count between 1 and 20.")
                    else:
                        top_k = number
                        output(f"Showing up to {top_k} passages per question.")
                    continue
                if not latest:
                    output("Ask a question first, then use /doc N to inspect a result.")
                    continue
                if not 1 <= number <= len(latest):
                    output(f"Choose a result between 1 and {len(latest)}.")
                    continue
                chunk, _ = latest[number - 1]
                document = session.documents.get(chunk.document_id)
                if document is None:
                    output("Full document unavailable. Start with --corpus pointing to its original corpus.")
                    continue
                output("-" * width)
                output(f"Full document: {document.id}")
                output(f"Source: {document.path or document.source}")
                _print_text(document.text, output, width)
                output("-" * width)
                continue

            started = perf_counter()
            try:
                results = session.search(text, top_k)
            except ValueError as exc:
                output(f"Search error: {exc}")
                continue
            latest = results
            output(f"\nRetrieved {len(results)} passages in {perf_counter() - started:.2f}s")
            for number, (chunk, score) in enumerate(results, 1):
                output("-" * width)
                output(f"[{number}] Similarity: {score:.3f} | Document: {chunk.document_id}")
                output(f"Source: {chunk.path or chunk.source}")
                output(f"Chunk: {chunk.id} | Characters: {chunk.start_char}:{chunk.end_char}")
                output("")
                _print_text(chunk.text, output, width)
            output("\nUse /doc N to read a full source, or type another question.")
    except (EOFError, KeyboardInterrupt):
        output("")
    output("Session ended.")
