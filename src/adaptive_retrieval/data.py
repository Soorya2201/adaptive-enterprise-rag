"""Parse the evaluation corpus into the project's Document contract."""

import json
import re
from pathlib import Path

from .models import Document
from .paths import PROCESSED_DIR


DEFAULT_CORPUS = PROCESSED_DIR / "subset_corpus.jsonl"


# Some benchmark files store their body as an escaped string: line breaks are the
# two characters backslash + n and quotes are backslash + quote.
_ESCAPE = re.compile(r'''\\([\\nrt"'])''')
_UNESCAPED = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "'": "'", "\\": "\\"}


# A partly escaped body: a break after sentence punctuation that starts a new prose
# line or "- " item. Code and protocol examples do not have this shape.
_PROSE_BREAK = re.compile(r"(?<=[a-z0-9)][.:;!?])\\n(?=(?:- )?[A-Z])")


def clean_document_text(text: str) -> str:
    """Decode escaped document bodies and escaped prose line breaks.

    A document counts as escaped when it has more literal backslash-n sequences
    than real line breaks. Documents that merely mention escapes (HTTP or SSE
    examples such as a literal backslash-n) keep them.
    """
    if text.count("\\n") <= text.count("\n"):
        # Several such breaks mean escaped prose; a single one may be a mention.
        if len(_PROSE_BREAK.findall(text)) >= 3:
            return _PROSE_BREAK.sub("\n", text)
        return text
    return _ESCAPE.sub(lambda match: _UNESCAPED[match.group(1)], text)


def load_documents(corpus_path: Path = DEFAULT_CORPUS) -> list[Document]:
    """Load one JSON object per line, preserving benchmark IDs and source paths."""
    corpus_path = Path(corpus_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. Run 'python src/runner.py build-subset' first."
        )

    documents: list[Document] = []
    seen: set[str] = set()
    with corpus_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                doc_id = record["doc_id"]
                text = record["text"]
                source = record["source"]
                relative_path = record.get("path")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid corpus record at {corpus_path}:{line_number}") from exc
            if not all(isinstance(value, str) and value for value in (doc_id, text, source)):
                raise ValueError(f"Empty or non-text field at {corpus_path}:{line_number}")
            if relative_path is not None and not isinstance(relative_path, str):
                raise ValueError(f"Invalid path at {corpus_path}:{line_number}")
            if doc_id in seen:
                raise ValueError(f"Duplicate document ID {doc_id!r} at {corpus_path}:{line_number}")
            seen.add(doc_id)
            documents.append(Document(doc_id, text, source, relative_path))
    if not documents:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    return documents
