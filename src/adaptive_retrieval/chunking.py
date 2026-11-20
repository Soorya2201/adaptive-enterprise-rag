"""Deterministic document chunking with exact offsets for source citations."""

import re
from collections.abc import Iterable
from dataclasses import asdict
import json
from pathlib import Path

from .models import Chunk, Document


WORD_RE = re.compile(r"\S+")


def _paragraph_boundaries(text: str, token_spans: list[tuple[int, int]]) -> set[int]:
    """Return token indices just after each blank-line-separated paragraph."""
    boundaries: set[int] = set()
    cursor = 0
    paragraph_end = 0
    token_index = 0
    for line in text.splitlines(keepends=True):
        cursor += len(line)
        if line.strip():
            paragraph_end = cursor
        elif paragraph_end:
            while token_index < len(token_spans) and token_spans[token_index][1] <= paragraph_end:
                token_index += 1
            boundaries.add(token_index)
            paragraph_end = 0
    boundaries.add(len(token_spans))
    return boundaries


def chunk_document(
    document: Document,
    *,
    strategy: str = "paragraph",
    max_words: int = 240,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Split text while keeping original character offsets and document identity.

    Paragraph mode prefers a blank-line boundary before the size cap. Fixed mode
    always cuts at the cap. Both use word overlap when the previous chunk is
    longer than the requested overlap.
    """
    if strategy not in {"paragraph", "fixed"}:
        raise ValueError("strategy must be 'paragraph' or 'fixed'")
    if max_words < 1 or not 0 <= overlap_words < max_words:
        raise ValueError("max_words must be positive and overlap_words must be smaller")

    token_spans = [(match.start(), match.end()) for match in WORD_RE.finditer(document.text)]
    if not token_spans:
        return []
    boundaries = _paragraph_boundaries(document.text, token_spans) if strategy == "paragraph" else set()
    chunks: list[Chunk] = []
    start = 0
    while start < len(token_spans):
        limit = min(start + max_words, len(token_spans))
        minimum = min(start + max(1, max_words // 2), len(token_spans))
        candidates = [boundary for boundary in boundaries if minimum <= boundary <= limit]
        end = max(candidates) if candidates else limit
        start_char, end_char = token_spans[start][0], token_spans[end - 1][1]
        chunks.append(
            Chunk(
                id=f"{document.id}::chunk-{len(chunks):04d}",
                document_id=document.id,
                text=document.text[start_char:end_char],
                source=document.source,
                path=document.path,
                start_char=start_char,
                end_char=end_char,
            )
        )
        if end == len(token_spans):
            break
        start = end - overlap_words if end - start > overlap_words else end
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    strategy: str = "paragraph",
    max_words: int = 240,
    overlap_words: int = 40,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            strategy=strategy,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    ]


def write_chunks(path: Path, chunks: list[Chunk], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"_meta": {**metadata, "chunk_count": len(chunks)}}) + "\n")
        for chunk in chunks:
            stream.write(json.dumps(asdict(chunk)) + "\n")


def load_chunks(path: Path) -> tuple[dict, list[Chunk]]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
        if not first_line:
            raise ValueError(f"Chunk file is empty: {path}")
        metadata = json.loads(first_line).get("_meta")
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing chunk metadata: {path}")
        chunks = [Chunk(**json.loads(line)) for line in stream if line.strip()]
    if len(chunks) != metadata.get("chunk_count"):
        raise ValueError(f"Chunk count does not match metadata: {path}")
    return metadata, chunks


def fit_chunks_to_token_limit(
    chunks: Iterable[Chunk], tokenizer, max_tokens: int, overlap_tokens: int = 24
) -> list[Chunk]:
    """Split oversized chunks by model-token offsets without losing provenance.

    This runs after the model is selected. It avoids silently truncating unusual
    text such as long identifiers, URLs, and code that use many subword tokens.
    """
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = max_tokens - special_tokens
    if content_limit < 1 or not 0 <= overlap_tokens < content_limit:
        raise ValueError("Invalid model token limit or overlap")

    fitted: list[Chunk] = []
    for chunk in chunks:
        offsets = tokenizer(
            chunk.text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )["offset_mapping"]
        if len(offsets) <= content_limit:
            fitted.append(chunk)
            continue

        start = 0
        part = 0
        while start < len(offsets):
            end = min(start + content_limit, len(offsets))
            local_start = offsets[start][0]
            local_end = offsets[end - 1][1]
            text = chunk.text[local_start:local_end]
            # Re-tokenizing a substring can change a boundary subword token.
            while len(tokenizer.encode(text, add_special_tokens=True)) > max_tokens:
                end -= 1
                if end <= start:
                    raise ValueError(f"Cannot fit a model token from {chunk.id}")
                local_end = offsets[end - 1][1]
                text = chunk.text[local_start:local_end]
            fitted.append(Chunk(
                id=f"{chunk.id}::part-{part:02d}",
                document_id=chunk.document_id,
                text=text,
                source=chunk.source,
                path=chunk.path,
                start_char=chunk.start_char + local_start,
                end_char=chunk.start_char + local_end,
            ))
            if end == len(offsets):
                break
            start = end - overlap_tokens if end - start > overlap_tokens else end
            part += 1
    return fitted
