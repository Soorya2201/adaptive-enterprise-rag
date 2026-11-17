"""Small JSONL vector index with exact chunk provenance for this research subset."""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .embedding import Embedder
from .models import Chunk


SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_index(
    path: Path,
    chunks: list[Chunk],
    embedder: Embedder,
    *,
    corpus_sha256: str,
    strategy: str,
    max_words: int,
    overlap_words: int,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Embed chunks and atomically write metadata followed by vector records."""
    if not chunks:
        raise ValueError("No non-empty document chunks to index")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            metadata = {
                "_meta": {
                    "schema_version": SCHEMA_VERSION,
                    "model": embedder.model_name,
                    "model_revision": getattr(embedder, "revision", None),
                    "query_instruction": getattr(embedder, "query_instruction", ""),
                    "normalized": True,
                    "corpus_sha256": corpus_sha256,
                    "strategy": strategy,
                    "max_words": max_words,
                    "overlap_words": overlap_words,
                    "chunk_count": len(chunks),
                }
            }
            stream.write(json.dumps(metadata) + "\n")
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors = embedder.embed_texts([chunk.text for chunk in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Embedder returned the wrong number of vectors")
                for chunk, vector in zip(batch, vectors):
                    if not vector or not all(math.isfinite(value) for value in vector):
                        raise ValueError(f"Invalid vector for {chunk.id}")
                    stream.write(json.dumps({"chunk": asdict(chunk), "vector": vector}) + "\n")
                if on_progress:
                    on_progress(min(start + batch_size, len(chunks)), len(chunks))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_index(path: Path) -> tuple[dict, list[tuple[Chunk, list[float]]]]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
        if not first_line:
            raise ValueError(f"Index is empty: {path}")
        metadata = json.loads(first_line).get("_meta")
        if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported index format: {path}")
        rows = []
        dimension = None
        for line_number, line in enumerate(stream, 2):
            record = json.loads(line)
            chunk = Chunk(**record["chunk"])
            vector = record["vector"]
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Invalid vector at {path}:{line_number}")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(f"Mixed vector dimensions at {path}:{line_number}")
            rows.append((chunk, vector))
    if len(rows) != metadata.get("chunk_count"):
        raise ValueError(f"Index row count does not match metadata: {path}")
    return metadata, rows


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def search_index(
    rows: list[tuple[Chunk, list[float]]], query_vector: list[float], top_k: int = 5
) -> list[tuple[Chunk, float]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = ((chunk, cosine_similarity(query_vector, vector)) for chunk, vector in rows)
    return sorted(ranked, key=lambda item: (-item[1], item[0].id))[:top_k]
