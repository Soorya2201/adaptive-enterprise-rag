"""Create an isolated corpus/index pair with one document removed."""

import json
from pathlib import Path

from .data import load_documents
from .vector_index import file_sha256, load_index


def remove_document(
    corpus_path: Path, index_path: Path, document_id: str, out_dir: Path,
) -> tuple[Path, Path]:
    """Reuse existing vectors, removing every passage from the selected document."""
    corpus_path, index_path, out_dir = map(Path, (corpus_path, index_path, out_dir))
    if not document_id.strip():
        raise ValueError("Document ID is required")
    metadata, rows = load_index(index_path)
    if file_sha256(corpus_path) != metadata["corpus_sha256"]:
        raise ValueError("The source corpus and index do not match; rebuild the index first")
    documents = load_documents(corpus_path)
    if document_id not in {document.id for document in documents}:
        raise ValueError(f"Document {document_id} is absent from the corpus")
    removed_passages = sum(chunk.document_id == document_id for chunk, _ in rows)
    if not removed_passages:
        raise ValueError(f"Document {document_id} has no indexed passages")

    target_corpus = out_dir / "corpus.jsonl"
    target_index = out_dir / "index.jsonl"
    if {target_corpus.resolve(), target_index.resolve()} & {
        corpus_path.resolve(), index_path.resolve(),
    }:
        raise ValueError("Ablation output would overwrite the source corpus or index")
    if target_corpus.exists() or target_index.exists():
        raise FileExistsError(f"Ablation files already exist in {out_dir}; choose another --out-dir")

    out_dir.mkdir(parents=True, exist_ok=True)
    with corpus_path.open(encoding="utf-8") as source, target_corpus.open(
        "w", encoding="utf-8"
    ) as stream:
        for line in source:
            if line.strip() and json.loads(line)["doc_id"] != document_id:
                # Preserve the source row and its evaluation metadata exactly.
                stream.write(line)

    new_metadata = dict(metadata)
    new_metadata["corpus_sha256"] = file_sha256(target_corpus)
    new_metadata["chunk_count"] = len(rows) - removed_passages
    new_metadata["ablation_removed_document_id"] = document_id
    with target_index.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"_meta": new_metadata}, ensure_ascii=False) + "\n")
        # Copy vector records byte for byte; no model download or re-embedding.
        with index_path.open(encoding="utf-8") as source:
            next(source)
            for line in source:
                if json.loads(line)["chunk"]["document_id"] != document_id:
                    stream.write(line)

    # Loading the pair checks both the row count and the corpus checksum.
    _, filtered_rows = load_index(target_index)
    if any(chunk.document_id == document_id for chunk, _ in filtered_rows):
        raise RuntimeError("Removed document still has indexed passages")
    return target_corpus, target_index
