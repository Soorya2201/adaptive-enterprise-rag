import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
# Instructions for query expansion:
INSTRUCTION = "Instruct: Given an enterprise support question, retrieve passages that contain the facts needed to answer the question.\nQuery: "


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(model, revision, download=False):
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, local_files_only=not download, cache_dir=str(ROOT / "data/processed/optimization-models"),
        allow_patterns=["*.json", "*.safetensors", "*.txt","*.model", "*.jinja", "LICENSE",],)


def load_encoder(path, device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 2048
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/optimization-suite")
    p.add_argument("--ref", default="87661f9d804af78de6d4a97380efbc5d02199109")
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output directory. Don't overwrite prepared data.")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True)
    ref = git("rev-parse", args.ref).strip()
    args.output.mkdir(parents=True)
    # Freeze the team's own stdlib loader/chunker under an isolated import path.
    team = args.output / "team-source/adaptive_retrieval"
    team.mkdir(parents=True)
    for name in ["__init__.py", "models.py","paths.py", "prepare_subset.py", "chunking.py", "data.py",]:
        (team / name).write_text(git("show", f"{ref}:src/adaptive_retrieval/{name}"))
    prefix = "data/subset_documents/"
    for file in git("ls-tree", "-r", "--name-only", ref, prefix).splitlines():
        target = args.output / "subset" / file.removeprefix(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(git("show", f"{ref}:{file}"))
    script = r"""
import sys,json
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,sys.argv[1])
from adaptive_retrieval.prepare_subset import prepare_subset
from adaptive_retrieval.data import load_documents
from adaptive_retrieval.chunking import chunk_documents
out=Path(sys.argv[2])
prepare_subset(out/'subset',out)
chunks=chunk_documents(load_documents(out/'subset_corpus.jsonl'),max_words=120,overlap_words=20)
(out/'chunks.jsonl').write_text(''.join(json.dumps(asdict(c))+'\n' for c in chunks))
"""
    subprocess.run([sys.executable, "-c", script, str(team.parent), str(args.output)], check=True)
    chunks = [json.loads(l) for l in (args.output / "chunks.jsonl").read_text().splitlines()]
    model_path = snapshot(MODEL, REVISION, args.download)
    reranker_path = snapshot(RERANKER, RERANKER_REVISION, args.download)
    started = time.perf_counter()
    encoder = load_encoder(model_path, args.device)
    texts = [c["text"] for c in chunks]
    if any(len(encoder.tokenizer.encode(t)) > 2048 for t in texts):
        raise ValueError("Oversized chunk, you should fit with token-aware chunker")
    vectors = encoder.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    np.save(args.output / "vectors.npy", np.asarray(vectors, dtype="float32"))
    metadata = {
        "team_ref": ref,
        "model": MODEL,
        "revision": REVISION,
        "model_path": model_path,
        "reranker": RERANKER,
        "reranker_revision": RERANKER_REVISION,
        "reranker_path": reranker_path,
        "device": args.device,
        "query_instruction": INSTRUCTION,
        "chunks": len(chunks),
        "max_words": 120,
        "overlap_words": 20,
        "encode_seconds": time.perf_counter() - started,
        "hashes": {
            name: sha(args.output / name)
            for name in ["chunks.jsonl", "vectors.npy", "subset_questions.jsonl", "subset_corpus.jsonl",
            ]
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
