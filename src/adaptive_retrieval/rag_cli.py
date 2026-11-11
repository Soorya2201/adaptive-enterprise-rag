"""Readable presentation output and a persistent local RAG session."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .generation import QwenGenerator
from .interactive import SearchSession
from .paths import DATA_DIR, PROJECT_ROOT
from .pipeline import RagPipeline


def format_result(result: dict, *, show_sources: bool = False) -> str:
    m = result["metrics"]
    lines = [
        f"\n{'ANSWER' if result['status'] == 'answer' else 'REFUSAL'}",
        result["answer"],
        f"\nEvidence check: {result['reason']}",
    ]
    for citation in result["citations"]:
        lines.extend([
            f"[{citation['source']}] {citation['path']}",
            f"  Chunk: {citation['chunk_id']} | chars {citation['start_char']}:{citation['end_char']}",
            f"  Quote: {citation['quote']}",
        ])
    lines.extend([
        "\nLIVE METRICS (startup excluded)",
        f"Query embedding: {m['embedding_ms']:.0f} ms | vector search: {m['vector_search_ms']:.0f} ms",
        f"Retrieval: {m['retrieval_ms']:.0f} ms | prompt: {m['prompt_ms']:.0f} ms",
        f"Generation: {m['generation_ms']:.0f} ms | validation: {m['validation_ms']:.0f} ms",
        f"Total: {m['total_ms'] / 1000:.2f} s",
        f"Prompt tokens: {m['prompt_tokens']} | output tokens: {m['output_tokens']}",
        f"Passages retrieved/sent: {m['retrieved_chunks']}/{m['context_chunks']} | "
        f"retrieval rounds: {m['retrieval_rounds']} | generation calls: {m['generation_calls']}",
    ])
    if m["first_token_ms"] is not None:
        lines.append(
            f"First token after generation starts: {m['first_token_ms']:.0f} ms | "
            f"decode: {m['decode_tokens_per_second']:.1f} tokens/s"
        )
    lines.extend([
        "Tokens include instructions, JSON, citations, and stop tokens.",
        "Local compute profile; dollar cost and energy are not measured.",
    ])
    if show_sources:
        lines.append(format_sources(result))
    return "\n".join(lines)


def format_sources(result: dict) -> str:
    lines = ["\nRETRIEVED PASSAGES (scores are not answer confidence)"]
    for rank, passage in enumerate(result["passages"], 1):
        label = passage["prompt_label"] or "omitted: context budget"
        lines.extend([
            f"\n#{rank} [{label}] cosine={passage['score']:.3f}",
            f"{passage['path'] or passage['source']} | {passage['id']}",
            f"chars {passage['start_char']}:{passage['end_char']}",
            passage["text"],
        ])
    return "\n".join(lines)


def run_rag(args) -> None:
    if not args.corpus.is_file():
        raise FileNotFoundError("RAG requires the source corpus; run 'prepare-subset'")
    started = perf_counter()
    print("Loading index and warming local models...", file=sys.stderr, flush=True)
    pipeline = RagPipeline(
        SearchSession(args.index, args.corpus),
        QwenGenerator(max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens),
    )
    pipeline.warmup()
    pipeline.startup["total_startup_ms"] = (perf_counter() - started) * 1000
    print(
        f"Ready in {pipeline.startup['total_startup_ms'] / 1000:.2f}s. "
        f"Embedding: {pipeline.session.metadata['model']} ({pipeline.session.embedder.device}); "
        f"generator: {pipeline.generator.model_name} (MLX).",
        file=sys.stderr, flush=True,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    trace = args.trace or DATA_DIR / "outputs" / f"rag_{stamp}_{os.getpid()}.jsonl"
    trace = Path(trace)
    trace.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.strip() or None
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.strip())
    last_update = 0.0

    def progress(tokens: int, ms: float) -> None:
        nonlocal last_update
        if sys.stderr.isatty() and ms - last_update >= 250:
            print(f"\rGenerating: {tokens} tokens | {ms / 1000:.1f}s", end="", file=sys.stderr, flush=True)
            last_update = ms

    def ask(question: str) -> dict:
        nonlocal last_update
        last_update = 0
        print("Retrieving passages and generating...", file=sys.stderr, flush=True)
        result = pipeline.ask(question, args.top_k, progress)
        result["git_commit"] = commit
        result["working_tree_dirty"] = dirty
        if sys.stderr.isatty():
            print("\r" + " " * 70 + "\r", end="", file=sys.stderr)
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        if getattr(args, "json", False):
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(format_result(result, show_sources=args.show_sources))
        return result

    if args.command == "ask":
        ask(args.question)
    else:
        print("Ask a question. /sources shows the last passages; /quit exits.")
        latest = None
        try:
            while True:
                question = input("\nQuestion> ").strip()
                if question.lower() in {"/quit", "exit", "quit"}:
                    break
                if question == "/sources":
                    print(format_sources(latest) if latest else "Ask a question first.")
                    continue
                if question == "/help":
                    print("Ask a question; /sources shows evidence; /quit exits.")
                    continue
                if question:
                    try:
                        latest = ask(question)
                    except ValueError as exc:
                        print(f"Question error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
    print(f"Trace saved: {trace}", file=sys.stderr)
