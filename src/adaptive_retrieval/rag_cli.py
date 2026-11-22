"""Readable presentation output and a persistent local RAG session."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .cache import VerifiedCache
from .expansion import MlxExpander, OllamaExpander
from .generation import QwenGenerator
from .interactive import SearchSession
from .judge import make_judge
from .hybrid_search import EXPANDING_RETRIEVERS, OptimizedSearchSession
from .paths import DATA_DIR, PROJECT_ROOT
from .pipeline import RagPipeline


def open_search_session(args, generator: QwenGenerator | None = None) -> OptimizedSearchSession:
    """Build the selected retriever; only expanding modes create an expander."""
    expander = None
    if args.retriever in EXPANDING_RETRIEVERS:
        if args.expander == "ollama":
            expander = OllamaExpander(args.expansion_model)
        else:
            # Reuse the answer generator so only one MLX model is loaded.
            expander = MlxExpander(generator or QwenGenerator())
    return OptimizedSearchSession(
        SearchSession(args.index, args.corpus), retriever=args.retriever, expander=expander,
    )


def format_result(result: dict, *, show_sources: bool = False) -> str:
    m = result["metrics"]
    lines = [
        f"\n{'ANSWER' if result['status'] == 'answer' else 'REFUSAL'}",
        result["answer"],
        f"\nEvidence check: {result['reason']}"
        + (f" (cache hit, similarity {result['cache']['matched_similarity']:.3f}, "
           f"from: {result['cache']['matched_question']})" if result.get("route") == "cache_hit" else ""),
    ]
    for citation in result["citations"]:
        lines.extend([
            f"[{citation['source']}] {citation['path']}",
            f"  Chunk: {citation['chunk_id']} | chars {citation['start_char']}:{citation['end_char']}",
            f"  Quote: {citation['quote']}",
        ])
    lines.extend([
        "\nLIVE METRICS (startup excluded)",
        f"Retriever: {result.get('retriever', 'cosine')} | route: {result.get('route')} | embedding: {m.get('embedding_ms', 0):.0f} ms | vector search: {m.get('vector_search_ms', 0):.0f} ms",
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
    if "hybrid_search_ms" in m:
        lines.append(
            f"Search incl. worker: {m['hybrid_search_ms']:.0f} ms | lexical: "
            f"{m['lexical_search_ms']:.0f} ms | fusion: {m.get('fusion_ms', 0):.0f} ms"
        )
    retrieval = result.get("retrieval") or {}
    if retrieval.get("retriever") in {"hybrid_lamer", "adaptive"}:
        if retrieval.get("gate_fired"):
            lines.append(
                f"Gate: expanded ({retrieval.get('gate_reason')}) | LameR: "
                f"{m.get('expansion_ms', 0):.0f} ms, {m['expansion_input_tokens']}/"
                f"{m['expansion_output_tokens']} tokens in/out"
            )
        else:
            lines.append("Gate: dense and keyword search agree on the top document; no expansion")
    if m.get("judge_calls"):
        verdict = result.get("cache", {}).get("judgment") or {}
        lines.append(
            f"Cache judge: {verdict.get('verdict')} | {m['judge_ms']:.0f} ms, "
            f"{m['judge_input_tokens']}/{m['judge_output_tokens']} tokens in/out"
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
            f"\n#{rank} [{label}] {result.get('score_type', 'cosine')}={passage['score']:.3f}",
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
    generator = QwenGenerator(max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens)
    session = open_search_session(args, generator)
    cache = None
    if getattr(args, "cache", None):
        cache = VerifiedCache(
            args.cache, threshold=args.cache_threshold,
            judge=make_judge(args.cache_judge, generator, args.expansion_model),
        )
        print(f"Answer cache: {args.cache} ({len(cache)} entries, threshold {args.cache_threshold})",
              file=sys.stderr, flush=True)
    try:
        _run_rag_session(args, RagPipeline(session, generator, cache), started)
    finally:
        session.close()


def run_rag_eval(args) -> None:
    from .rag_eval import format_rag_summary, print_progress, run_rag_evaluation

    if not args.corpus.is_file():
        raise FileNotFoundError("RAG evaluation requires the source corpus; run 'prepare-subset'")
    print("Loading index and warming local models...", file=sys.stderr, flush=True)
    generator = QwenGenerator(max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens)
    session = open_search_session(args, generator)
    try:
        run_dir, summary = run_rag_evaluation(
            RagPipeline(session, generator), args.questions, args.out_dir,
            split=None if args.split == "all" else args.split, limit=args.limit,
            top_k=args.top_k, progress=print_progress,
        )
    finally:
        session.close()
    print(format_rag_summary(summary))
    print(f"\nSaved per-question results and summary: {run_dir}")


def run_cache_eval(args) -> None:
    from .cache_eval import format_cache_summary, print_progress, run_cache_evaluation

    if not args.corpus.is_file():
        raise FileNotFoundError("Cache evaluation requires the source corpus; run 'prepare-subset'")
    print("Loading index and warming local models...", file=sys.stderr, flush=True)
    generator = QwenGenerator(max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens)
    session = open_search_session(args, generator)
    cache = VerifiedCache(threshold=args.cache_threshold, require_exact_quotes=args.require_exact_quotes)
    judges = {name: make_judge(name, generator, args.expansion_model) for name in args.judges}
    try:
        run_dir, summary = run_cache_evaluation(
            RagPipeline(session, generator, cache), judges, args.seeds, args.questions, args.out_dir,
            top_k=args.top_k, progress=print_progress,
        )
    finally:
        session.close()
    print(format_cache_summary(summary))
    print(f"\nSaved warm-up, variant lookups and summary: {run_dir}")


def _run_rag_session(args, pipeline: RagPipeline, started: float) -> None:
    pipeline.warmup()
    if pipeline.cache_discard:
        discard = pipeline.cache_discard
        print(
            f"Cache thrown out ({', '.join(discard['changed'])} changed): {discard['entries']} entries "
            f"moved to {discard['moved_to']}. It will rebuild as questions are answered.",
            file=sys.stderr, flush=True,
        )
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
