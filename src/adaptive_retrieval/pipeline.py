"""Retrieval (one round, or two when the gate expands) then grounded generation."""

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

from .generation import CheckedAnswer, QwenGenerator, check_answer
from .interactive import SearchSession


class RagPipeline:
    def __init__(self, session: SearchSession, generator: QwenGenerator) -> None:
        self.session = session
        self.generator = generator
        self.startup: dict[str, float] = {}
        self.ready = False

    def warmup(self) -> dict[str, float]:
        if self.ready:
            return self.startup
        started = perf_counter()
        self.session.embedder.warmup()
        embedded = perf_counter()
        self.generator.warmup()
        generated = perf_counter()
        expander = getattr(self.session, "expander", None)
        if expander is not None and getattr(self.session, "retriever", "") in {"hybrid_lamer", "adaptive"}:
            expander.warmup()
        self.startup = {
            "embedding_load_and_warmup_ms": (embedded - started) * 1000,
            "generator_load_and_warmup_ms": (generated - embedded) * 1000,
            "expander_load_and_warmup_ms": (perf_counter() - generated) * 1000,
        }
        self.ready = True
        return self.startup

    def ask(
        self, question: str, top_k: int = 5,
        on_progress: Callable[[int, float], None] | None = None,
    ) -> dict:
        if not question.strip():
            raise ValueError("Enter a question")
        if not 1 <= top_k <= 20:
            raise ValueError("top-k must be between 1 and 20")
        # Keep startup out of per-question cost even for callers outside the CLI.
        self.warmup()
        started = perf_counter()
        passages = self.session.search(question, top_k)
        retrieved = perf_counter()
        prompt = self.generator.prepare(question, passages)
        prepared = perf_counter()
        generation = None
        if prompt.sources:
            generation = self.generator.generate(prompt, on_progress)
        generated = perf_counter()
        if generation is None:
            checked = CheckedAnswer(
                "refusal", "No source passages fit the available context budget.",
                (), "no_usable_context",
            )
        else:
            checked = check_answer(generation, prompt.sources)
        finished = perf_counter()
        labels = {chunk.id: label for label, chunk in prompt.sources.items()}
        retrieval = dict(getattr(self.session, "last_trace", {}) or {})
        rounds = retrieval.get("retrieval_rounds", 1)
        if checked.status != "answer":
            route = "refusal"
        else:
            route = "second_pass" if rounds > 1 else "single_pass"
        expansion_in = retrieval.get("expansion_input_tokens", 0)
        expansion_out = retrieval.get("expansion_output_tokens", 0)
        generation_tokens = (
            (generation.prompt_tokens + generation.output_tokens) if generation else 0
        )
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retriever": getattr(self.session, "retriever", "cosine"),
            "score_type": getattr(self.session, "score_type", "cosine"),
            "status": checked.status,
            "route": route,
            "retrieval": retrieval,
            "answer": checked.answer,
            "reason": checked.reason,
            "citations": list(checked.citations),
            "passages": [
                {**asdict(chunk), "score": score, "prompt_label": labels.get(chunk.id)}
                for chunk, score in passages
            ],
            "models": {
                "embedding": self.session.metadata["model"],
                "embedding_revision": self.session.metadata.get("model_revision"),
                "embedding_device": getattr(self.session.embedder, "device", "unknown"),
                "generator": self.generator.model_name,
                "generator_revision": self.generator.revision,
                "generator_runtime": "mlx",
                "corpus_sha256": self.session.metadata["corpus_sha256"],
                "query_instruction": self.session.metadata.get("query_instruction", ""),
            },
            "settings": {
                "top_k": top_k, "max_new_tokens": self.generator.max_new_tokens,
                "max_prompt_tokens": self.generator.max_prompt_tokens,
                "temperature": 0.0, "prompt_version": "midterm-v3",
                "retriever": getattr(self.session, "retriever", "cosine"),
                "expander": (
                    getattr(getattr(self.session, "expander", None), "model_name", None)
                    if getattr(self.session, "retriever", "") in {"hybrid_lamer", "adaptive"} else None
                ),
            },
            "metrics": {
                **self.session.last_timings,
                "retrieval_ms": (retrieved - started) * 1000,
                "prompt_ms": (prepared - retrieved) * 1000,
                "generation_ms": (generated - prepared) * 1000,
                "validation_ms": (finished - generated) * 1000,
                "total_ms": (finished - started) * 1000,
                "first_token_ms": generation.first_token_ms if generation else None,
                "decode_tokens_per_second": (
                    generation.decode_tokens_per_second if generation else None
                ),
                "prompt_tokens": generation.prompt_tokens if generation else 0,
                "output_tokens": generation.output_tokens if generation else 0,
                "retrieved_chunks": len(passages),
                "context_chunks": len(prompt.sources),
                "retrieval_rounds": rounds,
                "expansion_input_tokens": expansion_in,
                "expansion_output_tokens": expansion_out,
                "total_llm_tokens": generation_tokens + expansion_in + expansion_out,
                "generation_calls": int(generation is not None),
                "expansion_calls": len(retrieval.get("expansion_texts", [])),
                "api_calls": 0,
                "cost_usd": None,
            },
            "startup": self.startup,
            # Retain the raw result for diagnosing failed output/citation checks.
            "raw_generation": generation.text if generation else None,
        }
