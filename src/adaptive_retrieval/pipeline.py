"""Retrieval (one round, or two when the gate expands) then grounded generation."""

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from time import perf_counter

from .cache import CacheDecision, VerifiedCache
from .generation import (
    CHECKER_VERSION, PROMPT_VERSION, SYSTEM_PROMPT, CheckedAnswer, QwenGenerator, check_answer,
)
from .interactive import SearchSession


class RagPipeline:
    def __init__(
        self, session: SearchSession, generator: QwenGenerator,
        cache: VerifiedCache | None = None,
    ) -> None:
        self.session = session
        self.generator = generator
        self.cache = cache
        self.startup: dict[str, float] = {}
        self.ready = False
        self._chunks: dict | None = None
        self.cache_discard: dict | None = None

    @property
    def chunks(self) -> dict:
        if self._chunks is None:
            self._chunks = {chunk.id: chunk for chunk, _ in self.session.rows}
        return self._chunks

    def versions(self, top_k: int = 5) -> dict:
        """Everything that must match for a cached answer to be reusable."""
        retriever = getattr(self.session, "retriever", "cosine")
        expanding = retriever in {"hybrid_lamer", "adaptive"}
        expander = getattr(self.session, "expander", None) if expanding else None
        metadata = self.session.metadata
        return {
            # Documents and embeddings are frozen; a change to either discards the cache.
            "corpus_sha256": metadata.get("corpus_sha256"),
            "index_sha256": metadata.get("index_sha256"),
            "embedding": metadata["model"],
            "embedding_revision": metadata.get("model_revision"),
            "query_instruction": metadata.get("query_instruction", ""),
            "generator": self.generator.model_name,
            "generator_revision": self.generator.revision,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "checker_version": CHECKER_VERSION,
            # Experiment arms must never share answers, but do not invalidate each other.
            "scope": {
                "retriever": retriever,
                "candidate_depth": getattr(self.session, "candidate_depth", None),
                "expansion_candidates": (
                    getattr(self.session, "expansion_candidates", None) if expanding else None
                ),
                "expander": getattr(expander, "backend_info", None),
                "top_k": top_k,
                "max_new_tokens": self.generator.max_new_tokens,
                "max_prompt_tokens": self.generator.max_prompt_tokens,
            },
        }

    def lookup_cache(self, question: str, top_k: int = 5) -> tuple[CacheDecision, list[float], dict]:
        """Embed once and check the cache, calling the judge at most once."""
        started = perf_counter()
        vector = self.session.embed_question(question)
        embedded = perf_counter()
        decision = self.cache.lookup(question, vector, self.versions(top_k), self.chunks)
        judgment = decision.judgment
        timings = {
            "embedding_ms": (embedded - started) * 1000,
            "judge_ms": judgment.judge_ms if judgment else 0.0,
            "judge_calls": int(judgment is not None),
            "judge_input_tokens": judgment.input_tokens if judgment else 0,
            "judge_output_tokens": judgment.output_tokens if judgment else 0,
            "cache_lookup_ms": (perf_counter() - started) * 1000,
        }
        return decision, vector, timings

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
        expanded = perf_counter()
        if self.cache is not None:
            # After the generator loads, because its revision is part of the versions.
            self.cache_discard = self.cache.discard_if_stale(self.versions())
            if self.cache.judge is not None:
                self.cache.judge.warmup()
        self.startup = {
            "embedding_load_and_warmup_ms": (embedded - started) * 1000,
            "generator_load_and_warmup_ms": (generated - embedded) * 1000,
            "expander_load_and_warmup_ms": (expanded - generated) * 1000,
            "judge_load_and_warmup_ms": (perf_counter() - expanded) * 1000,
        }
        self.ready = True
        return self.startup

    def ask(
        self, question: str, top_k: int = 5,
        on_progress: Callable[[int, float], None] | None = None,
        *, use_cache: bool = True, admit: bool = True, origin: str | None = None,
    ) -> dict:
        """Answer one question. With a cache: reuse a verified answer when every
        check passes; otherwise answer normally and store the answer if it passes."""
        if not question.strip():
            raise ValueError("Enter a question")
        if not 1 <= top_k <= 20:
            raise ValueError("top-k must be between 1 and 20")
        # Keep startup out of per-question cost even for callers outside the CLI.
        self.warmup()
        started = perf_counter()
        vector, cache_info, lookup = None, None, {}
        embedding_ms = None
        if self.cache is not None and use_cache:
            decision, vector, lookup = self.lookup_cache(question, top_k)
            cache_info = {**decision.summary(), **lookup}
            if decision.hit:
                return self._cached_result(question, top_k, decision, lookup, started)
            embedding_ms = lookup["embedding_ms"]
        elif self.cache is not None and admit:
            vector = self.session.embed_question(question)
            embedding_ms = (perf_counter() - started) * 1000
        search_started = perf_counter()
        if vector is None:
            passages = self.session.search(question, top_k)
        else:
            passages = self.session.search(question, top_k, vector=vector)
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
        judge_tokens = lookup.get("judge_input_tokens", 0) + lookup.get("judge_output_tokens", 0)
        result = {
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
                "temperature": 0.0, "prompt_version": PROMPT_VERSION,
                "checker_version": CHECKER_VERSION,
                "retriever": getattr(self.session, "retriever", "cosine"),
                "expander": (
                    getattr(getattr(self.session, "expander", None), "model_name", None)
                    if getattr(self.session, "retriever", "") in {"hybrid_lamer", "adaptive"} else None
                ),
            },
            "metrics": {
                **self.session.last_timings,
                **lookup,
                **({"embedding_ms": embedding_ms} if embedding_ms is not None else {}),
                # The question embedding belongs to retrieval; the judge does not.
                "retrieval_ms": (retrieved - search_started) * 1000 + (embedding_ms or 0.0),
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
                "total_llm_tokens": generation_tokens + expansion_in + expansion_out + judge_tokens,
                "generation_calls": int(generation is not None),
                "expansion_calls": len(retrieval.get("expansion_texts", [])),
                "api_calls": 0,
                "cost_usd": None,
            },
            "startup": self.startup,
            # Retain the raw result for diagnosing failed output/citation checks.
            "raw_generation": generation.text if generation else None,
        }
        if self.cache is not None:
            admitted, admission = (False, "admission_disabled")
            if admit:
                admitted, admission = self.cache.admit(
                    question, vector, result, self.versions(top_k), self.chunks, origin=origin,
                )
            result["cache"] = {
                **(cache_info or {"hit": False, "reason": "lookup_disabled"}),
                "admitted": admitted, "admission": admission,
            }
        return result

    def _cached_result(self, question, top_k, decision, lookup, started) -> dict:
        entry = decision.entry
        finished = perf_counter()
        # A hit skips retrieval, so show the passages that the stored answer cites.
        cited = {}
        for citation in entry.citations:
            cited.setdefault(citation["chunk_id"], citation["source"])
        similarity = next(c.similarity for c in decision.candidates if c.entry_id == entry.entry_id)
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retriever": getattr(self.session, "retriever", "cosine"),
            "score_type": "question_similarity",
            "status": "answer",
            "route": "cache_hit",
            "retrieval": {"gate_fired": False, "retrieval_rounds": 0},
            "answer": entry.answer,
            "reason": "cached_verified_answer",
            "citations": [dict(citation) for citation in entry.citations],
            "passages": [
                {**asdict(self.chunks[chunk_id]), "score": similarity, "prompt_label": label}
                for chunk_id, label in cited.items()
            ],
            "cache": {
                **decision.summary(), **lookup, "matched_similarity": similarity,
                "admitted": False, "admission": "cache_hit",
            },
            "models": {
                "embedding": self.session.metadata["model"],
                "generator": self.generator.model_name,
                "corpus_sha256": self.session.metadata["corpus_sha256"],
            },
            "settings": {
                "top_k": top_k, "retriever": getattr(self.session, "retriever", "cosine"),
                "prompt_version": PROMPT_VERSION, "checker_version": CHECKER_VERSION,
                "cache_threshold": self.cache.threshold,
            },
            "metrics": {
                **lookup,
                "retrieval_ms": lookup["embedding_ms"],
                "prompt_ms": 0.0, "generation_ms": 0.0, "validation_ms": 0.0,
                "total_ms": (finished - started) * 1000,
                "first_token_ms": None, "decode_tokens_per_second": None,
                "prompt_tokens": 0, "output_tokens": 0,
                "retrieved_chunks": 0, "context_chunks": 0,
                "retrieval_rounds": 0, "expansion_input_tokens": 0, "expansion_output_tokens": 0,
                "total_llm_tokens": lookup["judge_input_tokens"] + lookup["judge_output_tokens"],
                "generation_calls": 0, "expansion_calls": 0,
                "api_calls": 0, "cost_usd": None,
            },
            "startup": self.startup,
            "raw_generation": None,
        }
