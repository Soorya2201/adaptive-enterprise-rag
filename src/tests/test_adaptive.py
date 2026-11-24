import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.evaluate import load_questions, ranking_metrics, run_evaluation, summarize_records
from adaptive_retrieval.expansion import (
    PINNED_OLLAMA_DIGESTS, Expansion, OllamaExpander, build_prompt, expanded_query,
)
from adaptive_retrieval.hybrid_search import OptimizedSearchSession, fuse
from adaptive_retrieval.interactive import SearchSession
from adaptive_retrieval.models import Chunk
from adaptive_retrieval.pipeline import RagPipeline
from adaptive_retrieval.rag_eval import outcome, run_rag_evaluation, summarize
from adaptive_retrieval.vector_index import write_index
from runner import create_parser
from test_rag import CHUNK, FakeGenerator


class FakeEmbedder:
    model_name = "test-hybrid"

    def embed_texts(self, texts):
        return [[0.8, 0.2] if "refund" in text else [1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]


class FakeExpander:
    model_name = "fake-lamer"

    def __init__(self, texts=("banana policy",)):
        self.texts = texts
        self.calls = []
        self.warmups = 0

    def warmup(self):
        self.warmups += 1

    def expand(self, question, candidates):
        self.calls.append((question, [chunk.id for chunk in candidates]))
        return Expansion(self.texts, 12.0, 100, 20, self.model_name, 0, tuple(c.id for c in candidates))


def make_base(directory):
    index = Path(directory) / "index.jsonl"
    chunks = [
        Chunk("apple-1", "apple-doc", "apple policy", "wiki", "apple.txt", 0, 12),
        Chunk("banana-1", "banana-doc", "banana policy", "wiki", "banana.txt", 0, 13),
        Chunk("apple-2", "apple-doc", "apple refund", "wiki", "apple.txt", 20, 32),
    ]
    write_index(index, chunks, FakeEmbedder(), corpus_sha256="fixture", strategy="paragraph",
                max_words=240, overlap_words=40)
    return SearchSession(index, Path(directory) / "missing.jsonl", embedder=FakeEmbedder())


class FusionAndExpansionTests(unittest.TestCase):
    def test_rrf_sums_reciprocal_ranks_with_deterministic_ties(self):
        fused = fuse([(1, 0.9), (2, 0.5)], [(2, 7.0), (3, 1.0)])
        self.assertEqual([index for index, _ in fused], [2, 1, 3])
        self.assertAlmostEqual(fused[0][1], 1 / 62 + 1 / 61)
        self.assertEqual([i for i, _ in fuse([(5, 1)], [(4, 1)])], [4, 5])

    def test_lamer_prompt_uses_candidates_and_keeps_question_weight(self):
        prompt = build_prompt("Who rotates keys?", [CHUNK])
        self.assertIn("Who rotates keys?", prompt)
        self.assertIn("replacement key", prompt)
        self.assertEqual(expanded_query("q", ["a", "b"]), "q a q b")


class GateTests(unittest.TestCase):
    def test_expanding_modes_require_an_expander(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expander"):
                OptimizedSearchSession(make_base(directory), "adaptive")

    def test_agreement_skips_expansion_and_matches_hybrid(self):
        with TemporaryDirectory() as directory:
            base = make_base(directory)
            expander = FakeExpander()
            with OptimizedSearchSession(base, "hybrid") as hybrid, \
                    OptimizedSearchSession(base, "adaptive", expander=expander) as adaptive:
                self.assertEqual(adaptive.search("apple", 3), hybrid.search("apple", 3))
                self.assertFalse(adaptive.last_trace["gate_fired"])
                self.assertFalse(adaptive.last_trace["top_document_disagreement"])
                self.assertEqual(adaptive.last_trace["retrieval_rounds"], 1)
                self.assertEqual(expander.calls, [])

    def test_disagreement_expands_with_keyword_candidates(self):
        with TemporaryDirectory() as directory:
            base = make_base(directory)
            expander = FakeExpander()
            with OptimizedSearchSession(base, "adaptive", expander=expander,
                                        expansion_candidates=1) as session:
                # Dense favors apple-doc ("refund" vector); BM25 ties to banana-1 first.
                results = session.search("banana refund", 3)
                trace = session.last_trace
                self.assertTrue(trace["top_document_disagreement"])
                self.assertTrue(trace["gate_fired"])
                self.assertEqual(trace["gate_reason"], "top_document_disagreement")
                self.assertEqual(trace["retrieval_rounds"], 2)
                self.assertEqual(trace["expansion_texts"], ["banana policy"])
                self.assertEqual(expander.calls, [("banana refund", ["banana-1"])])
                self.assertIn("expansion_ms", session.last_timings)
                self.assertEqual(len(results), 3)

    def test_always_on_and_bm25_modes(self):
        with TemporaryDirectory() as directory:
            base = make_base(directory)
            expander = FakeExpander()
            with OptimizedSearchSession(base, "hybrid_lamer", expander=expander) as session:
                session.search("apple", 2)
                self.assertEqual(session.last_trace["gate_reason"], "always")
                self.assertEqual(len(expander.calls), 1)
            with OptimizedSearchSession(base, "bm25") as session:
                results = session.search("banana", 3)
                self.assertEqual([chunk.id for chunk, _ in results], ["banana-1"])
                self.assertEqual(session.score_type, "bm25")

    def test_cli_accepts_all_ablation_modes(self):
        for mode in ("cosine", "bm25", "hybrid", "hybrid_lamer", "adaptive"):
            args = create_parser().parse_args(["rag-eval", "--retriever", mode, "--expander", "mlx"])
            self.assertEqual(args.retriever, mode)
            self.assertEqual(args.split, "dev")


class MetricTests(unittest.TestCase):
    def test_binary_relevance_metrics(self):
        metrics = ranking_metrics(["x", "gold", "y"], {"gold"})
        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["hit@5"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@5"], 1 / math.log2(3))
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        missed = ranking_metrics(["a", "b", "c", "d", "e", "gold"], {"gold"})
        self.assertEqual((missed["ndcg@5"], missed["recall@5"]), (0.0, 0.0))
        self.assertAlmostEqual(missed["reciprocal_rank"], 1 / 6)
        self.assertIsNone(ranking_metrics(["a"], set())["ndcg@5"])

    def test_split_and_limit_filtering(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "q.jsonl"
            rows = [
                {"question_id": f"q{i}", "question": "apple", "expected_doc_ids": ["apple-doc"],
                 "split": "dev" if i < 3 else "test"} for i in range(5)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(len(load_questions(path, "dev")), 3)
            self.assertEqual([q["question_id"] for q in load_questions(path, "test", 1)], ["q3"])
            with self.assertRaises(ValueError):
                load_questions(path, "val")

    def test_retrieval_evaluation_reports_gate_and_latency(self):
        with TemporaryDirectory() as directory:
            base = make_base(directory)
            questions = Path(directory) / "q.jsonl"
            questions.write_text("".join(json.dumps(row) + "\n" for row in [
                {"question_id": "q1", "question": "apple", "expected_doc_ids": ["apple-doc"], "split": "dev",
                 "question_type": "basic"},
                {"question_id": "q2", "question": "banana refund", "expected_doc_ids": ["banana-doc"],
                 "split": "dev", "question_type": "semantic"},
            ]))
            with OptimizedSearchSession(base, "adaptive", expander=FakeExpander()) as session:
                run, summary = run_evaluation(session, questions, Path(directory) / "out", split="dev")
            self.assertEqual(summary["gate_rate"], 0.5)
            self.assertEqual(summary["run"]["expander"], "fake-lamer")
            self.assertEqual(set(summary["by_question_type"]), {"basic", "semantic"})
            self.assertIsNotNone(summary["p95_ms"])
            self.assertIn("ndcg@5", summary)
            self.assertTrue(run.name.startswith("retrieval_eval_adaptive_dev_"))


class MidtermProtocolTests(unittest.TestCase):
    def test_repeats_average_quality_per_question_before_across_questions(self):
        rows = [
            {"question_id": "q1", "answerable": True, "ndcg@5": 1.0, "retrieval_ms": 10.0, "gate_fired": True},
            {"question_id": "q1", "answerable": True, "ndcg@5": 0.0, "retrieval_ms": 30.0, "gate_fired": False},
            {"question_id": "q2", "answerable": True, "ndcg@5": 1.0, "retrieval_ms": 20.0, "gate_fired": False},
        ]
        for row in rows:
            row.update({"hit@1": row["ndcg@5"], "hit@5": row["ndcg@5"], "recall@5": row["ndcg@5"],
                        "reciprocal_rank": row["ndcg@5"], "expansion_tokens": 0})
        summary = summarize_records(rows)
        self.assertEqual((summary["questions"], summary["requests"]), (2, 3))
        self.assertEqual(summary["ndcg@5"], 0.75)
        self.assertEqual(summary["mean_ms"], 20.0)
        self.assertAlmostEqual(summary["gate_rate"], 1 / 3)

    def test_evaluation_runs_seeded_shuffled_passes_and_records_hardware(self):
        with TemporaryDirectory() as directory:
            questions = Path(directory) / "q.jsonl"
            questions.write_text("".join(json.dumps({
                "question_id": f"q{i}", "question": "apple", "expected_doc_ids": ["apple-doc"],
                "split": "dev", "question_type": "basic",
            }) + "\n" for i in range(4)))
            orders = []
            for _ in range(2):
                with OptimizedSearchSession(make_base(directory), "hybrid") as session:
                    run, summary = run_evaluation(
                        session, questions, Path(directory) / f"out{len(orders)}", split="dev", repeats=3,
                    )
                records = [json.loads(line) for line in (run / "per_question.jsonl").read_text().splitlines()]
                orders.append([(r["repeat"], r["question_id"]) for r in records])
            self.assertEqual(orders[0], orders[1])
            self.assertEqual(len(orders[0]), 12)
            self.assertEqual(summary["questions"], 4)
            self.assertEqual(summary["run"]["repeats"], 3)
            self.assertIn("platform", summary["run"]["hardware"])

    def test_ollama_expander_refuses_unpinned_weights_and_records_backend(self):
        pinned = PINNED_OLLAMA_DIGESTS["llama3.2:3b"]

        def fake_post(digest):
            def post(endpoint, payload=None):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "llama3.2:3b", "digest": digest,
                                        "details": {"quantization_level": "Q4_K_M"}}]}
                return {"version": "9.9.9"}
            return post

        expander = OllamaExpander()
        expander.expand = lambda question, candidates: None
        expander._post = fake_post("0" * 64)
        with self.assertRaisesRegex(RuntimeError, "not the pinned"):
            expander.warmup()
        expander._post = fake_post(pinned)
        expander.warmup()
        self.assertEqual(expander.backend_info["digest"], pinned)
        self.assertEqual(expander.backend_info["ollama_version"], "9.9.9")
        self.assertEqual(expander.backend_info["options"]["seeds"], [42, 43])


class RagEvaluationTests(unittest.TestCase):
    def test_correct_refusals_are_not_counted_as_false_refusal_reasons(self):
        row = lambda answerable, reason: {
            "answerable": answerable, "status": "refusal", "reason": reason, "route": "refusal",
            "outcome": "false_refusal" if answerable else "correct_refusal", "answer_correct": None,
            "context_has_gold": None, "cites_gold": None, "passages_ndcg@5": None,
            "metrics": {}, "retrieval": {},
        }
        summary = summarize([row(True, "quote_not_in_source"), row(False, "insufficient"), row(False, "insufficient")])
        self.assertEqual(summary["false_refusal_reasons"], {"quote_not_in_source": 1})
        self.assertEqual(summary["correct_refusal_reasons"], {"insufficient": 2})

    def test_outcomes(self):
        self.assertEqual(outcome(True, True), "answered")
        self.assertEqual(outcome(True, False), "false_refusal")
        self.assertEqual(outcome(False, False), "correct_refusal")
        self.assertEqual(outcome(False, True), "unsupported_answer")

    def test_batch_run_scores_refusals_citations_and_routes(self):
        session = SimpleNamespace(
            embedder=SimpleNamespace(warmup=lambda: None, device="test"),
            search=lambda question, top_k: [(CHUNK, .8)],
            metadata={"model": "test-embedding", "corpus_sha256": "corpus-version"},
            last_timings={"embedding_ms": 0.1, "vector_search_ms": 0.2},
            last_trace={"retriever": "adaptive", "gate_fired": True, "retrieval_rounds": 2,
                        "expansion_input_tokens": 100, "expansion_output_tokens": 20,
                        "expansion_texts": ["a", "b"]},
            retriever="adaptive", expander=FakeExpander(), candidate_depth=50, expansion_candidates=5,
        )
        generator = FakeGenerator()
        pipeline = RagPipeline(session, generator)
        with TemporaryDirectory() as directory:
            questions = Path(directory) / "q.jsonl"
            questions.write_text("".join(json.dumps(row) + "\n" for row in [
                {"question_id": "q1", "question": "How do I rotate a key?", "expected_doc_ids": ["doc"],
                 "split": "dev", "gold_answer": "Rotate it."},
                {"question_id": "q2", "question": "Gmail password?", "expected_doc_ids": [], "split": "dev"},
            ]))
            # Progress runs after each question; switch to a refusal for the second one.
            answers = iter([{"status": "insufficient_evidence"}, None])

            def progress(number, total, record):
                following = next(answers)
                if following is not None:
                    generator.record = following

            run, summary = run_rag_evaluation(
                pipeline, questions, Path(directory) / "out", split="dev", limit=None, top_k=5,
                progress=progress,
            )
            records = [json.loads(line) for line in (run / "per_question.jsonl").read_text().splitlines()]
        self.assertEqual([r["outcome"] for r in records], ["answered", "correct_refusal"])
        self.assertEqual(records[0]["route"], "second_pass")
        self.assertTrue(records[0]["cites_gold"])
        self.assertEqual(records[0]["gold_answer"], "Rotate it.")
        self.assertIsNone(records[0]["answer_correct"])
        self.assertEqual(records[0]["metrics"]["retrieval_rounds"], 2)
        self.assertEqual(records[0]["metrics"]["total_llm_tokens"], 160 + 45 + 120)
        self.assertEqual(summary["refusal_accuracy_unanswerable"], 1.0)
        self.assertEqual(summary["false_refusal_rate"], 0.0)
        self.assertEqual(summary["gate_rate"], 1.0)
        self.assertEqual(summary["run"]["expander"], "fake-lamer")
        self.assertEqual(session.expander.warmups, 1)


if __name__ == "__main__":
    unittest.main()
