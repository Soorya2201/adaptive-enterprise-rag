import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.embedding import LocalEmbedder, QUERY_INSTRUCTION
from adaptive_retrieval.generation import (
    GenerationOutput, PreparedPrompt, check_answer, prepare_prompt,
)
from adaptive_retrieval.models import Chunk
from adaptive_retrieval.pipeline import RagPipeline
from adaptive_retrieval.rag_cli import format_result


CHUNK = Chunk(
    "doc::chunk-0", "doc",
    "Create a replacement key, update clients, then revoke the old key.",
    "wiki", "wiki/keys.txt", 0, 65,
)


def output(record, finish="stop"):
    return GenerationOutput(
        json.dumps(record), 160, 45, 20.0, 4.0, 30.0, finish,
    )


def valid_record():
    return {
        "status": "answer",
        "answer": "Create a replacement key, update clients, then revoke the old key. [S1]",
        "citations": [{"source": "S1", "quote": CHUNK.text}],
    }


class CitationTests(unittest.TestCase):
    def test_valid_quote_preserves_source_address(self):
        checked = check_answer(output(valid_record()), {"S1": CHUNK})
        self.assertEqual(checked.status, "answer")
        self.assertEqual(checked.citations[0]["chunk_id"], CHUNK.id)
        self.assertEqual(checked.citations[0]["path"], CHUNK.path)

    def test_fail_closed_for_bad_or_incomplete_evidence(self):
        cases = [
            ({"status": "insufficient_evidence"}, "model_reported_insufficient_evidence"),
            ({**valid_record(), "citations": []}, "missing_or_excess_citations"),
            ({**valid_record(), "citations": [{"source": "S99", "quote": CHUNK.text}]}, "unknown_citation"),
            ({**valid_record(), "citations": [{"source": "S1", "quote": "Revoke the old key before anything else."}]}, "quote_not_in_source"),
            ({**valid_record(), "answer": "First revoke the old key [S9]."}, "citation_markers_do_not_match"),
            (["invalid structure"], "invalid_response_format"),
            ({**valid_record(), "answer": ""}, "empty_answer"),
        ]
        for record, reason in cases:
            with self.subTest(reason=reason):
                checked = check_answer(output(record), {"S1": CHUNK})
                self.assertEqual(checked.status, "refusal")
                self.assertEqual(checked.reason, reason)
        self.assertEqual(
            check_answer(output(valid_record(), "length"), {"S1": CHUNK}).reason,
            "incomplete_generation",
        )

    def test_whitespace_normalization_does_not_allow_changed_words(self):
        record = valid_record()
        record["citations"][0]["quote"] = CHUNK.text.replace(", ", ",\n")
        self.assertEqual(check_answer(output(record), {"S1": CHUNK}).status, "answer")
        record["citations"][0]["quote"] = "Create a replacement key and immediately revoke the old key."
        self.assertEqual(check_answer(output(record), {"S1": CHUNK}).status, "refusal")


class ContextTests(unittest.TestCase):
    def test_budget_keeps_whole_passages_and_skips_overflow(self):
        large = Chunk("large", "large", "irrelevant " * 2000, "wiki", None, 0, 22000)
        prompt = prepare_prompt(
            "How do I rotate a key?", [(large, .9), (CHUNK, .8), (CHUNK, .7)],
            render=lambda system, user: system + "\n" + user,
            count_tokens=lambda text: len(text.split()), max_prompt_tokens=500,
        )
        self.assertEqual(prompt.sources, {"S1": CHUNK})
        self.assertLessEqual(prompt.prompt_tokens, 500)
        self.assertIn(CHUNK.text, prompt.text)
        self.assertNotIn("irrelevant", prompt.text)

    def test_query_instruction_is_not_added_to_documents(self):
        encoder = LocalEmbedder()
        seen = []
        encoder.embed_texts = lambda texts: seen.extend(texts) or [[1., 0.]]
        encoder.embed_query("rotate a key")
        encoder.embed_texts(["document content"])
        self.assertEqual(seen, [QUERY_INSTRUCTION + "rotate a key", "document content"])


class FakeGenerator:
    model_name = "test-generator"
    revision = "test-revision"
    max_new_tokens = 512
    max_prompt_tokens = 4096

    def __init__(self):
        self.warmups = 0
        self.calls = 0
        self.record = valid_record()
        self.sources = {"S1": CHUNK}

    def warmup(self):
        self.warmups += 1

    def prepare(self, question, passages):
        return PreparedPrompt("test prompt", self.sources, 160)

    def generate(self, prompt, on_progress=None):
        self.calls += 1
        return output(self.record)


class PipelineTests(unittest.TestCase):
    def make_pipeline(self):
        self.warmups = 0

        def warmup():
            self.warmups += 1

        session = SimpleNamespace(
            embedder=SimpleNamespace(warmup=warmup, device="test"),
            search=lambda question, top_k: [(CHUNK, .8)],
            metadata={"model": "test-embedding", "corpus_sha256": "corpus-version"},
            last_timings={"embedding_ms": 0.1, "vector_search_ms": 0.2},
        )
        return RagPipeline(session, FakeGenerator())

    def test_reuse_models_and_measure_both_answer_and_refusal(self):
        pipeline = self.make_pipeline()
        answered = pipeline.ask("How do I rotate a key?")
        pipeline.generator.record = {"status": "insufficient_evidence"}
        refused = pipeline.ask("How do I reset a Gmail password?")
        self.assertEqual(answered["route"], "single_pass")
        self.assertEqual(refused["route"], "refusal")
        self.assertEqual(pipeline.generator.warmups, 1)
        self.assertEqual(self.warmups, 1)
        self.assertEqual(pipeline.generator.calls, 2)
        for result in (answered, refused):
            self.assertEqual(result["metrics"]["output_tokens"], 45)
            self.assertIsNone(result["metrics"]["cost_usd"])
            self.assertGreaterEqual(result["metrics"]["total_ms"], 0)
            self.assertEqual(result["models"]["corpus_sha256"], "corpus-version")
            self.assertIn("startup excluded", format_result(result))

    def test_no_context_skips_generation_and_reports_zero_usage(self):
        pipeline = self.make_pipeline()
        pipeline.generator.sources = {}
        result = pipeline.ask("A question")
        self.assertEqual(result["reason"], "no_usable_context")
        self.assertEqual(result["metrics"]["generation_calls"], 0)
        self.assertEqual(result["metrics"]["output_tokens"], 0)
        self.assertEqual(pipeline.generator.calls, 0)

    def test_bad_input_does_not_load_models(self):
        pipeline = self.make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.ask(" ")
        with self.assertRaises(ValueError):
            pipeline.ask("question", top_k=0)
        self.assertEqual(pipeline.generator.warmups, 0)


if __name__ == "__main__":
    unittest.main()
