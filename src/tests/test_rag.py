import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.embedding import LocalEmbedder, QUERY_INSTRUCTION
from adaptive_retrieval.generation import (
    GenerationOutput, PreparedPrompt, check_answer, citation_markers, prepare_prompt, quote_match,
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


LIST_CHUNK = Chunk(
    "runbook::chunk-0", "runbook",
    "Mitigation:\n- Use an idempotency key to avoid double processing.\n"
    "- Retry with **exponential backoff** and jitter.\n"
    "> The `api_key` header is required for every retry.",
    "wiki", "wiki/runbook.md", 0, 160,
)


class TolerantQuoteTests(unittest.TestCase):
    def test_formatting_the_model_drops_still_matches(self):
        cases = {
            "Use an idempotency key to avoid double processing.": "exact",
            "use an idempotency key to avoid double processing": "formatting",
            "Retry with exponential backoff and jitter.": "formatting",
            "The api_key header is required for every retry.": "formatting",
            "\u201cRetry with exponential backoff and jitter.\u201d": "formatting",
            "...Use an idempotency key to avoid double processing...": "formatting",
        }
        for quote, kind in cases.items():
            with self.subTest(quote=quote):
                self.assertEqual(quote_match(quote, LIST_CHUNK.text), kind)

    def test_changed_reordered_or_tiny_segments_do_not_match(self):
        for quote in [
            "Use an idempotency key to avoid triple processing.",
            "Retry with exponential backoff... use an idempotency key",
            "Use an... retry with exponential backoff",
            "Use an idempotency key... retry with exponential backoff",
            "Retry with exponential backoff - jitter",
            "Retry with exponential backoff and jitter. Always.",
        ]:
            with self.subTest(quote=quote):
                self.assertIsNone(quote_match(quote, LIST_CHUNK.text))

    def test_cropping_or_formatting_cannot_change_the_meaning(self):
        cases = [
            ("5 business days for delivery", "We need 15 business days for delivery."),
            ("5 business days for delivery", "We need 1.5 business days for delivery."),
            ("096 bytes is the header limit", "4,096 bytes is the header limit"),
            ("30 minutes is the typical time", "15-30 minutes is the typical time"),
            ("30 minutes is the typical time", "15 to 30 minutes is the typical time"),
            ("30 minutes is the typical time", "15 - 30 minutes is the typical time"),
            ("the typical wait in minutes is 15", "the typical wait in minutes is 15 to 30"),
            ("latency target is 15", "latency target is 150 ms"),
            ("latency target is 15", "latency target is 15 %"),
            ("the budget ceiling is 5", "the budget ceiling is 5 million"),
            ("5% margin change expected", "-5% margin change expected"),
            ("available in us-east region", "The API is unavailable in us-east region."),
            ("available in us-east region", "The API is not available in us-east region."),
            ("available in us-east region", "The API isn't currently available in us-east region."),
            ("production traffic is allowed", "non-production traffic is allowed"),
            ("The service is ... available in us-east", "The service is not currently available in us-east."),
            ("Acme response time ... is 4 hours", "Acme response time is 9 hours. Globex is 4 hours."),
            ("the limit is 23 requests", "the limit is 2*3 requests"),
            ("the capacity is 102 units", "the capacity is 10" + chr(0xB2) + " units"),
            ("set maxtokens to the limit", "set max_tokens to the limit"),
            ("500 ms triggers a page", "Rule:" + chr(10) + "> 500 ms triggers a page"),
            ("2000ms for 10 minutes", "P95 latency > 2000ms for 10 minutes."),
            ("2000ms for 10 minutes", "P95 latency >2000ms for 10 minutes."),
            ("link speed is 10 MB per second", "link speed is 10 Mb per second"),
            ("ask us for the report", "ask US for the report"),
            ("5) Emit detailed telemetry", "3) Emit detailed telemetry"),
            ("045) to ensure the proxy restarts", "(p = 0.045) to ensure the proxy restarts"),
        ]
        for quote, source in cases:
            with self.subTest(quote=quote, source=source):
                self.assertIsNone(quote_match(quote, source))

    def test_legitimate_crops_still_match(self):
        for quote, source in [
            ("eviction_rate > 0.5% for pages", "thresholds (eviction_rate > 0.5% for pages), then page"),
            ("30 seconds is the new timeout", "It moved to 30 seconds is the new timeout"),
            ("available in us-east region", "No downtime, available in us-east region."),
            ("Emit detailed telemetry", "3) Emit **detailed** telemetry"),
            ("3) Emit detailed telemetry", "3) Emit **detailed** telemetry"),
            ("5 replicas per region", "Plan:" + chr(10) + "- 5 replicas per region"),
            ("5 business days for delivery", "15 business days for delivery or 5 business days for delivery."),
        ]:
            with self.subTest(quote=quote):
                self.assertIsNotNone(quote_match(quote, source))

    def test_cropped_digit_cannot_become_a_verified_answer(self):
        chunk = Chunk("c", "d", "We need 15 business days for delivery.", "w", None, 0, 38)
        record = {"status": "answer", "answer": "5 business days. [S1]",
                  "citations": [{"source": "S1", "quote": "5 business days for delivery"}]}
        self.assertEqual(check_answer(output(record), {"S1": chunk}).reason, "quote_not_in_source")

    def test_grouped_markers_are_parsed_and_other_brackets_ignored(self):
        self.assertEqual(
            citation_markers("Retry [S1, S2] and log [S3][S4] before [DONE] or [S5 and 6]."),
            {"S1", "S2", "S3", "S4"},
        )

    def test_markers_must_be_quoted_sources_but_quotes_need_not_be_marked(self):
        sources = {"S1": CHUNK, "S2": LIST_CHUNK}
        record = {
            "status": "answer",
            "answer": "Rotate the key, then retry with backoff. [S1, S2]",
            "citations": [
                {"source": "S1", "quote": CHUNK.text},
                {"source": "S2", "quote": "- Retry with exponential backoff and jitter."},
            ],
        }
        checked = check_answer(output(record), sources)
        self.assertEqual(checked.status, "answer")
        self.assertEqual([c["quote_match"] for c in checked.citations], ["exact", "formatting"])
        record["answer"] = "Rotate the key. [S1]"
        self.assertEqual(check_answer(output(record), sources).status, "answer")
        record["answer"] = "Rotate the key. [S1][S2][S3]"
        self.assertEqual(check_answer(output(record), sources).reason, "citation_markers_do_not_match")
        record["answer"] = "Rotate the key."
        self.assertEqual(check_answer(output(record), sources).reason, "citation_markers_do_not_match")

    def test_quote_must_come_from_the_cited_source(self):
        record = {
            "status": "answer",
            "answer": "Retry with backoff. [S1]",
            "citations": [{"source": "S1", "quote": "Retry with exponential backoff and jitter."}],
        }
        checked = check_answer(output(record), {"S1": CHUNK, "S2": LIST_CHUNK})
        self.assertEqual(checked.reason, "quote_not_in_source")


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
