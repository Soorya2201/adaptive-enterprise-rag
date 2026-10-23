import unittest
from unittest.mock import patch
from adaptive_retrieval.experiments.expansion import expand, expanded_query


class ExpansionTests(unittest.TestCase):
    @patch("adaptive_retrieval.experiments.expansion.request_json")
    def test_query2doc_four_examples_one_generation(self, request):
        request.return_value = {
            "response": "A pseudo-document",
            "done": True,
            "prompt_eval_count": 100,
            "eval_count": 20,
        }
        texts, calls = expand(
            "target question",
            [],
            [],
            method="query2doc",
            concise=False,
            model="fixture",
        )
        self.assertEqual(request.call_count, 1)
        self.assertEqual(calls[0]["request"]["prompt"].count("Query:"), 5)
        self.assertEqual(texts, ("A pseudo-document",))
        self.assertEqual(
            expanded_query("query", texts, "query2doc"),
            "query query query query query A pseudo-document",
        )

    @patch("adaptive_retrieval.experiments.expansion.request_json")
    def test_lamer_uses_only_candidates_and_distinct_seeds(self, request):
        request.side_effect = [
            {"response": "First answer", "done": True},
            {"response": "Second answer", "done": True},
        ]
        chunks = [{"text": "retrieved evidence"}, {"text": "not retrieved"}]
        texts, calls = expand(
            "question",
            [(0, 1.0)],
            chunks,
            method="lamer",
            concise=True,
            model="fixture",
        )
        self.assertEqual(texts, ("First answer", "Second answer"))
        self.assertIn("retrieved evidence", calls[0]["request"]["prompt"])
        self.assertNotIn("not retrieved", calls[0]["request"]["prompt"])
        self.assertNotEqual(
            calls[0]["request"]["options"]["seed"],
            calls[1]["request"]["options"]["seed"],
        )
        self.assertEqual(
            expanded_query("query", texts, "lamer"),
            "query First answer query Second answer",
        )

    @patch("adaptive_retrieval.experiments.expansion.request_json")
    def test_empty_or_incomplete_generation_fails(self, request):
        for response in [
            {"response": "", "done": True},
            {"response": "partial", "done": False},
        ]:
            request.return_value = response
            with self.assertRaises(RuntimeError):
                expand(
                    "question",
                    [],
                    [],
                    method="query2doc",
                    concise=True,
                    model="fixture",
                )
