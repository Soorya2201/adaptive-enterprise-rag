import unittest
from types import SimpleNamespace
from unittest.mock import patch
from adaptive_retrieval.experiments.combinations import run_combination
from adaptive_retrieval.experiments.backends import fuse


class CombinationTests(unittest.TestCase):
    def suite(self, same_document=False):
        class Index:
            def __init__(self, rankings):
                self.rankings = iter(rankings)

            def search(self, query, depth):
                return next(self.rankings)

        chunks = [
            {"id": "a0", "document_id": "a"},
            {"id": "a1", "document_id": "a" if same_document else "b"},
            {"id": "c0", "document_id": "c"},
        ]
        return SimpleNamespace(
            config={
                "candidate_depth": 3,
                "expansion_candidates": 2,
                "generation_model": "fixture",
            },
            encoder=lambda q: [1.0, 0.0],
            chunks=chunks,
            dense={"flat": Index([[(0, 0.9), (2, 0.7), (1, 0.6)]])},
            lexical={
                "bm25s": Index(
                    [[(1, 3.0), (0, 2.0), (2, 1.0)], [(2, 4.0), (1, 2.0), (0, 1.0)]]
                )
            },
        )

    def test_gate_compares_parent_documents_and_skips_generation(self):
        trace = {}
        with patch("adaptive_retrieval.experiments.combinations.expand") as generate:
            result = run_combination(
                self.suite(True), "query", "hybrid_lamer_gated", trace
            )
        generate.assert_not_called()
        self.assertFalse(trace["gate_fired"])
        self.assertEqual(
            result, fuse([(0, 0.9), (2, 0.7), (1, 0.6)], [(1, 3.0), (0, 2.0), (2, 1.0)])
        )

    def test_disagreement_and_always_expand_have_same_path(self):
        results = []
        for method in ["hybrid_lamer_concise", "hybrid_lamer_gated"]:
            trace = {}
            with patch(
                "adaptive_retrieval.experiments.combinations.expand",
                return_value=(("answer one", "answer two"), []),
            ) as generate:
                results.append(run_combination(self.suite(), "query", method, trace))
            self.assertTrue(trace["gate_fired"])
            self.assertEqual(generate.call_args.args[1], [(1, 3.0), (0, 2.0)])
            self.assertEqual(trace["candidate_chunk_ids"], ["a1", "a0"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(
            results[0],
            fuse([(0, 0.9), (2, 0.7), (1, 0.6)], [(2, 4.0), (1, 2.0), (0, 1.0)]),
        )
