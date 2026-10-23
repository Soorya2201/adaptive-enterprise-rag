import copy
import math
import unittest
from adaptive_retrieval.experiments.metrics import quality
from scripts.analyze_optimizations import complete_passes, paired_comparison


class MetricTests(unittest.TestCase):
    def test_metrics_and_unanswerable_questions(self):
        scores = quality(["wrong", "a", "b"], {"a", "b"}, 2)
        self.assertEqual(scores["recall@2"], 0.5)
        self.assertEqual(scores["mrr@2"], 0.5)
        self.assertAlmostEqual(
            scores["ndcg@2"], (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        )
        self.assertIsNone(quality(["a"], set(), 2)["recall@2"])
        with self.assertRaises(ValueError):
            quality(["a", "a"], {"a"}, 2)

    def records(self):
        return [
            {"question_id": question, "method": method, "repeat": repeat}
            for repeat in [0, 1]
            for question in ["a", "b"]
            for method in ["flat", "hybrid"]
        ]

    def test_complete_passes_exclude_partial_pass(self):
        records = self.records()[:-1]
        rows, repeats = complete_passes(records, ["a", "b"], ["flat", "hybrid"])
        self.assertEqual(repeats, [0])
        self.assertEqual(len(rows), 4)

    def test_failed_duplicate_and_incomplete_runs_are_rejected(self):
        records = self.records()
        failed = copy.deepcopy(records)
        failed[0]["error"] = "Generation timed out"
        for invalid in [failed, records + [records[0]], records[:1]]:
            with self.assertRaises(ValueError):
                complete_passes(invalid, ["a", "b"], ["flat", "hybrid"])

    def test_paired_statistics_count_questions_not_repeats(self):
        left = [
            {"question_id": "a", "repeat": repeat, "recall@5": 1.0}
            for repeat in range(3)
        ]
        right = [
            {"question_id": "a", "repeat": repeat, "recall@5": 0.0}
            for repeat in range(3)
        ]
        result = paired_comparison(left, right, "recall@5")
        self.assertEqual(result["paired_queries"], 1)
        self.assertEqual(result["mean_delta"], 1.0)
        self.assertEqual(result["bootstrap_95_interval"], [1.0, 1.0])
