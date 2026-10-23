import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adaptive_retrieval.experiments.backends import (
    DenseIndex,
    LexicalIndex,
    fuse,
    document_ids,
    should_rerank,
)
from adaptive_retrieval.experiments.metrics import summarize


class BackendTests(unittest.TestCase):
    def test_exact_and_hnsw_neighbors(self):
        vectors = np.random.default_rng(42).normal(size=(120, 32)).astype("float32")
        exact = DenseIndex(vectors, "flat")
        query = vectors[17]
        expected = [i for i, _ in exact.search(query, 5)]
        self.assertEqual(
            expected, [i for i, _ in DenseIndex(vectors, "numpy").search(query, 5)]
        )
        self.assertEqual(
            expected, [i for i, _ in DenseIndex(vectors, "python").search(query, 5)]
        )
        self.assertEqual(
            expected,
            [i for i, _ in DenseIndex(vectors, "hnsw", ef_search=120).search(query, 5)],
        )

    def test_pq_training_guard_and_search(self):
        with self.assertRaises(ValueError):
            DenseIndex(np.ones((20, 16)), "ivfpq", nlist=4, nprobe=4, pq_m=4, pq_bits=4)
        # Enough for 16 PQ centroids; test actual trained/compressed search.
        import faiss

        faiss.omp_set_num_threads(2)
        v = np.random.default_rng(4).normal(size=(650, 16)).astype("float32")
        index = DenseIndex(v, "ivfpq", nlist=4, nprobe=4, pq_m=4, pq_bits=4)
        result = index.search(v[5], 10)
        self.assertEqual(len(result), 10)
        self.assertIn(5, [i for i, _ in result])

    def test_bm25s_matches_reference_including_repeated_and_unknown_terms(self):
        texts = [
            "reset password password",
            "reset account",
            "vpn connection",
            "invoice billing",
            "",
        ]
        a, b = LexicalIndex(texts), LexicalIndex(texts, "bm25s")
        for query in ["password", "reset reset", "unknown", "", "account billing"]:
            np.testing.assert_allclose(
                a.scores(query), b.scores(query), rtol=1e-5, atol=1e-6
            )
            self.assertEqual(
                [i for i, _ in a.search(query, 5)], [i for i, _ in b.search(query, 5)]
            )

    def test_fusion_dedup_and_gate(self):
        ranking = fuse([(0, 9), (1, 8)], [(1, 4), (2, 3)])
        self.assertEqual(ranking[0][0], 1)
        chunks = [{"document_id": "a"}, {"document_id": "a"}, {"document_id": "b"}]
        self.assertEqual(document_ids(ranking, chunks, 5), ["a", "b"])
        self.assertTrue(should_rerank([(0, 0.8), (1, 0.79)], 0.05))
        self.assertFalse(should_rerank([(0, 0.8), (1, 0.5)], 0.05))

    def test_bootstrap_uses_queries_not_repetitions(self):
        rows = []
        for repeat in range(3):
            for method, recall in [("flat", 0.0), ("hnsw", 1.0)]:
                rows.append(
                    {
                        "question_id": "q1",
                        "repeat": repeat,
                        "method": method,
                        "total_ms": 1.0,
                        "recall@5": recall,
                        "mrr@5": recall,
                        "ndcg@5": recall,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "truncated_generations": 0,
                        "gate_fired": False,
                    }
                )
        result = summarize(rows, {"variants": ["flat", "hnsw"], "top_k": 5, "seed": 42})
        self.assertEqual(result["hnsw"]["paired_recall"]["query_count"], 1)
        self.assertEqual(result["hnsw"]["paired_recall"]["mean_delta"], 1.0)
