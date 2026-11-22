import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.evaluate import run_evaluation
from adaptive_retrieval.hybrid_search import OptimizedSearchSession
from adaptive_retrieval.interactive import SearchSession, run_interactive
from adaptive_retrieval.models import Chunk
from adaptive_retrieval.vector_index import write_index
from runner import create_parser, main


class FakeEmbedder:
    model_name = "test-hybrid"

    def embed_texts(self, texts):
        return [[0.8, 0.2] if "refund" in text else [1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]


class HybridSearchTests(unittest.TestCase):
    def make_session(self, directory):
        index = Path(directory) / "index.jsonl"
        chunks = [
            Chunk("apple-1", "apple-doc", "apple policy", "wiki", "apple.txt", 0, 12),
            Chunk("banana-1", "banana-doc", "banana policy", "wiki", "banana.txt", 0, 13),
            Chunk("apple-2", "apple-doc", "apple refund", "wiki", "apple.txt", 20, 32),
        ]
        write_index(index, chunks, FakeEmbedder(), corpus_sha256="fixture", strategy="fixed", max_words=120, overlap_words=20)
        return SearchSession(index, Path(directory) / "missing.jsonl", embedder=FakeEmbedder())

    def test_default_cli_and_explicit_baseline(self):
        for command in (["search", "apple"], ["interactive"], ["chat"], ["evaluate"]):
            self.assertEqual(create_parser().parse_args(command).retriever, "hybrid")
            self.assertEqual(create_parser().parse_args(command + ["--retriever", "cosine"]).retriever, "cosine")

    def test_cosine_preserves_original_results(self):
        with TemporaryDirectory() as directory:
            base = self.make_session(directory)
            with OptimizedSearchSession(base, "cosine") as session:
                expected = base.search("apple", 3)
                with patch.object(base, "search", side_effect=AssertionError("slow Python search")):
                    actual = session.search("apple", 3)
                # Equal scores may be ordered differently; FAISS uses float32.
                self.assertEqual(actual[0][0].document_id, expected[0][0].document_id)
                self.assertEqual({c.id for c, _ in actual}, {c.id for c, _ in expected})
                for (_, score), (_, reference) in zip(actual, expected):
                    self.assertAlmostEqual(score, reference, places=5)
                self.assertTrue(session.process.is_alive(), "cosine must use the shared FAISS worker")

    def test_unmatched_keywords_do_not_boost_unrelated_documents(self):
        with TemporaryDirectory() as directory:
            base = self.make_session(directory)
            # The fake query vector points to banana, but "kiwi" matches no passage.
            with OptimizedSearchSession(base, "hybrid", candidate_depth=1) as session:
                self.assertEqual(session.search("kiwi", 1)[0][0].document_id, "banana-doc")

    def test_fusion_provenance_reuse_and_cleanup(self):
        with TemporaryDirectory() as directory:
            base = self.make_session(directory)
            with OptimizedSearchSession(base, "hybrid") as session:
                process = session.process
                apple = session.search("apple", 3)
                self.assertEqual(apple[0][0].document_id, "apple-doc")
                self.assertAlmostEqual(apple[0][1], 2 / 61)
                self.assertIs(apple[0][0], base.rows[0][0])
                self.assertEqual(session.search("banana", 1)[0][0].document_id, "banana-doc")
                self.assertIs(session.process, process)
                self.assertTrue(process.is_alive())
                self.assertIn("fusion_ms", session.last_timings)
            self.assertFalse(process.is_alive())
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.search("apple")

    def test_evaluation_uses_same_candidates_and_records_mode(self):
        with TemporaryDirectory() as directory:
            base = self.make_session(directory)
            questions = Path(directory) / "questions.jsonl"
            questions.write_text(json.dumps({"question_id": "q1", "question": "banana", "expected_doc_ids": ["banana-doc"]}) + "\n")
            with OptimizedSearchSession(base, "hybrid", candidate_depth=1) as session:
                self.assertEqual(session.search("banana", 1), session.search("banana", len(base.rows)))
                run, summary = run_evaluation(session, questions, Path(directory) / "results")
                self.assertEqual(summary["hit@5"], 1)
                self.assertEqual(summary["run"]["retriever"], "hybrid")
                self.assertEqual(summary["run"]["candidate_depth"], 1)
                record = json.loads((run / "per_question.jsonl").read_text())
                self.assertEqual(record["retrieved"][0]["best_chunk_id"], "banana-1")

    def test_cli_and_interactive_use_the_selected_score_label(self):
        with TemporaryDirectory() as directory:
            base = self.make_session(directory)
            for mode, cli_label, ui_label in [("hybrid", "rrf=", "RRF:"), ("cosine", "cosine=", "Similarity:")]:
                with self.subTest(retriever=mode):
                    # Patch where the CLI looks the class up, or the real index and models load.
                    with patch("adaptive_retrieval.rag_cli.SearchSession", return_value=base) as opened, \
                            patch("builtins.print") as output:
                        main(["search", "banana", "--retriever", mode])
                        opened.assert_called_once()
                        self.assertTrue(any(cli_label in str(call) for call in output.call_args_list))
                    with OptimizedSearchSession(base, mode) as session:
                        commands = iter(["banana", "/quit"])
                        output = []
                        run_interactive(session, input_fn=lambda _: next(commands), output_fn=output.append)
                        transcript = "\n".join(output)
                        self.assertIn(ui_label, transcript)
                        self.assertNotIn("Similarity:" if mode == "hybrid" else "RRF:", transcript)

    def test_bad_vectors_and_requests_fail_cleanly(self):
        with TemporaryDirectory() as directory:
            base = self.make_session(directory)
            with OptimizedSearchSession(base, "hybrid") as session:
                with self.assertRaises(ValueError):
                    session.search("")
                with self.assertRaises(ValueError):
                    session.search("apple", 0)
                with patch.object(base.embedder, "embed_texts", return_value=[[0, 0]]):
                    with self.assertRaisesRegex(RuntimeError, "nonzero"):
                        session.search("apple")
                self.assertTrue(session.search("banana"))
            base.rows[0] = (base.rows[0][0], [float("nan"), 0])
            with self.assertRaisesRegex(RuntimeError, "finite"):
                OptimizedSearchSession(base, "hybrid")


if __name__ == "__main__":
    unittest.main()
