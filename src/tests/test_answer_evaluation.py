import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_retrieval.answer_evaluation import main, paired_comparison, report, score_answers


class AnswerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"question_id": "a", "question": "Who?", "expected_doc_ids": ["d"], "split": "test"},
            {"question_id": "b", "question": "Missing?", "expected_doc_ids": [], "split": "dev"},
        ]
        self.traces = [
            {"question": "Missing?", "status": "refusal", "answer": "No evidence"},
            {"question": "Who?", "status": "answer", "answer": "Admin [S1]"},
        ]
        self.judgments = [{"question_id": "a", "answer": "Admin [S1]", "judge": "human:reviewer", "correct": True, "nli": "entailment"}]

    def test_reordered_traces_join_and_refusal_denominators(self):
        result = report(score_answers(self.questions, self.traces, self.judgments))
        self.assertEqual(result["overall"]["answer_correctness"], 1)
        self.assertEqual(result["overall"]["refusal_accuracy"], 1)
        self.assertEqual(result["overall"]["unanswerable_refusal_recall"], 1)
        self.assertEqual(result["overall"]["answerable_false_refusal_rate"], 0)
        self.assertEqual(result["by_split"]["dev"]["nli_entailment_rate"], None)

    def test_missing_judgments_are_not_automatic_successes(self):
        rows = score_answers(self.questions, self.traces, [])
        self.assertIsNone(rows[0]["correct"])
        self.assertEqual(report(rows)["overall"]["correctness_coverage"], .5)
        with self.assertRaises(ValueError):
            paired_comparison(rows, rows)

    def test_false_refusal_and_unanswerable_answer_are_incorrect(self):
        traces = [
            {"question_id": "a", "status": "refusal", "answer": "No"},
            {"question_id": "b", "status": "answer", "answer": "Invented"},
        ]
        overall = report(score_answers(self.questions, traces, []))["overall"]
        self.assertEqual(overall["answer_correctness"], 0)
        self.assertEqual(overall["refusal_accuracy"], 0)

    def test_reject_duplicates_missing_unknown_and_stale_judgments(self):
        bad_inputs = [self.traces[:1], self.traces + self.traces[:1],
                      [{"question_id": "unknown", "status": "answer"}] + self.traces]
        for traces in bad_inputs:
            with self.subTest(traces=traces), self.assertRaises(ValueError):
                score_answers(self.questions, traces, [])
        for change in ({"answer": "old output"}, {"correct": "yes"}, {"nli": "supported"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_answers(self.questions, self.traces, [{**self.judgments[0], **change}])

    def test_ambiguous_text_requires_id(self):
        questions = [*self.questions, {**self.questions[0], "question_id": "c"}]
        with self.assertRaises(ValueError):
            score_answers(questions, self.traces, [])

    def test_paired_bootstrap_matches_ids_and_is_reproducible(self):
        rows = score_answers(self.questions, self.traces, self.judgments)
        baseline = [{**row, "correct": False, "refusal_correct": False} for row in reversed(rows)]
        result = paired_comparison(rows, baseline, samples=100)
        self.assertEqual(result["correct"]["delta"], 1)
        self.assertEqual(result["correct"]["ci95"], [1, 1])
        self.assertEqual(result, paired_comparison(rows, baseline, samples=100))
        self.assertEqual(paired_comparison(rows, rows, samples=100)["correct"]["delta"], 0)

    def test_cli_writes_report_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (("questions", self.questions), ("traces", self.traces), ("judgments", self.judgments)):
                (root / name).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            args = ["--questions", str(root / "questions"), "--traces", str(root / "traces"),
                    "--judgments", str(root / "judgments"), "--out", str(root / "report.json")]
            main(args)
            self.assertEqual(json.loads((root / "report.json").read_text())["overall"]["questions"], 2)
            with self.assertRaises(SystemExit):
                main(args)


if __name__ == "__main__":
    unittest.main()
