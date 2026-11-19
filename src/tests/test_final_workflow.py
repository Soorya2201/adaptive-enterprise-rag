import copy
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_retrieval.final_benchmark import build_benchmark, select_final, verify_benchmark
from adaptive_retrieval.final_experiment import ask_policy, collect, main as experiment_main
from adaptive_retrieval.answer_review import export_review
from adaptive_retrieval.nli import claims, judge_trace
from adaptive_retrieval.answer_evaluation import score_answers


class FakeJudge:
    threshold = .8

    def assess(self, premise, hypothesis):
        label = "entailment" if hypothesis.strip() in premise else "neutral"
        return {"label": label, "max_entailment": float(label == "entailment"), "max_contradiction": 0, "windows": 1}


def answer(text="Admins rotate keys."):
    return {"question": "Who rotates keys?", "status": "answer", "answer": text + " [S1]",
            "passages": [{"prompt_label": "S1", "text": "Admins rotate keys."},
                         {"prompt_label": None, "text": "Everyone rotates keys."}],
            "citations": [{"source": "S1"}], "reason": "citation_quotes_valid",
            "metrics": {"prompt_tokens": 10, "output_tokens": 5, "generation_calls": 1,
                        "retrieval_rounds": 1, "retrieval_ms": 1, "generation_ms": 2}}


class FakePipeline:
    def __init__(self, results):
        self.results, self.depths = results, []

    def warmup(self):
        pass

    def ask(self, question, top_k):
        self.depths.append(top_k)
        return copy.deepcopy(self.results[len(self.depths) - 1])


class FinalWorkflowTests(unittest.TestCase):
    def test_adaptive_retry_is_bounded_and_cost_includes_both_attempts(self):
        pipeline = FakePipeline([answer("Everyone rotates keys."), answer()])
        result = ask_policy(pipeline, FakeJudge(), "Who rotates keys?", mode="adaptive")
        self.assertEqual(pipeline.depths, [5, 10])
        self.assertEqual(result["route"], "second_pass")
        self.assertEqual(result["metrics"]["prompt_tokens"], 20)
        self.assertEqual(result["metrics"]["retrieval_rounds"], 2)
        self.assertEqual(result["metrics"]["generation_calls"], 2)

    def test_fixed_uses_same_gate_but_does_not_retry(self):
        pipeline = FakePipeline([answer("Everyone rotates keys.")])
        result = ask_policy(pipeline, FakeJudge(), "Who rotates keys?", mode="fixed")
        self.assertEqual(result["status"], "refusal")
        self.assertEqual(result["reason"], "nli_not_entailed")
        self.assertEqual(result["citations"], [])
        self.assertEqual(pipeline.depths, [5])

    def test_success_does_not_trigger_extra_retrieval(self):
        pipeline = FakePipeline([answer()])
        result = ask_policy(pipeline, FakeJudge(), "Who rotates keys?", mode="adaptive")
        self.assertEqual(result["route"], "single_pass")
        self.assertEqual(pipeline.depths, [5])

    def test_final_retry_refusal_and_pipeline_refusals(self):
        refusal = {**answer(), "status": "refusal", "answer": "No evidence", "citations": []}
        result = ask_policy(FakePipeline([refusal, refusal]), FakeJudge(), "Who?", mode="adaptive")
        self.assertEqual(result["route"], "refusal")
        self.assertEqual(len(result["attempts"]), 2)

    def test_nli_context_excludes_omitted_passages_and_binds_evidence(self):
        question = {"question_id": "q", "question": "Who rotates keys?", "expected_doc_ids": ["doc"], "gold_answer": "Admins rotate keys."}
        trace = answer()
        judgment = judge_trace(question, trace, FakeJudge())
        self.assertTrue(judgment["correct"])
        self.assertEqual(judgment["judgment_kind"], "automatic_nli_proxy")
        trace["question_id"] = "q"
        self.assertTrue(score_answers([question], [trace], [judgment])[0]["correct"])
        trace["passages"][0]["text"] = "Changed source"
        with self.assertRaises(ValueError):
            score_answers([question], [trace], [judgment])
        omitted = judge_trace(question, answer("Everyone rotates keys."), FakeJudge())
        self.assertFalse(omitted["correct"])

    def test_review_export_hides_mode_and_requires_explicit_reviewer(self):
        question = {"question_id": "q", "question": "Who rotates keys?", "expected_doc_ids": ["d"], "gold_answer": "Admins rotate keys."}
        trace = {**answer(), "question_id": "q", "experiment_mode": "adaptive"}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "review"
            self.assertEqual(export_review([question], [trace], out), 1)
            task = json.loads((out / "tasks.jsonl").read_text())
            self.assertNotIn("experiment_mode", task)
            label = json.loads((out / "judgments.jsonl").read_text())
            with self.assertRaises(ValueError):
                score_answers([question], [trace], [label])
            label.update(judge="human:reviewer", correct=True)
            self.assertTrue(score_answers([question], [trace], [label])[0]["correct"])

    def test_claims_strip_citations_and_keep_numeric_decimals(self):
        self.assertEqual(claims("Cost is 1.5 dollars. [S1]\nAdmins approve. [S2]"),
                         ["Cost is 1.5 dollars.", "Admins approve."])

    def test_batch_collection_ids_exclusive_output_and_no_gold_to_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            question = {"question_id": "q", "question": "Who rotates keys?", "gold_answer": "Secret gold"}
            protocol = {"top_k": 5, "retry_top_k": 10, "protocol_sha256": "hash"}
            collect(FakePipeline([answer()]), FakeJudge(), [question], path, mode="fixed", protocol=protocol)
            record = json.loads(path.read_text())
            self.assertEqual(record["question_id"], "q")
            self.assertNotIn("gold_answer", record)
            with self.assertRaises(FileExistsError):
                collect(FakePipeline([answer()]), FakeJudge(), [question], path, mode="fixed", protocol=protocol)


class BenchmarkTests(unittest.TestCase):
    def question(self, key, docs):
        return {"question_id": key, "question": key, "question_type": "basic", "expected_doc_ids": docs}

    def test_selection_is_order_independent_and_groups_gold_documents(self):
        questions = [self.question(str(i), ["shared" if i < 3 else str(i)]) for i in range(20)]
        selected = select_final(questions, [], quotas={"basic": 15})
        random.Random(4).shuffle(questions)
        self.assertEqual(selected, select_final(questions, [], quotas={"basic": 15}))
        splits = {q["split"] for q in selected if "shared" in q["expected_doc_ids"]}
        self.assertLessEqual(len(splits), 1)

    def test_legacy_exposure_is_dev_only_and_legacy_ids_excluded(self):
        legacy = [self.question("old", ["seen"])]
        selected = select_final([*legacy, self.question("new", ["seen"])], legacy, quotas={"basic": 1})
        self.assertEqual(selected[0]["question_id"], "new")
        self.assertEqual(selected[0]["split"], "dev")
        self.assertTrue(selected[0]["legacy_document_exposure"])

    def test_build_keeps_duplicate_id_source_variants_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, legacy, archive = root / "source", root / "legacy", root / "docs.zip"
            source.write_text(json.dumps(self.question("new", ["d1", "d2"])) + "\n")
            legacy.write_text(json.dumps(self.question("old", ["d1"])) + "\n")
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("wiki/d1__first.txt", "First source")
                stream.writestr("wiki/d1__second.txt", "Second source")
                stream.writestr("wiki/d2__other.txt", "Other source")
            with patch("adaptive_retrieval.final_benchmark.QUOTAS", {"basic": 1}):
                manifest = build_benchmark(source, legacy, archive, root / "out")
            self.assertEqual(manifest["multi_document_questions"], 1)
            self.assertEqual(len(manifest["duplicate_id_documents"]["d1"]), 2)
            verify_benchmark(root / "out")
            with (root / "out" / "questions.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaises(ValueError):
                verify_benchmark(root / "out")

    def test_freeze_locks_settings_and_unsupported_runtime_starts_no_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, legacy, archive = root / "source", root / "legacy", root / "docs.zip"
            source.write_text(json.dumps(self.question("new", ["d1"])) + "\n")
            legacy.write_text(json.dumps(self.question("old", ["d1"])) + "\n")
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("wiki/d1__first.txt", "First source")
            with patch("adaptive_retrieval.final_benchmark.QUOTAS", {"basic": 1}):
                manifest = build_benchmark(source, legacy, archive, root / "benchmark")
            index = root / "index.jsonl"
            index.write_text(json.dumps({"_meta": {"corpus_sha256": manifest["corpus_sha256"]}}) + "\n")
            args = ["--benchmark", str(root / "benchmark"), "--index", str(index),
                    "--out", str(root / "experiment"), "--split", "dev"]
            experiment_main([*args, "--freeze"])
            self.assertTrue((root / "experiment" / "questions.jsonl").is_file())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment_main([*args, "--top-k", "6"])
            with patch("adaptive_retrieval.final_experiment.platform.system", return_value="Windows"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    experiment_main(args)
            self.assertFalse((root / "experiment" / "fixed.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
