import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.chunking import chunk_documents
from adaptive_retrieval.embedding import LocalEmbedder
from adaptive_retrieval.interactive import SearchSession, run_interactive
from adaptive_retrieval.models import Document
from adaptive_retrieval.vector_index import file_sha256, write_index


class FakeEmbedder:
    model_name = "test-encoder"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        if "invalid query" in texts:
            raise ValueError("Example query validation error")
        return [[float("apple" in text), float("banana" in text)] for text in texts]


class InteractiveTests(unittest.TestCase):
    def make_session(self, directory):
        root = Path(directory)
        corpus_path = root / "corpus.jsonl"
        documents = [
            Document("apple-doc", "apple policy\n\nThe full document has this extra section.", "wiki", "wiki/apple.txt"),
            Document("banana-doc", "banana policy", "wiki", "wiki/banana.txt"),
        ]
        corpus_path.write_text("\n".join(json.dumps({
            "doc_id": document.id, "text": document.text,
            "source": document.source, "path": document.path,
        }) for document in documents) + "\n", encoding="utf-8")
        encoder = FakeEmbedder()
        index_path = root / "index.jsonl"
        write_index(
            index_path, chunk_documents(documents, max_words=2, overlap_words=0), encoder,
            corpus_sha256=file_sha256(corpus_path), strategy="fixed", max_words=2, overlap_words=0,
        )
        encoder.calls.clear()
        return SearchSession(index_path, corpus_path, embedder=encoder)

    def test_repeated_queries_reuse_encoder_and_full_document_is_available(self):
        with TemporaryDirectory() as directory:
            session = self.make_session(directory)
            commands = iter(["apple", "/doc 1", "/top 1", "banana", "/quit"])
            output = []
            run_interactive(session, input_fn=lambda _: next(commands), output_fn=output.append)
            transcript = "\n".join(output)
            self.assertEqual(session.embedder.calls, [["apple"], ["banana"]])
            self.assertIn("Full document: apple-doc", transcript)
            self.assertIn("The full document has this extra section.", transcript)
            self.assertIn("Source: wiki/apple.txt", transcript)
            self.assertIn("Showing up to 1 passages", transcript)
            self.assertIn("Document: banana-doc", transcript)
            self.assertEqual(output[-1], "Session ended.")

    def test_bad_commands_and_query_errors_allow_next_question(self):
        with TemporaryDirectory() as directory:
            session = self.make_session(directory)
            commands = iter(["", "/doc 1", "/top 0", "/top hello", "/unknown", "invalid query", "apple", "/doc 99", "exit"])
            output = []
            run_interactive(session, input_fn=lambda _: next(commands), output_fn=output.append)
            transcript = "\n".join(output)
            self.assertIn("Ask a question first", transcript)
            self.assertIn("Usage: /top N", transcript)
            self.assertIn("Unknown command", transcript)
            self.assertIn("Search error: Example query validation error", transcript)
            self.assertIn("Document: apple-doc", transcript)
            self.assertIn("Choose a result between", transcript)

    def test_eof_and_interrupt_exit_cleanly(self):
        with TemporaryDirectory() as directory:
            session = self.make_session(directory)
            for exception in (EOFError, KeyboardInterrupt):
                with self.subTest(exception=exception):
                    output = []

                    def interrupted_input(_):
                        raise exception()

                    run_interactive(session, input_fn=interrupted_input, output_fn=output.append)
                    self.assertEqual(output[-1], "Session ended.")
            self.assertEqual(session.embedder.calls, [])

    def test_changed_corpus_is_rejected_before_search(self):
        with TemporaryDirectory() as directory:
            self.make_session(directory)
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus changed"):
                SearchSession(Path(directory) / "index.jsonl", corpus, embedder=FakeEmbedder())

    def test_model_and_query_instruction_mismatches_are_rejected(self):
        with TemporaryDirectory() as directory:
            self.make_session(directory)
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "different embedding model"):
                SearchSession(root / "index.jsonl", root / "corpus.jsonl", embedder=LocalEmbedder())
            # Even equal-dimensional vectors are incompatible when their
            # query instruction differs from the one used to define the index.
            lines = (root / "index.jsonl").read_text().splitlines()
            header = json.loads(lines[0])
            header["_meta"]["model"] = LocalEmbedder().model_name
            header["_meta"]["query_instruction"] = "an obsolete instruction"
            lines[0] = json.dumps(header)
            (root / "index.jsonl").write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "different query instruction"):
                SearchSession(root / "index.jsonl", root / "corpus.jsonl", embedder=LocalEmbedder())


if __name__ == "__main__":
    unittest.main()
