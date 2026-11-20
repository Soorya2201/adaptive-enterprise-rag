from contextlib import redirect_stdout
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.chunking import chunk_document, load_chunks, write_chunks
from adaptive_retrieval.data import clean_document_text, load_documents
from adaptive_retrieval.models import Document
from adaptive_retrieval import check_subset
from adaptive_retrieval.prepare_subset import prepare_subset
from adaptive_retrieval.vector_index import load_index, search_index, write_index


class DataPipelineTests(unittest.TestCase):
    def test_prepare_checked_in_style_subset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            out = Path(directory) / "processed"
            (root / "wiki").mkdir(parents=True)
            (root / "wiki" / "gold__guide.txt").write_text("Gold text", encoding="utf-8")
            (root / "wiki" / "negative__guide.txt").write_text("Other text", encoding="utf-8")
            question = {
                "question_id": "q1", "expected_doc_ids": ["gold"],
                "hard_negative_ids": ["negative"],
            }
            (root / "questions_100.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")

            self.assertEqual(prepare_subset(root, out), (1, 2))
            self.assertEqual(json.loads((out / "subset_questions.jsonl").read_text()), question)
            rows = [json.loads(line) for line in (out / "subset_corpus.jsonl").read_text().splitlines()]
            self.assertEqual([row["doc_id"] for row in rows], ["gold", "negative"])
            self.assertEqual(rows[0]["path"], "wiki/gold__guide.txt")
            self.assertTrue(rows[0]["is_gold"])
            self.assertEqual(rows[1]["mined_for"], ["q1"])

    def test_escaped_document_bodies_are_decoded(self):
        slash = chr(92)
        escaped = ("Subject\n\nFrom: Dylan" + slash + "nTo: Tom" + slash + "n" + slash
                   + "nHi " + slash + '"team' + slash + '", see C:' + slash + slash + "logs")
        self.assertEqual(
            clean_document_text(escaped),
            'Subject\n\nFrom: Dylan\nTo: Tom\n\nHi "team", see C:' + slash + "logs",
        )

    def test_documents_that_only_mention_escapes_are_unchanged(self):
        slash = chr(92)
        text = "Heartbeat fix\nSend " + '":' + slash + "n" + slash + 'n"' + " every 15s.\nDone."
        self.assertEqual(clean_document_text(text), text)

    def test_prepare_and_check_subset_decode_escaped_documents(self):
        slash = chr(92)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            out = Path(directory) / "processed"
            (root / "mail").mkdir(parents=True)
            body = "Subject" + slash + "n" + slash + "nHello " + slash + '"team' + slash + '"'
            (root / "mail" / "gold__mail.txt").write_text(body, encoding="utf-8")
            question = {
                "question_id": "q1", "question_type": "basic",
                "expected_doc_ids": ["gold"], "hard_negative_ids": [],
            }
            (root / "questions_100.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")
            prepare_subset(root, out)
            row = json.loads((out / "subset_corpus.jsonl").read_text())
            self.assertEqual(row["text"], 'Subject\n\nHello "team"')
            with redirect_stdout(io.StringIO()):
                check_subset.main([
                    "--questions", str(out / "subset_questions.jsonl"),
                    "--corpus", str(out / "subset_corpus.jsonl"), "--docs-dir", str(root),
                ])

    def test_parse_corpus_and_preserve_provenance(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(json.dumps({
                "doc_id": "source-1", "text": "First paragraph.", "source": "jira",
                "path": "jira/ticket.txt", "is_gold": True,
            }) + "\n", encoding="utf-8")
            self.assertEqual(
                load_documents(path),
                [Document("source-1", "First paragraph.", "jira", "jira/ticket.txt")],
            )

    def test_paragraph_chunks_preserve_exact_source_offsets(self):
        document = Document("d1", "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta", "wiki", "wiki/a.txt")
        chunks = chunk_document(document, max_words=5, overlap_words=1)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.text, document.text[chunk.start_char:chunk.end_char])
            self.assertEqual(chunk.document_id, document.id)
            self.assertEqual(chunk.path, document.path)
            self.assertLessEqual(len(chunk.text.split()), 5)
        self.assertEqual(chunks[0].text, "alpha beta gamma")

    def test_chunk_file_to_vector_index_round_trip(self):
        class FakeEmbedder:
            model_name = "test-encoder"

            def embed_texts(self, texts):
                return [[float("apple" in text), float("banana" in text)] for text in texts]

        chunks = chunk_document(Document("d1", "apple banana", "wiki"), max_words=1, overlap_words=0)
        with TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "chunks.jsonl"
            index_path = Path(directory) / "index.jsonl"
            write_chunks(chunks_path, chunks, {"corpus_sha256": "example"})
            _, restored = load_chunks(chunks_path)
            self.assertEqual(restored, chunks)
            write_index(
                index_path, restored, FakeEmbedder(), corpus_sha256="example",
                strategy="fixed", max_words=1, overlap_words=0,
            )
            metadata, rows = load_index(index_path)
            self.assertEqual(metadata["chunk_count"], 2)
            results = search_index(rows, [1.0, 0.0], top_k=1)
            self.assertEqual(results[0][0].text, "apple")


if __name__ == "__main__":
    unittest.main()
