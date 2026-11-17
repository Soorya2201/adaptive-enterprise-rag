import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_retrieval.ablation import remove_document
from adaptive_retrieval.chunking import chunk_documents
from adaptive_retrieval.interactive import SearchSession
from adaptive_retrieval.models import Document
from adaptive_retrieval.vector_index import file_sha256, load_index, write_index


class FakeEmbedder:
    model_name = "test-encoder"

    def embed_texts(self, texts):
        return [[float("minutes" in text), float("unrelated" in text)] for text in texts]


class AblationTests(unittest.TestCase):
    def test_separate_pair_removes_all_target_passages_and_preserves_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, index = root / "corpus.jsonl", root / "index.jsonl"
            documents = [
                Document("answer-doc", "Restore in tens of minutes.\n\nMore detail on minutes.", "wiki", "answer.txt"),
                Document("other-doc", "An unrelated policy.", "wiki", "other.txt"),
            ]
            corpus.write_text("".join(json.dumps({
                "doc_id": d.id, "text": d.text, "source": d.source, "path": d.path,
            }) + "\n" for d in documents))
            write_index(
                index, chunk_documents(documents, max_words=5, overlap_words=0), FakeEmbedder(),
                corpus_sha256=file_sha256(corpus), strategy="paragraph",
                max_words=5, overlap_words=0,
            )
            original_hashes = file_sha256(corpus), file_sha256(index)

            new_corpus, new_index = remove_document(
                corpus, index, "answer-doc", root / "variant",
            )
            metadata, rows = load_index(new_index)
            self.assertEqual([r["doc_id"] for r in map(json.loads, new_corpus.read_text().splitlines())], ["other-doc"])
            self.assertTrue(rows)
            self.assertEqual({chunk.document_id for chunk, _ in rows}, {"other-doc"})
            self.assertEqual(metadata["corpus_sha256"], file_sha256(new_corpus))
            self.assertEqual(metadata["ablation_removed_document_id"], "answer-doc")
            self.assertEqual((file_sha256(corpus), file_sha256(index)), original_hashes)
            self.assertEqual(len(SearchSession(new_index, new_corpus, embedder=FakeEmbedder()).rows), len(rows))


if __name__ == "__main__":
    unittest.main()
