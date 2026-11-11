"""Use the team's search-session contract with optional BM25S + FAISS retrieval."""

import multiprocessing as mp
import re
from time import perf_counter


def _search_worker(connection, texts, vectors, threads):
    # Import FAISS only in this process; the team's PyTorch embedder stays in the parent.
    try:
        import faiss
        import bm25s
        import numpy as np

        faiss.omp_set_num_threads(threads)
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
            raise ValueError("Expected a nonempty finite vector matrix")
        if (np.linalg.norm(vectors, axis=1) == 0).any():
            raise ValueError("Hybrid retrieval requires nonzero document vectors")
        faiss.normalize_L2(vectors)
        dense = faiss.IndexFlatIP(vectors.shape[1])
        dense.add(vectors)
        corpus = [re.findall(r"\w+", text.casefold()) for text in texts]
        if not any(corpus):
            raise ValueError("The corpus contains no searchable words")
        lexical = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        lexical.index(corpus, show_progress=False)
        connection.send((True, None))
        while True:
            request = connection.recv()
            if request is None:
                break
            question, vector, depth = request
            try:
                started = perf_counter()
                query = np.asarray(vector, dtype="float32").reshape(1, -1)
                if query.shape[1] != dense.d or not np.isfinite(query).all() or not np.linalg.norm(query):
                    raise ValueError("Query vector must be finite, nonzero and match the index dimensions")
                faiss.normalize_L2(query)
                _, dense_ids = dense.search(query, min(depth, dense.ntotal))
                searched = perf_counter()
                query_tokens = re.findall(r"\w+", question.casefold())
                lexical_scores = lexical.get_scores(query_tokens) if query_tokens else np.zeros(len(texts))
                matching_ids = np.flatnonzero(lexical_scores > 0)
                lexical_ids = matching_ids[
                    np.lexsort((matching_ids, -lexical_scores[matching_ids]))[:depth]
                ]
                lexical_done = perf_counter()
                scores = {}
                for ranking in (dense_ids[0], lexical_ids):
                    for rank, index in enumerate(ranking, 1):
                        index = int(index)
                        scores[index] = scores.get(index, 0.0) + 1 / (60 + rank)
                ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                timings = {
                    "vector_search_ms": (searched - started) * 1000,
                    "lexical_search_ms": (lexical_done - searched) * 1000,
                    "fusion_ms": (perf_counter() - lexical_done) * 1000,
                }
                connection.send((True, (ranking, timings)))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except EOFError:
        pass
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class OptimizedSearchSession:
    """Wrap an existing SearchSession; preserve its chunks, embedder and provenance."""

    def __init__(self, session, retriever="cosine", candidate_depth=50, faiss_threads=4):
        if retriever not in {"cosine", "hybrid"}:
            raise ValueError("retriever must be cosine or hybrid")
        if candidate_depth < 1 or faiss_threads < 1:
            raise ValueError("candidate_depth and faiss_threads must be positive")
        self.session = session
        self.retriever = retriever
        self.candidate_depth = candidate_depth
        self.score_type = "rrf" if retriever == "hybrid" else "cosine"
        self.rows = session.rows
        self.metadata = session.metadata
        self.embedder = session.embedder
        self.documents = session.documents
        self.last_timings = {}
        self.process = None
        self.connection = None
        self.closed = False
        self.chunks = [chunk for chunk, vector in session.rows]
        if retriever == "hybrid":
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(
                target=_search_worker,
                args=(child, [chunk.text for chunk in self.chunks],
                      [vector for chunk, vector in session.rows], faiss_threads),
                daemon=True,
            )
            try:
                self.process.start()
                child.close()
                self._receive()
            except BaseException:
                child.close()
                self.close()
                raise

    def _receive(self):
        if not self.connection.poll(240):
            raise TimeoutError("Hybrid search worker did not respond within 240 seconds")
        try:
            ok, result = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError("Hybrid search worker stopped unexpectedly") from exc
        if not ok:
            raise RuntimeError(result)
        return result

    def search(self, question, top_k=3):
        if self.closed:
            raise RuntimeError("Search session is closed")
        question = question.strip()
        if not question:
            raise ValueError("Enter a question to search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.retriever == "cosine":
            results = self.session.search(question, top_k)
            self.last_timings = dict(getattr(self.session, "last_timings", {}))
            return results
        started = perf_counter()
        if hasattr(self.embedder, "embed_query"):
            vector = self.embedder.embed_query(question)
        else:
            vector = self.embedder.embed_texts([question])[0]
        encoded = perf_counter()
        self.connection.send((question, vector, self.candidate_depth))
        ranking, timings = self._receive()
        self.last_timings = {
            "embedding_ms": (encoded - started) * 1000,
            # Includes worker communication, unlike the individual search-stage timers.
            "hybrid_search_ms": (perf_counter() - encoded) * 1000,
            **timings,
        }
        return [(self.chunks[index], score) for index, score in ranking[:top_k]]

    def close(self):
        self.closed = True
        try:
            if self.process is not None and self.process.pid is not None:
                if self.process.is_alive():
                    try:
                        self.connection.send(None)
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    self.process.join(5)
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join()
        finally:
            if self.connection is not None:
                self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
