"""BM25S + FAISS retrieval modes, including the disagreement-gated LameR expansion.

Retriever modes (the ablation ladder in the final report):
- ``cosine``       dense-only exact cosine search (bare-bones baseline)
- ``bm25``         keyword-only BM25S
- ``hybrid``       RRF fusion of dense and BM25S (H1)
- ``hybrid_lamer`` hybrid with LameR expansion on every question
- ``adaptive``     hybrid; expand only when dense and BM25S disagree on the top
                   document (H2, the project's proposed method)
"""

import multiprocessing as mp
import re
from time import perf_counter

from .expansion import Expander, expanded_query


RETRIEVERS = ("cosine", "bm25", "hybrid", "hybrid_lamer", "adaptive")
EXPANDING_RETRIEVERS = {"hybrid_lamer", "adaptive"}
RRF_CONSTANT = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def fuse(*rankings: list[tuple[int, float]], constant: int = RRF_CONSTANT) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion; ties break on chunk index for determinism."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (index, _) in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _search_worker(connection, texts, vectors, threads):
    # Import FAISS only in this process; the PyTorch embedder stays in the parent.
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
        corpus = [tokenize(text) for text in texts]
        if not any(corpus):
            raise ValueError("The corpus contains no searchable words")
        lexical = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        lexical.index(corpus, show_progress=False)

        def keyword(text, depth):
            query_tokens = tokenize(text)
            scores = lexical.get_scores(query_tokens) if query_tokens else np.zeros(len(texts))
            matching = np.flatnonzero(scores > 0)
            ids = matching[np.lexsort((matching, -scores[matching]))[:depth]]
            return [(int(index), float(scores[index])) for index in ids]

        connection.send((True, None))
        while True:
            request = connection.recv()
            if request is None:
                break
            try:
                kind = request[0]
                if kind == "lexical":
                    _, text, depth = request
                    started = perf_counter()
                    ranking = keyword(text, depth)
                    connection.send((True, (ranking, (perf_counter() - started) * 1000)))
                    continue
                if kind == "search":
                    _, question, vector, depth = request
                elif kind == "dense":
                    _, vector, depth = request
                else:
                    raise ValueError(f"Unknown worker request {kind!r}")
                started = perf_counter()
                query = np.asarray(vector, dtype="float32").reshape(1, -1)
                if query.shape[1] != dense.d or not np.isfinite(query).all() or not np.linalg.norm(query):
                    raise ValueError("Query vector must be finite, nonzero and match the index dimensions")
                faiss.normalize_L2(query)
                scores, ids = dense.search(query, min(depth, dense.ntotal))
                dense_ranking = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
                searched = perf_counter()
                if kind == "dense":
                    connection.send((True, (dense_ranking, (searched - started) * 1000)))
                    continue
                lexical_ranking = keyword(question, depth)
                timings = {
                    "vector_search_ms": (searched - started) * 1000,
                    "lexical_search_ms": (perf_counter() - searched) * 1000,
                }
                connection.send((True, (dense_ranking, lexical_ranking, timings)))
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

    def __init__(
        self, session, retriever="hybrid", candidate_depth=50, faiss_threads=4, *,
        expander: Expander | None = None, expansion_candidates=5,
    ):
        if retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {', '.join(RETRIEVERS)}")
        if candidate_depth < 1 or faiss_threads < 1 or expansion_candidates < 1:
            raise ValueError("candidate_depth, faiss_threads and expansion_candidates must be positive")
        if retriever in EXPANDING_RETRIEVERS and expander is None:
            raise ValueError(f"The {retriever} retriever needs a query expander")
        self.session = session
        self.retriever = retriever
        self.candidate_depth = candidate_depth
        self.expander = expander
        self.expansion_candidates = expansion_candidates
        self.score_type = {"cosine": "cosine", "bm25": "bm25"}.get(retriever, "rrf")
        self.rows = session.rows
        self.metadata = session.metadata
        self.embedder = session.embedder
        self.documents = session.documents
        self.last_timings: dict[str, float] = {}
        self.last_trace: dict = {}
        self.process = None
        self.connection = None
        self.closed = False
        self.chunks = [chunk for chunk, vector in session.rows]
        # Every mode uses the same worker and backend, so latencies are comparable.
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

    def _request(self, *request):
        self.connection.send(request)
        return self._receive()

    def _top_document(self, ranking):
        return self.chunks[ranking[0][0]].document_id if ranking else None

    def embed_question(self, question):
        return self.session.embed_question(question)

    def _check(self, question, top_k):
        if self.closed:
            raise RuntimeError("Search session is closed")
        question = question.strip()
        if not question:
            raise ValueError("Enter a question to search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return question

    def search(self, question, top_k=3, *, vector=None):
        """Rank chunks; pass ``vector`` to reuse a question embedding computed earlier."""
        question = self._check(question, top_k)
        if self.retriever == "cosine":
            started = perf_counter()
            if vector is None:
                vector = self.embed_question(question)
            encoded = perf_counter()
            ranking, dense_ms = self._request("dense", vector, self.candidate_depth)
            self.last_timings = {
                "embedding_ms": (encoded - started) * 1000,
                "vector_search_ms": dense_ms,
                "hybrid_search_ms": (perf_counter() - encoded) * 1000,
            }
            self.last_trace = {"retriever": "cosine", "gate_fired": False, "retrieval_rounds": 1}
            return [(self.chunks[index], score) for index, score in ranking[:top_k]]
        started = perf_counter()
        if self.retriever == "bm25":
            # Keyword-only search never needs the query embedding.
            ranking, lexical_ms = self._request("lexical", question, self.candidate_depth)
            self.last_timings = {
                "lexical_search_ms": lexical_ms,
                "hybrid_search_ms": (perf_counter() - started) * 1000,
            }
            self.last_trace = {"retriever": "bm25", "gate_fired": False, "retrieval_rounds": 1}
            return [(self.chunks[index], score) for index, score in ranking[:top_k]]
        if vector is None:
            vector = self.embed_question(question)
        encoded = perf_counter()
        dense, lexical, timings = self._request("search", question, vector, self.candidate_depth)
        searched = perf_counter()
        self.last_timings = {"embedding_ms": (encoded - started) * 1000, **timings}
        trace = {
            "retriever": self.retriever,
            "dense_top_document": self._top_document(dense),
            "lexical_top_document": self._top_document(lexical),
            "gate_fired": False,
            "retrieval_rounds": 1,
        }
        trace["top_document_disagreement"] = (
            not dense or not lexical
            or trace["dense_top_document"] != trace["lexical_top_document"]
        )
        expand = self.retriever == "hybrid_lamer" or (
            self.retriever == "adaptive" and trace["top_document_disagreement"]
        )
        if expand:
            candidates = [self.chunks[i] for i, _ in lexical[: self.expansion_candidates]]
            expansion = self.expander.expand(question, candidates)
            expanded_lexical, second_ms = self._request(
                "lexical", expanded_query(question, expansion.texts), self.candidate_depth,
            )
            tick = perf_counter()
            ranking = fuse(dense, expanded_lexical)
            self.last_timings.update({
                "expansion_ms": expansion.expansion_ms,
                "expanded_lexical_search_ms": second_ms,
                "fusion_ms": (perf_counter() - tick) * 1000,
            })
            trace.update({
                "gate_fired": True,
                "gate_reason": "always" if self.retriever == "hybrid_lamer" else "top_document_disagreement",
                "retrieval_rounds": 2,
                "expansion_model": expansion.model,
                "expansion_texts": list(expansion.texts),
                "expansion_input_tokens": expansion.input_tokens,
                "expansion_output_tokens": expansion.output_tokens,
                "expansion_truncated": expansion.truncated,
                "expansion_candidate_chunk_ids": list(expansion.candidate_chunk_ids),
            })
        else:
            tick = perf_counter()
            ranking = fuse(dense, lexical)
            self.last_timings["fusion_ms"] = (perf_counter() - tick) * 1000
        # Includes worker communication and any expansion, unlike the stage timers.
        self.last_timings["hybrid_search_ms"] = (perf_counter() - encoded) * 1000
        self.last_timings["first_round_ms"] = (searched - started) * 1000
        self.last_trace = trace
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
