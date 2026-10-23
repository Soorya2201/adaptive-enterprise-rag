from collections import Counter
import math
import re
import numpy as np


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def normalized(values):
    x = np.array(values, dtype="float32", order="C", copy=True)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite vector matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero vectors cannot define cosine similarity")
    return x / norms


def top(scores, k):
    if k < 1:
        raise ValueError("k must be positive")
    # Deterministic tie order for both lexical engines **
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))[:k]
    return [(int(i), float(scores[i])) for i in order]


def fuse(*rankings, constant=60):
    scores = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def document_ids(ranking, chunks, k):
    result = []
    seen = set()
    for i, _ in ranking:
        doc = chunks[i]["document_id"]
        if doc not in seen:
            result.append(doc)
            seen.add(doc)
        if len(result) == k:
            break
    return result


class DenseIndex:
    def __init__(self, vectors, kind="flat", *, hnsw_m=32, ef_search=64, nlist=16, nprobe=8, pq_m=32, pq_bits=4, seed=42,):
        import faiss

        self.vectors = normalized(vectors)
        n, d = self.vectors.shape
        self.kind = kind
        if kind in {"python", "numpy"}:
            self.index = None
            self.python_vectors = self.vectors.tolist() if kind == "python" else None
            return
        if kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = ef_search
        elif kind == "ivfpq":
            if d % pq_m or not 1 <= nprobe <= nlist:
                raise ValueError("Size error. Follow: 1 <= nprobe <= nlist")
            minimum = 39 * max(nlist, 2**pq_bits)
            if n < minimum:
                raise ValueError(f"IVF-PQ should have >= {minimum} training vectors for these settings instead of {n}")
            index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, pq_m, pq_bits, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = seed
            index.pq.cp.seed = seed
            index.train(self.vectors)
            index.nprobe = nprobe
        else:
            raise ValueError(f"Unknown dense index {kind}")
        index.add(self.vectors)
        self.index = index

    def search(self, vector, k):
        if k < 1:
            raise ValueError("k must be positive")
        q = normalized(np.asarray(vector).reshape(1, -1))
        if q.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query/index dimensions differ")
        if self.kind == "python":
            query = q[0].tolist()
            qnorm = math.sqrt(sum(v * v for v in query))
            scores = [
                sum(a * b for a, b in zip(query, v))
                / (qnorm * math.sqrt(sum(a * a for a in v)))
                for v in self.python_vectors
            ]
            return top(scores, k)
        if self.kind == "numpy":
            return top(self.vectors @ q[0], k)
        scores, ids = self.index.search(q, min(k, len(self.vectors)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def size_bytes(self):
        if self.index is None:
            return self.vectors.nbytes
        import faiss
        return len(faiss.serialize_index(self.index))


class LexicalIndex:
    # Used simple per-term NumPy scoring as reference so that we can isolate BM25S eager scoring from changes to BM25
    def __init__(self, texts, engine="reference", k1=1.5, b=0.75):
        self.engine, self.k1, self.b = engine, k1, b
        self.count = len(texts)
        corpus = [tokens(text) for text in texts]
        if not corpus or not any(corpus):
            raise ValueError("Empty corpus")
        if engine == "bm25s":
            import bm25s

            self.index = bm25s.BM25(method="lucene", k1=k1, b=b)
            self.index.index(corpus, show_progress=False)
        elif engine == "reference":
            self.counts = [Counter(t) for t in corpus]
            self.lengths = np.array([len(t) for t in corpus])
            self.avg = self.lengths.mean()
            df = Counter(t for c in self.counts for t in c)
            self.idf = {
                t: math.log(1 + (len(corpus) - freq + 0.5) / (freq + 0.5))
                for t, freq in df.items()
            }
        else:
            raise ValueError("Unknown lexical engine")

    def scores(self, question):
        query = tokens(question)
        if not query:
            return np.zeros(self.count)
        if self.engine == "bm25s":
            return self.index.get_scores(query)
        scores = np.zeros(len(self.counts), dtype="float64")
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)
        for term in query:
            if term not in self.idf:
                continue
            tf = np.array([doc[term] for doc in self.counts])
            scores += self.idf[term] * tf / (norm + tf)
        return scores

    def search(self, question, k):
        return top(self.scores(question), k)


def should_rerank(ranking, margin):
    return len(ranking) < 2 or ranking[0][1] - ranking[1][1] < margin # experimental heuristic


class CrossEncoderReranker:
    def __init__(self, model_path, device="mps"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(model_path), device=device, local_files_only=True, trust_remote_code=False, max_length=512)

    def rerank(self, question, ranking, chunks):
        if not ranking:
            return []
        pairs = [(question, chunks[i]["text"]) for i, _ in ranking]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        return sorted(
            [(i, float(s)) for (i, _), s in zip(ranking, scores)],
            key=lambda x: (-x[1], x[0]),
        )
