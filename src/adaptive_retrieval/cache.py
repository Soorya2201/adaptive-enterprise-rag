"""Verified-answer cache (H3): reuse a checked answer only for the same question.

Assumption: documents and embeddings are frozen while a cache is in use. If they,
or the generator, prompt or checker, change, the whole cache is thrown out and
rebuilt. Retrieval and answer settings (``scope``) keep separate entries instead. Only generated answers that passed ``check_answer`` are stored. Lookup:

1. ``similar``   question similarity is at least the threshold. This only shortlists;
                 it cannot tell "same question" from "same topic".
2. ``key_terms`` numbers, identifiers and regions match exactly (a cheap veto).
3. ``sources``   safety assertion: cited chunks and quotes are intact, which also
                 catches re-chunking of an unchanged corpus.
4. The most similar entry passing 1-3 goes to the judge. Only a strict YES is a hit.

Without a judge there are never hits. A match does not make the stored answer
correct: a wrong answer that passed the quote checker will be repeated.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from .generation import quote_match
from .judge import Judgment, SameQuestionJudge


DEFAULT_THRESHOLD = 0.80
CHECKS = ("similar", "key_terms", "sources")

_NUMBER_WORDS = {
    word: str(value) for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
                      "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100"})
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_REGION = re.compile(r"\b(?:us|eu|ap|ca|sa|me|af)-(?:east|west|north|south|central)(?:-\d+)?\b")


def key_terms(question: str) -> dict[str, list[str]]:
    """Numbers (including spelled-out ones), snake_case identifiers and region codes.

    Numbers are compared as values: "12KB" and "12 KB" both give 12, "4,096" gives
    4096, and "two weeks" gives 2. Leading zeros are dropped ("02" == "2").
    """
    text = _THOUSANDS.sub("", question.casefold())
    words = re.findall(r"[a-z]+", text)
    numbers = {str(float(value)).rstrip("0").rstrip(".") for value in _NUMBER.findall(text)}
    numbers |= {_NUMBER_WORDS[word] for word in words if word in _NUMBER_WORDS}
    return {
        "numbers": sorted(numbers),
        "identifiers": sorted(set(_IDENTIFIER.findall(text))),
        "regions": sorted(set(_REGION.findall(text))),
    }


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_versions(versions: dict) -> dict:
    """The part of the key whose change throws the cache out; ``scope`` only isolates entries."""
    return {key: value for key, value in versions.items() if key != "scope"}


def _unit(vector):
    # Imported here so commands that never use the cache work without optional packages.
    import numpy as np

    array = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(array))
    if not norm or not np.isfinite(array).all():
        raise ValueError("Cache vectors must be finite and nonzero")
    return array / norm


@dataclass
class CacheEntry:
    entry_id: str
    question: str
    vector: list[float]
    key_terms: dict
    answer: str
    citations: list[dict]
    versions: dict
    created_utc: str
    origin: str | None = None
    hits: int = 0


@dataclass
class Candidate:
    entry_id: str
    origin: str | None
    question: str
    similarity: float
    checks: dict[str, bool]
    detail: str = ""

    def passes(self, threshold: float, skip: tuple[str, ...] = ()) -> bool:
        """Similarity is re-tested against ``threshold``; other checks use stored results."""
        if "similar" not in skip and self.similarity < threshold:
            return False
        return all(ok for name, ok in self.checks.items() if name != "similar" and name not in skip)


@dataclass
class CacheDecision:
    hit: bool
    entry: CacheEntry | None
    reason: str
    candidates: list[Candidate] = field(default_factory=list)
    judgment: Judgment | None = None

    def summary(self) -> dict:
        best = self.candidates[0] if self.candidates else None
        return {
            "hit": self.hit,
            "reason": self.reason,
            "entry_id": self.entry.entry_id if self.entry else None,
            "matched_origin": self.entry.origin if self.entry else None,
            "matched_question": self.entry.question if self.entry else None,
            "best_similarity": best.similarity if best else None,
            "judgment": asdict(self.judgment) if self.judgment else None,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


class VerifiedCache:
    """In-memory store with optional JSONL persistence (one entry per line)."""

    def __init__(
        self, path: Path | None = None, *, threshold: float = DEFAULT_THRESHOLD,
        require_exact_quotes: bool = False, max_candidates: int = 5,
        judge: SameQuestionJudge | None = None,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.path = Path(path) if path else None
        self.threshold = threshold
        self.judge = judge
        self.require_exact_quotes = require_exact_quotes
        self.max_candidates = max_candidates
        self.entries: list[CacheEntry] = []
        self._matrix = None
        if self.path and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._add(CacheEntry(**json.loads(line)), persist=False)

    def __len__(self) -> int:
        return len(self.entries)

    def similarity_to_origin(self, origin: str, vector) -> float | None:
        """Similarity between ``vector`` and the entry stored for ``origin``, if any."""
        for index, entry in enumerate(self.entries):
            if entry.origin == origin:
                return float(self._matrix[index] @ _unit(vector))
        return None

    def discard_if_stale(self, versions: dict) -> dict | None:
        """Throw out the whole cache if any entry was built under other versions.

        The file is renamed, not deleted, so opening a cache with the wrong index
        by mistake does not destroy it. The cache then rebuilds from empty.
        """
        current = frozen_versions(versions)
        stale = [entry for entry in self.entries if frozen_versions(entry.versions) != current]
        if not stale:
            return None
        changed = sorted({
            key for entry in stale for key in frozen_versions(entry.versions).keys() | current.keys()
            if entry.versions.get(key) != current.get(key)
        })
        moved = None
        if self.path and self.path.is_file():
            moved = self.path.with_name(f"{self.path.name}.stale-{uuid4().hex[:8]}")
            self.path.rename(moved)
        discarded = len(self.entries)
        self.entries = []
        self._matrix = None
        return {"entries": discarded, "changed": changed, "moved_to": str(moved) if moved else None}

    def _add(self, entry: CacheEntry, *, persist: bool) -> None:
        import numpy as np

        row = _unit(entry.vector)[None, :]
        self._matrix = row if not self.entries else np.vstack([self._matrix, row])
        self.entries.append(entry)
        if persist and self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def admit(
        self, question: str, vector, result: dict, versions: dict, chunks: dict,
        *, origin: str | None = None,
    ) -> tuple[bool, str]:
        """Store a checked answer. ``chunks`` maps chunk ID to the current Chunk."""
        if result.get("status") != "answer" or result.get("route") == "cache_hit":
            return False, "not_a_new_answer"
        if not versions.get("corpus_sha256"):
            return False, "no_corpus_version"
        citations = result.get("citations") or []
        if not citations:
            return False, "no_citations"
        if self.require_exact_quotes and any(c.get("quote_match") != "exact" for c in citations):
            return False, "quotes_not_exact"
        stored = []
        for citation in citations:
            chunk = chunks.get(citation["chunk_id"])
            if chunk is None:
                return False, "cited_chunk_missing"
            stored.append({**citation, "chunk_sha256": text_sha256(chunk.text)})
        entry = CacheEntry(
            entry_id=uuid4().hex, question=question.strip(),
            vector=[float(value) for value in vector], key_terms=key_terms(question),
            answer=result["answer"], citations=stored, versions=dict(versions),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), origin=origin,
        )
        self._add(entry, persist=True)
        return True, "admitted"

    def shortlist(
        self, question: str, vector, versions: dict, chunks: dict, *, threshold: float | None = None,
    ) -> tuple[list[Candidate], str]:
        """Check the ``max_candidates`` most similar live entries, without judging.

        All checks are evaluated regardless of the threshold, for offline replays.
        Returns the candidates and, if there are none, the reason.
        """
        import numpy as np

        threshold = self.threshold if threshold is None else threshold
        if not versions.get("corpus_sha256"):
            return [], "no_corpus_version"
        live = [index for index, entry in enumerate(self.entries) if entry.versions == versions]
        if not live:
            return [], "empty" if not self.entries else "version_mismatch"
        similarities = self._matrix[live] @ _unit(vector)
        order = np.argsort(-similarities)[: self.max_candidates]
        terms = key_terms(question)
        candidates = []
        for position in order:
            entry = self.entries[live[position]]
            sources_ok, detail = self._sources_unchanged(entry, chunks)
            similarity = float(similarities[position])
            candidates.append(Candidate(
                entry.entry_id, entry.origin, entry.question, similarity,
                {
                    "similar": similarity >= threshold,
                    "key_terms": terms == entry.key_terms,
                    "sources": sources_ok,
                },
                detail,
            ))
        return candidates, ""

    def lookup(
        self, question: str, vector, versions: dict, chunks: dict, *, threshold: float | None = None,
    ) -> CacheDecision:
        """At most one judge call; anything but a strict YES is a miss."""
        threshold = self.threshold if threshold is None else threshold
        candidates, reason = self.shortlist(question, vector, versions, chunks, threshold=threshold)
        if not candidates:
            return CacheDecision(False, None, reason)
        selected = next((c for c in candidates if c.passes(threshold)), None)
        if selected is None:
            failed = next(name for name in CHECKS if not candidates[0].checks[name])
            return CacheDecision(False, None, f"best_{failed}_failed", candidates)
        if self.judge is None:
            return CacheDecision(False, None, "no_judge", candidates)
        judgment = self.judge.judge(question, selected.question)
        if not judgment.same:
            return CacheDecision(False, None, f"judge_{judgment.verdict}", candidates, judgment)
        entry = next(e for e in self.entries if e.entry_id == selected.entry_id)
        entry.hits += 1
        return CacheDecision(True, entry, "hit", candidates, judgment)

    @staticmethod
    def _sources_unchanged(entry: CacheEntry, chunks: dict) -> tuple[bool, str]:
        for citation in entry.citations:
            chunk = chunks.get(citation["chunk_id"])
            if chunk is None:
                return False, f"chunk {citation['chunk_id']} no longer indexed"
            if text_sha256(chunk.text) != citation["chunk_sha256"]:
                return False, f"chunk {citation['chunk_id']} text changed"
            if quote_match(citation["quote"], chunk.text) is None:
                return False, f"quote no longer in {citation['chunk_id']}"
        return True, ""
