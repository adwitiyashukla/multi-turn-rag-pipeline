from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from turnrag.text import terms

FLUSH = 1_000_000


@dataclass(frozen=True)
class Hits:
    indices: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.size)


EMPTY = Hits(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))


def top_k(scores: np.ndarray, k: int, positive: bool = False) -> Hits:
    if positive:
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return EMPTY
        values = scores[candidates]
    else:
        candidates = np.arange(scores.size)
        values = scores
    k = min(k, values.size)
    if k <= 0:
        return EMPTY
    kth = np.partition(values, values.size - k)[values.size - k]
    chosen = np.flatnonzero(values >= kth)
    order = chosen[np.lexsort((candidates[chosen], -values[chosen]))][:k]
    return Hits(candidates[order].astype(np.int64), values[order].astype(np.float32))


class ImpactIndex:
    def __init__(self, matrix: sparse.csc_matrix) -> None:
        self.matrix = matrix.tocsc()
        self.matrix.sort_indices()
        self.indptr = self.matrix.indptr
        self.rows = self.matrix.indices
        self.data = self.matrix.data
        self.n_docs, self.n_terms = self.matrix.shape

    def scores(self, query: dict[int, float]) -> np.ndarray:
        rows = []
        values = []
        for term, weight in query.items():
            if not 0 <= term < self.n_terms or weight == 0:
                continue
            start, end = self.indptr[term], self.indptr[term + 1]
            if end > start:
                rows.append(self.rows[start:end])
                values.append(self.data[start:end].astype(np.float32) * np.float32(weight))
        if not rows:
            return np.zeros(self.n_docs, dtype=np.float32)
        total = np.bincount(np.concatenate(rows), np.concatenate(values), minlength=self.n_docs)
        return total.astype(np.float32)

    def search(self, query: dict[int, float], k: int) -> Hits:
        return top_k(self.scores(query), k, positive=True)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(folder / "postings.npz", self.matrix, compressed=False)

    @classmethod
    def load(cls, folder: Path) -> ImpactIndex:
        return cls(sparse.load_npz(folder / "postings.npz"))


class BM25Index:
    def __init__(self, index: ImpactIndex, vocab: dict[str, int]) -> None:
        self.index = index
        self.vocab = vocab

    @classmethod
    def build(cls, texts: Iterable[str], k1: float, b: float) -> BM25Index:
        vocab: dict[str, int] = {}
        parts: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
        buffer: tuple[list[int], list[int], list[int]] = ([], [], [])
        lengths: list[int] = []
        for doc, text in enumerate(texts):
            counts = Counter(terms(text))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                buffer[0].append(doc)
                buffer[1].append(vocab.setdefault(term, len(vocab)))
                buffer[2].append(tf)
            if len(buffer[0]) >= FLUSH:
                _flush(buffer, parts)
        _flush(buffer, parts)
        row = np.concatenate(parts[0]) if parts[0] else np.zeros(0, dtype=np.int32)
        col = np.concatenate(parts[1]) if parts[1] else np.zeros(0, dtype=np.int32)
        tf = np.concatenate(parts[2]) if parts[2] else np.zeros(0, dtype=np.float32)
        n_docs = len(lengths)
        length = np.asarray(lengths, dtype=np.float32)
        average = float(length.mean()) if n_docs else 1.0
        df = np.bincount(col, minlength=len(vocab)).astype(np.float64)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1.0 - b + b * length / max(average, 1e-9))
        weights = (idf[col] * tf / (tf + norm[row])).astype(np.float32)
        matrix = sparse.csc_matrix((weights, (row, col)), shape=(n_docs, len(vocab)))
        return cls(ImpactIndex(matrix), vocab)

    def encode(self, weighted_texts: Iterable[tuple[str, float]]) -> dict[int, float]:
        query: dict[int, float] = {}
        for text, weight in weighted_texts:
            for term, count in Counter(terms(text)).items():
                position = self.vocab.get(term)
                if position is not None:
                    query[position] = query.get(position, 0.0) + count * weight
        return query

    def search(self, weighted_texts: Iterable[tuple[str, float]], k: int) -> Hits:
        return self.index.search(self.encode(weighted_texts), k)

    def save(self, folder: Path) -> None:
        self.index.save(folder)
        ordered = sorted(self.vocab, key=self.vocab.__getitem__)
        (folder / "vocab.json").write_text(json.dumps(ordered), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> BM25Index:
        ordered = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        return cls(ImpactIndex.load(folder), {term: i for i, term in enumerate(ordered)})


def _flush(buffer: tuple[list[int], ...], parts: tuple[list[np.ndarray], ...]) -> None:
    if not buffer[0]:
        return
    parts[0].append(np.asarray(buffer[0], dtype=np.int32))
    parts[1].append(np.asarray(buffer[1], dtype=np.int32))
    parts[2].append(np.asarray(buffer[2], dtype=np.float32))
    for values in buffer:
        values.clear()
