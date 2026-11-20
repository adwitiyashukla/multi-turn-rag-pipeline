import math

import numpy as np
import pytest

from turnrag.index import BM25Index, top_k
from turnrag.text import terms


def test_bm25_matches_formula():
    docs = ["red apple pie", "green apple", "red red car", "blue sky"]
    index = BM25Index.build(docs, k1=1.2, b=0.75)
    hits = index.search([("red apple", 1.0)], 4)
    lengths = [len(terms(d)) for d in docs]
    average = sum(lengths) / len(lengths)

    def expected(doc: int) -> float:
        total = 0.0
        for word in ("red", "appl"):
            tf = terms(docs[doc]).count(word)
            df = sum(1 for d in docs if word in terms(d))
            if tf:
                idf = math.log1p((len(docs) - df + 0.5) / (df + 0.5))
                total += idf * tf / (tf + 1.2 * (1 - 0.75 + 0.75 * lengths[doc] / average))
        return total

    assert [int(i) for i in hits.indices] == [0, 2, 1]
    for doc, score in zip(hits.indices, hits.scores, strict=True):
        assert score == pytest.approx(expected(int(doc)), rel=1e-5)


def test_bm25_save_and_load(tmp_path):
    index = BM25Index.build(["alpha beta", "beta gamma"], 1.2, 0.75)
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.vocab == index.vocab
    assert loaded.search([("gamma", 1.0)], 2).indices.tolist() == [1]


def test_top_k_breaks_ties_by_index():
    scores = np.asarray([0.5, 0.9, 0.5, 0.9, 0.1], dtype=np.float32)
    hits = top_k(scores, 3)
    assert hits.indices.tolist() == [1, 3, 0]
    assert top_k(np.zeros(3, dtype=np.float32), 2, positive=True).indices.size == 0
