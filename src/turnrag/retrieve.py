from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from turnrag.build import index_dir
from turnrag.config import Config
from turnrag.data import DocStore
from turnrag.encoders import CrossEncoder, SparseQueryEncoder
from turnrag.index import BM25Index, Hits, ImpactIndex
from turnrag.rewrite import Variants


@dataclass
class Candidate:
    index: int
    fused: float
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class Ranked:
    index: int
    doc_id: str
    score: float
    fused: float
    rerank: float | None
    ranks: dict[str, int]


class Shelf:
    def __init__(self, cfg: Config, collection: str, legs: Sequence[str]) -> None:
        self.name = collection
        self.store = DocStore.open(cfg, collection)
        self.bm25 = BM25Index.load(index_dir(cfg, collection, "bm25")) if "bm25" in legs else None
        folder = index_dir(cfg, collection, "sparse")
        self.sparse = ImpactIndex.load(folder) if "sparse" in legs else None


class RerankCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.items: dict[str, float] = {}
        if path and path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.items[row["k"]] = row["s"]

    @staticmethod
    def key(query: str, doc_id: str) -> str:
        return f"{doc_id}\t{query}"

    def missing(self, query: str, doc_ids: Sequence[str]) -> list[int]:
        return [i for i, doc_id in enumerate(doc_ids) if self.key(query, doc_id) not in self.items]

    def get(self, query: str, doc_ids: Sequence[str]) -> np.ndarray:
        return np.asarray([self.items[self.key(query, d)] for d in doc_ids], dtype=np.float32)

    def put(self, query: str, doc_ids: Sequence[str], scores: np.ndarray) -> None:
        with self.lock:
            rows = []
            for doc_id, score in zip(doc_ids, scores, strict=True):
                key = self.key(query, doc_id)
                self.items[key] = float(score)
                rows.append(json.dumps({"k": key, "s": float(score)}))
            if self.path and rows:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(rows) + "\n")


class Retriever:
    def __init__(
        self,
        cfg: Config,
        collections: Sequence[str] | None = None,
        reranker: bool = True,
        cache: RerankCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.legs = [leg for leg, weight in cfg.retrieval.legs.items() if weight > 0]
        names = collections or cfg.data.collections
        self.shelves = {name: Shelf(cfg, name, self.legs) for name in names}
        self.query_encoder = SparseQueryEncoder(cfg.model_dir(cfg.sparse.model))
        self.cache = cache or RerankCache(None)
        self.reranker = None
        if reranker:
            section = cfg.rerank
            self.reranker = CrossEncoder(
                cfg.model_dir(section.model),
                section.onnx,
                section.max_length,
                section.token_budget,
                section.threads,
            )

    def leg(
        self, shelf: Shelf, leg: str, weighted: Sequence[tuple[str, float]], depth: int
    ) -> Hits:
        if leg == "bm25":
            return shelf.bm25.search(weighted, depth)
        return shelf.sparse.search(self.query_encoder.encode(weighted), depth)

    def fuse(
        self,
        collection: str,
        variants: Variants,
        legs: Mapping[str, float] | None = None,
        queries: Mapping[str, float] | None = None,
        depth: int | None = None,
    ) -> list[Candidate]:
        shelf = self.shelves[collection]
        legs = legs if legs is not None else self.cfg.retrieval.legs
        queries = queries if queries is not None else self.cfg.retrieval.queries
        depth = depth or self.cfg.retrieval.depth
        k = self.cfg.retrieval.rrf_k
        pool: dict[int, Candidate] = {}
        for query_name, query_weight in queries.items():
            weighted = variants.weighted(query_name)
            if query_weight <= 0 or not weighted:
                continue
            for leg_name, leg_weight in legs.items():
                if leg_weight <= 0 or leg_name not in self.legs:
                    continue
                hits = self.leg(shelf, leg_name, weighted, depth)
                tag = f"{query_name}:{leg_name}"
                for rank, index in enumerate(hits.indices.tolist(), start=1):
                    entry = pool.setdefault(index, Candidate(index, 0.0))
                    entry.fused += query_weight * leg_weight / (k + rank)
                    entry.ranks[tag] = rank
        return sorted(pool.values(), key=lambda c: (-c.fused, c.index))

    def rerank_scores(self, collection: str, query: str, indices: Sequence[int]) -> np.ndarray:
        shelf = self.shelves[collection]
        doc_ids = [shelf.store.ids[i] for i in indices]
        missing = self.cache.missing(query, doc_ids)
        if missing:
            texts = [shelf.store.text(indices[i]) for i in missing]
            scores = self.reranker.score(query, texts)
            self.cache.put(query, [doc_ids[i] for i in missing], scores)
        return self.cache.get(query, doc_ids)

    def search(
        self,
        collection: str,
        variants: Variants,
        k: int = 10,
        rerank: bool = True,
        candidates: int | None = None,
        **fuse_options,
    ) -> list[Ranked]:
        shelf = self.shelves[collection]
        pool = self.fuse(collection, variants, **fuse_options)
        width = candidates or self.cfg.rerank.candidates
        head = pool[:width] if rerank and self.reranker else pool[:k]
        if not head:
            return []
        fused = np.asarray([c.fused for c in head], dtype=np.float32)
        if rerank and self.reranker:
            scores = self.rerank_scores(collection, variants.standalone, [c.index for c in head])
            blend = self.cfg.rerank.blend
            final = (1 - blend) * _z(scores) + blend * _z(fused) if blend > 0 else scores
        else:
            scores = None
            final = fused
        order = sorted(range(len(head)), key=lambda i: (-float(final[i]), head[i].index))[:k]
        return [
            Ranked(
                head[i].index,
                shelf.store.ids[head[i].index],
                float(final[i]),
                float(fused[i]),
                None if scores is None else float(scores[i]),
                head[i].ranks,
            )
            for i in order
        ]


def _z(values: np.ndarray) -> np.ndarray:
    spread = float(values.std())
    return (values - values.mean()) / spread if spread > 0 else values - values.mean()
