from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

from turnrag.config import Config
from turnrag.data import DocStore, sha256_file
from turnrag.encoders import SparseDocEncoder
from turnrag.index import BM25Index, ImpactIndex

SHARD = 2048
VOCAB = 30522


def index_dir(cfg: Config, collection: str, leg: str) -> Path:
    return cfg.path("indexes") / collection / leg


def _done(folder: Path) -> bool:
    return (folder / "manifest.json").exists()


def _manifest(folder: Path, payload: dict) -> None:
    (folder / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_docstore(cfg: Config, collection: str) -> DocStore:
    folder = DocStore.folder_for(cfg, collection)
    if _done(folder):
        return DocStore(folder)
    started = time.time()
    store = DocStore.build(cfg, collection)
    _manifest(folder, {"passages": len(store), "seconds": round(time.time() - started, 1)})
    return store


def build_bm25(cfg: Config, collection: str, store: DocStore, log: Callable[[str], None]) -> None:
    folder = index_dir(cfg, collection, "bm25")
    if _done(folder):
        return
    started = time.time()
    index = BM25Index.build((store.text(i) for i in range(len(store))), cfg.bm25.k1, cfg.bm25.b)
    index.save(folder)
    seconds = round(time.time() - started, 1)
    _manifest(
        folder,
        {
            "k1": cfg.bm25.k1,
            "b": cfg.bm25.b,
            "postings": int(index.index.matrix.nnz),
            "terms": len(index.vocab),
            "seconds": seconds,
        },
    )
    log(f"{collection} bm25: {index.index.matrix.nnz} postings in {seconds}s")


def sparse_encoder(cfg: Config) -> SparseDocEncoder:
    section = cfg.sparse
    return SparseDocEncoder(
        cfg.model_dir(section.model),
        section.onnx,
        section.max_length,
        section.token_budget,
        section.threads,
        max_terms=section.max_terms,
    )


def _save_shard(path: Path, encoded: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    lengths = np.asarray([terms.size for terms, _ in encoded], dtype=np.int64)
    terms = np.concatenate([terms for terms, _ in encoded]).astype(np.int32)
    weights = np.concatenate([values for _, values in encoded]).astype(np.float16)
    partial = path.with_suffix(".tmp.npz")
    np.savez(partial, lengths=lengths, terms=terms, weights=weights)
    partial.replace(path)


def build_sparse(cfg: Config, collection: str, store: DocStore, log: Callable[[str], None]) -> None:
    folder = index_dir(cfg, collection, "sparse")
    if _done(folder):
        return
    shards = folder / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    encoder = sparse_encoder(cfg)
    total = len(store)
    count = (total + SHARD - 1) // SHARD
    started = time.time()
    fresh = 0
    for shard in range(count):
        path = shards / f"{shard:05d}.npz"
        if path.exists():
            continue
        low, high = shard * SHARD, min(total, (shard + 1) * SHARD)
        _save_shard(path, encoder.encode([store.text(i) for i in range(low, high)]))
        fresh += high - low
        elapsed = time.time() - started
        left = total - high
        rate = fresh / max(elapsed, 1e-9)
        log(
            f"{collection} sparse: {high}/{total} passages, {rate:.1f}/s, "
            f"about {left / max(rate, 1e-9) / 60:.0f} min left"
        )
    rows, cols, vals = [], [], []
    offset = 0
    for shard in range(count):
        with np.load(shards / f"{shard:05d}.npz") as part:
            lengths = part["lengths"]
            rows.append(np.repeat(np.arange(offset, offset + lengths.size), lengths))
            cols.append(part["terms"])
            vals.append(part["weights"].astype(np.float32))
            offset += lengths.size
    matrix = sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(total, VOCAB)
    )
    ImpactIndex(matrix).save(folder)
    for path in shards.glob("*.npz"):
        path.unlink()
    shards.rmdir()
    _manifest(
        folder,
        {
            "model": cfg.sparse.model,
            "onnx_sha256": sha256_file(cfg.model_dir(cfg.sparse.model) / cfg.sparse.onnx),
            "max_terms": cfg.sparse.max_terms,
            "postings": int(matrix.nnz),
            "seconds_this_run": round(time.time() - started, 1),
        },
    )
    log(f"{collection} sparse: done, {matrix.nnz} postings")


def build_all(
    cfg: Config,
    collections: Sequence[str],
    legs: Sequence[str] = ("bm25", "sparse"),
    log: Callable[[str], None] = print,
) -> None:
    for collection in collections:
        store = build_docstore(cfg, collection)
        if "bm25" in legs:
            build_bm25(cfg, collection, store, log)
        if "sparse" in legs:
            build_sparse(cfg, collection, store, log)
        store.close()
