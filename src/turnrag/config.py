from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

COLLECTIONS = ("clapnq", "cloud", "fiqa", "govt")
LEGS = ("bm25", "sparse")
QUERIES = ("last", "context", "rewrite", "keywords")
ENV_PREFIX = "TURNRAG__"


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(Section):
    data: Path = Path("data")
    models: Path = Path("models")
    indexes: Path = Path("indexes")
    runs: Path = Path("runs")
    cache: Path = Path("cache")
    gate: Path = Path("gate.json")


class Data(Section):
    source: str = "https://raw.githubusercontent.com/IBM/mt-rag-benchmark"
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    collections: tuple[str, ...] = COLLECTIONS

    @field_validator("collections")
    @classmethod
    def known_collections(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one collection is required")
        unknown = sorted(set(value) - set(COLLECTIONS))
        if unknown:
            raise ValueError(f"unknown collections: {unknown}")
        return value


class BM25(Section):
    k1: float = Field(1.2, gt=0, le=10)
    b: float = Field(0.75, ge=0, le=1)


class Encoder(Section):
    model: str
    onnx: str = "onnx/model_int8.onnx"
    max_length: int = Field(512, ge=16, le=8192)
    token_budget: int = Field(4096, ge=128)
    threads: int = Field(0, ge=0, le=256)


class Sparse(Encoder):
    max_terms: int = Field(256, ge=8, le=30522)
    activation: Literal["log1p", "log1p_log1p"] = "log1p"


class Rerank(Encoder):
    candidates: int = Field(30, ge=1, le=500)
    blend: float = Field(0.0, ge=0, le=1)


class Retrieval(Section):
    depth: int = Field(100, ge=10, le=2000)
    rrf_k: float = Field(60.0, gt=0)
    context_turns: int = Field(3, ge=1, le=20)
    context_decay: float = Field(0.5, gt=0, le=1)
    legs: dict[str, float]
    queries: dict[str, float]

    @field_validator("legs")
    @classmethod
    def known_legs(cls, value: dict[str, float]) -> dict[str, float]:
        return _weights(value, LEGS, "legs")

    @field_validator("queries")
    @classmethod
    def known_queries(cls, value: dict[str, float]) -> dict[str, float]:
        return _weights(value, QUERIES, "queries")


class LLM(Section):
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3.5-2b"
    timeout: float = Field(900, gt=0)
    seed: int = 7
    temperature: float = Field(0.0, ge=0, le=2)
    retries: int = Field(2, ge=0, le=10)


class Rewrite(Section):
    history_turns: int = Field(6, ge=1, le=50)
    agent_words: int = Field(60, ge=0, le=5000)
    max_tokens: int = Field(160, ge=16, le=4096)


class Gate(Section):
    enabled: bool = True
    clarify: bool = True
    vague_rate: float = Field(0.05, gt=0, lt=1)


class Answer(Section):
    passages: int = Field(5, ge=1, le=20)
    evidence_tokens: int = Field(700, ge=64, le=16000)
    history_turns: int = Field(4, ge=0, le=50)
    agent_words: int = Field(80, ge=0, le=5000)
    max_tokens: int = Field(256, ge=16, le=4096)
    idk_text: str = "I do not have specific information about that in the documents I can search."


class Api(Section):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    max_sessions: int = Field(256, ge=1)
    session_ttl: float = Field(3600, gt=0)


class Eval(Section):
    bertscore_model: str = "deberta-xlarge-mnli"
    bertscore_layer: int = Field(40, ge=1, le=64)
    cached_texts: int = Field(64, ge=1, le=100000)
    judge_max_tokens: int = Field(96, ge=4, le=1024)


class Config(Section):
    root: Path = Path(".")
    paths: Paths = Paths()
    data: Data
    bm25: BM25 = BM25()
    sparse: Sparse
    rerank: Rerank
    retrieval: Retrieval
    llm: LLM = LLM()
    rewrite: Rewrite = Rewrite()
    gate: Gate = Gate()
    answer: Answer = Answer()
    api: Api = Api()
    eval: Eval = Eval()

    @model_validator(mode="after")
    def rerank_window(self) -> Config:
        if self.rerank.candidates > self.retrieval.depth:
            raise ValueError("rerank.candidates cannot exceed retrieval.depth")
        if self.answer.passages > self.rerank.candidates:
            raise ValueError("answer.passages cannot exceed rerank.candidates")
        return self

    def path(self, name: str) -> Path:
        return (self.root / getattr(self.paths, name)).resolve()

    def model_dir(self, name: str) -> Path:
        return self.path("models") / name


def _weights(value: dict[str, float], allowed: tuple[str, ...], label: str) -> dict[str, float]:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}, allowed: {list(allowed)}")
    if any(weight < 0 for weight in value.values()):
        raise ValueError(f"{label} weights must be non-negative")
    if not any(weight > 0 for weight in value.values()):
        raise ValueError(f"at least one of the {label} weights must be positive")
    return value


def _env_overrides(raw: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    for key, text in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("__")
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(text)
    return raw


def _parse_scalar(text: str) -> Any:
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def load_config(path: str | Path = "config.toml", environ: dict[str, str] | None = None) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    raw = _env_overrides(raw, dict(os.environ if environ is None else environ))
    raw["root"] = path.parent
    return Config.model_validate(raw)
