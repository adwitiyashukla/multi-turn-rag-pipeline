from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from turnrag.data import DocStore, Turn
from turnrag.index import BM25Index
from turnrag.retrieve import Ranked
from turnrag.rewrite import Variants, transcript
from turnrag.text import approx_tokens, normalize, sentences, terms, words

QUESTION_START = re.compile(
    r"^(what|who|whom|whose|when|where|why|how|which|is|are|was|were|do|does|did|can|could|"
    r"should|would|will|shall|may|might|has|have|had)\b",
    re.IGNORECASE,
)
FEATURES = (
    "rerank_top",
    "rerank_second",
    "rerank_mean3",
    "rerank_gap",
    "fused_top",
    "agreement",
    "question_words",
    "looks_like_question",
    "vague",
    "turn",
    "probe_no",
)
NO_ANSWER = "NO_ANSWER"
PROBE = "Before you answer, decide one thing. Can the last user message be answered with the documents above? Reply with only yes or no."

ANSWER_SYSTEM = """You answer the user's last message in a conversation, using only the documents given to you.
Write a direct answer in plain sentences, under 150 words. Do not mention the documents or their numbers.
If the documents do not contain the answer, reply with exactly NO_ANSWER and nothing else."""

CLARIFY_SYSTEM = """The user's last message is too vague to answer, because it could refer to several different things.
Write one short reply that asks the user which one they mean. If the documents show the options, name up to four of them. Do not answer the question."""


def probe_messages(chat: list[dict]) -> list[dict]:
    head = [dict(message) for message in chat]
    head[-1]["content"] = f"{head[-1]['content']}\n\n{PROBE}"
    return head


def probe_no(alternatives: dict[str, float] | None) -> float:
    if not alternatives:
        return 0.5
    yes = sum(math.exp(v) for t, v in alternatives.items() if t.strip().lower() == "yes")
    no = sum(math.exp(v) for t, v in alternatives.items() if t.strip().lower() == "no")
    return no / (yes + no) if yes + no > 0 else 0.5


def features(variants: Variants, ranked: Sequence[Ranked], turns: Sequence[Turn]) -> np.ndarray:
    reranks = [r.rerank if r.rerank is not None else r.score for r in ranked]
    padded = reranks + [min(reranks, default=-10.0)] * max(0, 3 - len(reranks))
    top = ranked[0] if ranked else None
    lists = {tag for r in ranked for tag in r.ranks}
    agreement = 0.0
    if top and lists:
        agreement = sum(1 for rank in top.ranks.values() if rank <= 10) / len(lists)
    question = variants.last
    return np.asarray(
        [
            padded[0],
            padded[1],
            float(np.mean(padded[:3])),
            padded[0] - padded[1],
            (top.fused if top else 0.0) * 60.0,
            agreement,
            math.log1p(len(words(question))),
            1.0 if ("?" in question or QUESTION_START.match(question.strip())) else 0.0,
            float(variants.vague),
            math.log1p(sum(1 for t in turns if t.speaker == "user")),
        ],
        dtype=np.float64,
    )


@dataclass
class Gate:
    weights: list[float]
    bias: float
    mean: list[float]
    scale: list[float]
    threshold: float
    vague_threshold: float = 1.0
    clarify: bool = True

    def probability(self, row: np.ndarray) -> float:
        z = (row - np.asarray(self.mean)) / np.asarray(self.scale)
        return float(1.0 / (1.0 + math.exp(-(float(z @ np.asarray(self.weights)) + self.bias))))

    def clarifies(self, vague: float) -> bool:
        return self.clarify and vague >= self.vague_threshold

    def decide(self, row: np.ndarray) -> tuple[str, float]:
        p = self.probability(row)
        return ("idk" if p < self.threshold else "answer"), p

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Gate:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def fit_logistic(
    rows: np.ndarray, labels: np.ndarray, l2: float = 1.0, steps: int = 50
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale[scale == 0] = 1.0
    x = np.hstack([(rows - mean) / scale, np.ones((rows.shape[0], 1))])
    positives = labels.mean()
    weight = np.where(labels == 1, 0.5 / max(positives, 1e-9), 0.5 / max(1 - positives, 1e-9))
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        gradient = x.T @ (weight * (p - labels)) + penalty * beta
        hessian = (x * (weight * p * (1 - p))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta[:-1], float(beta[-1]), mean, scale


def best_threshold(
    probability: np.ndarray, answer: np.ndarray, idk: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    order = np.argsort(probability, kind="stable")
    p = probability[order]
    totals = np.concatenate([[0.0], np.cumsum((weight * (idk - answer))[order])])
    cuts = np.concatenate([[0.0], (p[1:] + p[:-1]) / 2, [1.0]])
    valid = np.concatenate([[True], p[1:] > p[:-1], [True]])
    best = int(np.argmax(np.where(valid, totals, -np.inf)))
    return float(cuts[best]), float(totals[best] / weight.sum())


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def idf_vector(bm25: BM25Index) -> np.ndarray:
    matrix = bm25.index.matrix
    df = np.diff(matrix.indptr).astype(np.float64)
    n = matrix.shape[0]
    return np.log1p((n - df + 0.5) / (df + 0.5))


def evidence(
    store: DocStore,
    ranked: Sequence[Ranked],
    query: str,
    bm25: BM25Index,
    idf: np.ndarray,
    passages: int,
    budget: int,
) -> list[tuple[int, str]]:
    wanted = {bm25.vocab[t] for t in terms(query) if t in bm25.vocab}
    chosen: list[tuple[int, str]] = []
    share = max(48, budget // max(1, min(passages, len(ranked))))
    for number, hit in enumerate(ranked[:passages], start=1):
        parts = sentences(store.text(hit.index)) or [normalize(store.text(hit.index))]
        scored = []
        for position, sentence in enumerate(parts):
            present = {bm25.vocab[t] for t in terms(sentence) if t in bm25.vocab} & wanted
            value = float(sum(idf[t] for t in present)) - 0.01 * position
            scored.append((value, position, sentence))
        keep: list[tuple[int, str]] = []
        used = 0
        for _, position, sentence in sorted(scored, key=lambda item: (-item[0], item[1])):
            cost = approx_tokens(sentence)
            if keep and used + cost > share:
                continue
            keep.append((position, sentence))
            used += cost
        text = " ".join(sentence for _, sentence in sorted(keep))
        chosen.append((number, text))
    return chosen


def document_block(
    store: DocStore, ranked: Sequence[Ranked], chunks: Sequence[tuple[int, str]]
) -> str:
    lines = []
    for number, text in chunks:
        title = normalize(store.field("title", ranked[number - 1].index))
        header = f"[{number}] {title}" if title else f"[{number}]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def answer_messages(
    store: DocStore,
    ranked: Sequence[Ranked],
    chunks: Sequence[tuple[int, str]],
    turns: Sequence[Turn],
    history_turns: int,
    agent_words: int,
    idk_text: str,
) -> list[dict]:
    conversation = transcript(turns, max(1, history_turns * 2 - 1), agent_words)
    body = (
        f"Documents:\n{document_block(store, ranked, chunks)}\n\n"
        f"Conversation:\n{conversation}\n\nAnswer the last user message."
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM.format(idk=idk_text)},
        {"role": "user", "content": body},
    ]


def clarify_messages(
    store: DocStore,
    ranked: Sequence[Ranked],
    chunks: Sequence[tuple[int, str]],
    turns: Sequence[Turn],
    agent_words: int,
) -> list[dict]:
    conversation = transcript(turns, 5, agent_words)
    body = (
        f"Documents:\n{document_block(store, ranked, chunks)}\n\n"
        f"Conversation:\n{conversation}\n\nWrite the clarifying reply."
    )
    return [
        {"role": "system", "content": CLARIFY_SYSTEM},
        {"role": "user", "content": body},
    ]


def is_idk(text: str, idk_text: str) -> bool:
    lowered = normalize(text).lower()
    return lowered.startswith(normalize(idk_text).lower()[:40]) or lowered.startswith(
        "i do not have specific information"
    )


def attribute(
    answer: str, store: DocStore, ranked: Sequence[Ranked], bm25: BM25Index, idf: np.ndarray
) -> list[list[int]]:
    vectors = []
    for hit in ranked:
        ids = {bm25.vocab[t] for t in terms(store.text(hit.index)) if t in bm25.vocab}
        vectors.append(ids)
    cited = []
    for sentence in sentences(answer, min_words=1):
        ids = {bm25.vocab[t] for t in terms(sentence) if t in bm25.vocab}
        total = float(sum(idf[t] for t in ids)) or 1.0
        overlaps = [float(sum(idf[t] for t in ids & doc)) / total for doc in vectors]
        best = int(np.argmax(overlaps)) if overlaps else -1
        cited.append([best + 1] if best >= 0 and overlaps[best] >= 0.5 else [])
    return cited
