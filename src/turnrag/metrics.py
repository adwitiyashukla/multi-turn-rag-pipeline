from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

CUTOFFS = (1, 3, 5, 10)
IDK_LABELS = ("UNANSWERABLE", "CONVERSATIONAL")


def _unique(ranked: Sequence[str], limit: int) -> list[str]:
    seen: list[str] = []
    for doc_id in ranked:
        if doc_id not in seen:
            seen.append(doc_id)
            if len(seen) == limit:
                break
    return seen


def ndcg_at(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    gains = sorted((value for value in relevant.values() if value > 0), reverse=True)[:k]
    ideal = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    if ideal == 0:
        return 0.0
    dcg = sum(
        max(relevant.get(doc_id, 0), 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(_unique(ranked, k))
    )
    return dcg / ideal


def recall_at(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = {doc_id for doc_id, value in relevant.items() if value > 0}
    if not positives:
        return 0.0
    return len(positives.intersection(_unique(ranked, k))) / len(positives)


def retrieval_scores(
    ranked: Sequence[str], relevant: Mapping[str, int], cutoffs: Sequence[int] = CUTOFFS
) -> dict[str, float]:
    scores = {}
    for k in cutoffs:
        scores[f"ndcg@{k}"] = ndcg_at(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at(ranked, relevant, k)
    return scores


def aggregate(
    per_query: Mapping[str, Mapping[str, float]], groups: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for query_id, scores in per_query.items():
        buckets[groups[query_id]].append(scores)
        buckets["all"].append(scores)
    table: dict[str, dict[str, float]] = {}
    for name, rows in sorted(buckets.items()):
        keys = rows[0].keys()
        table[name] = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
        table[name]["count"] = float(len(rows))
    return table


def evaluate_runs(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    groups: Mapping[str, str],
    cutoffs: Sequence[int] = CUTOFFS,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    per_query = {
        query_id: retrieval_scores(runs.get(query_id, ()), relevant, cutoffs)
        for query_id, relevant in qrels.items()
        if query_id in groups
    }
    return aggregate(per_query, groups), per_query


def harmonic3(a: float, b: float, c: float) -> float:
    if a <= 0 or b <= 0 or c <= 0:
        return 0.0
    return 3 * a * b * c / (a * b + a * c + b * c)


def rb_alg(bert_recall: float, rouge_l: float, bert_k_precision: float) -> float:
    return harmonic3((bert_recall + 1) / 2, rouge_l, (bert_k_precision + 1) / 2)


def conditioned(answerability: str, idk: float, clarify: float, value: float) -> float:
    if answerability == "UNDERSPECIFIED":
        return 1.0 if clarify == 1 else 0.0
    if answerability in IDK_LABELS:
        return 1.0 if idk == 1 else 0.0
    if idk == 1:
        return 0.0
    return value
