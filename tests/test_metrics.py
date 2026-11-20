import random

import pytest
import pytrec_eval

from turnrag.metrics import conditioned, evaluate_runs, ndcg_at, rb_alg, recall_at


def test_matches_pytrec_eval_on_random_runs():
    rng = random.Random(0)
    docs = [f"d{i}" for i in range(40)]
    qrels, runs = {}, {}
    for q in range(60):
        qid = f"q{q}"
        qrels[qid] = {d: 1 for d in rng.sample(docs, rng.randint(1, 5))}
        runs[qid] = rng.sample(docs, 10)
    groups = {qid: "a" if i % 2 else "b" for i, qid in enumerate(qrels)}
    table, per_query = evaluate_runs(runs, qrels, groups)
    scored = {q: {d: float(10 - i) for i, d in enumerate(r)} for q, r in runs.items()}
    official = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.1,3,5,10", "recall.1,3,5,10"}
    ).evaluate(scored)
    for qid, values in official.items():
        for k in (1, 3, 5, 10):
            assert per_query[qid][f"ndcg@{k}"] == pytest.approx(values[f"ndcg_cut_{k}"], abs=1e-9)
            assert per_query[qid][f"recall@{k}"] == pytest.approx(values[f"recall_{k}"], abs=1e-9)
    assert table["all"]["count"] == 60


def test_duplicates_are_counted_once():
    relevant = {"a": 1, "b": 1}
    assert recall_at(["a", "a", "b"], relevant, 2) == 1.0
    assert ndcg_at(["a", "a", "b"], relevant, 2) == pytest.approx(1.0)


def test_rb_alg_matches_ibm_example():
    assert rb_alg(0.5707797408103941, 0.27027027027027, 0.581508338451385) == pytest.approx(
        0.4809335099944646, abs=1e-9
    )


def test_conditioning_rules():
    assert conditioned("UNANSWERABLE", 1, 0, 0.3) == 1.0
    assert conditioned("UNANSWERABLE", 0.5, 0, 0.3) == 0.0
    assert conditioned("ANSWERABLE", 1, 0, 0.3) == 0.0
    assert conditioned("ANSWERABLE", 0.5, 0, 0.3) == 0.3
    assert conditioned("UNDERSPECIFIED", 0, 1, 0.3) == 1.0
    assert conditioned("UNDERSPECIFIED", 1, 0, 0.3) == 0.0
