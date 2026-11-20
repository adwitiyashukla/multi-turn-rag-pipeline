from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from turnrag import data
from turnrag.config import Config
from turnrag.data import Task
from turnrag.llm import Cache, Client
from turnrag.metrics import evaluate_runs
from turnrag.retrieve import RerankCache, Retriever
from turnrag.rewrite import Variants, basic, rewrite

OFFICIAL = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}


def client(cfg: Config) -> Client:
    return Client(cfg.llm, Cache(cfg.path("cache") / "llm.jsonl"))


def retriever(cfg: Config, reranker: bool = True) -> Retriever:
    cache = RerankCache(cfg.path("cache") / f"rerank-{cfg.rerank.model}.jsonl")
    return Retriever(cfg, reranker=reranker, cache=cache)


def variants_path(cfg: Config, split: str) -> Path:
    return cfg.path("runs") / split / "variants.jsonl"


def load_variants(path: Path) -> dict[str, Variants]:
    found: dict[str, Variants] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task_id = row.pop("task_id")
                row["context"] = tuple(tuple(item) for item in row["context"])
                found[task_id] = Variants(**row)
    return found


def compute_variants(
    cfg: Config,
    tasks: Sequence[Task],
    split: str,
    workers: int = 2,
    log: Callable[[str], None] = print,
) -> dict[str, Variants]:
    path = variants_path(cfg, split)
    found = load_variants(path)
    todo = [task for task in tasks if task.task_id not in found]
    if not todo:
        return found
    llm = client(cfg)
    r, w = cfg.retrieval, cfg.rewrite
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def work(task: Task) -> tuple[str, Variants]:
        args = (r.context_turns, r.context_decay, w.history_turns, w.agent_words, w.max_tokens)
        return task.task_id, rewrite(llm, task.turns, *args)

    with ThreadPoolExecutor(workers) as pool, path.open("a", encoding="utf-8") as handle:
        for done, (task_id, value) in enumerate(pool.map(work, todo), start=1):
            found[task_id] = value
            handle.write(json.dumps({"task_id": task_id, **asdict(value)}) + "\n")
            handle.flush()
            if done % 25 == 0 or done == len(todo):
                rate = done / max(time.time() - started, 1e-9)
                left = (len(todo) - done) / max(rate, 1e-9) / 60
                log(f"{split} rewrites: {done}/{len(todo)}, about {left:.0f} min left")
    return found


def with_mixtral(variants: Variants, text: str) -> Variants:
    return Variants(variants.last, variants.context, text, None, variants.vague)


def retrieval_systems(include_mixtral: bool) -> dict[str, dict]:
    systems = {
        "bm25 last turn": {"legs": {"bm25": 1.0}, "queries": {"last": 1.0}, "rerank": False},
        "sparse last turn": {"legs": {"sparse": 1.0}, "queries": {"last": 1.0}, "rerank": False},
        "sparse qwen rewrite": {
            "legs": {"sparse": 1.0},
            "queries": {"rewrite": 1.0},
            "rerank": False,
        },
        "fusion": {"rerank": False},
        "fusion + rerank": {"rerank": True},
    }
    if include_mixtral:
        systems["bm25 mixtral rewrite"] = {
            "legs": {"bm25": 1.0},
            "queries": {"rewrite": 1.0},
            "rerank": False,
            "mixtral": True,
        }
        systems["sparse mixtral rewrite"] = {
            "legs": {"sparse": 1.0},
            "queries": {"rewrite": 1.0},
            "rerank": False,
            "mixtral": True,
        }
    return systems


def run_retrieval(
    cfg: Config,
    engine: Retriever,
    tasks: Iterable[Task],
    variants: dict[str, Variants],
    qrels: dict[str, dict[str, int]],
    system: dict,
    mixtral: dict[str, str] | None = None,
    k: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    runs: dict[str, list[tuple[str, float]]] = {}
    fuse = {key: system[key] for key in ("legs", "queries") if key in system}
    for task in tasks:
        if task.task_id not in qrels:
            continue
        value = variants.get(task.task_id) or basic(
            task.turns, cfg.retrieval.context_turns, cfg.retrieval.context_decay
        )
        if system.get("mixtral"):
            value = with_mixtral(value, mixtral[task.task_id])
        hits = engine.search(task.collection, value, k=k, rerank=system.get("rerank", True), **fuse)
        runs[task.task_id] = [(hit.doc_id, hit.score) for hit in hits]
    return runs


def score_runs(
    runs: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    tasks: Iterable[Task],
) -> dict[str, dict[str, float]]:
    groups = {task.task_id: task.collection for task in tasks if task.task_id in runs}
    ranked = {query: [doc for doc, _ in hits] for query, hits in runs.items()}
    table, _ = evaluate_runs(ranked, qrels, groups)
    return table


def write_official(
    path: Path, tasks: Iterable[Task], runs: dict[str, list[tuple[str, float]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            if task.task_id not in runs:
                continue
            row = {
                "conversation_id": task.conversation_id,
                "task_id": task.task_id,
                "Collection": OFFICIAL[task.collection],
                "contexts": [
                    {"document_id": doc_id, "score": score} for doc_id, score in runs[task.task_id]
                ],
            }
            handle.write(json.dumps(row) + "\n")


def retrieval_report(
    cfg: Config,
    split: str,
    systems: dict[str, dict] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    tasks = data.load_tasks(cfg, split)
    qrels = data.load_qrels(cfg, split)
    mixtral = data.load_mixtral_rewrites(cfg) if split == "dev" else None
    variants = load_variants(variants_path(cfg, split))
    engine = retriever(cfg)
    systems = systems or retrieval_systems(split == "dev")
    report = {}
    for name, system in systems.items():
        if system.get("rerank", True) and len(variants) < len(qrels):
            log(f"skipping {name}: rewrites missing")
            continue
        started = time.time()
        runs = run_retrieval(cfg, engine, tasks, variants, qrels, system, mixtral)
        table = score_runs(runs, qrels, tasks)
        report[name] = table
        write_official(cfg.path("runs") / split / "retrieval" / f"{_slug(name)}.jsonl", tasks, runs)
        row = table["all"]
        log(
            f"{split} {name}: ndcg@5 {row['ndcg@5']:.3f} recall@5 {row['recall@5']:.3f} "
            f"ndcg@10 {row['ndcg@10']:.3f} ({time.time() - started:.0f}s)"
        )
    out = cfg.path("runs") / split / "retrieval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def pipeline(cfg: Config):
    from turnrag.answer import Gate
    from turnrag.pipeline import Pipeline

    path = cfg.path("gate")
    loaded = Gate.load(path) if path.exists() else None
    return Pipeline(cfg, retriever(cfg), client(cfg), loaded)


def fit_gate(cfg: Config, log: Callable[[str], None] = print) -> dict:
    from turnrag.answer import FEATURES, Gate, auc, best_threshold, fit_logistic
    from turnrag.metrics import IDK_LABELS

    rows = [
        json.loads(line)
        for line in answers_path(cfg, "dev").read_text(encoding="utf-8").splitlines()
        if line
    ]
    scored = (cfg.path("runs") / "dev" / "scores.jsonl").read_text(encoding="utf-8")
    scores = {row["task_id"]: row["score"] for row in map(json.loads, scored.splitlines())}
    x = np.asarray([row["features"] + [row["probe_no"]] for row in rows])
    y = np.asarray([0 if row["answerability"] in IDK_LABELS else 1 for row in rows])
    folds = np.arange(len(rows)) % 5
    held = np.zeros(len(rows))
    for fold in range(5):
        train = folds != fold
        w, b, m, s = fit_logistic(x[train], y[train])
        held[~train] = 1 / (1 + np.exp(-(((x[~train] - m) / s) @ w + b)))
    weights, bias, mean, scale = fit_logistic(x, y)
    labels = [task.answerability for task in data.load_tasks(cfg, "dev")]
    idk_share = sum(label in IDK_LABELS for label in labels)
    sample_weight = np.where(
        y == 1,
        (len(labels) - idk_share) / max(int(y.sum()), 1),
        idk_share / max(int((y == 0).sum()), 1),
    )
    answered = np.asarray([scores[row["task_id"]] for row in rows])
    threshold, gain = best_threshold(held, answered, 1.0 - y, sample_weight)
    vague = np.asarray([v.vague for v in load_variants(variants_path(cfg, "dev")).values()])
    vague_threshold = float(np.quantile(vague, 1.0 - cfg.gate.vague_rate))
    gate = Gate(
        weights.tolist(),
        bias,
        mean.tolist(),
        scale.tolist(),
        threshold,
        vague_threshold,
        cfg.gate.clarify,
    )
    gate.save(cfg.path("gate"))
    abstain = held < threshold
    report = {
        "tasks": len(rows),
        "unanswerable": int((y == 0).sum()),
        "cross_validated_auc": auc(held, y),
        "probe_only_auc": auc(-x[:, -1], y),
        "rerank_only_auc": auc(x[:, 0], y),
        "threshold": threshold,
        "expected_gain_per_task": gain,
        "idk_recall_cv": float(abstain[y == 0].mean()),
        "answerable_abstain_rate_cv": float(abstain[y == 1].mean()),
        "vague_threshold": vague_threshold,
        "vague_flag_rate_dev": float((vague >= vague_threshold).mean()),
        "weights": dict(zip(FEATURES, weights.round(4).tolist(), strict=True)),
    }
    (cfg.path("runs") / "dev" / "gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log(
        f"gate: cv auc {report['cross_validated_auc']:.3f}, idk recall {report['idk_recall_cv']:.3f}, "
        f"answerable abstain {report['answerable_abstain_rate_cv']:.3f}, threshold {threshold:.3f}"
    )
    return report


def answers_path(cfg: Config, split: str) -> Path:
    return cfg.path("runs") / split / "answers.jsonl"


def run_answers(
    cfg: Config,
    split: str,
    task_ids: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
    workers: int = 2,
) -> list[dict]:
    tasks = data.load_tasks(cfg, split)
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [task for task in tasks if task.task_id in wanted]
    variants = load_variants(variants_path(cfg, split))
    path = answers_path(cfg, split)
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                done[row["task_id"]] = row
    todo = [task for task in tasks if task.task_id not in done]
    engine = pipeline(cfg)
    started = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)

    def work(task: Task) -> dict:
        response = engine.respond(task.collection, task.turns, variants.get(task.task_id))
        return {
            "task_id": task.task_id,
            "conversation_id": task.conversation_id,
            "Collection": OFFICIAL[task.collection],
            "collection": task.collection,
            "answerability": task.answerability,
            "input": [{"speaker": t.speaker, "text": t.text} for t in task.turns],
            "contexts": [
                {"document_id": p["document_id"], "text": p["text"], "score": p["score"]}
                for p in response.passages
            ],
            "predictions": [{"text": response.text}],
            "reference": task.reference,
            "action": response.action,
            "probability": response.probability,
            "rewrite": response.rewrite,
            "citations": response.citations,
            "timings": response.timings,
            "features": response.features,
            "probe_no": response.probe_no,
        }

    with ThreadPoolExecutor(workers) as pool, path.open("a", encoding="utf-8") as handle:
        for count, row in enumerate(pool.map(work, todo), start=1):
            done[row["task_id"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if count % 10 == 0 or count == len(todo):
                rate = count / max(time.time() - started, 1e-9)
                log(
                    f"{split} answers: {count}/{len(todo)}, about {(len(todo) - count) / rate / 60:.0f} min left"
                )
    return [done[task.task_id] for task in tasks if task.task_id in done]


def _official_eval(cfg: Config) -> tuple[dict, dict, list[dict]]:
    payload = json.loads(
        data.raw_path(cfg, "mtrag-human/evaluations/RAG.json").read_text(encoding="utf-8")
    )
    documents = {row["document_id"]: row["text"] for row in payload["documents"]}
    tasks = {row["task_id"]: row for row in payload["tasks"]}
    evaluations = [row for row in payload["evaluations"] if row["model_id"] != "target"]
    return documents, tasks, evaluations


def _label(task: dict) -> str:
    value = task.get("Answerability") or task.get("answerability") or ["ANSWERABLE"]
    return value[0] if isinstance(value, list) else value


def _value(row: dict, name: str) -> float:
    entry = row["annotations"][name]
    return float(next(iter(entry.values()))["value"])


def _save_report(cfg: Config, name: str, report: dict) -> None:
    path = cfg.path("runs") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def metric_check(cfg: Config, sample: int, bf16: bool, log: Callable[[str], None] = print) -> dict:
    from turnrag.scoring import Algorithmic

    documents, tasks, evaluations = _official_eval(cfg)
    rng = np.random.default_rng(0)
    picked = [evaluations[i] for i in rng.choice(len(evaluations), size=sample, replace=False)]
    rows = []
    for row in picked:
        task = tasks[row["task_id"]]
        rows.append(
            {
                "prediction": row["model_response"],
                "reference": task["targets"][0]["text"],
                "passages": [documents[c["document_id"]] for c in task["contexts"]],
            }
        )
    started = time.time()
    ours = Algorithmic(cfg, bf16=bf16).score(rows, log)
    seconds = time.time() - started
    report = {"sample": sample, "bf16": bf16, "seconds": round(seconds, 1)}
    pairs = (("rouge_l", "RougeL"), ("bert_recall", "Bert-Rec"), ("bert_k_precision", "Bert-KPrec"))
    for mine, theirs in pairs:
        diff = np.abs(
            np.asarray([o[mine] for o in ours]) - np.asarray([_value(r, theirs) for r in picked])
        )
        report[f"{mine}_max_abs_diff"] = float(diff.max())
        report[f"{mine}_mean_abs_diff"] = float(diff.mean())
    answered = [
        (o["rb_alg"], _value(r, "rb_agg"))
        for o, r in zip(ours, picked, strict=True)
        if _value(r, "conditional_idk") == 1
        and _label(tasks[r["task_id"]]) in ("ANSWERABLE", "PARTIAL")
    ]
    diff = np.abs(np.asarray([a for a, _ in answered]) - np.asarray([b for _, b in answered]))
    report["rb_alg_max_abs_diff"] = float(diff.max()) if len(diff) else float("nan")
    report["rb_alg_checked"] = len(answered)
    _save_report(cfg, "check-metrics.json", report)
    log(json.dumps(report, indent=2))
    return report


def judge_check(cfg: Config, sample: int, log: Callable[[str], None] = print) -> dict:
    from turnrag.scoring import Judges

    _, tasks, evaluations = _official_eval(cfg)
    rng = np.random.default_rng(1)
    picked = [evaluations[i] for i in rng.choice(len(evaluations), size=sample, replace=False)]
    judges = Judges(cfg, client(cfg))
    official, ours = [], []
    started = time.time()
    for count, row in enumerate(picked, start=1):
        task = tasks[row["task_id"]]
        flag = _value(row, "conditional_idk")
        unanswerable = _label(task) in ("UNANSWERABLE", "CONVERSATIONAL")
        official.append(flag == 1 if unanswerable else flag == 0)
        ours.append(judges.idk(task["input"], row["model_response"]) == 1)
        if count % 50 == 0:
            log(f"judge check: {count}/{sample} ({time.time() - started:.0f}s)")
    a, b = np.asarray(official), np.asarray(ours)
    agreement = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    report = {
        "sample": sample,
        "official_idk_rate": float(a.mean()),
        "local_idk_rate": float(b.mean()),
        "agreement": agreement,
        "kappa": (agreement - expected) / (1 - expected) if expected < 1 else float("nan"),
        "seconds": round(time.time() - started, 1),
    }
    _save_report(cfg, "check-judge.json", report)
    log(json.dumps(report, indent=2))
    return report


def judge_answers(cfg: Config, split: str, log: Callable[[str], None] = print) -> list[dict]:
    from turnrag.scoring import Judges

    path = cfg.path("runs") / split / "judged.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                done[row["task_id"]] = row
    rows = [
        json.loads(line)
        for line in answers_path(cfg, split).read_text(encoding="utf-8").splitlines()
        if line
    ]
    judges = Judges(cfg, client(cfg))
    started = time.time()
    with path.open("a", encoding="utf-8") as handle:
        for count, row in enumerate(rows, start=1):
            if row["task_id"] in done:
                continue
            text = row["predictions"][0]["text"]
            row["idk"] = judges.idk(row["input"], text)
            if row["answerability"] == "UNDERSPECIFIED" or row["action"] == "clarify":
                row["clarify"] = judges.clarify(text)
            done[row["task_id"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if count % 50 == 0:
                log(f"{split} judges: {count}/{len(rows)} ({time.time() - started:.0f}s)")
    return [done[row["task_id"]] for row in rows]


def score_answers(cfg: Config, split: str, bf16: bool, log: Callable[[str], None] = print) -> dict:
    from turnrag.metrics import conditioned
    from turnrag.scoring import Algorithmic

    path = cfg.path("runs") / split / "judged.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    needed = [
        row for row in rows if row["answerability"] in ("ANSWERABLE", "PARTIAL") and row["idk"] != 1
    ]
    algorithmic = Algorithmic(cfg, bf16=bf16).score(
        [
            {
                "prediction": row["predictions"][0]["text"],
                "reference": row["reference"],
                "passages": [c["text"] for c in row["contexts"]],
            }
            for row in needed
        ],
        log,
    )
    for row, values in zip(needed, algorithmic, strict=True):
        row.update(values)
    for row in rows:
        row["score"] = conditioned(
            row["answerability"], row["idk"], row.get("clarify", 0.0), row.get("rb_alg", 0.0)
        )
    report = {
        "tasks": len(rows),
        "bf16": bf16,
        "conditioned_rb_alg": mean(r["score"] for r in rows),
    }
    for key in ("answerability", "collection", "action"):
        groups: dict[str, list[float]] = {}
        for row in rows:
            groups.setdefault(row[key], []).append(row["score"])
        report[f"by_{key}"] = {
            name: {"n": len(v), "score": mean(v)} for name, v in sorted(groups.items())
        }
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = confusion.setdefault(row["answerability"], {})
        cell[row["action"]] = cell.get(row["action"], 0) + 1
    report["actions"] = confusion
    report["answered_rb_alg"] = mean(row["rb_alg"] for row in needed) if needed else 0.0
    out = cfg.path("runs") / split
    (out / "scores.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (out / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps({k: v for k, v in report.items() if not k.startswith("by_")}, indent=2))
    return report
