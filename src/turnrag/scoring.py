from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

from turnrag.config import Config
from turnrag.data import raw_path
from turnrag.llm import Cache, Client
from turnrag.metrics import conditioned, rb_alg

FIRST_WORD = re.compile(r"\s*([A-Za-z]+)")
DECISION = re.compile(r'"decision"\s*:\s*"([A-Za-z]+)"')


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise ValueError(f"{name} not found in {path}")


def official_prompts(cfg: Config) -> tuple[str, str]:
    folder = raw_path(cfg, "scripts/evaluation")
    idk_source = _function_source(folder / "judge_utils.py", "format_idk_judge")
    start = idk_source.index('f"""') + 4
    idk = idk_source[start : idk_source.index('"""', start)]
    tree = ast.parse((folder / "underspecified_eval.py").read_text(encoding="utf-8"))
    clarify = ""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "UNDERSPECIFIED_PROMPT"
        ):
            clarify = ast.literal_eval(node.value)
    if "{inquiry}" not in idk or "{{ response }}" not in clarify:
        raise ValueError("official judge prompts changed")
    return idk, clarify


def inquiry(turns: Sequence[dict]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns)


class Judges:
    def __init__(self, cfg: Config, client: Client) -> None:
        self.client = client
        self.max_tokens = cfg.eval.judge_max_tokens
        self.idk_prompt, self.clarify_prompt = official_prompts(cfg)

    def idk(self, turns: Sequence[dict], response: str) -> float:
        prompt = self.idk_prompt.replace("{inquiry}", inquiry(turns)).replace(
            "{response}", response
        )
        reply = self.client.chat([{"role": "user", "content": prompt}], 4)
        match = FIRST_WORD.match(reply)
        token = match.group(1).lower() if match else ""
        return 1.0 if token == "yes" else 0.0 if token == "no" else 0.5

    def clarify(self, response: str) -> float:
        prompt = self.clarify_prompt.replace("{{ response }}", response)
        reply = self.client.chat([{"role": "user", "content": prompt}], self.max_tokens)
        match = DECISION.search(reply)
        return 1.0 if match and match.group(1).lower() == "yes" else 0.0


class Algorithmic:
    def __init__(self, cfg: Config, bf16: bool = False) -> None:
        import bert_score
        import torch
        from rouge_score import rouge_scorer

        folder = cfg.model_dir(cfg.eval.bertscore_model)
        baseline = (
            Path(bert_score.__file__).parent
            / "rescale_baseline"
            / "en"
            / "microsoft"
            / "deberta-xlarge-mnli.tsv"
        )
        scorer = bert_score.BERTScorer(
            model_type=str(folder),
            num_layers=cfg.eval.bertscore_layer,
            lang="en",
            rescale_with_baseline=True,
            baseline_path=str(baseline),
            device="cpu",
        )
        scorer._tokenizer.model_max_length = 512
        if bf16:
            scorer._model.to(torch.bfloat16)
        self.torch = torch
        self.tokenizer = scorer._tokenizer
        self.model = scorer._model
        self.base_precision, self.base_recall = (float(v) for v in scorer.baseline_vals[:2])
        self.special = {self.tokenizer.sep_token_id, self.tokenizer.cls_token_id}
        self.texts: OrderedDict[str, tuple] = OrderedDict()
        self.size = cfg.eval.cached_texts
        self.tag = f"{cfg.eval.bertscore_model}:{cfg.eval.bertscore_layer}:{bf16}"
        self.cache = Cache(cfg.path("cache") / f"rbalg-{cfg.eval.bertscore_model}.jsonl")
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def rouge_l(self, prediction: str, target: str) -> float:
        return float(self.rouge.score(target, prediction)["rougeL"].fmeasure)

    def embed(self, text: str) -> tuple:
        from bert_score.utils import bert_encode, sent_encode

        if text in self.texts:
            self.texts.move_to_end(text)
            return self.texts[text]
        ids = sent_encode(self.tokenizer, text)
        x = self.torch.tensor([ids])
        vectors = bert_encode(self.model, x, self.torch.ones_like(x))[0].float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        weights = self.torch.tensor([0.0 if t in self.special else 1.0 for t in ids])
        total = float(weights.sum())
        entry = (vectors, weights / total if total > 0 else weights)
        self.texts[text] = entry
        while len(self.texts) > self.size:
            self.texts.popitem(last=False)
        return entry

    def bert(self, candidate: str, reference: str) -> tuple[float, float]:
        hyp, hyp_weights = self.embed(candidate)
        ref, ref_weights = self.embed(reference)
        precision = recall = 0.0
        if float(hyp_weights.sum()) > 0 and float(ref_weights.sum()) > 0:
            sim = hyp @ ref.T
            precision = float((sim.max(dim=1).values * hyp_weights).sum())
            recall = float((sim.max(dim=0).values * ref_weights).sum())
        return (
            (precision - self.base_precision) / (1 - self.base_precision),
            (recall - self.base_recall) / (1 - self.base_recall),
        )

    def _row(self, row: dict) -> dict:
        prediction = row["prediction"]
        _, recall = self.bert(prediction, row["reference"])
        best = max((self.bert(prediction, p)[0] for p in row["passages"]), default=-1.0)
        rouge = self.rouge_l(prediction, row["reference"])
        return {
            "bert_recall": recall,
            "rouge_l": rouge,
            "bert_k_precision": best,
            "rb_alg": rb_alg(recall, rouge, best) if row["passages"] else 0.0,
        }

    def score(self, rows: Sequence[dict], log: Callable[[str], None] | None = None) -> list[dict]:
        results = []
        started = time.time()
        for count, row in enumerate(rows, start=1):
            payload = [self.tag, row["prediction"], row["reference"], row["passages"]]
            key = hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
            cached = self.cache.get(key)
            if cached is None:
                cached = json.dumps(self._row(row))
                self.cache.put(key, cached)
            results.append(json.loads(cached))
            if log and (count % 10 == 0 or count == len(rows)):
                log(f"rb_alg: {count}/{len(rows)} ({time.time() - started:.0f}s)")
        return results


def conditioned_rows(rows: Sequence[dict]) -> list[float]:
    return [
        conditioned(row["answerability"], row["idk"], row.get("clarify", 0.0), row["rb_alg"])
        for row in rows
    ]
