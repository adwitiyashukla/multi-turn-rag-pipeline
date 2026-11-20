from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from turnrag import answer as answers
from turnrag.answer import Gate
from turnrag.config import Config
from turnrag.data import Turn
from turnrag.llm import Client
from turnrag.retrieve import Ranked, Retriever
from turnrag.rewrite import Variants, basic, rewrite


@dataclass
class Plan:
    collection: str
    variants: Variants
    ranked: list[Ranked]
    row: np.ndarray
    action: str
    probability: float
    timings: dict[str, float] = field(default_factory=dict)
    probe_no: float = 0.5


@dataclass
class Response:
    action: str
    text: str
    passages: list[dict]
    citations: list[list[int]]
    probability: float
    rewrite: str | None
    timings: dict[str, float]
    features: list[float]
    probe_no: float


class Pipeline:
    def __init__(
        self, cfg: Config, retriever: Retriever, client: Client, gate: Gate | None
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.client = client
        self.gate = gate
        self.idf = {
            name: answers.idf_vector(shelf.bm25) for name, shelf in retriever.shelves.items()
        }

    def variants(self, turns: Sequence[Turn]) -> Variants:
        r = self.cfg.retrieval
        if (
            self.cfg.retrieval.queries.get("rewrite", 0) <= 0
            and self.cfg.retrieval.queries.get("keywords", 0) <= 0
        ):
            return basic(turns, r.context_turns, r.context_decay)
        w = self.cfg.rewrite
        return rewrite(
            self.client,
            turns,
            r.context_turns,
            r.context_decay,
            w.history_turns,
            w.agent_words,
            w.max_tokens,
        )

    def plan(
        self, collection: str, turns: Sequence[Turn], variants: Variants | None = None
    ) -> Plan:
        timings = {}
        started = time.perf_counter()
        variants = variants or self.variants(turns)
        timings["rewrite"] = time.perf_counter() - started
        started = time.perf_counter()
        ranked = self.retriever.search(collection, variants, k=self.cfg.rerank.candidates)
        timings["retrieve"] = time.perf_counter() - started
        row = answers.features(variants, ranked, turns)
        gated = self.gate is not None and self.cfg.gate.enabled
        if not ranked:
            action = "idk"
        elif gated and self.cfg.gate.clarify and self.gate.clarifies(variants.vague):
            action = "clarify"
        else:
            action = "answer"
        return Plan(collection, variants, ranked, row, action, 1.0, timings)

    def decide(self, plan: Plan, chat: list[dict]) -> None:
        _, tokens = self.client.scored(answers.probe_messages(chat), 1, None, top=10)
        plan.probe_no = answers.probe_no(tokens[0][1] if tokens else None)
        if self.gate is not None and self.cfg.gate.enabled:
            plan.action, plan.probability = self.gate.decide(np.append(plan.row, plan.probe_no))

    def _chunks(self, plan: Plan) -> list[tuple[int, str]]:
        shelf = self.retriever.shelves[plan.collection]
        return answers.evidence(
            shelf.store,
            plan.ranked,
            plan.variants.standalone,
            shelf.bm25,
            self.idf[plan.collection],
            self.cfg.answer.passages,
            self.cfg.answer.evidence_tokens,
        )

    def messages(self, plan: Plan, turns: Sequence[Turn]) -> list[dict]:
        shelf = self.retriever.shelves[plan.collection]
        chunks = self._chunks(plan)
        a = self.cfg.answer
        if plan.action == "clarify":
            return answers.clarify_messages(shelf.store, plan.ranked, chunks, turns, a.agent_words)
        return answers.answer_messages(
            shelf.store, plan.ranked, chunks, turns, a.history_turns, a.agent_words, a.idk_text
        )

    def passages(self, plan: Plan) -> list[dict]:
        store = self.retriever.shelves[plan.collection].store
        rows = []
        for hit in plan.ranked[: self.cfg.answer.passages]:
            passage = store[hit.index]
            rows.append(
                {
                    "document_id": hit.doc_id,
                    "title": passage.title,
                    "url": passage.url,
                    "text": passage.text,
                    "score": hit.score,
                }
            )
        return rows

    def finish(self, plan: Plan, text: str) -> Response:
        shelf = self.retriever.shelves[plan.collection]
        action = plan.action
        if action == "answer" and answers.is_idk(text, self.cfg.answer.idk_text):
            action = "idk"
        citations = []
        if action == "answer":
            top = plan.ranked[: self.cfg.answer.passages]
            citations = answers.attribute(
                text, shelf.store, top, shelf.bm25, self.idf[plan.collection]
            )
        return Response(
            action,
            text,
            self.passages(plan),
            citations,
            plan.probability,
            plan.variants.rewrite,
            plan.timings,
            plan.row.tolist(),
            plan.probe_no,
        )

    def respond(
        self, collection: str, turns: Sequence[Turn], variants: Variants | None = None
    ) -> Response:
        plan = self.plan(collection, turns, variants)
        started = time.perf_counter()
        idk = self.cfg.answer.idk_text
        chat = self.messages(plan, turns) if plan.action != "idk" else []
        if plan.action == "answer":
            self.decide(plan, chat)
        if plan.action == "idk":
            text = idk
        else:
            text = self.client.chat(chat, self.cfg.answer.max_tokens)
            if plan.action == "answer" and text.strip().startswith(answers.NO_ANSWER):
                plan.action, text = "idk", idk
        plan.timings["generate"] = time.perf_counter() - started
        return self.finish(plan, text)

    def stream(self, collection: str, turns: Sequence[Turn]) -> Iterator[dict]:
        plan = self.plan(collection, turns)
        chat = self.messages(plan, turns) if plan.action != "idk" else []
        if plan.action == "answer":
            self.decide(plan, chat)
        yield {
            "type": "plan",
            "rewrite": plan.variants.rewrite,
            "action": plan.action,
            "passages": self.passages(plan),
        }
        started = time.perf_counter()
        idk = self.cfg.answer.idk_text
        if plan.action == "idk":
            text = idk
            yield {"type": "token", "text": text}
        else:
            pieces: list[str] = []
            for piece, _ in self.client.stream(chat, self.cfg.answer.max_tokens):
                pieces.append(piece)
                yield {"type": "token", "text": piece}
            text = "".join(pieces).strip()
            if plan.action == "answer" and text.startswith(answers.NO_ANSWER):
                plan.action, text = "idk", idk
        plan.timings["generate"] = time.perf_counter() - started
        response = self.finish(plan, text)
        yield {
            "type": "done",
            "action": response.action,
            "text": response.text,
            "probability": round(response.probability, 4),
            "citations": response.citations,
            "timings": {k: round(v, 3) for k, v in response.timings.items()},
        }
