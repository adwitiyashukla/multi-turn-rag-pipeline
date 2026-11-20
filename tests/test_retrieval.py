import numpy as np
from scipy import sparse

from turnrag.build import build_bm25, build_docstore, index_dir
from turnrag.data import Turn
from turnrag.encoders import SparseQueryEncoder
from turnrag.index import ImpactIndex
from turnrag.retrieve import RerankCache, Retriever
from turnrag.rewrite import Variants, basic, context_queries, messages, rewrite, transcript


class FakeReranker:
    def score(self, query, passages):
        wanted = set(query.lower().split())
        return np.asarray(
            [len(wanted & set(p.lower().replace(".", "").split())) for p in passages],
            dtype=np.float32,
        )


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def scored(self, chat, max_tokens, schema, top=5):
        self.seen.append(chat)
        head, tail = self.reply.split('"vague":')
        tokens = [
            (head + '"vague":', {}),
            (" false", {" false": -0.2, " true": -1.8}),
            (tail.replace(" false", ""), {}),
        ]
        return self.reply, tokens


def build(cfg):
    store = build_docstore(cfg, "fiqa")
    build_bm25(cfg, "fiqa", store, lambda message: None)
    encoder = SparseQueryEncoder(cfg.model_dir(cfg.sparse.model))
    rows, cols, vals = [], [], []
    for doc in range(len(store)):
        for token in set(encoder.tokenizer.encode(store.text(doc)).ids):
            if encoder.weights[token] > 0:
                rows.append(doc)
                cols.append(token)
                vals.append(1.0)
    matrix = sparse.csc_matrix((vals, (rows, cols)), shape=(len(store), encoder.weights.size))
    ImpactIndex(matrix).save(index_dir(cfg, "fiqa", "sparse"))
    (index_dir(cfg, "fiqa", "sparse") / "manifest.json").write_text("{}")
    engine = Retriever(cfg, ["fiqa"], reranker=False, cache=RerankCache(None))
    engine.reranker = FakeReranker()
    return engine


def test_context_weights_decay():
    turns = [
        Turn("user", "a"),
        Turn("agent", "x"),
        Turn("user", "b"),
        Turn("agent", "y"),
        Turn("user", "c"),
    ]
    assert context_queries(turns, 2, 0.5) == (("b", 0.5), ("c", 1.0))
    assert "agent: x" in transcript(turns, 6, 10)


def test_rewrite_parses_reply():
    turns = [
        Turn("user", "Tell me about Juno"),
        Turn("agent", "It is a probe."),
        Turn("user", "when did it arrive"),
    ]
    client = FakeClient(
        '{"query": "When did Juno reach Jupiter?", "keywords": "Juno Jupiter", "vague": false}'
    )
    value = rewrite(client, turns, 3, 0.5, 6, 60, 64)
    assert value.rewrite == "When did Juno reach Jupiter?"
    assert value.keywords == "Juno Jupiter"
    assert abs(value.vague - 0.1680) < 1e-3
    assert client.seen[0][-1]["content"].endswith("user: when did it arrive")
    assert len(messages("user: hi")) == 10


def test_fusion_and_rerank(cfg):
    engine = build(cfg)
    value = Variants(
        "when did it arrive",
        (("Juno probe", 0.5), ("when did it arrive", 1.0)),
        "When did Juno reach Jupiter",
        "Juno Jupiter",
    )
    fused = engine.fuse("fiqa", value)
    assert fused[0].fused >= fused[-1].fused
    top = {c.index for c in fused[:2]}
    assert top == {0, 4}
    ranked = engine.search("fiqa", value, k=3)
    assert ranked[0].doc_id in {"d1", "d5"}
    assert ranked[0].rerank is not None
    plain = engine.search("fiqa", basic([Turn("user", "Roth IRA tax")], 3, 0.5), k=2, rerank=False)
    assert plain[0].doc_id == "d3"


def test_rerank_cache_is_reused(cfg, tmp_path):
    engine = build(cfg)
    engine.cache = RerankCache(tmp_path / "rerank.jsonl")
    value = basic([Turn("user", "Arizona Cardinals stadium")], 3, 0.5)
    first = engine.search("fiqa", value, k=2)
    engine.reranker = None
    engine.reranker = FakeReranker()
    again = RerankCache(tmp_path / "rerank.jsonl")
    assert len(again.items) >= 1
    assert first[0].doc_id == "d4"


class ScriptedClient:
    def __init__(self, answer, vague):
        self.answer = answer
        self.vague = vague

    def scored(self, chat, max_tokens, schema, top=5):
        if schema is None:
            return "no", [("no", {"no": -0.1, "yes": -2.5})]
        flag = " true" if self.vague else " false"
        reply = '{"query": "Where do the Cardinals play?", "keywords": "Cardinals", "vague":'
        tokens = [(reply, {}), (flag, {" true": -0.1, " false": -2.4}), ("}", {})]
        if not self.vague:
            tokens[1] = (flag, {" true": -2.4, " false": -0.1})
        return reply + flag + "}", tokens

    def chat(self, chat, max_tokens, schema=None):
        return self.answer

    def stream(self, chat, max_tokens, top=0):
        for word in self.answer.split(" "):
            yield word + " ", None


def test_pipeline_actions(cfg):
    from turnrag.answer import FEATURES, Gate
    from turnrag.pipeline import Pipeline

    engine = build(cfg)
    turns = [Turn("user", "Where do the Arizona Cardinals play?")]
    size = len(FEATURES)
    gate = Gate([0.0] * size, 0.0, [0.0] * size, [1.0] * size, 0.0, 0.5)
    answer = Pipeline(cfg, engine, ScriptedClient("They play at State Farm Stadium.", False), gate)
    reply = answer.respond("fiqa", turns)
    assert reply.action == "answer"
    assert reply.passages[0]["document_id"] == "d4"
    assert reply.rewrite == "Where do the Cardinals play?"
    unknown = Pipeline(cfg, engine, ScriptedClient("NO_ANSWER", False), gate)
    reply = unknown.respond("fiqa", turns)
    assert reply.action == "idk" and reply.text == cfg.answer.idk_text
    vague = Pipeline(cfg, engine, ScriptedClient("Which team do you mean?", True), gate)
    events = list(vague.stream("fiqa", turns))
    assert events[0]["type"] == "plan" and events[0]["action"] == "clarify"
    assert events[-1]["type"] == "done" and events[-1]["text"] == "Which team do you mean?"
    strict = Gate([0.0] * size, 0.0, [0.0] * size, [1.0] * size, 0.9, 1.0)
    refused = Pipeline(
        cfg, engine, ScriptedClient("They play at State Farm Stadium.", False), strict
    )
    assert refused.respond("fiqa", turns).action == "idk"
