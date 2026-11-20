import json

from fastapi.testclient import TestClient

from turnrag.api import create_app


class FakeEngine:
    def __init__(self):
        self.calls = []

    def stream(self, collection, turns):
        self.calls.append((collection, [t.text for t in turns]))
        yield {
            "type": "plan",
            "rewrite": "q",
            "action": "answer",
            "probability": 0.9,
            "passages": [],
        }
        yield {"type": "token", "text": "hello"}
        yield {"type": "done", "action": "answer", "text": "hello", "citations": [], "timings": {}}


def events(response):
    return [
        json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")
    ]


def test_chat_flow(cfg):
    engine = FakeEngine()
    client = TestClient(create_app(cfg, engine))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/collections").json() == {"collections": ["fiqa"]}
    assert "turnrag chat" in client.get("/").text
    session = client.post("/sessions").json()["session_id"]
    first = client.post(
        f"/sessions/{session}/messages", json={"message": "hi there", "collection": "fiqa"}
    )
    assert [e["type"] for e in events(first)] == ["plan", "token", "done"]
    client.post(f"/sessions/{session}/messages", json={"message": "and then", "collection": "fiqa"})
    assert engine.calls[-1][1] == ["hi there", "hello", "and then"]
    turns = client.get(f"/sessions/{session}").json()["turns"]
    assert [t["speaker"] for t in turns] == ["user", "agent", "user", "agent"]


def test_bad_requests(cfg):
    client = TestClient(create_app(cfg, FakeEngine()))
    assert (
        client.post(
            "/sessions/nope/messages", json={"message": "x", "collection": "fiqa"}
        ).status_code
        == 404
    )
    session = client.post("/sessions").json()["session_id"]
    assert (
        client.post(
            f"/sessions/{session}/messages", json={"message": "x", "collection": "nope"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/sessions/{session}/messages", json={"message": "", "collection": "fiqa"}
        ).status_code
        == 422
    )
