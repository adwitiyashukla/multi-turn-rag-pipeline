from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from turnrag.config import Config
from turnrag.data import Turn


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    collection: str


class Sessions:
    def __init__(self, limit: int, ttl: float) -> None:
        self.limit = limit
        self.ttl = ttl
        self.lock = threading.Lock()
        self.items: OrderedDict[str, tuple[float, list[Turn]]] = OrderedDict()

    def create(self) -> str:
        key = uuid.uuid4().hex
        with self.lock:
            self._expire()
            self.items[key] = (time.time(), [])
            while len(self.items) > self.limit:
                self.items.popitem(last=False)
        return key

    def turns(self, key: str) -> list[Turn]:
        with self.lock:
            self._expire()
            if key not in self.items:
                raise KeyError(key)
            _, turns = self.items.pop(key)
            self.items[key] = (time.time(), turns)
            return turns

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for key in [k for k, (seen, _) in self.items.items() if seen < cutoff]:
            del self.items[key]


def create_app(cfg: Config, engine=None) -> FastAPI:
    app = FastAPI(title="turnrag")
    sessions = Sessions(cfg.api.max_sessions, cfg.api.session_ttl)
    state = {"engine": engine}

    def pipeline():
        if state["engine"] is None:
            from turnrag.evaluate import pipeline as build

            state["engine"] = build(cfg)
        return state["engine"]

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return resources.files("turnrag").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/collections")
    def collections() -> dict:
        return {"collections": list(cfg.data.collections)}

    @app.post("/sessions")
    def new_session() -> dict:
        return {"session_id": sessions.create()}

    @app.get("/sessions/{session_id}")
    def history(session_id: str) -> dict:
        try:
            turns = sessions.turns(session_id)
        except KeyError as missing:
            raise HTTPException(404, "unknown session") from missing
        return {"turns": [{"speaker": t.speaker, "text": t.text} for t in turns]}

    @app.post("/sessions/{session_id}/messages")
    def send(session_id: str, body: Message) -> StreamingResponse:
        if body.collection not in cfg.data.collections:
            raise HTTPException(422, "unknown collection")
        try:
            turns = sessions.turns(session_id)
        except KeyError as missing:
            raise HTTPException(404, "unknown session") from missing
        turns.append(Turn("user", body.message.strip()))
        snapshot = list(turns)

        def events():
            text = ""
            for event in pipeline().stream(body.collection, snapshot):
                if event["type"] == "done":
                    text = event["text"]
                yield f"data: {json.dumps(event)}\n\n"
            turns.append(Turn("agent", text))

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
