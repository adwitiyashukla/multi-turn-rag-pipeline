from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from turnrag.config import LLM

THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean(text: str) -> str:
    return THINK.sub("", text).replace("<think>", "").replace("</think>", "").strip()


class Cache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.items: dict[str, str] = {}
        if path and path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.items[row["key"]] = row["value"]

    def get(self, key: str) -> str | None:
        return self.items.get(key)

    def put(self, key: str, value: str) -> None:
        with self.lock:
            self.items[key] = value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "value": value}) + "\n")


class Client:
    def __init__(self, settings: LLM, cache: Cache | None = None) -> None:
        self.settings = settings
        self.cache = cache or Cache(None)
        self.http = httpx.Client(base_url=settings.base_url, timeout=settings.timeout)

    def _payload(self, messages: list[dict], max_tokens: int, schema: dict | None) -> dict:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.settings.temperature,
            "seed": self.settings.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema, "strict": True},
            }
        return payload

    def chat(self, messages: list[dict], max_tokens: int, schema: dict | None = None) -> str:
        payload = self._payload(messages, max_tokens, schema)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                response = self.http.post("/chat/completions", json=payload)
                response.raise_for_status()
                text = clean(response.json()["choices"][0]["message"].get("content") or "")
                self.cache.put(key, text)
                return text
            except (httpx.HTTPError, KeyError, ValueError) as caught:
                error = caught
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"llm request failed: {error}")

    def scored(
        self, messages: list[dict], max_tokens: int, schema: dict | None, top: int = 5
    ) -> tuple[str, list[tuple[str, dict[str, float]]]]:
        payload = self._payload(messages, max_tokens, schema)
        payload["logprobs"] = True
        payload["top_logprobs"] = top
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is None:
            error: Exception | None = None
            for attempt in range(self.settings.retries + 1):
                try:
                    response = self.http.post("/chat/completions", json=payload)
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    tokens = [
                        [
                            item["token"],
                            {alt["token"]: alt["logprob"] for alt in item["top_logprobs"]},
                        ]
                        for item in (choice.get("logprobs") or {}).get("content") or []
                    ]
                    text = clean(choice["message"].get("content") or "")
                    cached = json.dumps({"text": text, "tokens": tokens})
                    self.cache.put(key, cached)
                    break
                except (httpx.HTTPError, KeyError, ValueError) as caught:
                    error = caught
                    time.sleep(2 * (attempt + 1))
            if cached is None:
                raise RuntimeError(f"llm request failed: {error}")
        row = json.loads(cached)
        return row["text"], [(token, alts) for token, alts in row["tokens"]]

    def stream(
        self, messages: list[dict], max_tokens: int, top: int = 0
    ) -> Iterator[tuple[str, dict[str, float] | None]]:
        payload = self._payload(messages, max_tokens, None)
        payload["stream"] = True
        if top:
            payload["logprobs"] = True
            payload["top_logprobs"] = top
        inside = False
        first: dict[str, float] | None = None
        with self.http.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                choice = json.loads(body)["choices"][0]
                delta = choice.get("delta", {}).get("content") or ""
                if top and first is None:
                    content = (choice.get("logprobs") or {}).get("content") or []
                    if content:
                        first = {alt["token"]: alt["logprob"] for alt in content[0]["top_logprobs"]}
                if "<think>" in delta:
                    inside = True
                if inside:
                    if "</think>" in delta:
                        inside = False
                        delta = delta.split("</think>", 1)[1]
                    else:
                        continue
                if delta:
                    yield delta, first

    def json(self, messages: list[dict], max_tokens: int, schema: dict) -> dict:
        text = self.chat(messages, max_tokens, schema)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}
