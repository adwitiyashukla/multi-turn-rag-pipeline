from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from turnrag.data import Turn
from turnrag.llm import Client
from turnrag.text import normalize, truncate_words

SYSTEM = """You help a search engine understand the last user message of a conversation. Reply with JSON only, with three fields.
"query": the last user message rewritten as one standalone question that makes sense without the conversation. Replace words like it, they, this or that with what they refer to, and bring in the topic from earlier turns when the last message is short. Keep the user's own words where you can. Do not answer the question.
"keywords": 3 to 8 search keywords or names for the question, separated by spaces.
"vague": look at your rewritten query. Set true only when that query still cannot be answered without asking the user which specific thing they mean, because it points to one member of a group (for example "the bank", "that chain", "this phone") and the conversation never says which member. Short follow-ups that the conversation explains are not vague. Most questions are not vague."""

EXAMPLES = (
    (
        "user: What is the Score Agreement level if it is less than 0?\n"
        "agent: If the Score Agreement level is less than 0, it indicates poor agreement "
        "among the annotators.\n"
        "user: How many annotators are needed?\n"
        "agent: Inter-annotator agreement scores cannot be calculated unless two or more "
        "human annotators annotate the same documents.\n"
        "user: data volume",
        '{"query": "How does the data volume affect the number of annotators needed?", '
        '"keywords": "data volume number of annotators inter-annotator agreement", '
        '"vague": false}',
    ),
    (
        "user: How is federal income tax calculated?\n"
        "agent: Income tax is based on your taxable income, which is your gross income "
        "minus deductions, taxed at bracket rates.\n"
        "user: deductions",
        '{"query": "What deductions lower taxable income for federal income tax?", '
        '"keywords": "income tax deductions taxable income", "vague": false}',
    ),
    (
        "user: F-1 student investing in foreign markets\n"
        "agent: It is legal for an F-1 student to trade stocks. If you are a non-resident "
        "alien for tax purposes you file Form 1040NR.\n"
        "user: Can I not use the 1040NR-EZ?",
        '{"query": "Can an F-1 student use the 1040NR-EZ instead of the 1040NR for taxes?", '
        '"keywords": "F-1 student 1040NR-EZ 1040NR non-resident alien tax form", '
        '"vague": false}',
    ),
    (
        "user: I keep seeing store credit cards at checkout.\n"
        "agent: Many large retailers offer store credit cards with discounts on purchases "
        "at that store.\n"
        "user: What is the interest rate on it?",
        '{"query": "What is the interest rate on the store credit card?", '
        '"keywords": "store credit card interest rate APR", "vague": true}',
    ),
)

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "keywords": {"type": "string"},
        "vague": {"type": "boolean"},
    },
    "required": ["query", "keywords", "vague"],
}


@dataclass(frozen=True)
class Variants:
    last: str
    context: tuple[tuple[str, float], ...]
    rewrite: str | None = None
    keywords: str | None = None
    vague: float = 0.0

    def weighted(self, name: str) -> tuple[tuple[str, float], ...]:
        if name == "last":
            return ((self.last, 1.0),)
        if name == "context":
            return self.context
        value = getattr(self, name)
        return ((value, 1.0),) if value else ()

    @property
    def standalone(self) -> str:
        return self.rewrite or " ".join(text for text, _ in self.context)


def user_turns(turns: Sequence[Turn]) -> list[str]:
    return [turn.text for turn in turns if turn.speaker == "user"]


def context_queries(
    turns: Sequence[Turn], size: int, decay: float
) -> tuple[tuple[str, float], ...]:
    recent = user_turns(turns)[-size:]
    return tuple(
        (text, decay ** (len(recent) - 1 - position)) for position, text in enumerate(recent)
    )


def transcript(turns: Sequence[Turn], history_turns: int, agent_words: int) -> str:
    lines = []
    for turn in list(turns)[-history_turns:]:
        text = normalize(turn.text)
        if turn.speaker != "user" and agent_words:
            text = truncate_words(text, agent_words)
        lines.append(f"{turn.speaker}: {text}")
    return "\n".join(lines)


def messages(conversation: str) -> list[dict]:
    chat = [{"role": "system", "content": SYSTEM}]
    for example, reply in EXAMPLES:
        chat.append({"role": "user", "content": example})
        chat.append({"role": "assistant", "content": reply})
    chat.append({"role": "user", "content": conversation})
    return chat


def basic(turns: Sequence[Turn], size: int, decay: float) -> Variants:
    return Variants(turns[-1].text, context_queries(turns, size, decay))


def rewrite(
    client: Client,
    turns: Sequence[Turn],
    size: int,
    decay: float,
    history_turns: int,
    agent_words: int,
    max_tokens: int,
) -> Variants:
    base = basic(turns, size, decay)
    conversation = transcript(turns, history_turns, agent_words)
    text, tokens = client.scored(messages(conversation), max_tokens, SCHEMA)
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        reply = {}
    query = normalize(str(reply.get("query") or "")) or None
    keywords = normalize(str(reply.get("keywords") or "")) or None
    return Variants(base.last, base.context, query, keywords, vague_probability(tokens, reply))


def vague_probability(tokens: Sequence[tuple[str, dict[str, float]]], reply: dict) -> float:
    seen = ""
    for token, alternatives in tokens:
        if seen.rstrip().endswith('"vague":'):
            scores = {}
            for alt, logprob in alternatives.items():
                word = alt.strip().strip('"').lower()
                if word in ("true", "false"):
                    scores[word] = max(scores.get(word, -1e9), logprob)
            if "true" in scores and "false" in scores:
                high = max(scores.values())
                yes = math.exp(scores["true"] - high)
                return yes / (yes + math.exp(scores["false"] - high))
            if scores:
                return 1.0 if "true" in scores else 0.0
        seen += token
    return 1.0 if reply.get("vague") is True else 0.0
