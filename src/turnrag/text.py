from __future__ import annotations

import re
import unicodedata

import Stemmer

STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)
WORD = re.compile(r"[^\W_]+")
SPACE = re.compile(r"\s+")
POSSESSIVE = re.compile("['\\u2019]s\\b")
BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
LINES = re.compile(r"\s*\n\s*")
STEMMER = Stemmer.Stemmer("english")


def normalize(text: str) -> str:
    return SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def words(text: str) -> list[str]:
    text = POSSESSIVE.sub("", unicodedata.normalize("NFKC", text).lower())
    return WORD.findall(text)


def terms(text: str) -> list[str]:
    return STEMMER.stemWords([word for word in words(text) if word not in STOPWORDS])


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35) + 1


def sentences(text: str, min_words: int = 4) -> list[str]:
    pieces: list[str] = []
    for line in LINES.split(unicodedata.normalize("NFKC", text)):
        line = SPACE.sub(" ", line).strip()
        if line:
            pieces.extend(part.strip() for part in BOUNDARY.split(line) if part.strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece.split()) < min_words:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    if len(merged) > 1 and len(merged[0].split()) < min_words:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged = merged[1:]
    return merged


def truncate_words(text: str, limit: int) -> str:
    parts = text.split()
    if len(parts) <= limit:
        return text
    return " ".join(parts[:limit]) + " ..."
