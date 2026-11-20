from pathlib import Path

import pytest
from pydantic import ValidationError

from turnrag.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_is_valid():
    cfg = load_config(ROOT / "config.toml", environ={})
    assert cfg.rerank.candidates <= cfg.retrieval.depth
    assert set(cfg.retrieval.legs) <= {"bm25", "sparse"}


def test_environment_override():
    cfg = load_config(
        ROOT / "config.toml",
        environ={"TURNRAG__LLM__BASE_URL": "http://x:1/v1", "TURNRAG__SPARSE__THREADS": "2"},
    )
    assert cfg.llm.base_url == "http://x:1/v1"
    assert cfg.sparse.threads == 2


@pytest.mark.parametrize(
    "old,new",
    [
        ("bm25 = 0.25\nsparse = 1.0", "bm25 = -1.0\nsparse = 1.0"),
        ("bm25 = 0.25\nsparse = 1.0", "bm25 = 0.0\nsparse = 0.0"),
        ("candidates = 30", "candidates = 1000"),
        ('activation = "log1p"', 'activation = "relu"'),
        ("k1 = 1.2", "k1 = 1.2\nk3 = 7"),
    ],
)
def test_bad_values_are_rejected(tmp_path, old, new):
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    assert old in text
    (tmp_path / "config.toml").write_text(text.replace(old, new, 1))
    with pytest.raises(ValidationError):
        load_config(tmp_path / "config.toml", environ={})
