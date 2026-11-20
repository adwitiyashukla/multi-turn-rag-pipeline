import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from turnrag.config import load_config

CORPUS = [
    {
        "_id": "d1",
        "title": "Juno",
        "text": "Juno is a NASA probe that reached Jupiter in 2016.",
        "url": "",
    },
    {
        "_id": "d2",
        "title": "Great Wall",
        "text": "The Great Wall of China was built against nomads.",
        "url": "",
    },
    {
        "_id": "d3",
        "title": "Roth IRA",
        "text": "Roth IRA withdrawals are tax free after age 59.",
        "url": "",
    },
    {
        "_id": "d4",
        "title": "Cardinals",
        "text": "The Arizona Cardinals play at State Farm Stadium.",
        "url": "",
    },
    {
        "_id": "d5",
        "title": "Jupiter",
        "text": "Jupiter is the largest planet and Juno studies it.",
        "url": "",
    },
]


def write_tokenizer(folder: Path) -> None:
    words = sorted({w.lower().strip(".,") for row in CORPUS for w in row["text"].split()})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word in words + ["where", "does", "play", "probe", "tax"]:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.normalizer = None
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "idf.json").write_text(
        json.dumps({word: 1.0 + i % 3 for i, word in enumerate(vocab)})
    )
    (folder / "special_tokens_map.json").write_text(
        json.dumps(
            {"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]"}
        )
    )
    (folder / "config.json").write_text(json.dumps({"pad_token_id": 0}))


@pytest.fixture
def cfg(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    text = (root / "config.toml").read_text(encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        text.replace('collections = ["clapnq", "cloud", "fiqa", "govt"]', 'collections = ["fiqa"]')
    )
    corpus = tmp_path / "data" / "corpora"
    corpus.mkdir(parents=True)
    (corpus / "fiqa.jsonl").write_text("\n".join(json.dumps(row) for row in CORPUS) + "\n")
    write_tokenizer(tmp_path / "models" / "opensearch-sparse-doc-v2-mini")
    return load_config(tmp_path / "config.toml", environ={})
