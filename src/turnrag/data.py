from __future__ import annotations

import csv
import hashlib
import json
import mmap
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from turnrag.config import Config

FILES = {
    "corpora/passage_level/clapnq.jsonl.zip": "3f4bd79341b176070a83b6281bf380d7d6958270c137913f7377121b85a0747e",
    "corpora/passage_level/cloud.jsonl.zip": "a0c5427013c65556aeddfdc246dd6068f3a7fad08ce6ae7adddaa9e87aae246d",
    "corpora/passage_level/fiqa.jsonl.zip": "59f7cf5b043abe007fdb6521eb281bf055524de4376466e9cb4808466e761ee9",
    "corpora/passage_level/govt.jsonl.zip": "09adcc5a1a8d11e362c66d7dbf0ef2ea338df4327d655102bf91ea8e258399f6",
    "mtrag-human/generation_tasks/reference.jsonl": "b3dde4076f9f692a99513e424a0a39f57591dbe45600a4c1017aaa7a7edcfaac",
    "mtrag-human/retrieval_tasks/clapnq/qrels/dev.tsv": "0766bc3e8fb6f515c34b33a97e568394303214915a2f835533705979b2303d97",
    "mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv": "925635635bf0e60b20d8ea0b18160cff878897acf8e0bf54e7096cc0ade8bd3b",
    "mtrag-human/retrieval_tasks/fiqa/qrels/dev.tsv": "a10ec0b5cba634d6c4ad02dd86de715a258ebd313dbcf5fa2d0a77bb0e121abc",
    "mtrag-human/retrieval_tasks/govt/qrels/dev.tsv": "46acb9b882297b6ff68978a56fd4ec57ea1c340cd604c62787a9ad69ac880d34",
    "mtrag-human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl": "7e510ceae50c1897797c79c51c6002b56d11c252843282b5c62ccfe88baab44e",
    "mtrag-human/retrieval_tasks/cloud/cloud_rewrite.jsonl": "f999821b4179bd900a8c06fcf849a895f2a1eb56adb6ae10b5bc95bb468c6a4c",
    "mtrag-human/retrieval_tasks/fiqa/fiqa_rewrite.jsonl": "88706f60953539e3927b05e3e6c20b39e4d9211df9247cd7de4035ae50ac5f15",
    "mtrag-human/retrieval_tasks/govt/govt_rewrite.jsonl": "d84cf9f1598875a5cd28616a6648155948072c27363a31c85a51050cf4078835",
    "mtragun-human/generation_tasks/reference.jsonl": "68053f709c928570718fd2c88455eccf974746f5bfa2206d7fd39b969acb21d4",
    "mtragun-human/retrieval_tasks/qrels/clapnq.tsv": "880a583ceaaea10f55eafa02f1a3964a12d48ec06fe396861277dedd2c5c892c",
    "mtragun-human/retrieval_tasks/qrels/cloud.tsv": "4c9e5903b0c1a702ade78f3f3bb826712614d31095f40c91121a214de6d248bf",
    "mtragun-human/retrieval_tasks/qrels/fiqa.tsv": "8a224221b72f4b93b5c068be5f7cfff668d50ecf6ec4971ef22f2f183ef54c21",
    "mtragun-human/retrieval_tasks/qrels/govt.tsv": "59903b16cbd28bd62910b1cde7dc325320a989f02f50c150c7a4ef3e96bdc691",
    "mtrag-human/evaluations/RAG.json": "5f026f95cb0d794c93eec6fdab71f6c4c2cb386332ed250c60648952500c412e",
    "scripts/evaluation/judge_utils.py": "222bd11dc06fcf63144aa09a23ecd7ce8ef2f2bab90bf2943ecabc3e4fd23669",
    "scripts/evaluation/underspecified_eval.py": "bc8fdbf6e5bfcfc25afd0ab3faa8de0258cbf4b6a77f22c86b40032f68aa1127",
}

HUB = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
SPARSE = (
    "opensearch-sparse-doc-v2-mini",
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    "4af867a426867dfdd744097531046f4289a32fdd",
)
RERANK = (
    "ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "233902d25c440f23af6f7d6e94d2946bac0bee0a",
)
LLM = ("qwen3.5-2b", "unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae")
SCORER = (
    "deberta-xlarge-mnli",
    "microsoft/deberta-xlarge-mnli",
    "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c",
)
MODELS = (
    (
        "run",
        SPARSE,
        "config.json",
        "5324b88812089731800d6f0e397690f91840ac4c24521360181d23002db5a677",
    ),
    (
        "run",
        SPARSE,
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    (
        "run",
        SPARSE,
        "special_tokens_map.json",
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    ("run", SPARSE, "idf.json", "da23a1c0b9252776cc8c6d70fd14723e218f484d489cd9027ac6e4065d5b9edd"),
    (
        "export",
        SPARSE,
        "model.safetensors",
        "d90a4233e0ca352cf73d6c3adf3c41fac7fee9a49ccd9ee3e806a86c1e05368a",
    ),
    (
        "export",
        SPARSE,
        "tokenizer_config.json",
        "ae83fa6ca0333117ff12606020af925d648667ef70d92ff7f27d781ba0ca4544",
    ),
    (
        "export",
        SPARSE,
        "vocab.txt",
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
    (
        "run",
        RERANK,
        "config.json",
        "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    ),
    (
        "run",
        RERANK,
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    (
        "run",
        RERANK,
        "onnx/model_quint8_avx2.onnx",
        "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9",
    ),
    (
        "run",
        LLM,
        "Qwen3.5-2B-Q4_K_M.gguf",
        "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223",
    ),
    (
        "eval",
        SCORER,
        "config.json",
        "4d1e27d6929862798fb1f866788756b0a39986d3ea7b25f96b0413d725036cb0",
    ),
    (
        "eval",
        SCORER,
        "vocab.json",
        "ef13b2af772bf95f74e5512265bbe22b786bf25861df22d9f19fde0b3937fdea",
    ),
    (
        "eval",
        SCORER,
        "merges.txt",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    (
        "eval",
        SCORER,
        "tokenizer_config.json",
        "052b1ce2b5b3bf119a72dee8cbb2de441dd4ff8f2255be61cd8aa63fb24d8149",
    ),
    (
        "eval",
        SCORER,
        "pytorch_model.bin",
        "60a1cac3b855e6994dd09daee5f697bd13f34bc1bada78d60844574872b092c3",
    ),
)

ALIASES = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
    "ibmcloud": "cloud",
    "clapnq": "clapnq",
    "cloud": "cloud",
    "fiqa": "fiqa",
    "govt": "govt",
}

SPLITS = {
    "dev": (
        "mtrag-human/generation_tasks/reference.jsonl",
        "mtrag-human/retrieval_tasks/{collection}/qrels/dev.tsv",
    ),
    "test": (
        "mtragun-human/generation_tasks/reference.jsonl",
        "mtragun-human/retrieval_tasks/qrels/{collection}.tsv",
    ),
}

MIXTRAL_REWRITES = "mtrag-human/retrieval_tasks/{collection}/{collection}_rewrite.jsonl"
FIELDS = ("text", "title", "url")


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class Task:
    task_id: str
    conversation_id: str
    collection: str
    split: str
    turns: tuple[Turn, ...]
    answerability: str
    multi_turn: str
    question_types: tuple[str, ...]
    reference: str
    reference_ids: tuple[str, ...]

    @property
    def question(self) -> str:
        return self.turns[-1].text


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str
    url: str


def raw_path(cfg: Config, relative: str) -> Path:
    return cfg.path("data") / "raw" / relative


def corpus_path(cfg: Config, collection: str) -> Path:
    return cfg.path("data") / "corpora" / f"{collection}.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(client: httpx.Client, url: str, target: Path, digest: str) -> bool:
    if target.exists() and sha256_file(target) == digest:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    hasher = hashlib.sha256()
    with client.stream("GET", url) as response, partial.open("wb") as handle:
        response.raise_for_status()
        for block in response.iter_bytes(1 << 20):
            handle.write(block)
            hasher.update(block)
    if hasher.hexdigest() != digest:
        partial.unlink()
        raise ValueError(f"checksum mismatch for {url}")
    partial.replace(target)
    return True


def download_models(
    cfg: Config, groups: tuple[str, ...] = ("run",), log: Callable[[str], None] = print
) -> None:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for group, (folder, repo, revision), path, digest in MODELS:
            if group not in groups:
                continue
            url = HUB.format(repo=repo, revision=revision, path=path)
            if _fetch(client, url, cfg.model_dir(folder) / path, digest):
                log(f"downloaded {folder}/{path}")


def download(cfg: Config, log: Callable[[str], None] = print) -> None:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for relative, digest in FILES.items():
            url = f"{cfg.data.source}/{cfg.data.commit}/{relative}"
            if _fetch(client, url, raw_path(cfg, relative), digest):
                log(f"downloaded {relative}")
    for collection in cfg.data.collections:
        archive = raw_path(cfg, f"corpora/passage_level/{collection}.jsonl.zip")
        target = corpus_path(cfg, collection)
        with zipfile.ZipFile(archive) as bundle:
            info = bundle.getinfo(f"{collection}.jsonl")
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                for block in iter(lambda: source.read(1 << 20), b""):
                    sink.write(block)
        log(f"unpacked {collection}")


def iter_corpus(cfg: Config, collection: str) -> Iterator[Passage]:
    with corpus_path(cfg, collection).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield Passage(
                row["_id"], row.get("title") or "", row.get("text") or "", row.get("url") or ""
            )


class DocStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.ids = (folder / "ids.txt").read_text(encoding="utf-8").split("\n")
        self.positions = {doc_id: index for index, doc_id in enumerate(self.ids)}
        self.offsets = {name: np.load(folder / f"{name}.npy") for name in FIELDS}
        self.handles = {name: (folder / f"{name}.bin").open("rb") for name in FIELDS}
        self.blobs = {
            name: mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            if (folder / f"{name}.bin").stat().st_size > 0
            else b""
            for name, handle in self.handles.items()
        }

    @staticmethod
    def folder_for(cfg: Config, collection: str) -> Path:
        return cfg.path("indexes") / collection / "docs"

    @classmethod
    def build(cls, cfg: Config, collection: str) -> DocStore:
        folder = cls.folder_for(cfg, collection)
        folder.mkdir(parents=True, exist_ok=True)
        ids: list[str] = []
        offsets: dict[str, list[int]] = {name: [0] for name in FIELDS}
        sinks = {name: (folder / f"{name}.bin").open("wb") for name in FIELDS}
        try:
            for passage in iter_corpus(cfg, collection):
                ids.append(passage.doc_id)
                for name in FIELDS:
                    payload = getattr(passage, name).encode("utf-8")
                    sinks[name].write(payload)
                    offsets[name].append(offsets[name][-1] + len(payload))
        finally:
            for sink in sinks.values():
                sink.close()
        for name in FIELDS:
            np.save(folder / f"{name}.npy", np.asarray(offsets[name], dtype=np.int64))
        (folder / "ids.txt").write_text("\n".join(ids), encoding="utf-8")
        return cls(folder)

    @classmethod
    def open(cls, cfg: Config, collection: str) -> DocStore:
        return cls(cls.folder_for(cfg, collection))

    def close(self) -> None:
        for blob in self.blobs.values():
            if isinstance(blob, mmap.mmap):
                blob.close()
        for handle in self.handles.values():
            handle.close()

    def __len__(self) -> int:
        return len(self.ids)

    def field(self, name: str, index: int) -> str:
        offsets = self.offsets[name]
        start, end = int(offsets[index]), int(offsets[index + 1])
        return self.blobs[name][start:end].decode("utf-8")

    def __getitem__(self, index: int) -> Passage:
        return Passage(
            self.ids[index],
            self.field("title", index),
            self.field("text", index),
            self.field("url", index),
        )

    def text(self, index: int) -> str:
        return self.field("text", index)

    def index(self, doc_id: str) -> int:
        return self.positions[doc_id]


def _labels(row: dict, key: str) -> list[str]:
    value = row.get(key)
    if value is None:
        value = row.get(key.capitalize())
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def parse_task(row: dict, split: str) -> Task:
    turns = tuple(Turn(turn["speaker"], turn["text"]) for turn in row["input"])
    answerability = _labels(row, "answerability")
    multi_turn = _labels(row, "Multi-Turn")
    targets = row.get("targets") or []
    return Task(
        task_id=row["task_id"],
        conversation_id=row["conversation_id"],
        collection=ALIASES[row["Collection"]],
        split=split,
        turns=turns,
        answerability=answerability[0] if answerability else "ANSWERABLE",
        multi_turn=multi_turn[0] if multi_turn else "N/A",
        question_types=tuple(_labels(row, "Question Type")),
        reference=targets[0]["text"] if targets else "",
        reference_ids=tuple(context["document_id"] for context in row.get("contexts") or []),
    )


def load_tasks(cfg: Config, split: str) -> list[Task]:
    tasks_file, _ = SPLITS[split]
    with raw_path(cfg, tasks_file).open(encoding="utf-8") as handle:
        tasks = [parse_task(json.loads(line), split) for line in handle if line.strip()]
    return [task for task in tasks if task.collection in cfg.data.collections]


def load_qrels(cfg: Config, split: str) -> dict[str, dict[str, int]]:
    _, pattern = SPLITS[split]
    qrels: dict[str, dict[str, int]] = {}
    for collection in cfg.data.collections:
        with raw_path(cfg, pattern.format(collection=collection)).open(encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader)
            for query_id, doc_id, score in reader:
                qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels


def load_mixtral_rewrites(cfg: Config) -> dict[str, str]:
    rewrites: dict[str, str] = {}
    for collection in cfg.data.collections:
        path = raw_path(cfg, MIXTRAL_REWRITES.format(collection=collection))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rewrites[row["_id"]] = row["text"].replace("|user|:", "").strip()
    return rewrites
