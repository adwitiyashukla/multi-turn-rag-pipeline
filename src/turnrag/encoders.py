from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Encoding, Tokenizer


class OnnxModel:
    def __init__(
        self,
        folder: Path,
        onnx_file: str,
        max_length: int,
        token_budget: int,
        threads: int = 0,
        pair: bool = False,
    ) -> None:
        self.folder = folder
        self.max_length = max_length
        self.token_budget = token_budget
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(
            max_length, strategy="only_second" if pair else "longest_first"
        )
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.pad_id = int(config.get("pad_token_id") or 0)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(folder / onnx_file), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {item.name for item in self.session.get_inputs()}

    def batches(self, encodings: Sequence[Encoding]) -> Iterator[tuple[np.ndarray, dict]]:
        order = sorted(range(len(encodings)), key=lambda i: -len(encodings[i].ids))
        start = 0
        while start < len(order):
            longest = len(encodings[order[start]].ids)
            size = max(1, self.token_budget // max(longest, 1))
            chosen = order[start : start + size]
            start += len(chosen)
            ids = np.full((len(chosen), longest), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(chosen), longest), dtype=np.int64)
            types = np.zeros((len(chosen), longest), dtype=np.int64)
            for row, index in enumerate(chosen):
                encoding = encodings[index]
                length = len(encoding.ids)
                ids[row, :length] = encoding.ids
                mask[row, :length] = 1
                types[row, :length] = encoding.type_ids
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = types
            yield np.asarray(chosen), feed

    def run(self, feed: dict) -> np.ndarray:
        return self.session.run(None, feed)[0]


class SparseDocEncoder(OnnxModel):
    def __init__(self, *args, max_terms: int = 256, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_terms = max_terms

    def encode(self, texts: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        encodings = self.tokenizer.encode_batch(list(texts))
        output: list[tuple[np.ndarray, np.ndarray]] = [None] * len(texts)
        for chosen, feed in self.batches(encodings):
            weights = self.run(feed)
            for row, index in enumerate(chosen):
                values = weights[row]
                terms = np.flatnonzero(values > 0)
                if terms.size > self.max_terms:
                    keep = np.argpartition(-values[terms], self.max_terms - 1)[: self.max_terms]
                    terms = np.sort(terms[keep])
                output[index] = (terms.astype(np.int32), values[terms].astype(np.float32))
        return output


class SparseQueryEncoder:
    def __init__(self, folder: Path) -> None:
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(512)
        idf = json.loads((folder / "idf.json").read_text(encoding="utf-8"))
        self.weights = np.zeros(self.tokenizer.get_vocab_size(), dtype=np.float32)
        for token, weight in idf.items():
            position = self.tokenizer.token_to_id(token)
            if position is not None:
                self.weights[position] = weight
        special = json.loads((folder / "special_tokens_map.json").read_text(encoding="utf-8"))
        for token in special.values():
            position = self.tokenizer.token_to_id(
                token if isinstance(token, str) else token["content"]
            )
            if position is not None:
                self.weights[position] = 0.0

    def encode(self, weighted_texts: Sequence[tuple[str, float]]) -> dict[int, float]:
        query: dict[int, float] = {}
        for text, weight in weighted_texts:
            for token in set(self.tokenizer.encode(text).ids):
                value = float(self.weights[token]) * weight
                if value > 0:
                    query[token] = query.get(token, 0.0) + value
        return query


class CrossEncoder(OnnxModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, pair=True, **kwargs)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        encodings = self.tokenizer.encode_batch([(query, passage) for passage in passages])
        scores = np.zeros(len(passages), dtype=np.float32)
        for chosen, feed in self.batches(encodings):
            scores[chosen] = self.run(feed).reshape(len(chosen), -1)[:, 0]
        return scores
