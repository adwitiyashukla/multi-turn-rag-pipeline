from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from turnrag.config import Config

SAMPLES = (
    "The Arizona Cardinals play their home games at State Farm Stadium in Glendale.",
    "To reset the password for an IBM Cloud API key, open the access management page.",
    "Dividends from a stock held in a Roth IRA are not taxed when you withdraw them.",
    "Residents can book seats for the senior center events through the city website.",
)


def _module(torch, model, keep, double_log: bool):
    class Sparse(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("keep", keep)

        def forward(self, input_ids, attention_mask):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).to(logits.dtype)
            pooled = torch.max(logits * mask, dim=1)[0]
            weights = torch.log1p(torch.relu(pooled))
            if double_log:
                weights = torch.log1p(weights)
            return weights * self.keep

    return Sparse().eval()


def _compare(torch, module, tokenizer, onnx_path: Path) -> float:
    import onnxruntime as ort

    batch = tokenizer(list(SAMPLES), padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = module(batch["input_ids"], batch["attention_mask"]).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            "input_ids": batch["input_ids"].numpy().astype(np.int64),
            "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
        },
    )[0]
    numerator = (expected * actual).sum(axis=1)
    denominator = np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1)
    return float((numerator / np.maximum(denominator, 1e-12)).min())


def export_sparse(cfg: Config, log: Callable[[str], None] = print) -> dict[str, float]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(0)
    folder = cfg.model_dir(cfg.sparse.model)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForMaskedLM.from_pretrained(folder).float().eval()
    keep = torch.ones(model.config.vocab_size)
    keep[list(tokenizer.all_special_ids)] = 0
    module = _module(torch, model, keep, cfg.sparse.activation == "log1p_log1p")
    full = folder / "onnx" / "model.onnx"
    quantized = folder / cfg.sparse.onnx
    full.parent.mkdir(parents=True, exist_ok=True)
    batch = tokenizer(list(SAMPLES), padding=True, return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(
            module,
            (batch["input_ids"], batch["attention_mask"]),
            str(full),
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "output": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    quantize_dynamic(str(full), str(quantized), weight_type=QuantType.QInt8, per_channel=True)
    report = {
        "fp32_cosine": _compare(torch, module, tokenizer, full),
        "int8_cosine": _compare(torch, module, tokenizer, quantized),
    }
    log(f"exported {quantized}, int8 cosine to torch {report['int8_cosine']:.4f}")
    (folder / "onnx" / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
