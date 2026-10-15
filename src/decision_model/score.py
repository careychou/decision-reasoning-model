"""Local scorer: base model + trained LoRA adapter -> calibrated decision + route.

Binary tasks return P(Yes) and auto_yes / auto_no / to_person. Multiclass and ordinal
tasks return the top label and its probability; scalar tasks return the expected value and its
spread. Both route "auto" when confidence clears the threshold tuned on val, else "to_person".

    python -m decision_model.score --model-dir models/qwen2.5-1.5b-instruct --text "..."
    python -m decision_model.score --model-dir models/qwen2.5-1.5b-instruct --file recs.jsonl

models/<tag>/ is written by scripts/run_pipeline.sh (train + evaluate) and contains:
  adapter/        LoRA weights + tokenizer (adapter_config.json names the base model)
  metrics.json    eval metrics incl. calibration temperature + auto-yes/no thresholds
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_model import heads
from decision_model.common import pick_device, score, score_logits, set_task, setup_tokenizer


class DecisionModel:
    def __init__(self, model_dir: str | Path, device: str | None = None, dtype: str = "float32"):
        model_dir = Path(model_dir)
        adapter = model_dir / "adapter"
        self.output = set_task(adapter / "task.json" if (adapter / "task.json").exists() else None)
        base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
        m = json.loads((model_dir / "metrics.json").read_text())
        self.temperature = m["temperature"]
        if self.output["type"] == "binary":
            self.yes_t = next(v for k, v in m.items() if k.startswith("auto_yes"))["test"]["threshold"]
            self.no_t = next(v for k, v in m.items() if k.startswith("auto_no"))["test"]["threshold"]
        else:
            self.auto_t = next(v for k, v in m.items() if k.startswith("auto_act"))["test"]["threshold"]
        self.device = device or pick_device()
        self.tok = setup_tokenizer(AutoTokenizer.from_pretrained(adapter))
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, dtype))
        self.model = PeftModel.from_pretrained(model, adapter).merge_and_unload().to(self.device).eval()
        self.base = base

    def predict(self, texts: list[str], batch_size: int = 1) -> list[dict]:
        # batch_size=1 (no padding) makes each score independent of what it is batched with
        if self.output["type"] != "binary":
            return self._predict_k(texts, batch_size)
        log_odds = score(self.model, self.tok, texts, 768, batch_size, self.device)
        out = []
        for lo in log_odds:
            p = 1 / (1 + math.exp(-lo / self.temperature))
            if self.yes_t is not None and p >= self.yes_t:
                route = "auto_yes"
            elif self.no_t is not None and p <= self.no_t:
                route = "auto_no"
            else:
                route = "to_person"
            out.append({"p_yes": round(p, 5), "decision": "Yes" if p >= 0.5 else "No",
                        "route": route})
        return out

    def _predict_k(self, texts: list[str], batch_size: int) -> list[dict]:
        o, out = self.output, []
        for z in score_logits(self.model, self.tok, texts, 768, batch_size, self.device):
            p = heads.softmax(z, self.temperature)
            r = heads.read(p, o)
            auto = self.auto_t is not None and r["confidence"] >= self.auto_t
            if o["type"] == "scalar":
                r.pop("confidence")  # -spread; the spread itself is reported
            out.append({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
                       | {"probs": dict(zip(o["names"], (round(x, 5) for x in p))),
                          "route": "auto" if auto else "to_person"})
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", help="single input text")
    ap.add_argument("--file", help="jsonl with a 'text' field per line")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    a = ap.parse_args()

    dm = DecisionModel(a.model_dir, a.device, a.dtype)
    if a.text:
        rows = [{"text": a.text}]
    elif a.file:
        rows = [json.loads(l) for l in open(a.file) if l.strip()]
    else:
        rows = [{"text": sys.stdin.read()}]
    for r, pred in zip(rows, dm.predict([r["text"] for r in rows])):
        res = {k: r[k] for k in ("uid", "label") if k in r} | pred
        print(json.dumps(res))


if __name__ == "__main__":
    main()
