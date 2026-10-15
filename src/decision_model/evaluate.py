"""Evaluate a decision model: accuracy, calibration, auto-act coverage at a target precision,
paraphrase flip rate and run-to-run determinism. Works for zero-shot (no adapter) too.

Binary tasks keep the original Yes/No report (AUC, auto-yes / auto-no). Multiclass, ordinal
and scalar tasks get `evaluate_k` (argmax / expected-value reads, one auto-act threshold)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_model import heads
from decision_model.common import (load_jsonl, pick_device, score, score_logits, set_task,
                                   setup_tokenizer)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def auc(probs, labels) -> float:
    pairs = sorted(zip(probs, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):  # average ranks for ties
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(probs, labels, bins=15) -> float:
    buckets = defaultdict(list)
    for p, y in zip(probs, labels):
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    return sum(len(b) / len(probs) * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
               for b in buckets.values())


def nll(probs, labels) -> float:
    eps = 1e-7
    return -sum(math.log(max(p, eps)) if y else math.log(max(1 - p, eps))
                for p, y in zip(probs, labels)) / len(labels)


def fit_temperature(log_odds, labels) -> float:
    grid = [0.05 * k for k in range(1, 201)]  # 0.05 .. 10
    return min(grid, key=lambda t: nll([sigmoid(x / t) for x in log_odds], labels))


def metrics(probs, labels) -> dict:
    preds = [p >= 0.5 for p in probs]
    return {
        "n": len(labels),
        "acc": round(sum(int(p) == y for p, y in zip(preds, labels)) / len(labels), 4),
        "auc": round(auc(probs, labels), 4),
        "ece": round(ece(probs, labels), 4),
        "brier": round(sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels), 4),
    }


def auto_threshold(probs, labels, target_precision, side="yes", min_support=20):
    """Most permissive threshold whose precision on this set >= target."""
    best = None
    cands = sorted(set(round(p, 4) for p in probs))
    for t in (cands if side == "yes" else reversed(cands)):
        sel = [(p, y) for p, y in zip(probs, labels) if (p >= t if side == "yes" else p <= t)]
        if len(sel) < min_support:
            continue
        good = sum(y == (1 if side == "yes" else 0) for _, y in sel)
        if good / len(sel) >= target_precision:
            best = t
            break
    return best


def coverage(probs, labels, t, side):
    if t is None:
        return {"threshold": None, "coverage": 0.0, "precision": None}
    sel = [(p, y) for p, y in zip(probs, labels) if (p >= t if side == "yes" else p <= t)]
    good = sum(y == (1 if side == "yes" else 0) for _, y in sel)
    return {"threshold": t, "coverage": round(len(sel) / len(probs), 4),
            "precision": round(good / len(sel), 4) if sel else None, "n": len(sel)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for zero-shot")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--target-precision", type=float, default=0.98)
    ap.add_argument("--task", default=None, help="task.json (system prompt + question); default = generic Yes/No")
    a = ap.parse_args()
    out_spec = set_task(a.task)

    device = pick_device()
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(a.adapter or a.model))
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()

    d = Path(a.data)
    sets = {k: load_jsonl(d / f"{k}.jsonl") for k in ("val", "test", "ood", "perturb")}
    if out_spec["type"] != "binary":
        res, preds = evaluate_k(model, tok, sets, out_spec, a, device)
        write(a, res, preds)
        return
    lo = {k: score(model, tok, [r["text"] for r in v], a.max_len, 32, device) for k, v in sets.items()}
    y = {k: [r["label"] for r in v] for k, v in sets.items()}

    T = fit_temperature(lo["val"], y["val"])
    cal = {k: [sigmoid(x / T) for x in v] for k, v in lo.items()}
    raw = {k: [sigmoid(x) for x in v] for k, v in lo.items()}

    res = {"model": a.model, "adapter": a.adapter, "temperature": T}
    for k in ("test", "ood"):
        res[k] = {"raw": metrics(raw[k], y[k]), "calibrated": metrics(cal[k], y[k])}
        res[k]["by_kind_acc"] = {
            rt: round(sum((cal[k][i] >= 0.5) == bool(y[k][i]) for i in idx) / len(idx), 4)
            for rt in sorted({r.get("kind", "all") for r in sets[k]})
            for idx in [[i for i, r in enumerate(sets[k]) if r.get("kind", "all") == rt]]
        }

    # auto-yes / auto-no thresholds chosen on val, applied to test
    for side in ("yes", "no"):
        t = auto_threshold(cal["val"], y["val"], a.target_precision, side)
        res[f"auto_{side}@{a.target_precision}"] = {
            "test": coverage(cal["test"], y["test"], t, side),
            "ood": coverage(cal["ood"], y["ood"], t, side),
        }

    # paraphrase robustness
    groups = defaultdict(list)
    for r, p in zip(sets["perturb"], cal["perturb"]):
        groups[r["group"]].append(p >= 0.5)
    flip_groups = sum(len(set(v)) > 1 for v in groups.values())
    pair_flips = pair_total = 0
    for v in groups.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pair_total += 1
                pair_flips += v[i] != v[j]
    res["robustness"] = {
        "groups": len(groups),
        "group_flip_rate": round(flip_groups / len(groups), 4),
        "pairwise_flip_rate": round(pair_flips / pair_total, 4),
        "perturb_acc": metrics(cal["perturb"], y["perturb"])["acc"],
    }

    # determinism: identical inputs re-scored 3x (incl. different batch size)
    texts = [r["text"] for r in sets["test"][:200]]
    runs = [score(model, tok, texts, a.max_len, bs, device) for bs in (32, 32, 1)]
    same_bs = max(abs(p - q) for p, q in zip(runs[0], runs[1]))
    diff_bs = max(abs(sigmoid(p / T) - sigmoid(q / T)) for p, q in zip(runs[0], runs[2]))
    res["determinism"] = {
        "n": len(texts),
        "max_abs_logodds_diff_same_batch": same_bs,
        "decision_flips_same_batch": sum((p > 0) != (q > 0) for p, q in zip(runs[0], runs[1])),
        "max_abs_prob_diff_bs32_vs_bs1": round(diff_bs, 6),
        "decision_flips_bs32_vs_bs1": sum((p > 0) != (q > 0) for p, q in zip(runs[0], runs[2])),
    }

    write(a, res, [{"uid": r["uid"], "kind": r.get("kind"), "label": r["label"],
                    "p_yes": round(p, 5)} for r, p in zip(sets["test"], cal["test"])])


def write(a, res: dict, preds: list[dict]) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    with (out / "test_predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(json.dumps(res, indent=2))


def k_metrics(probs: list[list[float]], rows: list[dict], o: dict) -> dict:
    n = len(rows)
    tg = [heads.soft_target(r, o) for r in rows]
    m = {"n": n, "nll": round(sum(heads.cross_entropy(p, t) for p, t in zip(probs, tg)) / n, 4)}
    if o["type"] == "scalar":
        err = [heads.expectation(p, o["values"]) - float(r["target"]) for p, r in zip(probs, rows)]
        clip = [min(max(float(r["target"]), o["values"][0]), o["values"][-1]) for r in rows]
        med = sorted(clip)[n // 2]
        m |= {"mae": round(sum(map(abs, err)) / n, 4),
              "rmse": round(math.sqrt(sum(e * e for e in err) / n), 4),
              "within_tolerance": round(sum(abs(e) <= o["tolerance"] for e in err) / n, 4),
              "baseline_mae_zero": round(sum(abs(float(r["target"])) for r in rows) / n, 4),
              "baseline_mae_median": round(sum(abs(float(r["target"]) - med) for r in rows) / n, 4),
              "mean_spread": round(sum(heads.spread(p, o["values"]) for p in probs) / n, 4)}
        return m
    pred = [heads.argmax(p) for p in probs]
    y = [int(r["label"]) for r in rows]
    conf = [max(p) for p in probs]
    m["acc"] = round(sum(a == b for a, b in zip(pred, y)) / n, 4)
    m["ece_top"] = round(ece(conf, [a == b for a, b in zip(pred, y)]), 4)
    m["recall_by_label"] = {o["names"][k]: round(sum(p == k for p, t in zip(pred, y) if t == k)
                                                 / max(1, y.count(k)), 4) for k in range(len(o["labels"]))}
    m["label_counts"] = {o["names"][k]: y.count(k) for k in range(len(o["labels"]))}
    if o["type"] == "ordinal":
        m["within_one"] = round(sum(abs(a - b) <= 1 for a, b in zip(pred, y)) / n, 4)
        m["mae_bands"] = round(sum(abs(a - b) for a, b in zip(pred, y)) / n, 4)
        m["emd2"] = round(sum(heads.emd2(p, t) for p, t in zip(probs, tg)) / n, 4)
    return m


def evaluate_k(model, tok, sets: dict, o: dict, a, device):
    """Multiclass / ordinal / scalar report. Confidence = top probability (classes) or -spread (scalar)."""
    lg = {k: score_logits(model, tok, [r["text"] for r in v], a.max_len, 32, device)
          for k, v in sets.items()}
    T = heads.fit_temperature(lg["val"], [heads.soft_target(r, o) for r in sets["val"]])
    cal = {k: [heads.softmax(z, T) for z in v] for k, v in lg.items()}
    raw = {k: [heads.softmax(z) for z in v] for k, v in lg.items()}
    res = {"model": a.model, "adapter": a.adapter, "output": o, "temperature": T}
    for k in ("test", "ood"):
        res[k] = {"raw": k_metrics(raw[k], sets[k], o), "calibrated": k_metrics(cal[k], sets[k], o)}

    def conf_right(k):
        return ([heads.read(p, o)["confidence"] for p in cal[k]],
                [heads.is_right(p, r, o) for p, r in zip(cal[k], sets[k])])

    t = heads.auto_threshold(*conf_right("val"), a.target_precision)
    res[f"auto_act@{a.target_precision}"] = {k: heads.coverage(*conf_right(k), t) for k in ("test", "ood")}

    groups = defaultdict(list)
    for r, p in zip(sets["perturb"], cal["perturb"]):
        groups[r["group"]].append((p, r))

    def chosen(p, r):
        # menu-style tasks reorder options across paraphrases; compare the chosen option, not the letter
        i = heads.argmax(p)
        menu = r.get("menu")
        return menu[i] if menu and i < len(menu) else (f"none:{i}" if menu else i)
    flips = sum(len({chosen(p, r) for p, r in v}) > 1 for v in groups.values())
    groups = {g: [p for p, _ in v] for g, v in groups.items()}
    rob = {"groups": len(groups), "group_flip_rate": round(flips / max(1, len(groups)), 4),
           "perturb_right": round(sum(heads.is_right(p, r, o) for p, r in
                                      zip(cal["perturb"], sets["perturb"])) / len(sets["perturb"]), 4)}
    if o["type"] == "scalar":
        rob["mean_group_estimate_range"] = round(sum(
            max(e) - min(e) for v in groups.values()
            for e in [[heads.expectation(p, o["values"]) for p in v]]) / max(1, len(groups)), 4)
    res["robustness"] = rob

    texts = [r["text"] for r in sets["test"][:200]]
    runs = [score_logits(model, tok, texts, a.max_len, bs, device) for bs in (32, 32, 1)]
    res["determinism"] = {
        "n": len(texts),
        "max_abs_logit_diff_same_batch": max(abs(x - y) for p, q in zip(runs[0], runs[1])
                                             for x, y in zip(p, q)),
        "decision_flips_same_batch": sum(heads.argmax(p) != heads.argmax(q) for p, q in zip(runs[0], runs[1])),
        "max_abs_prob_diff_bs32_vs_bs1": round(max(abs(x - y) for p, q in zip(runs[0], runs[2])
                                                   for x, y in zip(heads.softmax(p, T), heads.softmax(q, T))), 6),
        "decision_flips_bs32_vs_bs1": sum(heads.argmax(p) != heads.argmax(q) for p, q in zip(runs[0], runs[2])),
    }
    preds = []
    for r, p in zip(sets["test"], cal["test"]):
        rec = {"uid": r["uid"], "kind": r.get("kind"), "label": r["label"],
               "probs": [round(x, 5) for x in p]} | {
            k: (round(v, 5) if isinstance(v, float) else v) for k, v in heads.read(p, o).items()}
        if "target" in r:
            rec["target"] = r["target"]
        preds.append(rec)
    return res, preds


if __name__ == "__main__":
    main()
