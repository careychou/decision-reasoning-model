"""Output heads for a DRM: turn K label-token logits into a decision. Pure Python (no torch).

Every task reads the next-token logits of a fixed set of single-token labels, so the model
always returns a deterministic probability distribution p over K labels. What we do with p
depends on the task's output type:

  binary      labels ["No", "Yes"]          P(Yes); auto-Yes / auto-No thresholds
  multiclass  e.g. ["A", "B", "C"]          argmax label; auto-act if max p >= threshold
  ordinal     ordered bands ["A".."D"]      as multiclass, plus within-one-band accuracy
  scalar      digit bins ["0".."8"] with    point = sum p_k * v_k, spread = std over bins;
              numeric bin centers values     auto-accept if spread <= threshold
"""

from __future__ import annotations

import math

TYPES = ("binary", "multiclass", "ordinal", "scalar")
DEFAULT_OUTPUT = {"type": "binary", "labels": ["No", "Yes"], "names": ["No", "Yes"]}


def normalize_output(out: dict | None) -> dict:
    o = dict(DEFAULT_OUTPUT if not out else out)
    o.setdefault("type", "binary")
    if o["type"] not in TYPES:
        raise ValueError(f"unknown output type {o['type']!r}; expected one of {TYPES}")
    if o["type"] == "binary":
        o.setdefault("labels", ["No", "Yes"])
    o.setdefault("names", list(o["labels"]))
    if len(o["names"]) != len(o["labels"]):
        raise ValueError("output.names must match output.labels")
    if o["type"] == "scalar":
        v = o.get("values")
        if not v or len(v) != len(o["labels"]) or any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("scalar output needs ascending 'values', one per label")
        o.setdefault("tolerance", (v[1] - v[0]) / 2)
    # loss: "ce" (default), or for ordered outputs "emd" (squared earth mover's distance over the
    # CDFs) / "ce+emd" (CE plus emd_weight * EMD^2). EMD charges more for a far miss than a near one.
    o.setdefault("loss", "ce")
    if o["loss"] not in ("ce", "emd", "ce+emd"):
        raise ValueError("output.loss must be 'ce', 'emd' or 'ce+emd'")
    if o["loss"] != "ce" and o["type"] not in ("ordinal", "scalar"):
        raise ValueError("EMD loss needs ordered labels (ordinal or scalar)")
    o.setdefault("emd_weight", 1.0)
    return o


def emd2(p: list[float], t: list[float]) -> float:
    """Squared earth mover's distance between two distributions over ordered labels (unit spacing)."""
    cp = ct = d = 0.0
    for a, b in zip(p, t):
        cp += a
        ct += b
        d += (cp - ct) ** 2
    return d


def softmax(z: list[float], temperature: float = 1.0) -> list[float]:
    s = [x / temperature for x in z]
    m = max(s)
    e = [math.exp(x - m) for x in s]
    t = sum(e)
    return [x / t for x in e]


def soft_target(row: dict, out: dict) -> list[float]:
    """Training target over the K labels.

    Classes get a one-hot target. A scalar gets a "two-hot" target split between the two nearest
    bin centers, so the expected value over bins reproduces the (clipped) true number exactly.
    """
    k = len(out["labels"])
    if out["type"] != "scalar" or "target" not in row:
        t = [0.0] * k
        t[int(row["label"])] = 1.0
        return t
    v = out["values"]
    y = min(max(float(row["target"]), v[0]), v[-1])
    t = [0.0] * k
    for i in range(k - 1):
        if v[i] <= y <= v[i + 1]:
            w = (y - v[i]) / (v[i + 1] - v[i])
            t[i], t[i + 1] = 1 - w, w
            return t
    t[-1] = 1.0
    return t


def cross_entropy(p: list[float], target: list[float]) -> float:
    return -sum(t * math.log(max(q, 1e-9)) for q, t in zip(p, target) if t)


def fit_temperature(logits: list[list[float]], targets: list[list[float]]) -> float:
    grid = [0.05 * k for k in range(1, 201)]  # 0.05 .. 10
    return min(grid, key=lambda T: sum(cross_entropy(softmax(z, T), t)
                                       for z, t in zip(logits, targets)))


def argmax(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def expectation(p: list[float], values: list[float]) -> float:
    return sum(q * v for q, v in zip(p, values))


def spread(p: list[float], values: list[float]) -> float:
    mu = expectation(p, values)
    return math.sqrt(max(sum(q * (v - mu) ** 2 for q, v in zip(p, values)), 0.0))


def read(p: list[float], out: dict) -> dict:
    """Decision read from one distribution. `confidence` is higher-is-surer for every type."""
    if out["type"] == "scalar":
        s = spread(p, out["values"])
        return {"estimate": expectation(p, out["values"]), "spread": s, "confidence": -s}
    i = argmax(p)
    r = {"index": i, "label": out["labels"][i], "name": out["names"][i], "confidence": p[i]}
    if out["type"] == "ordinal":
        # 1-based expected band; sort candidates by this to get an order of preference
        r["expected_band"] = sum((k + 1) * x for k, x in enumerate(p))
    return r


def is_right(p: list[float], row: dict, out: dict) -> bool:
    if out["type"] == "scalar":
        return abs(expectation(p, out["values"]) - float(row["target"])) <= out["tolerance"]
    return argmax(p) == int(row["label"])


def auto_threshold(conf: list[float], right: list[bool], target_precision: float,
                   min_support: int = 20) -> float | None:
    """Lowest confidence cut whose auto-acted set is right >= target_precision of the time."""
    for t in sorted(set(round(c, 6) for c in conf)):
        sel = [r for c, r in zip(conf, right) if c >= t]
        if len(sel) < min_support:
            return None
        if sum(sel) / len(sel) >= target_precision:
            return t
    return None


def coverage(conf: list[float], right: list[bool], t: float | None) -> dict:
    if t is None:
        return {"threshold": None, "coverage": 0.0, "precision": None, "n": 0}
    sel = [r for c, r in zip(conf, right) if c >= t]
    return {"threshold": t, "coverage": round(len(sel) / len(conf), 4),
            "precision": round(sum(sel) / len(sel), 4) if sel else None, "n": len(sel)}
