"""Shared helpers for synthetic DRM datasets: hidden latent -> hidden policy -> free-form text.

Each example generator (examples/*/gen.py) defines
  sample(rng, uid)              -> a latent dict (the facts, never shown to the model as-is)
  label(z)                      -> {"label": int} or {"label": int, "target": float} from a hidden rule
  render(rng, z, ood)           -> free-form text built from many phrasing templates

and calls `main(...)`, which writes the splits a DRM run expects:

  train / val / test : same phrasing pool, disjoint latents (train/val get a little label noise)
  ood                : held-out phrasing (the last option of every `pick`) to test new wording
  perturb            : K re-phrasings of the same latent, to measure how often the answer flips
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Callable


def pick(rng: random.Random, options: list[str], ood: bool) -> str:
    """Phrasing choice. The last option is reserved for the out-of-distribution split."""
    return options[-1] if ood else rng.choice(options[:-1])


def assemble(rng: random.Random, sections: list[tuple[str, str]]) -> str:
    """Lay the same sections out in one of a few formats so the model can't key on layout."""
    style = rng.randint(0, 2)
    if style == 0:
        return "\n\n".join(f"{k}:\n{v}" for k, v in sections)
    if style == 1:
        return "\n".join(f"### {k}\n{v}" for k, v in sections)
    return " ".join(v for _, v in sections)


def bullets_or_prose(rng: random.Random, facts: list[str], ood: bool) -> str:
    rng.shuffle(facts)
    if rng.random() < 0.4 and not ood:
        return "\n".join(f"- {f}" for f in facts)
    return " ".join(facts)


def main(sample: Callable, label: Callable, render: Callable, kind: Callable,
         task_json: Path, n_classes: int, noise: float = 0.04) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-ood", type=int, default=500)
    ap.add_argument("--n-perturb", type=int, default=200)
    ap.add_argument("--k-perturb", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    def example(z, ood=False, noisy=False):
        y = label(z)
        row = {"uid": z["uid"], "kind": kind(z), "text": render(rng, z, ood)} | y
        if noisy and rng.random() < noise:
            # flip binary labels; nudge class / scalar labels one step
            if n_classes == 2:
                row["label"] = 1 - row["label"]
            elif "target" not in row:
                row["label"] = min(max(row["label"] + rng.choice((-1, 1)), 0), n_classes - 1)
        row["clean_label"] = y["label"]
        row["latent"] = z
        return row

    def latents(prefix, n):
        return [sample(rng, f"{prefix}-{i}") for i in range(n)]

    splits = {
        "train": [example(z, noisy=True) for z in latents("tr", a.n_train)],
        "val": [example(z, noisy=True) for z in latents("va", a.n_val)],
        "test": [example(z) for z in latents("te", a.n_test)],
        "ood": [example(z, ood=True) for z in latents("oo", a.n_ood)],
        "perturb": [example(z) | {"group": z["uid"], "variant": k}
                    for z in latents("pt", a.n_perturb) for k in range(a.k_perturb)],
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"label_noise_train_val": noise}
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts = [sum(r["label"] == k for r in rows) for k in range(n_classes)]
        stats[name] = {"n": len(rows), "label_counts": counts}
    shutil.copy(task_json, out / "task.json")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
