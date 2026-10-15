"""Render results/*/{zero_shot,finetuned}/metrics.json into a markdown comparison table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(root: str) -> None:
    rows, k_rows = [], []
    for m in sorted(Path(root).glob("*/*/metrics.json")):
        r = json.loads(m.read_text())
        if r.get("output", {}).get("type", "binary") != "binary":
            k_rows.append((m, r))
            continue
        aa = next(v for k, v in r.items() if k.startswith("auto_yes"))
        ar = next(v for k, v in r.items() if k.startswith("auto_no"))
        rows.append([
            m.parent.parent.name, m.parent.name,
            r["test"]["calibrated"]["acc"], r["test"]["calibrated"]["auc"],
            r["test"]["raw"]["ece"], r["test"]["calibrated"]["ece"],
            r["ood"]["calibrated"]["acc"],
            f'{aa["test"]["coverage"]:.0%} @ {aa["test"]["precision"]}',
            f'{ar["test"]["coverage"]:.0%} @ {ar["test"]["precision"]}',
            r["robustness"]["group_flip_rate"], r["robustness"]["pairwise_flip_rate"],
            r["determinism"]["decision_flips_same_batch"],
            r["determinism"]["decision_flips_bs32_vs_bs1"],
        ])
    if k_rows:
        print_k(k_rows)
    if not rows:
        return
    hdr = ["model", "mode", "test acc", "test AUC", "ECE raw", "ECE cal", "OOD acc",
           "auto-yes cov @ prec", "auto-no cov @ prec", "group flip", "pair flip",
           "rerun flips", "bs flips"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def print_k(k_rows) -> None:
    """Multiclass / ordinal / scalar runs."""
    hdr = ["model", "mode", "type", "test", "OOD", "auto-act cov @ prec", "group flip",
           "rerun flips", "bs flips"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for m, r in k_rows:
        t, o = r["test"]["calibrated"], r["ood"]["calibrated"]
        if r["output"]["type"] == "scalar":
            fmt = lambda x: f'MAE {x["mae"]} (prior-only {x["baseline_mae_zero"]}), in-tol {x["within_tolerance"]}'
        elif r["output"]["type"] == "ordinal":
            fmt = lambda x: f'acc {x["acc"]}, within-1 {x["within_one"]}'
        else:
            fmt = lambda x: f'acc {x["acc"]}'
        aa = next(v for k, v in r.items() if k.startswith("auto_act"))["test"]
        print("| " + " | ".join(str(x) for x in [
            m.parent.parent.name, m.parent.name, r["output"]["type"], fmt(t), fmt(o),
            f'{aa["coverage"]:.0%} @ {aa["precision"]}', r["robustness"]["group_flip_rate"],
            r["determinism"]["decision_flips_same_batch"],
            r["determinism"]["decision_flips_bs32_vs_bs1"]]) + " |")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
