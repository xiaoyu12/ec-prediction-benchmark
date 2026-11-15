#!/usr/bin/env python
"""Where are the remaining errors? Per-enzyme F1 by top-hit identity to the training set.

    python scripts/stratify.py --hybrid hyb_ens6_6b_msaf --model ens6_6b --single esm2_3b_last_esmc600_saemax \\
        --homology mmseqs foldseek_afdb

Each query's identity is the highest top-hit identity over the homology runs (``hits.npz``; 0 when
no tool finds a hit at e-value 1e-3). Queries are binned (no hit, <30%, 30-50%, 50-70%, >=70%) and
each run is scored per bin with the usual decision rule. Tests use the full-protocol runs, id50 the
id50-protocol runs. "Error share" is a bin's part of the total per-enzyme F1 deficit,
sum(1 - F1_i), over the pooled test queries: where improvements can still come from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from infer import call_ec  # noqa: E402
from train import read_labels  # noqa: E402
from train_feat import TESTS  # noqa: E402

RUNS = ROOT / "runs" / "feat"
BINS = [("no hit", -1, 0), ("<30%", 0, 0.3), ("30-50%", 0.3, 0.5), ("50-70%", 0.5, 0.7), (">=70%", 0.7, 1.01)]


def load(protocol, run):
    data = np.load(RUNS / f"{protocol}_{run}" / "logits.npz")
    return {k: (data[k], data[f"{k}__entries"].astype(str)) for k in data.files if "__" not in k}


def per_enzyme_f1(logits, truth, classes):
    out = []
    for calls, true in zip(call_ec(logits.astype(np.float32)), truth):
        pred = set(classes[calls])
        tp = len(pred & true)
        out.append(2 * tp / (len(pred) + len(true)) if tp else 0.0)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hybrid", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--homology", nargs="+", required=True)
    args = ap.parse_args()
    classes = np.load(ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    train = read_labels(ROOT / "features" / "split100_min5.csv")
    truth_of = {"id50": train, **{n: read_labels(ROOT / "features" / "tests" / f"{n}.csv") for n in TESTS}}
    runs = {"hybrid": args.hybrid, "models": args.model, "single": args.single, **{h: h for h in args.homology}}

    pooled = {k: [] for k in runs}
    pooled_ident = []
    for name in ["id50", *TESTS]:
        protocol = "id50" if name == "id50" else "full"
        entries = load(protocol, args.hybrid)[name][1]
        ident = np.full(len(entries), -1.0)
        for h in args.homology:
            hits = np.load(RUNS / f"{protocol}_{h}" / "hits.npz")[name]
            have = load(protocol, h)[name][1]
            where = {e: i for i, e in enumerate(have)}
            ident = np.maximum(ident, np.where(hits[[where[e] for e in entries], 1] > 0,
                                               hits[[where[e] for e in entries], 0], -1.0))
        truth = [set(truth_of[name][e][0]) for e in entries]
        f1 = {}
        for key, run in runs.items():
            values, have = load(protocol, run)[name]
            where = {e: i for i, e in enumerate(have)}
            f1[key] = per_enzyme_f1(values[[where[e] for e in entries]], truth, classes)
        print(f"\n## {name} ({len(entries)} queries; {protocol} protocol)")
        print("| bin | n | " + " | ".join(runs) + " |")
        print("|---|---:|" + "---:|" * len(runs))
        for label, lo, hi in BINS:
            m = ident < 0 if label == "no hit" else (ident >= lo) & (ident < hi)
            if m.sum():
                print(f"| {label} | {m.sum()} | " + " | ".join(f"{f1[k][m].mean():.3f}" for k in runs) + " |")
        print(f"| all | {len(entries)} | " + " | ".join(f"{f1[k].mean():.3f}" for k in runs) + " |")
        if name != "id50":
            for k in runs:
                pooled[k].append(f1[k])
            pooled_ident.append(ident)

    ident = np.concatenate(pooled_ident)
    f1 = {k: np.concatenate(v) for k, v in pooled.items()}
    deficit = (1 - f1["hybrid"]).sum()
    print(f"\n## pooled test queries ({len(ident)})")
    print("| bin | n | share of queries | " + " | ".join(runs) + " | hybrid error share |")
    print("|---|---:|---:|" + "---:|" * len(runs) + "---:|")
    for label, lo, hi in BINS:
        if label == "no hit":
            m = ident < 0
        else:
            m = (ident >= lo) & (ident < hi)
        print(f"| {label} | {m.sum()} | {m.mean():.3f} | " + " | ".join(f"{f1[k][m].mean():.3f}" for k in runs)
              + f" | {(1 - f1['hybrid'][m]).sum() / deficit:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
