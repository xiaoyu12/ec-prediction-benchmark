#!/usr/bin/env python
"""Per-enzyme F1 of prediction files by top-hit identity and by EC rarity (pooled full-protocol tests).

    python3 scripts/bins_from_predictions.py --out reports/manuscript/data/published_bins.json \\
        "Homology only=features/predictions_published/Homology_only.csv" \\
        "EnzymeHunter=features/predictions_published/EnzymeHunter.csv" ...

Takes ``Entry,EC number`` files (``;``-separated calls, as ``bootstrap_ci.py --dump-predictions`` writes
them), so methods that are not ``runs/feat`` runs (published checkpoints, external tools) are binned the
same way as ours. Identity bins follow ``scripts/stratify.py`` (the higher of the MMseqs2 and
Foldseek-AFDB top-hit identities, full protocol); rarity bins follow ``scripts/rare_eval.py`` (training
count in all of split100 of the query's rarest true EC). The six test sets are pooled (3,394 queries).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rare_eval import BINS as RARITY, training_counts
from stratify import BINS as IDENTITY, ROOT, RUNS, TESTS, load, read_labels


def read_calls(path: Path) -> dict[str, set[str]]:
    with open(path) as handle:
        return {r["Entry"]: {e for e in (r["EC number"] or "").split(";") if e} for r in csv.DictReader(handle)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", metavar="NAME=CSV")
    ap.add_argument("--homology", nargs="+", default=["mmseqs", "foldseek_afdb"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    classes = np.load(ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    counts = training_counts("full", classes)
    models = {m.split("=", 1)[0]: read_calls(Path(m.split("=", 1)[1])) for m in args.models}
    ident, rarity, f1 = [], [], {name: [] for name in models}
    for name in TESTS:
        labels = read_labels(ROOT / "features" / "tests" / f"{name}.csv")
        entries = load("full", args.homology[0])[name][1]
        best = np.full(len(entries), -1.0)
        for h in args.homology:
            hits = np.load(RUNS / f"full_{h}" / "hits.npz")[name]
            where = {e: i for i, e in enumerate(load("full", h)[name][1])}
            rows = hits[[where[e] for e in entries]]
            best = np.maximum(best, np.where(rows[:, 1] > 0, rows[:, 0], -1.0))
        ident.append(best)
        for e in entries:
            true = set(labels[e][0])
            rarity.append(min(counts.get(ec, 0) for ec in true))
            for m, calls in models.items():
                pred = calls[e]
                tp = len(pred & true)
                f1[m].append(2 * tp / (len(pred) + len(true)) if tp else 0.0)
    ident, rarity = np.concatenate(ident), np.array(rarity)
    f1 = {m: np.array(v) for m, v in f1.items()}
    out = {"n": len(ident), "identity": [], "rarity": [], "all": {m: float(v.mean()) for m, v in f1.items()}}
    for label, lo, hi in IDENTITY:
        mask = ident < 0 if label == "no hit" else (ident >= lo) & (ident < hi)
        out["identity"].append({"bin": label, "n": int(mask.sum()), "f1": {m: float(v[mask].mean()) for m, v in f1.items()}})
    for label, lo, hi in RARITY:
        mask = (rarity >= lo) & (rarity < hi)
        if mask.any():
            out["rarity"].append({"bin": label, "n": int(mask.sum()), "f1": {m: float(v[mask].mean()) for m, v in f1.items()}})
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    names = list(models)
    for key in ("identity", "rarity"):
        print(f"## {key}\n{'bin':8s} {'n':>5s} " + " ".join(f"{m[:13]:>13s}" for m in names))
        for b in out[key]:
            print(f"{b['bin']:8s} {b['n']:5d} " + " ".join(f"{b['f1'][m]:13.3f}" for m in names))
    print("all", {m: round(v, 3) for m, v in out["all"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
