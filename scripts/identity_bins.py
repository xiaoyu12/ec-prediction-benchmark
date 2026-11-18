#!/usr/bin/env python
"""Per-enzyme F1 of any set of runs, by top-hit identity to training, written as JSON for the manuscript.

    python3 scripts/identity_bins.py --runs hom_best_mmseqs_foldseek_afdb esm2_3b_last ... \\
        --homology mmseqs foldseek_afdb --out reports/manuscript/data/identity_bins.json

Bins and pooling follow ``scripts/stratify.py``: identity is the highest top-hit identity over the
homology runs, and the six full-protocol test sets are pooled (3,394 queries). id50 uses the
id50-protocol runs. Each bin also gets a paired bootstrap interval (1,000 resamples over queries)
for every run's difference from the first run.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from stratify import BINS, ROOT, RUNS, TESTS, load, per_enzyme_f1, read_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="first run is the reference for differences")
    ap.add_argument("--homology", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    classes = np.load(ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    train = read_labels(ROOT / "features" / "split100_min5.csv")
    truth_of = {"id50": train, **{n: read_labels(ROOT / "features" / "tests" / f"{n}.csv") for n in TESTS}}
    rng = np.random.default_rng(0)
    result = {}
    for scope, names in (("id50", ["id50"]), ("tests", TESTS)):
        protocol = "id50" if scope == "id50" else "full"
        idents, f1 = [], {r: [] for r in args.runs}
        for name in names:
            entries = load(protocol, args.runs[0])[name][1]
            ident = np.full(len(entries), -1.0)
            for h in args.homology:
                hits = np.load(RUNS / f"{protocol}_{h}" / "hits.npz")[name]
                where = {e: i for i, e in enumerate(load(protocol, h)[name][1])}
                rows = hits[[where[e] for e in entries]]
                ident = np.maximum(ident, np.where(rows[:, 1] > 0, rows[:, 0], -1.0))
            idents.append(ident)
            truth = [set(truth_of[name][e][0]) for e in entries]
            for r in args.runs:
                values, have = load(protocol, r)[name]
                where = {e: i for i, e in enumerate(have)}
                f1[r].append(per_enzyme_f1(values[[where[e] for e in entries]], truth, classes))
        ident = np.concatenate(idents)
        f1 = {r: np.concatenate(v) for r, v in f1.items()}
        bins = []
        for label, lo, hi in BINS:
            m = ident < 0 if label == "no hit" else (ident >= lo) & (ident < hi)
            idx = np.flatnonzero(m)
            row = {"bin": label, "n": int(m.sum()), "f1": {r: float(f1[r][m].mean()) for r in args.runs}, "diff_ci": {}}
            draws = rng.integers(0, len(idx), size=(1000, len(idx)))
            for r in args.runs[1:]:
                d = (f1[r] - f1[args.runs[0]])[idx][draws].mean(axis=1)
                row["diff_ci"][r] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            bins.append(row)
        result[scope] = {"n": len(ident), "bins": bins, "all": {r: float(f1[r].mean()) for r in args.runs}}
        print(f"## {scope} ({len(ident)} queries)")
        print(f"{'bin':8s} {'n':>5s} " + " ".join(f"{r[:14]:>14s}" for r in args.runs))
        for b in bins:
            print(f"{b['bin']:8s} {b['n']:5d} " + " ".join(f"{b['f1'][r]:14.3f}" for r in args.runs))
    with open(args.out, "w") as handle:
        json.dump({"runs": args.runs, "homology": args.homology, **result}, handle, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
