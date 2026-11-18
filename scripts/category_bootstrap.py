#!/usr/bin/env python
"""Paired bootstrap of several runs against one reference run, per test set and for the test mean.

    python3 scripts/category_bootstrap.py --ref hom_best_mmseqs_foldseek_afdb \\
        --runs esm2_3b_last ens6_6b hyb_ens6_6b_msaf --out reports/manuscript/data/bootstrap_vs_homology.json

Uses the full-protocol runs' saved logits and the usual decision rule. Each resample draws the
enzymes of every test set with replacement (2,000 resamples, seed 0); all runs are scored on the same
draws, so differences are paired. The test-mean difference averages the six per-set differences of
the same resample. An interval excluding 0 marks a significant difference.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from stratify import ROOT, TESTS, load, per_enzyme_f1, read_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    classes = np.load(ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    rng = np.random.default_rng(0)
    diffs = {r: {} for r in args.runs}
    for name in TESTS:
        labels = read_labels(ROOT / "features" / "tests" / f"{name}.csv")
        entries = load("full", args.ref)[name][1]
        truth = [set(labels[e][0]) for e in entries]

        def f1(run):
            values, have = load("full", run)[name]
            where = {e: i for i, e in enumerate(have)}
            return per_enzyme_f1(values[[where[e] for e in entries]], truth, classes)

        ref = f1(args.ref)
        draws = rng.integers(0, len(entries), size=(args.resamples, len(entries)))
        for r in args.runs:
            d = f1(r) - ref
            diffs[r][name] = (float(d.mean()), d[draws].mean(axis=1))
    out = {"ref": args.ref, "resamples": args.resamples, "runs": {}}
    print(f"difference from {args.ref} (95% CI; * excludes 0)")
    for r in args.runs:
        row = {}
        for name in TESTS:
            point, boot = diffs[r][name]
            row[name] = [point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        mean_boot = np.mean([diffs[r][n][1] for n in TESTS], axis=0)
        point = float(np.mean([diffs[r][n][0] for n in TESTS]))
        row["test_mean"] = [point, float(np.percentile(mean_boot, 2.5)), float(np.percentile(mean_boot, 97.5))]
        out["runs"][r] = row
        print(f"{r:34s} " + "  ".join(
            f"{k[:6]} {v[0]:+.3f}{'*' if v[1] > 0 or v[2] < 0 else ' '}" for k, v in row.items()))
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
