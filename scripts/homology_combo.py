#!/usr/bin/env python
"""Homology-only baselines that combine two homology.py runs, with no learned model.

    PYTHONPATH=code .venv/bin/python scripts/homology_combo.py --protocol full --runs mmseqs foldseek_afdb

Two parameter-free rules, both scored per enzyme like every other run:

- ``union``: an EC is called if either tool transfers it (element-wise max of the transfer logits);
- ``best``: per query, keep the calls of the tool whose top hit has the higher identity.

Results go to ``runs/feat/<protocol>_hom_<rule>_<run1>_<run2>`` in the usual format.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from ensemble import RUNS, align, load
from infer import call_ec
from train import PACKAGE_ROOT, read_labels
from train_feat import TESTS, score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=["id50", "full"], required=True)
    p.add_argument("--runs", nargs=2, required=True)
    args = p.parse_args()
    classes = np.load(PACKAGE_ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    a, b = (load(args.protocol, r) for r in args.runs)
    hits = [np.load(RUNS / f"{args.protocol}_{r}" / "hits.npz") for r in args.runs]
    train = read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv")
    for rule in ("union", "best"):
        metrics, logits, entries = {}, {}, {}
        for n in a:
            entries[n] = a[n][1]
            la, lb = a[n][0], align(b[n], entries[n])
            ia = align((hits[0][n][:, 0], a[n][1]), entries[n])
            ib = align((hits[1][n][:, 0], b[n][1]), entries[n])
            logits[n] = np.maximum(la, lb) if rule == "union" else np.where((ia >= ib)[:, None], la, lb)
            labels = train if n == "id50" else read_labels(PACKAGE_ROOT / "features" / "tests" / f"{n}.csv")
            metrics[n] = score(logits[n], [labels[str(e)][0] for e in entries[n]], classes)
        name = f"{args.protocol}_hom_{rule}_{'_'.join(args.runs)}"
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"features": [f"hom_{rule}({'+'.join(args.runs)})"],
                                                     "protocol": args.protocol, "arch": "homology"}, indent=2) + "\n")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        np.savez_compressed(out / "logits.npz", **{n: v.astype(np.float16) for n, v in logits.items()},
                            **{f"{n}__entries": entries[n] for n in logits})
        with open(out / "predictions.csv", "w") as handle:
            handle.write("Entry,EC number\n")
            written = set()
            for n in TESTS:
                if n not in logits:
                    continue
                for e, calls in zip(entries[n], call_ec(logits[n])):
                    if str(e) not in written:
                        handle.write(f"{e},{';'.join(classes[calls])}\n")
                        written.add(str(e))
        tests = [metrics[n]["sample"] for n in TESTS if n in metrics]
        print(name, "  ".join(f"{n}: {m['sample']:.3f}" for n, m in metrics.items()),
              f" test mean: {np.mean(tests):.3f}" if len(tests) == len(TESTS) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
