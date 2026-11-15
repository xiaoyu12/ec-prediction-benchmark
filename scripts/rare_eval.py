#!/usr/bin/env python
"""Per-enzyme F1 by EC rarity, and post-hoc logit adjustment for rare classes.

    python scripts/rare_eval.py --runs c3_s1 c3_proto --protocol id50
    python scripts/rare_eval.py --runs esm2_3b_last_esmc600_saemax --protocol full --adjust 0.5:50 1:50
    python scripts/rare_eval.py --runs c3_s1 --protocol id50 --adjust 0.5:50 --write

A query's rarity is the training count of its rarest true EC (training set of the run's protocol:
split100 minus the id50 fold for ``id50``, all of split100 for ``full``). Bins: unseen, 1-9, 10-49,
50-199, >=200. "tests" pools the six test sets' queries; "test mean" is the usual unweighted mean
of the six per-set F1s.

``--adjust tau:n0`` adds ``tau * max(0, log(n0 / n_c))`` to every class's logit (``n_c`` its
training count, at least 1): classes rarer than ``n0`` are boosted, common ones untouched. This is
logit adjustment (Menon et al. 2021) restricted to the tail. ``--write`` saves each adjusted run as
``runs/feat/<protocol>_<run>_la<tau>_<n0>`` (same files as a training run) for ensembling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from infer import call_ec  # noqa: E402
from train import MIN_LENGTH, read_labels  # noqa: E402
from train_feat import FOLDS, TESTS, score  # noqa: E402

RUNS = ROOT / "runs" / "feat"
BINS = [("unseen", 0, 1), ("1-9", 1, 10), ("10-49", 10, 50), ("50-199", 50, 200), (">=200", 200, 10 ** 9)]


def training_counts(protocol: str, classes: np.ndarray) -> dict[str, int]:
    train = read_labels(ROOT / "features" / "split100_min5.csv")
    skip = set()
    if protocol == "id50":
        skip = set((FOLDS / "id50_val.txt").read_text().split()) | set((FOLDS / "id50_exclude.txt").read_text().split())
    known = set(classes)
    counts: dict[str, int] = {}
    for entry, (ecs, length) in train.items():
        if entry in skip or (length is not None and length < MIN_LENGTH):
            continue
        for ec in set(ecs) & known:
            counts[ec] = counts.get(ec, 0) + 1
    return counts


def per_enzyme_f1(logits: np.ndarray, truth: list[set], classes: np.ndarray) -> np.ndarray:
    out = []
    for calls, true in zip(call_ec(logits.astype(np.float32)), truth):
        pred = set(classes[calls])
        tp = len(pred & true)
        out.append(2 * tp / (len(pred) + len(true)) if tp else 0.0)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run names without the protocol prefix")
    ap.add_argument("--protocol", choices=["id50", "full"], required=True)
    ap.add_argument("--adjust", nargs="*", default=[], help="tau:n0 pairs")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--shift", type=float, default=0.0, help="global logit shift added before the decision rule")
    args = ap.parse_args()
    classes = np.load(ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    counts = training_counts(args.protocol, classes)
    n_c = np.array([max(counts.get(ec, 0), 1) for ec in classes], dtype=np.float64)
    truth_of = {n: read_labels(ROOT / "features" / "tests" / f"{n}.csv") for n in TESTS}
    truth_of["id50"] = read_labels(ROOT / "features" / "split100_min5.csv")

    variants = [(run, None) for run in args.runs]
    variants += [(run, tuple(float(x) for x in a.split(":"))) for run in args.runs for a in args.adjust]
    header = ["set", *[b[0] for b in BINS], "all"]
    print(f"rarity = training count of the query's rarest true EC ({args.protocol} training set)\n")
    for run, adjust in variants:
        data = np.load(RUNS / f"{args.protocol}_{run}" / "logits.npz")
        shift = np.full(len(classes), args.shift, dtype=np.float32)
        if adjust:
            tau, n0 = adjust
            shift = shift + (tau * np.maximum(0.0, np.log(n0 / n_c))).astype(np.float32)
        label = run + (f"  +adjust tau={adjust[0]:g} n0={adjust[1]:g}" if adjust else "")
        rows, pooled, logits_out, metrics = [], {"f1": [], "rare": []}, {}, {}
        names = [k for k in data.files if "__" not in k]
        for name in names:
            entries = data[f"{name}__entries"].astype(str)
            logits = data[name].astype(np.float32) + shift
            truth = [set(truth_of[name][e][0]) for e in entries]
            rare = np.array([min(counts.get(ec, 0) for ec in t) for t in truth])
            f1 = per_enzyme_f1(logits, truth, classes)
            logits_out[name] = logits
            metrics[name] = score(logits, [truth_of[name][e][0] for e in entries], classes)
            if name == "id50":
                rows.append(("id50", f1, rare))
            else:
                pooled["f1"].append(f1)
                pooled["rare"].append(rare)
        rows.append(("tests (pooled)", np.concatenate(pooled["f1"]), np.concatenate(pooled["rare"])))
        test_mean = np.mean([metrics[n]["sample"] for n in TESTS])
        print(f"## {label}   test mean {test_mean:.3f}" + (f"   id50 {metrics['id50']['sample']:.3f}"
                                                         if "id50" in metrics else ""))
        print("| " + " | ".join(header) + " |")
        print("|---|" + "---:|" * (len(header) - 1))
        for set_name, f1, rare in rows:
            cells = []
            for _, lo, hi in BINS:
                m = (rare >= lo) & (rare < hi)
                cells.append(f"{f1[m].mean():.3f} ({m.sum()})" if m.any() else "–")
            print(f"| {set_name} | " + " | ".join(cells) + f" | {f1.mean():.3f} |")
        print()
        if adjust and args.write:
            out = RUNS / f"{args.protocol}_{run}_la{adjust[0]:g}_{adjust[1]:g}"
            out.mkdir(parents=True, exist_ok=True)
            config = json.loads((RUNS / f"{args.protocol}_{run}" / "config.json").read_text())
            config.update(base_run=f"{args.protocol}_{run}", logit_adjust={"tau": adjust[0], "n0": adjust[1]})
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            np.savez_compressed(out / "logits.npz", **{n: v.astype(np.float16) for n, v in logits_out.items()},
                                **{f"{n}__entries": data[f"{n}__entries"] for n in names})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
