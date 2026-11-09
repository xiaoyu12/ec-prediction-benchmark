#!/usr/bin/env python
"""Ensembles of saved ``logits.npz`` (``train_feat.py`` / ``knn_feat.py`` / ``homology.py`` runs).

    python ensemble.py --protocol id50 --runs esmc_mid esm2_bce_e100 --name ens_a
    python ensemble.py --protocol id50 --runs esmc_mid --homology mmseqs --name hyb_a

Model probabilities are averaged (``--weights``, default equal). With ``--homology`` the MMseqs2
transfer is blended in per query by the top hit's identity: ``p = (1 - a) p_model + a p_hom`` with
``a = clip((identity - lo) / (hi - lo), 0, 1) * amax``; ``lo``/``hi``/``amax`` are chosen on id50
(``--protocol id50``) or passed explicitly (``--protocol full``, from the id50 choice).
Run directories are ``runs/feat/<protocol>_<run>``; output goes to ``runs/feat/<protocol>_<name>``
in the same format, so ensembles appear in ``scripts/feat_table.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from infer import call_ec
from train import PACKAGE_ROOT, read_labels
from train_feat import TESTS, FOLDS, score

RUNS = PACKAGE_ROOT / "runs" / "feat"
EPS = 1e-6


def load(protocol: str, run: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = np.load(RUNS / f"{protocol}_{run}" / "logits.npz")
    return {k: (data[k].astype(np.float32), data[f"{k}__entries"]) for k in data.files if "__" not in k}


def align(source: tuple[np.ndarray, np.ndarray], entries: np.ndarray) -> np.ndarray:
    values, have = source
    where = {e: i for i, e in enumerate(have)}
    return values[[where[e] for e in entries]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=["id50", "full"], required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--homology", nargs="+", help="homology.py run name(s) (e.g. mmseqs foldseek)")
    p.add_argument("--hom-combine", choices=["mean", "max"], default="mean",
                   help="several homology runs: average (mean) or union (max) of their EC calls; the gate "
                        "uses the highest top-hit identity")
    p.add_argument("--lo", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--hi", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    p.add_argument("--amax", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--name", required=True)
    p.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"))
    args = p.parse_args()
    classes = np.load(args.classes, allow_pickle=True).astype(str)
    weights = np.array(args.weights or [1.0] * len(args.runs))
    weights = weights / weights.sum()
    models = [load(args.protocol, r) for r in args.runs]
    names = list(models[0])
    entries = {n: models[0][n][1] for n in names}

    truth: dict[str, list[list[str]]] = {}
    train = read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv")
    for n in names:
        labels = train if n == "id50" else read_labels(PACKAGE_ROOT / "features" / "tests" / f"{n}.csv")
        truth[n] = [labels[str(e)][0] for e in entries[n]]

    prob = {n: sum(w / (1 + np.exp(-align(m[n], entries[n]))) for w, m in zip(weights, models)) for n in names}

    hom_prob, identity = {}, {}
    if args.homology:
        for h in args.homology:
            hom = load(args.protocol, h)
            hits = np.load(RUNS / f"{args.protocol}_{h}" / "hits.npz")
            for n in names:
                calls = (align(hom[n], entries[n]) >= 0).astype(np.float32)
                ident = align((hits[n][:, 0], hom[n][1]), entries[n])
                if n not in hom_prob:
                    hom_prob[n], identity[n] = calls / len(args.homology) if args.hom_combine == "mean" else calls, ident
                else:
                    hom_prob[n] = hom_prob[n] + calls / len(args.homology) if args.hom_combine == "mean" \
                        else np.maximum(hom_prob[n], calls)
                    identity[n] = np.maximum(identity[n], ident)

    def combine(n: str, lo: float, hi: float, amax: float) -> np.ndarray:
        p_ = prob[n]
        if args.homology:
            a = (np.clip((identity[n] - lo) / max(hi - lo, 1e-6), 0, 1) * amax)[:, None]
            p_ = (1 - a) * p_ + a * hom_prob[n]
        p_ = np.clip(p_, EPS, 1 - EPS)
        return np.log(p_ / (1 - p_))

    grid = [(0.0, 1.0, 0.0)]
    if args.homology:
        grid = [(lo, hi, am) for lo, hi, am in itertools.product(args.lo, args.hi, args.amax) if hi > lo]
    results = {g: {n: score(combine(n, *g), truth[n], classes) for n in names} for g in grid}
    if args.protocol == "id50" and len(grid) > 1:
        for g, r in sorted(results.items(), key=lambda kv: -kv[1]["id50"]["sample"])[:5]:
            print(f"lo={g[0]} hi={g[1]} amax={g[2]}  " + "  ".join(f"{n}: {r[n]['sample']:.3f}" for n in names))
    chosen = max(results, key=lambda g: results[g]["id50"]["sample"]) if "id50" in names else grid[0]
    metrics = results[chosen]
    print("chosen", chosen, "  ".join(f"{n}: {metrics[n]['sample']:.3f}/{metrics[n]['micro']:.3f}" for n in names))

    out = RUNS / f"{args.protocol}_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    logits = {n: combine(n, *chosen) for n in names}
    config = {"features": [f"ens({'+'.join(args.runs)})" + (f"+hom({'+'.join(args.homology)})" if args.homology else "")],
              "protocol": args.protocol, "arch": "ensemble", "loss": "-", "hidden": 0, "layers": 0,
              "runs": args.runs, "weights": weights.tolist(), "homology": args.homology,
              "lo_hi_amax": list(chosen)}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "logits.npz", **{n: v.astype(np.float16) for n, v in logits.items()},
                        **{f"{n}__entries": entries[n] for n in names})
    with open(out / "predictions.csv", "w") as handle:
        handle.write("Entry,EC number\n")
        written = set()
        for n in TESTS:
            for e, calls in zip(entries[n], call_ec(logits[n])):
                if str(e) not in written:
                    handle.write(f"{e},{';'.join(classes[calls])}\n")
                    written.add(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
