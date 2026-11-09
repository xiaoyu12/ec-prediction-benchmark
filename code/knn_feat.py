#!/usr/bin/env python
"""Nearest-neighbour EC transfer in embedding space (cosine similarity), on the same features,
data and protocols as ``train_feat.py``.

    python knn_feat.py --features esmc-600m:mid --protocol id50 --k 10 --temperature 0.02 --output-dir ../runs/feat/knn_x

Each block is standardized with training statistics and L2-normalized; blocks are concatenated
with equal weight (each block normalized again to unit length / sqrt(#blocks)). For a query the
score of EC ``c`` is ``sum_j w_j y_jc`` over its ``k`` nearest training proteins, with
``w = softmax(sim / T)``; the logit of that score goes through ``infer.call_ec`` like any model.
Writes the same ``metrics.json``, ``logits.npz`` and ``predictions.csv`` as ``train_feat.py``,
plus ``neighbours.npz`` (top-1 similarity per query, for gating ensembles).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from train import PACKAGE_ROOT, read_labels
from train_feat import FOLDS, TESTS, Bank, labelled, score

EPS = 1e-6


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--protocol", choices=["id50", "full"], required=True)
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--temperature", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    classes = np.load(args.classes, allow_pickle=True).astype(str)
    class_index = {ec: i for i, ec in enumerate(classes)}
    log1p = {s for s in args.features if "sae" in s}
    bank = Bank(args.features, log1p, args.device)
    train_rows, train_ecs, train_classes = labelled(bank, read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv"),
                                                    class_index, training=True)
    eval_sets = {}
    if args.protocol == "id50":
        held = set((FOLDS / "id50_val.txt").read_text().split())
        dropped = set((FOLDS / "id50_exclude.txt").read_text().split())
        entries = [bank.entries[r] for r in train_rows]
        val = [i for i, e in enumerate(entries) if e in held]
        keep = [i for i, e in enumerate(entries) if e not in held and e not in dropped]
        eval_sets["id50"] = (train_rows[val], [train_ecs[i] for i in val])
        train_rows, train_classes = train_rows[keep], [train_classes[i] for i in keep]
    for name in TESTS:
        rows, ecs, _ = labelled(bank, read_labels(PACKAGE_ROOT / "features" / "tests" / f"{name}.csv"), class_index, False)
        eval_sets[name] = (rows, ecs)

    rows = torch.from_numpy(train_rows).to(args.device)
    bank.fit(rows)

    def embed(index: np.ndarray | torch.Tensor) -> torch.Tensor:
        index = torch.as_tensor(index, device=args.device)
        parts = []
        for chunk in index.split(8192):
            blocks = [torch.nn.functional.normalize(b, dim=1) for b in bank.batch(chunk)]
            parts.append((torch.cat(blocks, 1) / len(blocks) ** 0.5).half())
        return torch.cat(parts)

    reference = embed(rows)  # (n_train, D) half
    labels = torch.zeros(len(train_classes), len(classes), dtype=torch.bool)
    for i, c in enumerate(train_classes):
        labels[i, c] = True
    labels = labels.to(args.device)
    kmax = max(args.k)

    neighbours = {}
    for name, (r, _) in eval_sets.items():
        sims, idx = [], []
        for chunk in embed(r).split(1024):
            s, i = (chunk @ reference.T).float().topk(kmax, dim=1)
            sims.append(s)
            idx.append(i)
        neighbours[name] = (torch.cat(sims), torch.cat(idx))

    def logits_for(name: str, k: int, t: float) -> np.ndarray:
        s, i = neighbours[name]
        w = torch.softmax(s[:, :k] / t, dim=1)
        prob = torch.zeros(len(s), len(classes), device=args.device)
        for j in range(k):
            prob += w[:, j:j + 1] * labels[i[:, j]].float()
        prob = prob.clamp(EPS, 1 - EPS)
        return torch.log(prob / (1 - prob)).cpu().numpy()

    select_on = "id50" if "id50" in eval_sets else "new"
    grid = {}
    for k in args.k:
        for t in args.temperature if k > 1 else [1.0]:
            grid[(k, t)] = {n: score(logits_for(n, k, t), ecs, classes) for n, (_, ecs) in eval_sets.items()}
            print(f"k={k:3d} T={t:<5} " + "  ".join(f"{n}: {m['sample']:.3f}" for n, m in grid[(k, t)].items()),
                  flush=True)
    best = max(grid, key=lambda key: grid[key][select_on]["sample"]) if args.protocol == "id50" else (args.k[0], args.temperature[0])
    k, t = best
    print(f"chosen k={k} T={t} (by {select_on} per-enzyme F1)" if args.protocol == "id50" else f"k={k} T={t}")
    logits = {n: logits_for(n, k, t) for n in eval_sets}
    metrics = grid[best]
    config = {"features": args.features, "protocol": args.protocol, "arch": "knn", "loss": "-", "hidden": k,
              "layers": 0, "k": k, "temperature": t, "grid": {f"k{a}_T{b}": v for (a, b), v in grid.items()}}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "logits.npz", **{n: v.astype(np.float16) for n, v in logits.items()},
                        **{f"{n}__entries": np.array([bank.entries[i] for i in r]) for n, (r, _) in eval_sets.items()})
    np.savez_compressed(out / "neighbours.npz", **{n: v[0][:, 0].cpu().numpy() for n, v in neighbours.items()})
    from infer import call_ec

    with open(out / "predictions.csv", "w") as handle:
        handle.write("Entry,EC number\n")
        written = set()
        for name in TESTS:
            r, _ = eval_sets[name]
            for row, calls in zip(r, call_ec(logits[name])):
                if bank.entries[row] not in written:
                    handle.write(f"{bank.entries[row]},{';'.join(classes[calls])}\n")
                    written.add(bank.entries[row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
