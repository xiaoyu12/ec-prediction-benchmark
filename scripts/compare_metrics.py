#!/usr/bin/env python
"""Compare released and retrained models with metrics that charge every false positive.

    python scripts/compare_metrics.py    # writes features/logs/compare_metrics.json and prints the tables

Metrics on the six test sets:

``weighted``  CLEAN's support-weighted F1 (the paper's); a wrong call of an EC absent from the set's
              truth is free.
``sample``    per-enzyme F1 averaged over enzymes (protein-centric); every wrong call costs.
``micro``     F1 pooled over all (enzyme, EC) decisions; every wrong call costs.
``Fmax``      best per-enzyme F1 over logit shifts 0..8 (CAFA-style): calibration-free, but chosen on
              the test set itself, so it ranks models rather than giving a deployable number.

Sigmoid models are scored at shift 0 (the paper's rule), at the shift in their calibration.json
(chosen on the id50 fold by weighted F1), and at a shift chosen on the id50 fold by per-enzyme F1
(using the id50-trained checkpoint of the same recipe). MIL-CL+ uses max-separation (no threshold).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from infer import apply_max_separation, call_ec  # noqa: E402
from models import load_milcl  # noqa: E402
from train import (  # noqa: E402
    SHIFT_GRID,
    attach_labels,
    build_model,
    calls_to_ecs,
    fetch,
    load_archive,
    load_archives,
    micro_prf,
    read_labels,
    sample_prf,
    subset,
    weighted_prf,
)

F = ROOT / "features"
TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
METRICS = {"weighted": weighted_prf, "sample": sample_prf, "micro": micro_prf}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SIGMOID = {  # label -> (model, full-data checkpoint, id50 checkpoint of the same recipe or None)
    "released MIL-BC": ("milbc", "checkpoints/milbc/model_v1.pth", None),
    "released MIL-BC+": ("milbc_plus", "checkpoints/milbc_plus/model_v1.pth", None),
    "retrained MIL-BC": ("milbc", "checkpoints/retrained/milbc/model.pth", "runs/id50/milbc/model.pth"),
    "retrained MIL-BC+": ("milbc_plus", "checkpoints/retrained/milbc_plus/model.pth", "runs/id50/milbc_plus/model.pth"),
    "retrained ESM2-BC": ("esm2_bc", "checkpoints/retrained/esm2_bc/model.pth", "runs/id50/esm2_bc/model.pth"),
}
MAXSEP = {
    "released MIL-CL+": ("checkpoints/milcl_plus/model_v2.pt", "checkpoints/milcl_plus/centroids.npy"),
    "retrained MIL-CL+": ("checkpoints/retrained/milcl_plus/model.pt", "checkpoints/retrained/milcl_plus/centroids.npy"),
}


def logits_of(model, name, store, rows):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            d, m, e = fetch(store, torch.from_numpy(rows[start:start + 2048]), DEVICE, local_only=True)
            out.append(model(d, m, None if name == "milbc" else e)[0].float().cpu().numpy())
    return np.concatenate(out)


def load(name, path):
    model = build_model(name)
    model.load_state_dict(torch.load(ROOT / path, map_location="cpu", weights_only=True))
    return model.to(DEVICE)


def scores(true_ecs, calls, classes):
    predicted = calls_to_ecs(calls, classes)
    out = {k: f(true_ecs, predicted)["f1"] for k, f in METRICS.items()}
    sp = sample_prf(true_ecs, predicted)
    out.update(sample_precision=sp["precision"], sample_recall=sp["recall"], calls=float(calls.sum(1).mean()))
    return out


def main() -> int:
    classes = np.load(ROOT / "checkpoints/classes.npy", allow_pickle=True).astype(str)
    index = {c: i for i, c in enumerate(classes)}
    tests = {}
    for n in TESTS:
        st = load_archive(F / f"tests/{n}.npz").to(DEVICE)
        tests[n] = (st, attach_labels(st, read_labels(F / f"tests/{n}.csv"), index, n, False))

    # id50 validation set, to choose a shift by per-enzyme F1 on checkpoints that never saw it
    train = load_archives([F / f"shards/split100_{i:02d}.npz" for i in range(16)]).to(DEVICE)
    labelled = attach_labels(train, read_labels(F / "split100_min5.csv"), index, "train", True)
    held = set((ROOT / "scripts/folds/id50_val.txt").read_text().split())
    val = subset(labelled, np.array([i for i, r in enumerate(labelled.rows) if train.entries[r] in held]))

    results: dict[str, dict] = {}
    for label, (name, full, id50) in SIGMOID.items():
        model = load(name, full)
        test_logits = {n: logits_of(model, name, st, ds.rows) for n, (st, ds) in tests.items()}
        rules = {"shift 0": 0.0}
        sidecar = (ROOT / full).parent / "calibration.json"
        if sidecar.is_file():
            rules["shift by weighted F1 on id50"] = json.loads(sidecar.read_text())["logit_shift"]
        if id50:
            val_logits = logits_of(load(name, id50), name, train, val.rows)
            curve = [(sample_prf(val.true_ecs, calls_to_ecs(call_ec(val_logits, s), classes))["f1"], -s) for s in SHIFT_GRID]
            rules["shift by per-enzyme F1 on id50"] = -max(curve)[1]
        for rule, shift in rules.items():
            key = f"{label} | {rule} ({shift:+.2f})"
            results[key] = {n: scores(tests[n][1].true_ecs, call_ec(lg, shift), classes) for n, lg in test_logits.items()}
            print(key, flush=True)
        fmax = {}
        for n, lg in test_logits.items():
            fmax[n] = max(sample_prf(tests[n][1].true_ecs, calls_to_ecs(call_ec(lg, s), classes))["f1"] for s in SHIFT_GRID)
        results[f"{label} | Fmax"] = {n: {"Fmax": v} for n, v in fmax.items()}

    for label, (ckpt, centroids) in MAXSEP.items():
        model, centers = load_milcl(str(ROOT / ckpt), str(ROOT / centroids), DEVICE)
        res = {}
        for n, (st, ds) in tests.items():
            chunks = []
            with torch.no_grad():
                for start in range(0, len(ds.rows), 2048):
                    d, m, e = fetch(st, torch.from_numpy(ds.rows[start:start + 2048]), DEVICE, local_only=False)
                    chunks.append(model(d, m, e).float().cpu().numpy())
            calls, _ = apply_max_separation(np.concatenate(chunks), centers)
            res[n] = scores(ds.true_ecs, calls, classes)
        results[f"{label} | max-separation"] = res
        print(label, flush=True)

    (F / "logs/compare_metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    short = {"new": "New", "price": "Price", "uniprot-all": "U-all", "uniprot-multi": "U-multi", "test70": "T-70", "test50": "T-50"}
    for metric in ["weighted", "sample", "micro", "Fmax", "calls"]:
        print(f"\n### {metric}")
        print(f"{'model | rule':62s}" + "".join(f"{short[n]:>8s}" for n in TESTS))
        for key, res in results.items():
            if metric not in res["new"]:
                continue
            print(f"{key[:62]:62s}" + "".join(f"{res[n][metric]:8.3f}" for n in TESTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
