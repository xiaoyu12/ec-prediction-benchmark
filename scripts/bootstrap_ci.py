#!/usr/bin/env python
"""Paired bootstrap confidence intervals for model scores and model differences on the test sets.

    python scripts/bootstrap_ci.py [--resamples 2000] [--seed 0]

Every model makes its calls once (the paper's rules: sigmoid >= 0.5 with call_ec's refinement, or
max-separation for MIL-CL+). Each resample draws the enzymes of a test set with replacement and
scores *every* model on the same draw, so a difference between two models gets a paired interval.
Metrics: per-enzyme F1 (``sample``), ``micro`` F1, and CLEAN's support-weighted F1 (``weighted``).

Writes ``features/logs/bootstrap_ci.json`` and prints, per metric, each model's score with its 95 %
percentile interval, then the paired differences with their interval and the share of resamples in
which the difference is <= 0 (a one-sided bootstrap p-value for "A beats B").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_metrics import DEVICE, MAXSEP, SIGMOID, TESTS, load, logits_of  # noqa: E402
from infer import apply_max_separation, call_ec  # noqa: E402
from models import load_milcl  # noqa: E402
from train import attach_labels, calls_to_ecs, fetch, load_archive, read_labels  # noqa: E402

F = ROOT / "features"
SHORT = {
    "released MIL-BC": "rel BC", "released MIL-BC+": "rel BC+", "released MIL-CL+": "rel CL+",
    "retrained MIL-BC": "ret BC", "retrained MIL-BC+": "ret BC+", "retrained ESM2-BC": "ret ESM2",
    "retrained MIL-CL+": "ret CL+",
}
EXTERNAL_REFERENCES = ["released MIL-BC+", "retrained MIL-BC+", "retrained ESM2-BC", "released MIL-CL+"]


def read_external(path: Path) -> dict[str, list[str]]:
    """``Entry -> predicted ECs`` from another tool's table (first EC-like column after Entry)."""
    import csv
    text = path.read_text().splitlines()
    delimiter = "\t" if text[0].count("\t") > text[0].count(",") else ","
    rows = list(csv.DictReader(text, delimiter=delimiter))
    column = next(c for c in rows[0] if c != "Entry" and "ec" in c.lower())
    return {r["Entry"]: [ec.strip() for ec in (r[column] or "").split(";") if ec.strip()] for r in rows}


COMPARISONS = [  # (A, B): is A better than B?
    ("retrained MIL-BC+", "released MIL-BC+"),
    ("retrained MIL-BC", "released MIL-BC"),
    ("retrained MIL-CL+", "released MIL-CL+"),
    ("released MIL-BC+", "retrained ESM2-BC"),
    ("retrained MIL-BC+", "retrained ESM2-BC"),
    ("released MIL-BC", "retrained ESM2-BC"),
    ("released MIL-BC+", "released MIL-BC"),
    ("released MIL-BC+", "released MIL-CL+"),
]


def per_enzyme(true_ecs, predicted):
    """Per-enzyme F1, true positives, number of calls and number of true labels."""
    f1, tp, called, support = [], [], [], []
    for t, p in zip(true_ecs, predicted):
        t, p = set(t), set(p)
        hit = len(t & p)
        pr = hit / len(p) if p else 0.0
        rc = hit / len(t) if t else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        tp.append(hit)
        called.append(len(p))
        support.append(len(t))
    return [torch.tensor(x, dtype=torch.float64, device=DEVICE) for x in (f1, tp, called, support)]


def indicator(rows, vocabulary):
    index = {ec: j for j, ec in enumerate(vocabulary)}
    out = torch.zeros(len(rows), len(vocabulary), dtype=torch.float64, device=DEVICE)
    for i, ecs in enumerate(rows):
        out[i, [index[ec] for ec in set(ecs)]] = 1.0
    return out


def weighted_f1(weights, truth, guess):
    """Support-weighted F1 for each row of ``weights`` (multiplicity of every enzyme in a resample)."""
    tp = weights @ (truth * guess)
    support = weights @ truth
    called = weights @ guess
    precision = torch.where(called > 0, tp / called.clamp_min(1e-12), torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support.clamp_min(1e-12), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp_min(1e-12),
                     torch.zeros_like(tp))
    return (f1 * support).sum(1) / support.sum(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--external", action="append", default=[], metavar="NAME=CSV",
                        help="predictions of another tool: a CSV/TSV with an Entry column and an EC column "
                             "(';'-separated); compared against the main models. Repeatable.")
    parser.add_argument("--pair", action="append", default=[], metavar="A|B",
                        help="extra paired comparison A - B between any two model names. Repeatable.")
    args = parser.parse_args()

    classes = np.load(ROOT / "checkpoints/classes.npy", allow_pickle=True).astype(str)
    index = {c: i for i, c in enumerate(classes)}
    tests = {}
    for n in TESTS:
        st = load_archive(F / f"tests/{n}.npz").to(DEVICE)
        tests[n] = (st, attach_labels(st, read_labels(F / f"tests/{n}.csv"), index, n, False))
    load_entries = {n: [st.entries[r] for r in ds.rows] for n, (st, ds) in tests.items()}

    predictions: dict[str, dict[str, list[list[str]]]] = {}
    for label, (name, full, _) in SIGMOID.items():
        model = load(name, full)
        predictions[label] = {n: calls_to_ecs(call_ec(logits_of(model, name, st, ds.rows)), classes)
                              for n, (st, ds) in tests.items()}
    for label, (ckpt, centroids) in MAXSEP.items():
        model, centers = load_milcl(str(ROOT / ckpt), str(ROOT / centroids), DEVICE)
        predictions[label] = {}
        for n, (st, ds) in tests.items():
            with torch.no_grad():
                chunks = [model(*fetch(st, torch.from_numpy(ds.rows[s:s + 2048]), DEVICE, local_only=False)).float().cpu().numpy()
                          for s in range(0, len(ds.rows), 2048)]
            predictions[label][n] = calls_to_ecs(apply_max_separation(np.concatenate(chunks), centers)[0], classes)
    comparisons = list(COMPARISONS)
    for spec in args.external:
        name, path = spec.split("=", 1)
        calls = read_external(Path(path).expanduser().resolve())
        predictions[name] = {}
        for n, (_, ds) in tests.items():
            entries = [load_entries[n][i] for i in range(len(ds.true_ecs))]
            missing = [e for e in entries if e not in calls]
            if missing:
                raise SystemExit(f"{path}: no prediction for {len(missing)} {n} enzymes, e.g. {missing[:3]}")
            predictions[name][n] = [calls[e] for e in entries]
        SHORT.setdefault(name, name[:9])
        comparisons += [(name, ref) for ref in EXTERNAL_REFERENCES]
    for spec in args.pair:
        a, b = spec.split("|", 1)
        for name in (a, b):
            if name not in predictions:
                raise SystemExit(f"--pair: unknown model {name!r}; known: {sorted(predictions)}")
        comparisons.append((a, b))
    labels = list(predictions)

    rng = np.random.default_rng(args.seed)
    results: dict = {"resamples": args.resamples, "seed": args.seed, "scores": {}, "differences": {}}
    for n, (_, ds) in tests.items():
        size = len(ds.true_ecs)
        draws = rng.integers(0, size, size=(args.resamples, size))
        weights = torch.zeros(args.resamples, size, dtype=torch.float64, device=DEVICE)
        weights.scatter_add_(1, torch.from_numpy(draws).to(DEVICE), torch.ones_like(weights))
        full = torch.ones(1, size, dtype=torch.float64, device=DEVICE)
        vocabulary = sorted({ec for t in ds.true_ecs for ec in t} | {ec for lab in labels for p in predictions[lab][n] for ec in p})
        truth = indicator(ds.true_ecs, vocabulary)
        samples = {}
        for lab in labels:
            f1, tp, called, support = per_enzyme(ds.true_ecs, predictions[lab][n])

            def micro(w):
                t, c, s = w @ tp, w @ called, w @ support
                p, r = t / c, t / s
                return 2 * p * r / (p + r)

            guess = indicator(predictions[lab][n], vocabulary)
            samples[lab] = {
                "sample": ((weights @ f1) / size, float(f1.mean())),
                "micro": (micro(weights), float(micro(full)[0])),
                "weighted": (weighted_f1(weights, truth, guess), float(weighted_f1(full, truth, guess)[0])),
            }
        for metric in ("sample", "micro", "weighted"):
            for lab in labels:
                boot, point = samples[lab][metric]
                lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64, device=DEVICE)).tolist()
                results["scores"].setdefault(metric, {}).setdefault(lab, {})[n] = {"f1": point, "lo": lo, "hi": hi}
            for a, b in comparisons:
                diff = samples[a][metric][0] - samples[b][metric][0]
                lo, hi = torch.quantile(diff, torch.tensor([0.025, 0.975], dtype=torch.float64, device=DEVICE)).tolist()
                results["differences"].setdefault(metric, {}).setdefault(f"{a} - {b}", {})[n] = {
                    "diff": samples[a][metric][1] - samples[b][metric][1], "lo": lo, "hi": hi,
                    "p_le_0": float((diff <= 0).double().mean()),
                }
        print(f"{n}: {size} enzymes", flush=True)

    (F / "logs/bootstrap_ci.json").write_text(json.dumps(results, indent=1) + "\n")
    head = {"new": "New", "price": "Price", "uniprot-all": "U-all", "uniprot-multi": "U-multi", "test70": "T-70", "test50": "T-50"}
    for metric in ("sample", "micro", "weighted"):
        print(f"\n### {metric} F1, 95% CI")
        print(f"{'model':10s}" + "".join(f"{head[n]:>21s}" for n in TESTS))
        for lab in labels:
            row = results["scores"][metric][lab]
            print(f"{SHORT[lab]:10s}" + "".join(f"  {row[n]['f1']:.3f} [{row[n]['lo']:.3f},{row[n]['hi']:.3f}]" for n in TESTS))
        print(f"\n### {metric} F1, paired difference A - B [95% CI] (* = CI excludes 0)")
        print(f"{'A - B':20s}" + "".join(f"{head[n]:>18s}" for n in TESTS))
        for a, b in comparisons:
            row = results["differences"][metric][f"{a} - {b}"]
            cells = []
            for n in TESTS:
                r = row[n]
                star = "*" if r["lo"] > 0 or r["hi"] < 0 else " "
                cells.append(f"{r['diff']:+.3f}[{r['lo']:+.2f},{r['hi']:+.2f}]{star}")
            print(f"{SHORT[a] + ' - ' + SHORT[b]:20s}" + "".join(f"{c:>18s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
