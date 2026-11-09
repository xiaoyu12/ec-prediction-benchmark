#!/usr/bin/env python
"""Stacked EC re-ranker over base models and homology (gradient-boosted trees), trained on id50.

    envs/esmc/bin/python code/stack.py --runs esm2_bce_e100 esmc600_last esmc600_saemax --homology mmseqs \\
        --name stack_a

Candidates per query: the top ``--top`` ECs of the mean model probability, plus every EC carried by a
homology hit. Features per (query, EC): each base model's probability, their mean and max, the
EC's rank under the mean, the EC's best-hit bit-score ratio (``bits_c / bits_top``, 0 if no hit
carries it), the top hit's identity and bit score, whether the top hit carries the EC, the number of
distinct ECs among hits, the margin to the best candidate's mean probability, and the EC's log
training frequency.

Training: the id50 fold, scored by the id50-protocol base runs (models and homology trained
without the fold). 5-fold cross-validation over id50 clusters (MMseqs2 30% id, GroupKFold) gives the reported id50 score, out of
fold. The model fitted on all of id50 then scores the test sets twice: with the id50-protocol base
runs (``runs/feat/id50_<name>``) and with the full-protocol base runs (``runs/feat/full_<name>``,
same feature definitions on bases trained on all of split100). Calls: ``call_ec`` on
``logit(p_stack)``, non-candidates at -inf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from train import PACKAGE_ROOT, read_labels
from train_feat import TESTS, score

RUNS = PACKAGE_ROOT / "runs" / "feat"
MMSEQS = Path.home() / ".local/opt/mmseqs2/bin/mmseqs"


def clusters(entries) -> np.ndarray:
    """MMseqs2 clusters (30% identity, 80% coverage) of the id50 proteins: CV groups, so near-identical
    proteins never sit on both sides of a fold. Cached in runs/feat/id50_groups.tsv."""
    import csv
    import subprocess
    import tempfile

    cache = RUNS / "id50_groups.tsv"
    if not cache.exists():
        with open(PACKAGE_ROOT / "features" / "domains" / "all_union.csv") as handle:
            seqs = {r["Entry"]: r["Sequence"].strip() for r in csv.DictReader(handle, delimiter="\t")}
        with tempfile.TemporaryDirectory(dir="/dev/shm") as work:
            fasta = Path(work) / "q.fasta"
            fasta.write_text("".join(f">{e}\n{seqs[e]}\n" for e in entries))
            subprocess.run([str(MMSEQS), "easy-cluster", str(fasta), f"{work}/clu", f"{work}/tmp", "--min-seq-id", "0.3",
                            "-c", "0.8", "--threads", "32"], check=True, stdout=subprocess.DEVNULL)
            cache.write_text(Path(f"{work}/clu_cluster.tsv").read_text())
    rep = {}
    for line in cache.read_text().splitlines():
        r, m = line.split("\t")
        rep[m] = r
    names = {r: i for i, r in enumerate(sorted(set(rep.values())))}
    return np.array([names[rep[e]] for e in entries])


def load(protocol, run):
    data = np.load(RUNS / f"{protocol}_{run}" / "logits.npz")
    return {k: (data[k].astype(np.float32), data[f"{k}__entries"].astype(str)) for k in data.files if "__" not in k}


def aligned(source, entries):
    values, have = source
    where = {e: i for i, e in enumerate(have)}
    return values[[where[e] for e in entries]]


def build(protocol: str, runs: list[str], homology: list[str], sets: list[str], top: int, log_freq: np.ndarray):
    """``{set: (query index, class index, features)}`` plus the entries per set. Homology features
    (bit-score ratio of the EC, top-hit identity and bits, top hit carries the EC, distinct hit ECs)
    are repeated per homology run (e.g. MMseqs2 and Foldseek)."""
    models = [load(protocol, r) for r in runs]
    homs = [(load(protocol, h), np.load(RUNS / f"{protocol}_{h}" / "hits.npz")) for h in homology]
    out = {}
    for name in sets:
        entries = models[0][name][1]
        probs = np.stack([1 / (1 + np.exp(-aligned(m[name], entries))) for m in models])  # (M, n, C)
        mean = probs.mean(0)
        ratios, top_hits = [], []
        for hom, hits in homs:
            ratio = aligned(hom[name], entries) / 50.0 + 0.95  # homology.py: logit = (ratio - 0.95) * 50
            ratio[ratio < 0] = 0.0
            ratios.append(ratio)
            top_hits.append(aligned((hits[name], hom[name][1]), entries))  # (n, 2): identity, bits
        any_hit = np.max(ratios, axis=0)
        order = np.argsort(-mean, axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(mean.shape[1])[None].repeat(len(mean), 0), axis=1)
        q_idx, c_idx = [], []
        for i in range(len(entries)):
            cand = set(order[i, :top].tolist()) | set(np.nonzero(any_hit[i] > 0)[0].tolist())
            q_idx += [i] * len(cand)
            c_idx += sorted(cand)
        q, c = np.array(q_idx), np.array(c_idx)
        best = mean[np.arange(len(mean)), order[:, 0]]
        hom_features = []
        for ratio, top_hit in zip(ratios, top_hits):
            hom_features += [ratio[q, c], top_hit[q, 0], np.log1p(top_hit[q, 1]), (ratio[q, c] >= 0.999).astype(np.float32),
                             np.log1p((ratio > 0).sum(1)[q])]
        features = np.column_stack([
            probs[:, q, c].T, mean[q, c], probs[:, q, c].max(0), np.log1p(rank[q, c]), *hom_features,
            best[q] - mean[q, c], log_freq[c]])
        out[name] = (q, c, features.astype(np.float32), entries)
    return out


def calls_logits(q, c, p, n, n_classes):
    logits = np.full((n, n_classes), -1e4, dtype=np.float32)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logits[q, c] = np.log(p / (1 - p))
    return logits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--homology", nargs="+", default=["mmseqs"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    classes = np.load(PACKAGE_ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    train = read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv")
    freq = np.zeros(len(classes))
    index = {ec: i for i, ec in enumerate(classes)}
    for ecs, _ in train.values():
        for ec in ecs:
            freq[index[ec]] += 1
    log_freq = np.log1p(freq).astype(np.float32)
    truth_of = {"id50": train, **{n: read_labels(PACKAGE_ROOT / "features" / "tests" / f"{n}.csv") for n in TESTS}}

    def labels_for(name, q, c, entries):
        return np.array([classes[ci] in truth_of[name][entries[qi]][0] for qi, ci in zip(q, c)], dtype=np.int8)

    def make_model():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=args.seed)

    id50 = build("id50", args.runs, args.homology, ["id50", *TESTS], args.top, log_freq)
    q, c, x, entries = id50["id50"]
    y = labels_for("id50", q, c, entries)
    truth = [truth_of["id50"][e][0] for e in entries]
    oof = np.zeros(len(y), dtype=np.float32)
    groups = clusters(list(entries))
    for fold, (tr_q, te_q) in enumerate(GroupKFold(5).split(np.arange(len(entries)), groups=groups)):
        tr, te = np.isin(q, tr_q), np.isin(q, te_q)
        oof[te] = make_model().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    metrics = {"id50": score(calls_logits(q, c, oof, len(entries), len(classes)), truth, classes)}
    print(f"id50 (5-fold, cluster-grouped, out of fold): {metrics['id50']}", flush=True)
    model = make_model().fit(x, y)

    results = {}
    for protocol, built in [("id50", id50), ("full", build("full", args.runs, args.homology, TESTS, args.top, log_freq))]:
        res = dict(metrics) if protocol == "id50" else {}
        logits = {}
        for name in TESTS:
            qq, cc, xx, ee = built[name]
            p = model.predict_proba(xx)[:, 1]
            logits[name] = calls_logits(qq, cc, p, len(ee), len(classes))
            res[name] = score(logits[name], [truth_of[name][e][0] for e in ee], classes)
        if protocol == "id50":
            logits["id50"] = calls_logits(q, c, oof, len(entries), len(classes))
        results[protocol] = res
        print(protocol, "  ".join(f"{n}: {r['sample']:.3f}/{r['micro']:.3f}" for n, r in res.items()), flush=True)
        out = RUNS / f"{protocol}_{args.name}"
        out.mkdir(parents=True, exist_ok=True)
        config = {"features": [f"stack({'+'.join(args.runs)})+hom({'+'.join(args.homology)})"], "protocol": protocol,
                  "arch": "stack-gbt", "loss": "-", "hidden": args.top, "layers": 0, "runs": args.runs}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
        ents = {n: built[n][3] for n in TESTS}
        if protocol == "id50":
            ents["id50"] = entries
        np.savez_compressed(out / "logits.npz", **{n: np.maximum(v, -60).astype(np.float16) for n, v in logits.items()},
                            **{f"{n}__entries": ents[n] for n in logits})
        from infer import call_ec

        with open(out / "predictions.csv", "w") as handle:
            handle.write("Entry,EC number\n")
            written = set()
            for n in TESTS:
                for e, calls in zip(ents[n], call_ec(logits[n])):
                    if e not in written:
                        handle.write(f"{e},{';'.join(classes[calls])}\n")
                        written.add(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
