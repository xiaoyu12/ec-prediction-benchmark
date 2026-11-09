#!/usr/bin/env python
"""MMseqs2 homology transfer of EC numbers, on the same data and protocols as ``train_feat.py``.

    python homology.py --protocol id50 --output-dir ../runs/feat/id50_mmseqs

Queries (the eval sets) are searched against the training proteins of the protocol (``easy-search``,
sensitivity ``-s 7.5``, e-value 1e-3, up to 300 hits). The score of EC ``c`` is the highest bit
score among hits carrying ``c`` divided by the best hit's bit score, so the top hit's ECs score 1;
``--ratio`` turns that into calls (``logit = (score - ratio) * 50``: ECs within the ratio of the top
hit are called). Queries with no hit get flat ``-50`` logits (``call_ec`` then calls an arbitrary
EC, a miss; ensembles fill these from a model). Writes ``metrics.json``, ``logits.npz``,
``predictions.csv`` and ``hits.npz`` (top-hit identity and bit score per query, 0 if none).

``--tool foldseek`` searches structure instead: Foldseek over 3Di + amino-acid sequences
(``--alignment-type 2``, default sensitivity, e-value 1e-3, up to 300 hits), with the 3Di states
predicted from sequence by ProstT5 (``features/plm/prostt5/3di.fasta``, from ``extract_plm.py``).
No 3D coordinates are needed for this alignment mode; identity is the amino-acid identity of the
alignment. ``--states`` swaps in another 3Di FASTA, e.g. ``features/plm/afdb/3di.fasta`` (3Di of
AlphaFold DB structures where available, from ``scripts/afdb_3di.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from infer import call_ec
from train import PACKAGE_ROOT, read_labels
from train_feat import FOLDS, TESTS, score

MMSEQS = Path.home() / ".local/opt/mmseqs2/bin/mmseqs"
FOLDSEEK = next(Path.home().glob(".local/opt/foldseek_dl/gpu/*/bin/foldseek"), Path.home() / ".local/opt/foldseek/bin/foldseek")
THREE_DI = PACKAGE_ROOT / "features" / "plm" / "prostt5" / "3di.fasta"


def foldseek_search(query: Path, target: Path, states: dict[str, str], hits: Path, work: Path, threads: int) -> None:
    """Foldseek databases from FASTA plus predicted 3Di: an amino-acid DB (MMseqs2 ``createdb``) and
    its ``_ss`` twin built from the 3Di FASTA in the same entry order, then search + convertalis."""
    import shutil

    for fasta in (query, target):
        db = work / fasta.stem
        names = [line[1:] for line in fasta.read_text().splitlines() if line.startswith(">")]
        ss = work / f"{fasta.stem}.3di.fasta"
        ss.write_text("".join(f">{e}\n{states[e]}\n" for e in names))
        subprocess.run([str(MMSEQS), "createdb", str(fasta), str(db)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([str(MMSEQS), "createdb", str(ss), str(work / "ss_tmp")], check=True, stdout=subprocess.DEVNULL)
        for suffix in ("", ".index", ".dbtype"):
            shutil.move(str(work / f"ss_tmp{suffix}"), str(work / f"{fasta.stem}_ss{suffix}"))
    qdb, tdb, aln = work / query.stem, work / target.stem, work / "aln"
    subprocess.run([str(FOLDSEEK), "search", str(qdb), str(tdb), str(aln), str(work / "tmp"), "--alignment-type", "2",
                    "-e", "1e-3", "--max-seqs", "300", "--threads", str(threads)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([str(FOLDSEEK), "convertalis", str(qdb), str(tdb), str(aln), str(hits) + ".tmp",
                    "--format-output", "query,target,fident,bits"], check=True, stdout=subprocess.DEVNULL)
    Path(str(hits) + ".tmp").rename(hits)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=["id50", "full"], required=True)
    p.add_argument("--ratio", type=float, default=0.95)
    p.add_argument("--tool", choices=["mmseqs", "foldseek"], default="mmseqs")
    p.add_argument("--states", default=str(THREE_DI), help="3Di FASTA for --tool foldseek")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"))
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    work = out / args.tool
    work.mkdir(parents=True, exist_ok=True)
    classes = np.load(args.classes, allow_pickle=True).astype(str)
    class_index = {ec: i for i, ec in enumerate(classes)}
    with open(PACKAGE_ROOT / "features" / "domains" / "all_union.csv") as handle:
        sequences = {r["Entry"]: r["Sequence"].strip() for r in csv.DictReader(handle, delimiter="\t")}

    train = read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv")
    eval_sets: dict[str, dict] = {}
    if args.protocol == "id50":
        held = set((FOLDS / "id50_val.txt").read_text().split())
        dropped = set((FOLDS / "id50_exclude.txt").read_text().split())
        eval_sets["id50"] = {e: train[e] for e in train if e in held}
        train = {e: v for e, v in train.items() if e not in held and e not in dropped}
    for name in TESTS:
        eval_sets[name] = read_labels(PACKAGE_ROOT / "features" / "tests" / f"{name}.csv")

    target, query = work / "train.fasta", work / "query.fasta"
    target.write_text("".join(f">{e}\n{sequences[e]}\n" for e in train))
    queries = sorted({e for labels in eval_sets.values() for e in labels})
    query.write_text("".join(f">{e}\n{sequences[e]}\n" for e in queries))
    hits_file = work / "hits.m8"
    if not hits_file.exists() and args.tool == "foldseek":
        names = Path(args.states).read_text().split()
        foldseek_search(query, target, dict(zip((n[1:] for n in names[::2]), names[1::2])), hits_file, work,
                        args.threads)
    if not hits_file.exists():
        subprocess.run([str(MMSEQS), "easy-search", str(query), str(target), str(hits_file) + ".tmp", str(work / "tmp"),
                        "-s", "7.5", "-e", "1e-3", "--max-seqs", "300", "--threads", str(args.threads),
                        "--format-output", "query,target,fident,bits"], check=True, stdout=subprocess.DEVNULL)
        (Path(str(hits_file) + ".tmp")).rename(hits_file)

    best: dict[str, dict[int, float]] = defaultdict(dict)  # query -> class -> best bit score
    top: dict[str, tuple[float, float]] = {}  # query -> (identity, bits) of the best hit
    for line in hits_file.read_text().splitlines():
        q, t, ident, bits = line.split("\t")
        bits = float(bits)
        if q not in top or bits > top[q][1]:
            top[q] = (float(ident), bits)
        for ec in train[t][0]:
            c = class_index[ec]
            if bits > best[q].get(c, 0.0):
                best[q][c] = bits

    def logits_for(entries: list[str]) -> np.ndarray:
        logits = np.full((len(entries), len(classes)), -50.0, dtype=np.float32)
        for i, e in enumerate(entries):
            if e in top:
                for c, bits in best[e].items():
                    logits[i, c] = (bits / top[e][1] - args.ratio) * 50.0
            else:
                logits[i, 0] = -49.0  # one arbitrary call, not a 5,242-way tie
        return logits

    logits, metrics = {}, {}
    for name, labels in eval_sets.items():
        entries = list(labels)
        logits[name] = logits_for(entries)
        metrics[name] = score(logits[name], [labels[e][0] for e in entries], classes)
        metrics[name]["coverage"] = float(np.mean([e in top for e in entries]))
        print(f"{name}: per-enzyme F1 {metrics[name]['sample']:.3f}  micro {metrics[name]['micro']:.3f}  "
              f"hit coverage {metrics[name]['coverage']:.3f}", flush=True)

    states = "afdb" if "afdb" in args.states else "prostt5"
    config = {"features": ["mmseqs2" if args.tool == "mmseqs" else f"foldseek({states}-3di+aa)"], "protocol": args.protocol, "arch": "homology", "loss": "-", "hidden": 0,
              "layers": 0, "ratio": args.ratio}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "logits.npz", **{n: v.astype(np.float16) for n, v in logits.items()},
                        **{f"{n}__entries": np.array(list(l)) for n, l in eval_sets.items()})
    np.savez_compressed(out / "hits.npz", **{
        n: np.array([top.get(e, (0.0, 0.0)) for e in l], dtype=np.float32) for n, l in eval_sets.items()})
    with open(out / "predictions.csv", "w") as handle:
        handle.write("Entry,EC number\n")
        written = set()
        for name in TESTS:
            entries = list(eval_sets[name])
            for e, calls in zip(entries, call_ec(logits[name])):
                if e not in written:
                    handle.write(f"{e},{';'.join(classes[calls])}\n")
                    written.add(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
