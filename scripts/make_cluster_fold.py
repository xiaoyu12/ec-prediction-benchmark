#!/usr/bin/env python
"""Build a sequence-identity-clustered validation fold of split100 for train.py.

    python scripts/make_cluster_fold.py              # 50 % identity, ~10 % of enzymes held out
    python scripts/make_cluster_fold.py --min-seq-id 0.7 --name id70

1. MMseqs2 ``easy-cluster`` on split100 at ``--min-seq-id`` (coverage ``-c 0.8``).
2. Whole clusters, in random order, go to validation until it holds ``--fraction`` of the enzymes.
   A cluster is skipped when taking it would leave one of its EC numbers without any training
   enzyme, so every validation label stays learnable.
3. Strict filter: ``easy-search`` validation against the remaining training set; a validation
   enzyme with any hit at ``--min-seq-id`` (coverage ``--search-cov``) is dropped from both sides,
   because its cluster mates could otherwise leak through it.

Writes ``scripts/folds/<name>_val.txt`` and ``scripts/folds/<name>_exclude.txt`` (one Entry per line), for
``train.py --val-entries ... --exclude-entries ...``, plus ``scripts/folds/<name>_stats.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # scripts/
ROOT = HERE.parent
DEFAULT_MMSEQS = Path.home() / ".local/opt/mmseqs2/bin/mmseqs"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def write_fasta(path: Path, entries: list[str], sequences: dict[str, str]) -> None:
    with path.open("w") as handle:
        for entry in entries:
            handle.write(f">{entry}\n{sequences[entry]}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvdata", default=str(ROOT / "features" / "split100_min5.csv"))
    parser.add_argument("--name", default="id50")
    parser.add_argument("--min-seq-id", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8, help="clustering coverage (-c)")
    parser.add_argument("--search-cov", type=float, default=0.5, help="coverage for the leak filter")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--mmseqs", default=str(DEFAULT_MMSEQS))
    args = parser.parse_args()

    with open(args.csvdata) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    sequences = {r["Entry"]: r["Sequence"].strip() for r in rows}
    labels = {r["Entry"]: [ec.strip() for ec in r["EC number"].split(";") if ec.strip()] for r in rows}
    entries = list(sequences)
    out_dir = HERE / "folds"
    out_dir.mkdir(exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="clusterfold-", dir="/dev/shm"))
    try:
        fasta = work / "all.fasta"
        write_fasta(fasta, entries, sequences)
        run([args.mmseqs, "easy-cluster", str(fasta), str(work / "clu"), str(work / "tmp"),
             "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage),
             "--threads", str(args.threads)])
        clusters: dict[str, list[str]] = {}
        with open(work / "clu_cluster.tsv") as handle:
            for line in handle:
                representative, member = line.split()
                clusters.setdefault(representative, []).append(member)
        cluster_list = list(clusters.values())
        print(f"{len(entries)} enzymes in {len(cluster_list)} clusters", flush=True)

        # 2. pick whole clusters, keeping at least one training enzyme for every EC
        remaining = Counter(ec for entry in entries for ec in labels[entry])
        target = int(round(args.fraction * len(entries)))
        order = np.random.default_rng(args.seed).permutation(len(cluster_list))
        val: list[str] = []
        skipped = 0
        for position in order:
            if len(val) >= target:
                break
            members = cluster_list[position]
            taken = Counter(ec for entry in members for ec in labels[entry])
            if any(remaining[ec] - count < 1 for ec, count in taken.items()):
                skipped += 1
                continue
            remaining.subtract(taken)
            val.extend(members)
        val_set = set(val)
        train = [entry for entry in entries if entry not in val_set]
        print(f"picked {len(val)} validation enzymes; skipped {skipped} clusters to keep ECs trainable")

        # 3. leak filter: drop validation enzymes with any training hit at the identity threshold
        write_fasta(work / "val.fasta", val, sequences)
        write_fasta(work / "train.fasta", train, sequences)
        run([args.mmseqs, "easy-search", str(work / "val.fasta"), str(work / "train.fasta"),
             str(work / "hits.m8"), str(work / "tmp2"), "--min-seq-id", str(args.min_seq_id),
             "-c", str(args.search_cov), "--cov-mode", "0", "-s", "7.5", "--max-seqs", "1000",
             "--threads", str(args.threads)])
        leaked = set()
        with open(work / "hits.m8") as handle:
            for line in handle:
                leaked.add(line.split("\t", 1)[0])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    final_val = [entry for entry in val if entry not in leaked]
    exclude = sorted(leaked)
    (out_dir / f"{args.name}_val.txt").write_text("\n".join(final_val) + "\n")
    (out_dir / f"{args.name}_exclude.txt").write_text("\n".join(exclude) + ("\n" if exclude else ""))

    train_counts = Counter(ec for entry in entries if entry not in val_set for ec in labels[entry])
    size_of = [min(train_counts[ec] for ec in labels[entry]) for entry in final_val]
    bins = [1, 2, 3, 6, 11, 31, 101, 1001, 10**9]
    stats = {
        "csvdata": args.csvdata, "min_seq_id": args.min_seq_id, "coverage": args.coverage,
        "search_cov": args.search_cov, "fraction": args.fraction, "seed": args.seed,
        "clusters": len(cluster_list), "validation": len(final_val), "excluded_leaks": len(exclude),
        "training": len(entries) - len(val_set),
        "validation_multi_label": sum(len(labels[e]) > 1 for e in final_val),
        "validation_ecs": len({ec for e in final_val for ec in labels[e]}),
        "validation_by_training_class_size": {
            f"{a}-{b - 1}": sum(a <= s < b for s in size_of) for a, b in zip(bins, bins[1:])
        },
    }
    (out_dir / f"{args.name}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
