#!/usr/bin/env python
"""Write mutated copies of orphan-class enzymes, the positives MIL-CL+ trains orphan ECs with.

    python orphans.py --csvdata ../dataset/split100.csv --output ../dataset/split100_orphans.csv
    python process.py --csvdata ../dataset/split100_orphans.csv
    python train.py --model milcl_plus ... --augment-npz ../dataset/split100_orphans.npz

An orphan EC has a single training enzyme, so contrastive learning has no second enzyme to pull
towards it. Following CLEAN, every enzyme carrying an orphan EC gets ``--copies`` mutated copies,
each with a fraction of its residues (drawn from N(0.10, 0.02)) masked at random positions after the
first. CLEAN writes ESM's ``<mask>`` token; DCTdomain reads plain sequences, so the masked residues
become ``X`` here. Copies are named ``<Entry>_<k>``, which is how ``train.py`` matches them back.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import random
from pathlib import Path


def mutate(sequence: str, rng: random.Random) -> str:
    residues = list(sequence)
    rate = rng.gauss(0.10, 0.02)
    for _ in range(max(0, math.ceil(len(residues) * rate))):
        residues[rng.randint(1, len(residues) - 1)] = "X"
    return "".join(residues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvdata", required=True, help="training table: Entry, EC number, Sequence")
    parser.add_argument("--output", help="output table (default: <csvdata>_orphans.csv)")
    parser.add_argument("--copies", type=int, default=10, help="mutated copies per orphan enzyme")
    parser.add_argument("--min-length", type=int, default=5, help="ignore shorter sequences, as train.py does")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.csvdata).expanduser().resolve()
    destination = (
        Path(args.output).expanduser().resolve() if args.output else source.with_name(f"{source.stem}_orphans.csv")
    )
    first = source.read_text().splitlines()[0]
    delimiter = "\t" if first.count("\t") >= first.count(",") else ","
    with source.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter=delimiter)
            if len((row["Sequence"] or "").strip()) >= args.min_length
        ]
    support = collections.Counter(ec for row in rows for ec in row["EC number"].split(";"))
    orphans = [row for row in rows if any(support[ec] == 1 for ec in row["EC number"].split(";"))]

    rng = random.Random(args.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Entry", "EC number", "Sequence"])
        for row in orphans:
            sequence = row["Sequence"].strip().upper()
            for copy in range(args.copies):
                writer.writerow([f"{row['Entry']}_{copy}", row["EC number"], mutate(sequence, rng)])
    orphan_ecs = sum(1 for count in support.values() if count == 1)
    print(f"{orphan_ecs} orphan ECs on {len(orphans)} enzymes -> {len(orphans) * args.copies} copies in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
