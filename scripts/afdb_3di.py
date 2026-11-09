#!/usr/bin/env python
"""3Di states from AlphaFold DB structures, with ProstT5-predicted 3Di as the fallback.

    python scripts/afdb_3di.py --afdb ~/.local/opt/foldseek_db/afdb_sp

Reads ``afsp_aa.fasta`` / ``afsp_3di.fasta`` (``foldseek convert2fasta`` of Foldseek's
``Alphafold/Swiss-Prot`` database; headers ``AF-<accession>-F1-model_v6``). A protein of
``features/domains/all_union.csv`` takes the AFDB 3Di string when its accession is in AFDB *and*
the AFDB sequence equals ours exactly, else the ProstT5 prediction
(``features/plm/prostt5/3di.fasta``). Writes ``features/plm/afdb/3di.fasta`` (same entry order
and names as the ProstT5 file) and prints coverage per data split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from train import read_labels  # noqa: E402
from train_feat import FOLDS, TESTS  # noqa: E402


def fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        else:
            out[name] = out.get(name, "") + line.strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--afdb", required=True)
    p.add_argument("--out", default=str(ROOT / "features" / "plm" / "afdb" / "3di.fasta"))
    args = p.parse_args()
    afdb = Path(args.afdb).expanduser()
    aa, ss = fasta(afdb / "afsp_aa.fasta"), fasta(afdb / "afsp_3di.fasta")
    by_acc = {}
    for name in aa:
        parts = name.split("-")
        if len(parts) >= 3 and parts[2] == "F1":
            by_acc[parts[1]] = (aa[name], ss[name])
    predicted = fasta(ROOT / "features" / "plm" / "prostt5" / "3di.fasta")
    with open(ROOT / "features" / "domains" / "all_union.csv") as handle:
        sequences = {r["Entry"]: r["Sequence"].strip() for r in csv.DictReader(handle, delimiter="\t")}
    used, lines = set(), []
    for entry, states in predicted.items():
        hit = by_acc.get(entry)
        if hit and hit[0] == sequences[entry] and len(hit[1]) == len(states):
            states = hit[1]
            used.add(entry)
        lines.append(f">{entry}\n{states}\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    in_afdb = {e for e in predicted if e in by_acc}
    print(f"{len(used)}/{len(predicted)} proteins take AFDB 3Di ({len(in_afdb)} accessions in AFDB Swiss-Prot)")
    train = read_labels(ROOT / "features" / "split100_min5.csv")
    held = set((FOLDS / "id50_val.txt").read_text().split())
    splits = {"train (split100)": list(train), "id50 fold": [e for e in train if e in held]}
    for name in TESTS:
        splits[name] = list(read_labels(ROOT / "features" / "tests" / f"{name}.csv"))
    for name, entries in splits.items():
        print(f"  {name}: {sum(e in used for e in entries) / len(entries):.3f} AFDB "
              f"({sum(e in in_afdb for e in entries) / len(entries):.3f} accession present)")
    agree = [sum(a == b for a, b in zip(by_acc[e][1], predicted[e])) / len(predicted[e]) for e in list(used)[:5000]]
    print(f"ProstT5 vs AFDB 3Di per-residue agreement (5,000 proteins): {sum(agree) / len(agree):.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
