#!/usr/bin/env python
"""Per-residue pLDDT from the AlphaFold DB Swiss-Prot archive, and pLDDT-masked 3Di for SaProt.

    python -I scripts/afdb_plddt.py --tar swissprot_pdb_v6.tar

Streams ``swissprot_pdb_v6.tar`` (members ``AF-<accession>-F1-model_v6.pdb.gz``) without
extracting it. For each protein of ``features/plm/afdb/3di.fasta`` whose accession has an F1 model
whose residue sequence equals ours exactly (the same rule ``afdb_3di.py`` used to take AFDB 3Di),
reads the CA B-factor column, which AFDB fills with pLDDT. Writes

- ``features/plm/afdb/plddt.npz``: one uint8 array per accession (pLDDT rounded), and
- ``features/plm/afdb/3di_plddt<cut>.fasta``: the 3Di of ``3di.fasta`` with every residue whose
  pLDDT is below ``--cut`` replaced by ``#`` (SaProt's mask token, as in the SaProt paper).
  Proteins without an AFDB model keep their ProstT5 3Di unchanged, so the only difference from
  the earlier SaProt run is the masking.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
         "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
         "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        else:
            out[name] = out.get(name, "") + line.strip()
    return out


def parse(item):
    acc, blob = item
    seq, plddt = [], []
    for line in gzip.decompress(blob).decode("ascii", "replace").splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA ":
            seq.append(THREE.get(line[17:20], "X"))
            plddt.append(float(line[60:66]))
    return acc, "".join(seq), np.array(plddt, dtype=np.float32)


def members(tar_path: Path, wanted: set[str]):
    with tarfile.open(tar_path, "r|") as tar:
        for m in tar:
            parts = m.name.rsplit("/", 1)[-1].split("-")
            if not m.isfile() or len(parts) < 3 or parts[2] != "F1" or parts[1] not in wanted:
                continue
            if not m.name.endswith(".pdb.gz") or m.size > 200_000_000:
                continue
            yield parts[1], tar.extractfile(m).read()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tar", required=True)
    p.add_argument("--cut", type=float, default=70.0)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    afdb = ROOT / "features" / "plm" / "afdb"
    states = fasta(afdb / "3di.fasta")
    with open(ROOT / "features" / "domains" / "all_union.csv") as handle:
        sequences = {r["Entry"]: r["Sequence"].strip() for r in csv.DictReader(handle, delimiter="\t")}
    wanted = set(states)
    plddt, seen, mismatch = {}, 0, 0
    with Pool(args.workers) as pool:
        for acc, seq, values in pool.imap_unordered(parse, members(Path(args.tar), wanted), chunksize=64):
            seen += 1
            if seq == sequences.get(acc) and len(values) == len(states[acc]):
                plddt[acc] = np.round(values).clip(0, 100).astype(np.uint8)
            else:
                mismatch += 1
            if seen % 20000 == 0:
                print(f"  {seen} models read, {len(plddt)} matched", flush=True)
    print(f"{seen} of our accessions in the archive; {len(plddt)} match our sequence ({mismatch} differ)")
    np.savez_compressed(afdb / "plddt.npz", **plddt)
    cut = args.cut
    lines, masked, total = [], 0, 0
    for entry, s in states.items():
        if entry in plddt:
            low = plddt[entry] < cut
            s = "".join("#" if m else c for c, m in zip(s, low))
            masked += int(low.sum())
            total += len(low)
        lines.append(f">{entry}\n{s}\n")
    out = afdb / f"3di_plddt{cut:g}.fasta"
    out.write_text("".join(lines))
    print(f"masked {masked}/{total} AFDB residues ({masked / max(total, 1):.3f}) below pLDDT {cut:g}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
