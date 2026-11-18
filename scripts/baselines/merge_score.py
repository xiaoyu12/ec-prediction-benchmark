"""Merge per-test prediction CSVs into one Entry,EC number file and print per-enzyme F1 per test set.
    python merge_score.py <pred_dir> <merged.csv>
"""
import csv, sys
from pathlib import Path
TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
D = Path(__file__).resolve().parents[2] / "dataset"
src, merged = Path(sys.argv[1]), Path(sys.argv[2])
rows, out = {}, []
for t in TESTS:
    pred = {r["Entry"]: set(filter(None, r["EC number"].split(";"))) for r in csv.DictReader(open(src / f"{t}.csv"))}
    f1s, calls = [], 0
    for r in csv.DictReader(open(D / f"{t}.csv"), delimiter="\t"):
        true, p = set(r["EC number"].split(";")), pred.get(r["Entry"], set())
        tp = len(true & p); calls += len(p)
        f1s.append(2 * tp / (len(true) + len(p)) if tp else 0.0)
        if r["Entry"] in rows and rows[r["Entry"]] != p:
            print("warn: differing calls for", r["Entry"])
        rows.setdefault(r["Entry"], p)
    out.append(sum(f1s) / len(f1s))
    print(f"{t:14s} n={len(f1s):5d} F1={out[-1]:.3f} calls/enz={calls/len(f1s):.2f} missing={sum(1 for x in csv.DictReader(open(D/f'{t}.csv'),delimiter=chr(9)) if x['Entry'] not in pred)}")
print(f"mean {sum(out)/len(out):.3f}")
with open(merged, "w") as h:
    h.write("Entry,EC number\n" + "".join(f"{e},{';'.join(sorted(p))}\n" for e, p in rows.items()))
