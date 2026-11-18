#!/usr/bin/env python
"""Collect the numbers behind the manuscript's figures and tables into one JSON file.

    python3 scripts/manuscript_data.py   # -> reports/manuscript/data/results.json

For each model it reads ``runs/feat/id50_<run>/metrics.json`` (id50 score) and
``runs/feat/full_<run>/metrics.json`` (six test sets). Models with seed replicates (``_s1``, ``_s2``)
are averaged over seeds, and the per-seed test means are kept for the range. Published methods that
were re-run outside ``runs/feat`` come from ``features/logs/compare_metrics.json`` and the
EnzymeHunter bootstrap log; their id50 scores are the ones recorded in reports/AUTORESEARCH.md.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "feat"
TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]

# (category, display name, run, seeds?)
MODELS = [
    ("Homology only", "MMseqs2", "mmseqs", False),
    ("Homology only", "Foldseek, ProstT5 3Di", "foldseek", False),
    ("Homology only", "Foldseek, AFDB 3Di", "foldseek_afdb", False),
    ("Homology only", "MMseqs2 + Foldseek-AFDB, best hit", "hom_best_mmseqs_foldseek_afdb", False),
    ("Homology only", "MMseqs2 + Foldseek-AFDB, union", "hom_union_mmseqs_foldseek_afdb", False),
    ("Sequence PLM", "ESM2-650M", "esm2_bce_e100", True),
    ("Sequence PLM", "ESM2-3B", "esm2_3b_last", True),
    ("Sequence PLM", "ESMC-600M", "esmc600_last", True),
    ("Sequence PLM", "ESMC-6B", "esmc6b_last", False),
    ("Sequence PLM", "ProtT5-XL", "prott5_last", False),
    ("PLM + SAE features", "ESMC-600M SAE, max-pooled", "esmc600_saemax", True),
    ("PLM + SAE features", "ESMC-6B SAE, max-pooled", "esmc6b_saemax", True),
    ("Structure-aware PLM", "ProstT5", "prostt5_last", False),
    ("Structure-aware PLM", "SaProt-650M", "saprot_last", False),
    ("Structure-aware PLM", "SaProt-650M, pLDDT-masked", "saprotp70_last", False),
    ("Fine-tuned PLM", "ESMC-600M, fine-tuned", "ft_esmc600", False),
    ("PLM combination", "c3 concatenation", "esm2_3b_last_esmc600_saemax", False),
    ("PLM combination", "Six-head ensemble", "ens6_6b", False),
    ("PLM + homology", "ESM2-650M + MMseqs2", "hyb_esm2_mmseqs", False),
    ("PLM + homology", "ESM2-3B + homology", "hyb_esm2_3b_msaf", False),
    ("PLM + homology", "ESMC-600M SAE + homology", "hyb_esmc600_saemax_msaf", False),
    ("PLM + homology", "c3 + homology", "hyb_c3_msaf", False),
    ("PLM + homology", "Six-head ensemble + homology", "hyb_ens6_6b_msaf", False),
    ("PLM + homology", "Gradient-boosted re-ranker", "stack5_mmseqs", False),
]

# Re-run published methods: key in compare_metrics.json, id50 from AUTORESEARCH.md "Reference models".
PUBLISHED = [
    ("ESM2-BC (retrained)", "retrained ESM2-BC | shift 0 (+0.00)", 0.877),
    ("MIL-BC+ (released)", "released MIL-BC+ | shift 0 (+0.00)", 0.868),
    ("MIL-CL+ (released)", "released MIL-CL+ | max-separation", None),
]
ENZYMEHUNTER = {"new": 0.620, "price": 0.416, "uniprot-all": 0.656, "uniprot-multi": 0.596,
                "test70": 0.539, "test50": 0.453}  # HANDOFF_MILBC.md, scored by scripts/bootstrap_ci.py


def metrics(protocol: str, run: str) -> dict | None:
    path = RUNS / f"{protocol}_{run}" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def entry(category: str, name: str, run: str, seeds: bool) -> dict:
    runs = [run] + ([f"{run}_s1", f"{run}_s2"] if seeds else [])
    id50 = [m["id50"]["sample"] for r in runs if (m := metrics("id50", r)) and "id50" in m]
    full = [m for r in runs if (m := metrics("full", r))]
    sets = {t: sum(m[t]["sample"] for m in full) / len(full) for t in TESTS}
    out = {"category": category, "name": name, "runs": runs,
           "id50": sum(id50) / len(id50) if id50 else None, "id50_seeds": id50,
           "tests": sets, "test_mean": sum(sets.values()) / len(TESTS),
           "test_mean_seeds": [sum(m[t]["sample"] for t in TESTS) / len(TESTS) for m in full],
           "weighted": {t: sum(m[t].get("weighted", float("nan")) for m in full) / len(full) for t in TESTS},
           "calls": {t: sum(m[t].get("calls", float("nan")) for m in full) / len(full) for t in TESTS}}
    return out


def main() -> int:
    rows = [entry(*m) for m in MODELS]
    compare = json.loads((ROOT / "features" / "logs" / "compare_metrics.json").read_text())
    for name, key, id50 in PUBLISHED:
        m = compare[key]
        sets = {t: m[t]["sample"] for t in TESTS}
        rows.append({"category": "Published method", "name": name, "id50": id50, "id50_seeds": [],
                     "tests": sets, "test_mean": sum(sets.values()) / len(TESTS), "test_mean_seeds": [],
                     "weighted": {t: m[t]["weighted"] for t in TESTS}, "calls": {t: m[t]["calls"] for t in TESTS}})
    rows.append({"category": "Published method", "name": "EnzymeHunter", "id50": None, "id50_seeds": [],
                 "tests": ENZYMEHUNTER, "test_mean": sum(ENZYMEHUNTER.values()) / len(TESTS),
                 "test_mean_seeds": [], "weighted": {}, "calls": {}})
    # Calls per enzyme against per-enzyme and support-weighted F1, with and without the logit shift
    # chosen on id50 by weighted F1 (HANDOFF_MILBC.md, "Logit shift for the retrained models").
    calib = []
    for name, key in [("ESM2-BC", "retrained ESM2-BC | shift 0 (+0.00)"),
                      ("ESM2-BC, shift +5", "retrained ESM2-BC | shift by weighted F1 on id50 (+5.00)"),
                      ("MIL-BC+ (retrained)", "retrained MIL-BC+ | shift 0 (+0.00)"),
                      ("MIL-BC+ (retrained), shift +5", "retrained MIL-BC+ | shift by weighted F1 on id50 (+5.00)"),
                      ("MIL-BC+ (released)", "released MIL-BC+ | shift 0 (+0.00)")]:
        m = compare[key]
        calib.append({"name": name, **{k: {t: m[t][k] for t in TESTS} for k in ("sample", "weighted", "calls")}})
    for r in rows:
        if r["name"] in ("MMseqs2 + Foldseek-AFDB, best hit", "Six-head ensemble + homology"):
            calib.append({"name": r["name"], "sample": r["tests"], "weighted": r["weighted"], "calls": r["calls"]})
    (ROOT / "reports" / "manuscript" / "data" / "calibration.json").write_text(json.dumps(calib, indent=1) + "\n")
    out = ROOT / "reports" / "manuscript" / "data" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{'category':22s} {'model':36s} {'id50':>6s} {'test':>6s}  " + " ".join(f"{t[:7]:>7s}" for t in TESTS))
    for r in rows:
        id50 = f"{r['id50']:.3f}" if r["id50"] else "  –  "
        print(f"{r['category']:22s} {r['name']:36s} {id50:>6s} {r['test_mean']:.3f}  "
              + " ".join(f"{r['tests'][t]:7.3f}" for t in TESTS))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
