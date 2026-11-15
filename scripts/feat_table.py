#!/usr/bin/env python
"""Collect ``runs/feat/*/metrics.json`` into a markdown table (per-enzyme F1, the primary metric).

    python scripts/feat_table.py [--metric sample|micro|weighted] [--protocol id50|full] [--sort id50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = ["id50", "new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
SHORT = {"uniprot-all": "U-all", "uniprot-multi": "U-multi", "test70": "T-70", "test50": "T-50",
         "new": "New", "price": "Price", "id50": "id50"}


def describe(config: dict) -> str:
    parts = [config["arch"], config["loss"], f"h{config['hidden']}x{config['layers']}"]
    if config.get("out", "linear") != "linear":
        parts.append(config["out"])
    if config.get("balance"):
        parts.append(f"bal{config['balance']}")
    if config.get("ema"):
        parts.append(f"ema{config['ema']}")
    if config.get("block_dropout"):
        parts.append(f"bd{config['block_dropout']}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="sample", choices=["sample", "micro", "weighted"])
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--sort", default=None, help="column to sort by (descending)")
    args = parser.parse_args()
    rows = []
    for run in sorted((ROOT / "runs" / "feat").glob("*/metrics.json")):
        config = json.loads((run.parent / "config.json").read_text())
        if args.protocol and config["protocol"] != args.protocol:
            continue
        metrics = json.loads(run.read_text())
        tests = [metrics[s][args.metric] for s in SETS[1:] if s in metrics]
        rows.append({"run": run.parent.name, "features": " + ".join(config["features"]), "head": describe(config),
                     **{s: metrics.get(s, {}).get(args.metric) for s in SETS}, "mean": sum(tests) / len(tests)})
    if args.sort:
        rows.sort(key=lambda r: -(r.get(args.sort) or 0))
    print("| run | features | head | " + " | ".join(SHORT[s] for s in SETS) + " | test mean |")
    print("|---|---|---|" + "---:|" * (len(SETS) + 1))
    for r in rows:
        cells = [f"{r[s]:.3f}" if r[s] is not None else "–" for s in SETS]
        print(f"| {r['run']} | {r['features']} | {r['head']} | " + " | ".join(cells) + f" | {r['mean']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
