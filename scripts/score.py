"""Score checkpoints on the six test sets with infer.py's decision rules and CLEAN's weighted P/R/F1.

    python scripts/score.py name=model:checkpoint [name=model:checkpoint ...]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from train import read_labels, weighted_prf  # noqa: E402

TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]


def score(model: str, checkpoint: str) -> dict:
    results = {}
    with tempfile.TemporaryDirectory(dir=ROOT / "features") as tmp:
        for name in TESTS:
            out = Path(tmp) / f"{name}.csv"
            subprocess.run(
                [sys.executable, "infer.py", "--npzdata", str(ROOT / f"features/tests/{name}.npz"), "--model", model,
                 "--checkpoint", checkpoint, "--output", str(out)],
                cwd=ROOT / "code", check=True, capture_output=True,
            )
            truth = read_labels(ROOT / f"features/tests/{name}.csv")
            predicted = {}
            for line in out.read_text().splitlines()[1:]:
                entry, ecs, _ = line.split(",")[:3]
                predicted[entry] = ecs.split(";")
            entries = list(predicted)
            results[name] = weighted_prf([truth[e][0] for e in entries], [predicted[e] for e in entries])
    return results


if __name__ == "__main__":
    table = {}
    for spec in sys.argv[1:]:
        label, rest = spec.split("=", 1)
        model, checkpoint = rest.split(":", 1)
        table[label] = score(model, str(Path(checkpoint).resolve()))
        row = "  ".join(f"{t} {table[label][t]['f1']:.3f}" for t in TESTS)
        print(f"{label:24s} F1  {row}", flush=True)
    print(json.dumps(table, indent=1))
