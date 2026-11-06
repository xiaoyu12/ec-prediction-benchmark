#!/usr/bin/env python
"""Choose a logit shift for an already-trained sigmoid model on a held-out validation set.

    python calibrate.py --model milbc_plus --checkpoint ../runs/id50/milbc_plus/model.pth \\
        --npzdata ../features/shards/split100_{00..15}.npz --csvdata ../features/split100_min5.csv \\
        --val-entries ../features/folds/id50_val.txt \\
        --eval-set ../features/tests/test70.npz ../features/tests/test70.csv \\
        --apply-to ../checkpoints/retrained/milbc_plus

Same selection as ``train.py --select-shift`` (maximum F1 on the validation set -- per-enzyme by
default, see ``--shift-criterion`` -- ties to the smaller shift). Writes ``calibration.json`` beside ``--checkpoint`` and, with
``--apply-to DIR``, a copy into each DIR, so ``infer.py`` uses the shift for the checkpoint there.
The model must not have trained on the validation entries: pick the shift on a checkpoint trained
without them (e.g. ``runs/id50/*``) and carry it over to the full-data checkpoint of the same recipe.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from train import (
    PACKAGE_ROOT,
    attach_labels,
    build_model,
    load_archive,
    load_archives,
    read_labels,
    select_logit_shift,
    subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["milbc", "milbc_plus", "esm2_bc"])
    parser.add_argument("--checkpoint", required=True, help="model.pth trained WITHOUT the validation entries")
    parser.add_argument("--npzdata", required=True, nargs="+", help="training archive(s) holding the validation entries")
    parser.add_argument("--csvdata", required=True)
    parser.add_argument("--val-entries", required=True, help="Entry ids the checkpoint did not train on")
    parser.add_argument("--eval-set", nargs=2, action="append", default=[], metavar=("NPZ", "CSV"),
                        help="also report these sets at shift 0 and at the chosen shift (not used to choose)")
    parser.add_argument("--apply-to", nargs="+", default=[], help="directories to copy calibration.json into")
    parser.add_argument("--shift-criterion", choices=["sample", "micro", "weighted"], default="sample",
                        help="F1 maximised on the validation set (default: per-enzyme)")
    parser.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    classes = np.load(Path(args.classes).expanduser().resolve(), allow_pickle=True).astype(str)
    class_index = {ec: i for i, ec in enumerate(classes)}
    store = load_archives([Path(p).expanduser().resolve() for p in args.npzdata])
    labelled = attach_labels(store, read_labels(Path(args.csvdata).expanduser().resolve()), class_index,
                             "training set", training=True)
    val_path = Path(args.val_entries).expanduser().resolve()
    held = set(val_path.read_text().split())
    positions = np.array([i for i, row in enumerate(labelled.rows) if store.entries[row] in held], dtype=np.int64)
    if len(positions) != len(held):
        raise SystemExit(f"{val_path}: {len(held) - len(positions)} entries are not in the archives")
    name = val_path.stem.removesuffix("_val") or "valid"
    eval_sets = {name: (store.to(args.device), subset(labelled, positions))}
    for npz, csv_path in args.eval_set:
        npz_path = Path(npz).expanduser().resolve()
        eval_store = load_archive(npz_path).to(args.device)
        eval_sets[npz_path.stem] = (eval_store, attach_labels(eval_store, read_labels(Path(csv_path).expanduser().resolve()),
                                                              class_index, npz_path.stem, training=False))

    model = build_model(args.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.to(args.device)
    options = argparse.Namespace(validation_set=name, output_dir=checkpoint.parent, device=args.device,
                                 eval_batch_size=args.eval_batch_size, amp=False,
                                 shift_criterion=args.shift_criterion)
    select_logit_shift(model, args.model, eval_sets, classes, options)
    source = checkpoint.parent / "calibration.json"
    payload = json.loads(source.read_text())
    payload["chosen_with_checkpoint"] = str(checkpoint)
    source.write_text(json.dumps(payload, indent=2) + "\n")
    for directory in args.apply_to:
        target = Path(directory).expanduser().resolve()
        if not target.is_dir():
            raise SystemExit(f"--apply-to: {target} is not a directory")
        shutil.copyfile(source, target / "calibration.json")
        print(f"copied calibration.json to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
