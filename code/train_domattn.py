#!/usr/bin/env python
"""Train the domain-attention model (``models.DomainAttentionEC``): ESM2-BC + label-specific attention
over per-domain ESM2 embeddings, optionally with the EC hierarchy.

    F=../features
    python train_domattn.py --npzdata $F/shards/split100_{00..15}.npz --csvdata $F/split100_min5.csv \\
        --domains $F/domains/domain_esm.npz \\
        --val-entries ../scripts/folds/id50_val.txt --exclude-entries ../scripts/folds/id50_exclude.txt \\
        --eval-set $F/tests/test70.npz $F/tests/test70.csv --output-dir ../runs/domattn/id50_hier

Inputs: the global ESM2 embedding comes from the ``process.py`` archives (their ``.compact.npz``
caches), the domain features from ``domain_esm.py``. Calls use ``infer.call_ec`` (sigmoid >= 0.5 with
the at-least-one-label refinement) on the per-EC logits; for the hierarchical model those are
``logit(P(EC | x))`` with ``P`` the product of the conditional probabilities down the tree.

Hierarchical loss: binary cross-entropy on every node's conditional logit, counted only where the
node's parent is a positive of the protein (level-1 nodes always), i.e. a child head learns
``P(child | parent, x)`` from proteins known to belong to the parent. A label resolved only to a
prefix (``1.1.1.-``) marks that prefix positive and its children's negatives unresolved (masked).

Writes ``model.pth``, ``config.json``, ``history.json`` and ``predictions.csv`` (``Entry``,
``EC number`` predicted, one row per enzyme of every ``--eval-set``) into ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from infer import call_ec
from models import DomainAttentionEC, ESM_DIM
from train import (
    PACKAGE_ROOT,
    attach_labels,
    calls_to_ecs,
    load_archive,
    load_archives,
    micro_prf,
    read_labels,
    sample_prf,
    subset,
    weighted_prf,
)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


class DomainStore:
    """Global ESM2 embeddings plus a flat table of domain features, indexed by entry."""

    def __init__(self, entries: list[str], esm2: torch.Tensor, domain_file: Path, device: str) -> None:
        with np.load(domain_file, allow_pickle=False) as data:
            names = [str(e) for e in data["entries"]]
            offsets, flat = data["offsets"], data["domains"]
        where = {e: i for i, e in enumerate(names)}
        missing = [e for e in entries if e not in where]
        if missing:
            raise SystemExit(f"{domain_file}: no domain features for {len(missing)} entries, e.g. {missing[:3]}")
        pos = np.array([where[e] for e in entries])
        self.entries = entries
        self.esm2 = esm2.to(device)
        self.offset = torch.from_numpy(offsets[pos]).to(device)
        self.count = torch.from_numpy(offsets[pos + 1] - offsets[pos]).to(device)
        self.domains = torch.from_numpy(flat).to(device)  # float16, (total, 1280)

    def batch(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = self.count[rows]
        width = int(count.max())
        steps = torch.arange(width, device=rows.device)
        mask = steps[None, :] < count[:, None]
        index = (self.offset[rows][:, None] + steps[None, :]).masked_fill(~mask, 0)
        domains = self.domains[index].float() * mask[..., None]
        return domains, mask, self.esm2[rows]


def load_store(paths: list[Path], domain_file: Path, device: str, single: bool = False):
    features = load_archive(paths[0]) if single else load_archives(paths)
    return features, DomainStore(features.entries, features.esm2, domain_file, device)


def node_targets(true_ecs: list[list[str]], nodes: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded tables of every enzyme's positive nodes and unresolved nodes (``-1`` padding).

    A complete EC makes all four of its prefixes positive; ``a.b.c.-`` makes ``a``, ``a.b``,
    ``a.b.c`` positive and ``a.b.c`` unresolved (its children's negatives are not known).
    """
    position = {n: i for i, n in enumerate(nodes)}
    positives, unresolved = [], []
    for ecs in true_ecs:
        pos, unres = set(), set()
        for ec in ecs:
            parts = ec.split(".")
            known = [p for p in parts if p not in {"-", ""}]
            prefixes = [".".join(parts[: d + 1]) for d in range(len(known))]
            # Every prefix that is an output becomes positive: all four in the hierarchical model,
            # only the full EC in the flat one (an earlier version stopped at the first prefix that
            # was not an output, which left the flat model with no positives at all).
            pos.update(position[prefix] for prefix in prefixes if prefix in position)
            if len(known) < 4 and prefixes and prefixes[-1] in position:
                unres.add(position[prefixes[-1]])
        positives.append(sorted(pos))
        unresolved.append(sorted(unres))

    def pad(rows):
        table = torch.full((len(rows), max(1, max(len(r) for r in rows))), -1, dtype=torch.long)
        for i, r in enumerate(rows):
            table[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return table

    return pad(positives), pad(unresolved)


def multi_hot(table: torch.Tensor, width: int) -> torch.Tensor:
    out = torch.zeros(table.shape[0], width + 1, device=table.device)
    out.scatter_(1, table.masked_fill(table < 0, width), 1.0)
    return out[:, :width]


def hierarchical_loss(model: DomainAttentionEC, logits, positive, unresolved) -> torch.Tensor:
    """Masked BCE on conditional logits: a node counts where its parent is positive (level 1: always),
    except a negative child of an unresolved parent."""
    parent = model.parent
    top = parent < 0
    parent_positive = torch.where(top[None, :], torch.ones_like(positive), positive[:, parent.clamp(min=0)])
    parent_unresolved = torch.where(top[None, :], torch.zeros_like(unresolved), unresolved[:, parent.clamp(min=0)])
    mask = parent_positive * (1 - parent_unresolved * (1 - positive))
    loss = F.binary_cross_entropy_with_logits(logits, positive, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def leaf_logits(model, store: DomainStore, rows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(rows), batch_size):
        r = torch.from_numpy(rows[start:start + batch_size]).to(store.esm2.device)
        logits, _ = model(*store.batch(r))
        out.append(model.leaf_logits(logits).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, eval_sets, classes) -> dict:
    results = {}
    for name, (store, data) in eval_sets.items():
        predicted = calls_to_ecs(call_ec(leaf_logits(model, store, data.rows)), classes)
        results[name] = {
            "sample": sample_prf(data.true_ecs, predicted)["f1"],
            "micro": micro_prf(data.true_ecs, predicted)["f1"],
            "weighted": weighted_prf(data.true_ecs, predicted)["f1"],
            "calls": float(np.mean([len(p) for p in predicted])),
        }
    return results


def format_results(results: dict) -> str:
    return "  ".join(f"{n}: F1 sample {r['sample']:.3f} micro {r['micro']:.3f} weighted {r['weighted']:.3f} "
                     f"({r['calls']:.2f} calls)" for n, r in results.items())


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npzdata", required=True, nargs="+", help="training archives (global ESM2)")
    parser.add_argument("--csvdata", required=True)
    parser.add_argument("--domains", required=True, help="domain_esm.py output covering training and eval entries")
    parser.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"))
    parser.add_argument("--eval-set", nargs=2, action="append", default=[], metavar=("NPZ", "CSV"))
    parser.add_argument("--val-entries", help="Entry ids held out as the validation set")
    parser.add_argument("--exclude-entries", help="Entry ids dropped from training (with --val-entries)")
    parser.add_argument("--flat", action="store_true", help="no EC hierarchy: independent sigmoids over the 5,242 ECs")
    parser.add_argument("--width", type=int, default=64, help="attention key/value width d")
    parser.add_argument("--no-domains", action="store_true", help="ablation: drop the domain-attention branch")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.exclude_entries and not args.val_entries:
        parser.error("--exclude-entries goes with --val-entries")
    return args


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = np.load(Path(args.classes).expanduser().resolve(), allow_pickle=True).astype(str)
    class_index = {ec: i for i, ec in enumerate(classes)}
    domain_file = Path(args.domains).expanduser().resolve()

    features, store = load_store([Path(p).expanduser().resolve() for p in args.npzdata], domain_file, args.device)
    train_set = attach_labels(features, read_labels(Path(args.csvdata).expanduser().resolve()), class_index,
                              "training set", training=True)
    eval_sets = {}
    if args.val_entries:
        held = set(Path(args.val_entries).expanduser().resolve().read_text().split())
        dropped = set(Path(args.exclude_entries).expanduser().resolve().read_text().split()) if args.exclude_entries else set()
        entry_of = [features.entries[r] for r in train_set.rows]
        val = np.array([i for i, e in enumerate(entry_of) if e in held], dtype=np.int64)
        keep = np.array([i for i, e in enumerate(entry_of) if e not in held and e not in dropped], dtype=np.int64)
        name = Path(args.val_entries).stem.removesuffix("_val") or "valid"
        eval_sets[name] = (store, subset(train_set, val))
        train_set = subset(train_set, keep)
    for npz, csv_path in args.eval_set:
        npz_path = Path(npz).expanduser().resolve()
        set_features, set_store = load_store([npz_path], domain_file, args.device, single=True)
        eval_sets[npz_path.stem] = (set_store, attach_labels(set_features, read_labels(Path(csv_path).expanduser().resolve()),
                                                              class_index, npz_path.stem, training=False))

    model = DomainAttentionEC(list(classes), ESM_DIM, ESM_DIM, args.width, hierarchical=not args.flat,
                              dropout_rate=args.dropout, use_domains=not args.no_domains).to(args.device)
    positive_table, unresolved_table = (t.to(args.device) for t in node_targets(train_set.true_ecs, model.nodes))
    n_out = len(model.nodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = torch.from_numpy(train_set.rows).to(args.device)
    steps = math.ceil(len(rows) / args.batch_size)
    config = {k: v for k, v in vars(args).items()}
    config.update(training_enzymes=len(rows), outputs=n_out, parameters=sum(p.numel() for p in model.parameters()),
                  eval_sets={n: len(d.rows) for n, (_, d) in eval_sets.items()})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        order = torch.randperm(len(rows), device=args.device)
        for step in range(steps):
            positions = order[step * args.batch_size:(step + 1) * args.batch_size]
            logits, _ = model(*store.batch(rows[positions]))
            positive = multi_hot(positive_table[positions], n_out)
            if args.flat:
                loss = F.binary_cross_entropy_with_logits(logits, positive)
            else:
                loss = hierarchical_loss(model, logits, positive, multi_hot(unresolved_table[positions], n_out))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        record = {"epoch": epoch, "loss": total / steps, "seconds": round(time.time() - started, 1)}
        if eval_sets and (epoch % args.eval_every == 0 or epoch == args.epochs):
            record["eval"] = evaluate(model, eval_sets, classes)
        history.append(record)
        print(f"epoch {epoch:3d}/{args.epochs}  loss {record['loss']:.6f}  {record['seconds']:.0f}s  "
              + format_results(record.get("eval", {})), flush=True)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "model.pth")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    with open(out_dir / "predictions.csv", "w") as handle:
        handle.write("Entry,EC number\n")
        written = set()
        for name, (set_store, data) in eval_sets.items():
            if name in {Path(args.val_entries).stem.removesuffix("_val") if args.val_entries else None}:
                continue
            calls = calls_to_ecs(call_ec(leaf_logits(model, set_store, data.rows)), classes)
            for row, ecs in zip(data.rows, calls):
                entry = set_store.entries[row]
                if entry not in written:
                    handle.write(f"{entry},{';'.join(ecs)}\n")
                    written.add(entry)
    print(f"wrote {out_dir / 'model.pth'} and {out_dir / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
