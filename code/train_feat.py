#!/usr/bin/env python
"""EC classifiers on fixed per-protein embeddings (one or several, concatenated): the autoresearch
workbench.

    python train_feat.py --features esmc-600m:mid esm2-650m:last --protocol id50 --output-dir ../runs/feat/x

Features are ``<model>:<key>`` arrays from ``features/plm/<model>/<key>.npy`` (``extract_plm.py``;
``esm2-650m:last`` is the original global ESM2 embedding), rows aligned with ``entries.txt``.
Each block is standardized with training-set statistics; ``--log1p`` blocks (SAE features) are
``log1p``-transformed first.

Protocols, on the same data as every other model in this repo:

- ``id50``: train on split100 minus the identity-clustered validation fold and its leak-excluded
  neighbours (``scripts/folds``), select on the fold. Use this to choose configurations.
- ``full``: train on all of split100. Use this for the test-set table.

Both evaluate the six test sets every ``--eval-every`` epochs, with ``infer.call_ec`` (sigmoid >= 0.5,
at-least-one fallback, cap of five). Writes ``config.json``, ``history.json``, ``metrics.json``
(final epoch), ``predictions.csv`` (test-set calls) and ``logits.npz`` (final logits of every
eval set, for ensembling).

Heads (``--arch``):

- ``mlp``: per-block ``Linear -> LayerNorm`` projection to ``--proj``, concatenated, then
  ``--layers`` x (``Linear -> LayerNorm -> GELU -> Dropout``) of width ``--hidden``, then the
  5,242-way output. ``--layers 2 --hidden 320`` on one block is ESM2-BC's head.
- ``gated``: the projected blocks are mixed by a learned per-protein softmax gate before the MLP.
- ``linear``: a single linear layer on the standardized concatenation.

Losses: ``bce`` (the paper's), ``asl`` (asymmetric loss, Ridnik et al. 2021: focusing on negatives
plus a probability margin), ``focal``.

Output layer (``--out``, for every head but ``linear``):

- ``linear`` (default): one weight vector and bias per EC.
- ``cosine``: ``s * cos(h, w_c) + b`` with learned weights, one shared learned scale ``s`` and bias
  ``b``. No per-class bias, so no learned class prior.
- ``proto``: ``s * cos(h, mu_c) + b`` where ``mu_c`` is the mean of the L2-normalized hidden vectors
  of the training proteins of EC ``c``. ``mu_c`` is not learned: it is recomputed from the
  current network every ``--proto-every`` steps and before evaluation; hidden vectors are centered
  on the training mean before the cosine. A class with one training protein
  gets a prototype as good as that protein's embedding; the network learns a metric space in which
  proteins sit near their ECs' means.

Rare-class rebalancing: ``--balance beta`` weights each training protein's loss by
``n^-beta`` (normalized to mean 1), ``n`` = training count of its rarest EC.
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
from torch import nn
from train import PACKAGE_ROOT, MIN_LENGTH, micro_prf, read_labels, sample_prf, weighted_prf

FEATURE_ROOT = PACKAGE_ROOT / "features" / "plm"
TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
FOLDS = PACKAGE_ROOT / "scripts" / "folds"


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def load_block(spec: str) -> tuple[list[str], np.ndarray]:
    model, key = spec.split(":")
    folder = FEATURE_ROOT / model
    entries = (folder / "entries.txt").read_text().split()
    return entries, np.load(folder / f"{key}.npy", mmap_mode="r")


class Bank:
    """Feature blocks on the GPU (float16), standardized on the fly with training statistics."""

    def __init__(self, specs: list[str], log1p: set[str], device: str) -> None:
        self.specs, self.blocks, self.entries = specs, [], None
        for spec in specs:
            entries, array = load_block(spec)
            if self.entries is None:
                self.entries = entries
            elif entries != self.entries:
                raise SystemExit(f"{spec}: entries differ from {specs[0]}")
            block = torch.empty(array.shape, dtype=torch.float16, device=device)
            for start in range(0, len(array), 16384):  # chunked: never a full float32 copy in RAM or on GPU
                chunk = torch.from_numpy(np.array(array[start:start + 16384])).to(device)
                if spec in log1p:
                    chunk = torch.log1p(chunk.float().clamp_min(0)).half()
                block[start:start + 16384] = chunk
            self.blocks.append(block)
        self.index = {e: i for i, e in enumerate(self.entries)}
        self.dims = [b.shape[1] for b in self.blocks]
        self.mean = self.std = None

    def fit(self, rows: torch.Tensor) -> None:
        self.mean, self.std = [], []
        for block in self.blocks:
            total = torch.zeros(block.shape[1], device=block.device, dtype=torch.float64)
            square = torch.zeros_like(total)
            for chunk in rows.split(16384):
                x = block[chunk].double()
                total += x.sum(0)
                square += (x * x).sum(0)
            mean = total / len(rows)
            std = (square / len(rows) - mean * mean).clamp_min(0).sqrt()
            self.mean.append(mean.float())
            self.std.append(std.float().clamp_min(1e-3 * float(std.mean()) + 1e-6))

    def batch(self, rows: torch.Tensor) -> list[torch.Tensor]:
        return [(b[rows].float() - m) / s for b, m, s in zip(self.blocks, self.mean, self.std)]


def labelled(bank: Bank, labels: dict, class_index: dict[str, int], training: bool):
    rows, true_ecs, classes = [], [], []
    for entry, (ecs, length) in labels.items():
        if entry not in bank.index:
            raise SystemExit(f"{entry} has no features")
        if training and length is not None and length < MIN_LENGTH:
            continue
        rows.append(bank.index[entry])
        true_ecs.append(ecs)
        classes.append(sorted({class_index[ec] for ec in ecs if ec in class_index}))
    return np.asarray(rows, dtype=np.int64), true_ecs, classes


def target_table(classes: list[list[int]], device: str) -> torch.Tensor:
    width = max(1, max(len(c) for c in classes))
    table = torch.full((len(classes), width), -1, dtype=torch.long)
    for i, c in enumerate(classes):
        table[i, :len(c)] = torch.tensor(c, dtype=torch.long)
    return table.to(device)


def multi_hot(table: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(table.shape[0], n + 1, device=table.device)
    out.scatter_(1, table.masked_fill(table < 0, n), 1.0)
    return out[:, :n]


# --------------------------------------------------------------------------------------------
# Model and losses
# --------------------------------------------------------------------------------------------


ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class ResBlock(nn.Module):
    """Pre-norm residual feed-forward block: x + Dropout(W2 act(W1 LN(x)))."""

    def __init__(self, width: int, dropout: float, act: str, expand: int = 2) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, expand * width), ACTS[act](), nn.Dropout(dropout),
                               nn.Linear(expand * width, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class FeatureHead(nn.Module):
    """Head on per-protein embedding blocks. Every block is projected to ``proj`` (Linear + LayerNorm).

    - ``mlp``: concatenate, then ``layers`` x (Linear, LayerNorm, act, Dropout).
    - ``gated``: softmax-weighted sum of the projected blocks, then the same body.
    - ``resmlp``: concatenate, Linear to ``hidden``, ``layers`` pre-norm residual blocks (2x expansion), LayerNorm.
    - ``blockattn``: each projected block is a token (plus a learned block embedding) next to a [CLS]
      token; ``layers`` pre-norm Transformer encoder layers (``heads`` heads, feed-forward ``hidden``);
      the [CLS] output is the representation.
    - ``linear``: logistic regression on the concatenated blocks.
    """

    def __init__(self, dims: list[int], n_out: int, arch: str, proj: int, hidden: int, layers: int,
                 dropout: float, input_dropout: float, block_dropout: float, act: str = "gelu", heads: int = 8) -> None:
        super().__init__()
        self.arch, self.block_dropout = arch, block_dropout
        self.input_dropout = nn.Dropout(input_dropout)
        if arch == "linear":
            self.out = nn.Linear(sum(dims), n_out)
            return
        self.proj = nn.ModuleList(nn.Sequential(nn.Linear(d, proj), nn.LayerNorm(proj)) for d in dims)
        if arch == "blockattn":
            self.cls = nn.Parameter(torch.zeros(1, 1, proj))
            self.block_embedding = nn.Parameter(torch.randn(1, len(dims), proj) * 0.02)
            layer = nn.TransformerEncoderLayer(proj, heads, hidden, dropout, activation=act if act != "silu" else F.silu,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(proj)
            self.out = nn.Linear(proj, n_out)
            return
        width = proj if arch == "gated" else proj * len(dims)
        if arch == "gated":
            self.gate = nn.Linear(proj * len(dims), len(dims))
        if arch == "resmlp":
            self.body = nn.Sequential(ACTS[act](), nn.Dropout(dropout), nn.Linear(width, hidden),
                                      *[ResBlock(hidden, dropout, act) for _ in range(layers)], nn.LayerNorm(hidden))
            self.out = nn.Linear(hidden, n_out)
            return
        body = []
        for _ in range(layers):
            body += [nn.Linear(width, hidden), nn.LayerNorm(hidden), ACTS[act](), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(ACTS[act](), nn.Dropout(dropout), *body) if body else nn.Identity()
        self.out = nn.Linear(width, n_out)

    def forward(self, blocks: list[torch.Tensor], hidden: bool = False) -> torch.Tensor:
        blocks = [self.input_dropout(b) for b in blocks]
        if self.arch == "linear":
            return torch.cat(blocks, 1) if hidden else self.out(torch.cat(blocks, 1))
        z = [p(b) for p, b in zip(self.proj, blocks)]
        if self.training and self.block_dropout > 0 and len(z) > 1:
            keep = (torch.rand(z[0].shape[0], len(z), device=z[0].device) > self.block_dropout).float()
            keep[keep.sum(1) == 0, torch.randint(len(z), (1,))] = 1.0  # keep at least one block
            z = [zi * keep[:, i:i + 1] * len(z) / keep.sum(1, keepdim=True) for i, zi in enumerate(z)]
        if self.arch == "blockattn":
            tokens = torch.stack(z, 1) + self.block_embedding
            tokens = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], 1)
            h = self.norm(self.body(tokens)[:, 0])
        elif self.arch == "gated":
            weights = torch.softmax(self.gate(torch.cat(z, 1)), dim=1)
            h = self.body(sum(w[:, None] * zi for w, zi in zip(weights.unbind(1), z)))
        else:
            h = self.body(torch.cat(z, 1))
        return h if hidden else self.out(h)


class CosineOut(nn.Module):
    """``scale * cos(h, w_c) + bias``: learned (``cosine``) or class-mean (``proto``) vectors ``w_c``."""

    def __init__(self, width: int, n_out: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))
        self.bias = nn.Parameter(torch.tensor(-8.0))
        if kind == "cosine":
            self.weight = nn.Parameter(torch.randn(n_out, width) / math.sqrt(width))
        else:
            self.register_buffer("weight", torch.zeros(n_out, width))
        self.register_buffer("center", torch.zeros(width))  # proto: mean training hidden vector
        self.in_features = width

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = F.normalize(h - self.center, dim=1)
        return self.log_scale.exp() * h @ F.normalize(self.weight, dim=1).T + self.bias


@torch.no_grad()
def refresh_prototypes(model: "FeatureHead", bank, rows: torch.Tensor, table: torch.Tensor) -> None:
    """Class means of the normalized hidden vectors of the training proteins (multi-label: each EC)."""
    was_training = model.training
    model.eval()
    h = torch.cat([model(bank.batch(c), hidden=True).float() for c in rows.split(8192)])
    center = h.mean(0)
    h = F.normalize(h - center, dim=1)
    model.out.center.copy_(center)
    proto = torch.zeros_like(model.out.weight)
    for j in range(table.shape[1]):
        valid = table[:, j] >= 0
        proto.index_add_(0, table[valid, j], h[valid])
    model.out.weight.copy_(F.normalize(proto, dim=1))
    model.train(was_training)


def loss_fn(name: str, logits: torch.Tensor, target: torch.Tensor, smoothing: float,
            weight: torch.Tensor | None = None) -> torch.Tensor:
    if smoothing:  # soften the positives only: 5,241 negatives would otherwise swamp the targets
        target = target * (1 - smoothing)
    if weight is not None:  # per-protein weights (mean 1): BCE only
        assert name == "bce"
        return (F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(1) * weight).mean()
    if name == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    if name == "focal":
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.where(target > 0.5, p, 1 - p)
        return ((1 - pt) ** 2 * ce).mean()
    # ASL: gamma_neg=4, gamma_pos=0, clip=0.05; summed over labels, averaged over the batch as in the
    # paper, rescaled by the label count so learning rates stay comparable with BCE.
    p_neg = (p - 0.05).clamp_min(0)
    pos = target * F.logsigmoid(logits)
    neg = (1 - target) * p_neg.pow(4) * torch.log((1 - p_neg).clamp_min(1e-8))
    return -(pos + neg).mean()


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def predict(model: nn.Module, bank: Bank, rows: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    index = torch.from_numpy(rows).to(bank.blocks[0].device)
    return torch.cat([model(bank.batch(chunk)).float().cpu() for chunk in index.split(batch)]).numpy()


@torch.no_grad()
def hidden_knn(model, bank, train_rows: np.ndarray, train_classes, eval_sets, n_classes: int, ks, temperatures):
    """Nearest-neighbour EC transfer in the trained head's last hidden layer (a learned metric):
    ``{(k, T): {set: logits}}``."""
    model.eval()
    device = bank.blocks[0].device

    def embed(rows):
        index = torch.from_numpy(rows).to(device)
        return torch.cat([F.normalize(model(bank.batch(c), hidden=True).float(), dim=1).half() for c in index.split(4096)])

    reference = embed(train_rows)
    labels = torch.zeros(len(train_classes), n_classes, dtype=torch.bool)  # filled on CPU: 200k tiny GPU writes are slow
    for i, c in enumerate(train_classes):
        labels[i, c] = True
    labels = labels.to(device)
    out = {}
    kmax = max(ks)
    for name, (rows, _) in eval_sets.items():
        sims, idx = [], []
        for chunk in embed(rows).split(1024):
            s, i = (chunk @ reference.T).float().topk(kmax, dim=1)
            sims.append(s)
            idx.append(i)
        s, i = torch.cat(sims), torch.cat(idx)
        for k in ks:
            for t in temperatures:
                w = torch.softmax(s[:, :k] / t, dim=1)
                prob = torch.zeros(len(s), n_classes, device=device)
                for j in range(k):
                    prob += w[:, j:j + 1] * labels[i[:, j]].float()
                prob = prob.clamp(1e-6, 1 - 1e-6)
                out.setdefault((k, t), {})[name] = torch.log(prob / (1 - prob)).cpu().numpy()
    return out


def write_run(out: Path, config: dict, metrics: dict, logits: dict, eval_sets, bank, classes) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "logits.npz", **{n: v.astype(np.float16) for n, v in logits.items()},
                        **{f"{n}__entries": np.array([bank.entries[i] for i in r]) for n, (r, _) in eval_sets.items()})
    with open(out / "predictions.csv", "w") as handle:
        handle.write("Entry,EC number\n")
        written = set()
        for name in TESTS:
            r, _ = eval_sets[name]
            for row, calls in zip(r, call_ec(logits[name])):
                entry = bank.entries[row]
                if entry not in written:
                    handle.write(f"{entry},{';'.join(classes[calls])}\n")
                    written.add(entry)


def score(logits: np.ndarray, true_ecs: list[list[str]], classes: np.ndarray, shift: float = 0.0) -> dict:
    calls = call_ec(logits, shift)
    predicted = [list(classes[row]) for row in calls]
    return {"sample": sample_prf(true_ecs, predicted)["f1"], "micro": micro_prf(true_ecs, predicted)["f1"],
            "weighted": weighted_prf(true_ecs, predicted)["f1"], "calls": float(calls.sum(1).mean())}


def evaluate(model, bank, eval_sets, classes) -> dict:
    return {name: score(predict(model, bank, rows), true_ecs, classes) for name, (rows, true_ecs) in eval_sets.items()}


def summary(results: dict) -> str:
    return "  ".join(f"{n}: {r['sample']:.3f}/{r['micro']:.3f}/{r['weighted']:.3f}" for n, r in results.items())


# --------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True, help="<model>:<key> blocks")
    p.add_argument("--log1p", nargs="*", default=None, help="blocks to log1p (default: every *sae* block)")
    p.add_argument("--protocol", choices=["id50", "full"], required=True)
    p.add_argument("--arch", choices=["mlp", "gated", "resmlp", "blockattn", "linear"], default="mlp")
    p.add_argument("--act", choices=sorted(ACTS), default="gelu")
    p.add_argument("--heads", type=int, default=8, help="attention heads (blockattn)")
    p.add_argument("--out", choices=["linear", "cosine", "proto"], default="linear", help="output layer")
    p.add_argument("--proto-every", type=int, default=50, help="steps between prototype refreshes (proto)")
    p.add_argument("--balance", type=float, default=0.0,
                   help="weight proteins by (training count of their rarest EC)^-beta")
    p.add_argument("--proj", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--input-dropout", type=float, default=0.1)
    p.add_argument("--block-dropout", type=float, default=0.0)
    p.add_argument("--loss", choices=["bce", "asl", "focal"], default="bce")
    p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--hier", type=float, default=0.0,
                   help="weight of auxiliary BCE losses on EC prefixes (levels 1-3) from the hidden layer")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=0.03, help="fraction of steps")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the evaluated weights (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hknn", action="store_true", help="also write a kNN-in-hidden-space run (<name>_hknn)")
    p.add_argument("--hknn-k", type=int, nargs="+", default=[1, 3, 10])
    p.add_argument("--hknn-t", type=float, nargs="+", default=[0.02, 0.05])
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    classes = np.load(args.classes, allow_pickle=True).astype(str)
    class_index = {ec: i for i, ec in enumerate(classes)}
    log1p = set(args.log1p) if args.log1p is not None else {s for s in args.features if "sae" in s}
    started = time.time()
    bank = Bank(args.features, log1p, args.device)

    train_rows, train_ecs, train_classes = labelled(bank, read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv"),
                                                    class_index, training=True)
    eval_sets = {}
    if args.protocol == "id50":
        held = set((FOLDS / "id50_val.txt").read_text().split())
        dropped = set((FOLDS / "id50_exclude.txt").read_text().split())
        entries = [bank.entries[r] for r in train_rows]
        val = [i for i, e in enumerate(entries) if e in held]
        keep = [i for i, e in enumerate(entries) if e not in held and e not in dropped]
        eval_sets["id50"] = (train_rows[val], [train_ecs[i] for i in val])
        train_rows, train_classes = train_rows[keep], [train_classes[i] for i in keep]
    for name in TESTS:
        rows, ecs, _ = labelled(bank, read_labels(PACKAGE_ROOT / "features" / "tests" / f"{name}.csv"), class_index,
                                training=False)
        eval_sets[name] = (rows, ecs)

    rows = torch.from_numpy(train_rows).to(args.device)
    bank.fit(rows)
    table = target_table(train_classes, args.device)
    n_out = len(classes)
    model = FeatureHead(bank.dims, n_out, args.arch, args.proj, args.hidden, args.layers, args.dropout,
                        args.input_dropout, args.block_dropout, args.act, args.heads).to(args.device)
    if args.out != "linear":
        assert args.arch != "linear" and not args.ema
        model.out = CosineOut(model.out.in_features, n_out, args.out).to(args.device)
    sample_weight = None
    if args.balance:
        counts = np.bincount([c for cs in train_classes for c in cs], minlength=n_out)
        rarest = np.array([min(counts[c] for c in cs) if cs else 1 for cs in train_classes], dtype=np.float64)
        w = rarest ** -args.balance
        sample_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=args.device)
    aux, prefix_maps = nn.ModuleList(), []
    if args.hier:  # one linear head per EC level 1-3 on the shared hidden representation
        width = model.out.in_features
        for level in (1, 2, 3):
            names = sorted({".".join(c.split(".")[:level]) for c in classes})
            where = {n: i for i, n in enumerate(names)}
            m = torch.zeros(n_out, len(names), device=args.device)
            m[torch.arange(n_out), torch.tensor([where[".".join(c.split(".")[:level])] for c in classes])] = 1.0
            prefix_maps.append(m)
            aux.append(nn.Linear(width, len(names)))
        aux = aux.to(args.device)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema)) if args.ema else None
    optimizer = torch.optim.AdamW([*model.parameters(), *aux.parameters()], lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(len(rows) / args.batch_size)
    total, warm = steps * args.epochs, max(1, int(args.warmup * steps * args.epochs))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    config = dict(vars(args), log1p=sorted(log1p), dims=bank.dims, training_enzymes=len(rows),
                  parameters=sum(p.numel() for p in model.parameters()),
                  eval_sets={n: len(r) for n, (r, _) in eval_sets.items()})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        order = torch.randperm(len(rows), device=args.device)
        for step in range(steps):
            pick = order[step * args.batch_size:(step + 1) * args.batch_size]
            if args.out == "proto" and step % args.proto_every == 0:
                refresh_prototypes(model, bank, rows, table)
            target = multi_hot(table[pick], n_out)
            weight = sample_weight[pick] if sample_weight is not None else None
            if args.hier:
                h = model(bank.batch(rows[pick]), hidden=True)
                loss = loss_fn(args.loss, model.out(h), target, args.smoothing, weight)
                for head, m in zip(aux, prefix_maps):
                    loss = loss + args.hier * F.binary_cross_entropy_with_logits(head(h), (target @ m).clamp_max(1.0))
            else:
                loss = loss_fn(args.loss, model(bank.batch(rows[pick])), target, args.smoothing, weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            if ema is not None:
                ema.update_parameters(model)
            running += loss.item()
        record = {"epoch": epoch, "loss": running / steps, "seconds": round(time.time() - t0, 1)}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if args.out == "proto":
                refresh_prototypes(model, bank, rows, table)
            record["eval"] = evaluate(ema.module if ema else model, bank, eval_sets, classes)
        history.append(record)
        print(f"epoch {epoch:3d}  loss {record['loss']:.5f}  {record['seconds']:.0f}s  {summary(record.get('eval', {}))}",
              flush=True)

    final = ema.module if ema else model
    torch.save({k: v.detach().cpu() for k, v in final.state_dict().items()}, out / "model.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    logits = {name: predict(final, bank, r) for name, (r, _) in eval_sets.items()}
    write_run(out, config, history[-1]["eval"], logits, eval_sets, bank, classes)

    if args.hknn:  # learned-metric kNN in the head's hidden space, written as the sibling run <name>_hknn
        grid = hidden_knn(final, bank, train_rows, train_classes, eval_sets, n_out, args.hknn_k, args.hknn_t)
        scores = {key: {n: score(v, eval_sets[n][1], classes) for n, v in sets.items()} for key, sets in grid.items()}
        for key, sc in scores.items():
            print(f"hknn k={key[0]} T={key[1]}  {summary(sc)}", flush=True)
        key = max(scores, key=lambda q: scores[q]["id50"]["sample"]) if "id50" in eval_sets else next(iter(scores))
        write_run(out.with_name(out.name + "_hknn"), dict(config, arch="hknn", k=key[0], temperature=key[1]),
                  scores[key], grid[key], eval_sets, bank, classes)
    print(f"done in {(time.time() - started) / 60:.1f} min: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
