#!/usr/bin/env python
"""Train the enzyme-MIL models on a feature archive built by ``process.py``.

    # 1. features for the training set (once; shard split100.csv for a smaller memory peak)
    python process.py --csvdata ../dataset/split100.csv

    # 2. MIL-BC / MIL-BC+ (binary cross-entropy, 80 epochs, AdamW 2e-4, wd 0.01, batch 512)
    python train.py --model milbc      --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv
    python train.py --model milbc_plus --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv

    # ESM2-BC baseline: the same head and BCE recipe on the whole-sequence ESM2 embedding alone
    python train.py --model esm2_bc    --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv

    # 3. MIL-CL+: BCE-train the fused backbone, then refine it end-to-end with contrastive learning
    python train.py --model milcl_backbone --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv
    python train.py --model milcl_plus     --npzdata ../dataset/split100.npz --csvdata ../dataset/split100.csv \\
                    --backbone ../runs/milcl_backbone/backbone.pth

Every run writes into ``--output-dir`` (default ``../runs/<model>``) a checkpoint ``infer.py`` reads
unchanged, plus ``config.json`` and ``history.json``:

``milbc`` / ``milbc_plus``   ``model.pth``                  state dict of ``AttentionMIL``
``esm2_bc``                  ``model.pth``                  state dict of ``ESM2BC``
``milcl_backbone``           ``backbone.pth``               state dict of ``MILBCPlusBackbone``
``milcl_plus``               ``model.pt`` + ``centroids.npy``  ``JointMILCL`` and its EC centroids

Labels come from the ``EC number`` column of ``--csvdata`` (``;``-separated for multifunctional
enzymes), joined to the archive on ``Entry``. Sequences shorter than five residues are dropped, as
in the paper. ``--num-folds 5 --fold k`` holds out one fold of the training data for
cross-validation, and ``--eval-set NPZ CSV`` (repeatable) scores an independent test set while
training, with the same decision rule ``infer.py`` uses and CLEAN's support-weighted precision,
recall and F1.

MIL-CL+ follows CLEAN's SupCon-Hard recipe: every epoch draws one anchor per EC class, ``--n-pos``
positives from the anchor's class and ``--n-neg`` negatives from the ``--neg-knn`` EC classes whose
centroids lie nearest to it, re-mined every ``--mine-every`` epochs. Orphan classes (one training
enzyme) take their positives from ``--augment-npz``, features of mutated copies written by
``orphans.py``; without it the anchor itself is the positive and dropout supplies the second view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from infer import apply_max_separation, call_ec
from models import (
    DOMAIN_DIM,
    ESM_DIM,
    NUM_CLASSES,
    AttentionMIL,
    ESM2BC,
    JointMILCL,
    MILBCPlusBackbone,
)
from torch import Tensor, nn

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TOKENS = 16  # MIL-BC / MIL-BC+ read only the local domain tokens, not the whole-protein one
MIN_LENGTH = 5
FAR_AWAY = 1.0e4  # centroid coordinate for an EC with no training enzyme: never a nearest neighbour

# Hyper-parameters reported in the paper; any of them can be overridden on the command line.
PAPER_DEFAULTS = {
    "milbc": {"epochs": 80, "batch_size": 512, "lr": 2e-4, "weight_decay": 0.01},
    "milbc_plus": {"epochs": 80, "batch_size": 512, "lr": 2e-4, "weight_decay": 0.01},
    "esm2_bc": {"epochs": 80, "batch_size": 512, "lr": 2e-4, "weight_decay": 0.01},
    "milcl_backbone": {"epochs": 80, "batch_size": 512, "lr": 2e-4, "weight_decay": 0.01},
    # batch_size counts anchors; each brings 1 + n_pos + n_neg enzymes into the step.
    "milcl_plus": {"epochs": 1500, "batch_size": 256, "lr": 5e-4, "weight_decay": 0.0},
}


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass
class Features:
    """Compact, model-ready features: 16 local tokens plus the whole-protein token, as ``infer.py``."""

    entries: list[str]
    domains: Tensor  # (n, 17, 480) float16 -- DCT values are k/128, exact in half precision
    masks: Tensor  # (n, 17) bool
    esm2: Tensor  # (n, 1280) float32

    def __len__(self) -> int:
        return len(self.entries)

    def to(self, device: str) -> "Features":
        return Features(self.entries, self.domains.to(device), self.masks.to(device), self.esm2.to(device))


def convert_archive(path: Path) -> Features:
    """Read a ``process.py`` archive into compact arrays, with the checks ``infer.load_features`` makes."""
    with np.load(path, allow_pickle=True) as archive:
        if "dataset_samples" not in archive.files:
            raise SystemExit(f"{path} must contain a 'dataset_samples' array; build it with process.py")
        samples = archive["dataset_samples"]
    if samples.ndim != 1 or len(samples) == 0:
        raise SystemExit(f"{path}: dataset_samples must be a non-empty vector")
    token_count = np.asarray(samples[0]["dct_feature"]).shape[0]
    if token_count != LOCAL_TOKENS:
        raise SystemExit(f"{path}: expected {LOCAL_TOKENS} domain tokens, found {token_count}")

    n = len(samples)
    domains = np.zeros((n, LOCAL_TOKENS + 1, DOMAIN_DIM), dtype=np.float16)
    masks = np.zeros((n, LOCAL_TOKENS + 1), dtype=bool)
    esm2 = np.zeros((n, ESM_DIM), dtype=np.float32)
    entries: list[str] = []
    for row in range(n):
        sample = samples[row]
        entry = str(sample["entry"])
        feature = np.asarray(sample["dct_feature"], dtype=np.float32)
        whole = np.asarray(sample["dct_global"], dtype=np.float32).reshape(-1)
        mask = np.asarray(sample["dct_mask"], dtype=bool)
        embedding = np.asarray(sample["esm2_feature"], dtype=np.float32).reshape(-1)
        if feature.shape[0] != LOCAL_TOKENS or feature.shape[1] not in {DOMAIN_DIM, 2 * DOMAIN_DIM}:
            raise SystemExit(f"{path}: unexpected DCT shape {feature.shape} for {entry}")
        if mask.shape != (LOCAL_TOKENS,) or whole.shape != (DOMAIN_DIM,) or embedding.shape != (ESM_DIM,):
            raise SystemExit(f"{path}: inconsistent mask, global or ESM shape for {entry}")
        if not mask.any():
            raise SystemExit(f"{path}: {entry} has no valid domain")
        if not (np.isfinite(feature).all() and np.isfinite(whole).all() and np.isfinite(embedding).all()):
            raise SystemExit(f"{path}: non-finite feature for {entry}")
        domains[row, :LOCAL_TOKENS] = feature[:, :DOMAIN_DIM]
        domains[row, -1] = whole
        masks[row, :LOCAL_TOKENS] = mask
        masks[row, -1] = True
        esm2[row] = embedding
        entries.append(entry)
        samples[row] = None  # release the float64 sample as soon as it is copied
    del samples
    return Features(entries, torch.from_numpy(domains), torch.from_numpy(masks), torch.from_numpy(esm2))


def load_archive(path: Path, use_cache: bool = True) -> Features:
    """Load one archive, through a ``<stem>.compact.npz`` cache beside it.

    ``process.py`` archives hold every token in float64 and must be unpickled whole, roughly 120 kB
    per enzyme; the cache is about 21 kB per enzyme and loads without unpickling. A cache whose
    archive is absent (e.g. only the caches were copied to another machine) is used as is.
    """
    cache = path.with_name(f"{path.stem}.compact.npz")
    fresh = cache.is_file() and (not path.exists() or cache.stat().st_mtime >= path.stat().st_mtime)
    if use_cache and fresh:
        with np.load(cache, allow_pickle=False) as data:
            return Features(
                [str(entry) for entry in data["entries"]],
                torch.from_numpy(data["domains"]),
                torch.from_numpy(data["masks"]),
                torch.from_numpy(data["esm2"]),
            )
    print(f"reading {path}", flush=True)
    features = convert_archive(path)
    if use_cache:
        try:
            np.savez(
                cache,
                entries=np.asarray(features.entries),
                domains=features.domains.numpy(),
                masks=features.masks.numpy(),
                esm2=features.esm2.numpy(),
            )
            print(f"cached compact features in {cache}", flush=True)
        except OSError as error:
            print(f"[warn] could not write {cache}: {error}", flush=True)
    return features


def load_archives(paths: list[Path], use_cache: bool = True) -> Features:
    """Concatenate several archives, e.g. ``process.py`` run on shards of split100."""
    parts = [load_archive(path, use_cache) for path in paths]
    entries = [entry for part in parts for entry in part.entries]
    if len(set(entries)) != len(entries):
        raise SystemExit("duplicate Entry across the feature archives")
    if len(parts) == 1:
        return parts[0]
    return Features(
        entries,
        torch.cat([part.domains for part in parts]),
        torch.cat([part.masks for part in parts]),
        torch.cat([part.esm2 for part in parts]),
    )


def read_labels(path: Path) -> dict[str, tuple[list[str], int | None]]:
    """``Entry -> (EC numbers, sequence length or None)`` from a tab- or comma-separated table."""
    text = path.read_text()
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = "\t" if first.count("\t") >= first.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows or "Entry" not in rows[0] or "EC number" not in rows[0]:
        raise SystemExit(f"{path} needs 'Entry' and 'EC number' columns")
    labels: dict[str, tuple[list[str], int | None]] = {}
    for row in rows:
        entry = (row["Entry"] or "").strip()
        ecs = [ec.strip() for ec in (row["EC number"] or "").split(";") if ec.strip()]
        if not entry or not ecs:
            raise SystemExit(f"{path}: {entry or 'a row'} has no Entry or no EC number")
        if entry in labels:
            raise SystemExit(f"{path}: duplicate Entry {entry}")
        sequence = row.get("Sequence")
        labels[entry] = (ecs, len(sequence.strip()) if sequence else None)
    return labels


@dataclass
class LabelledSet:
    """Rows of a ``Features`` store with their labels."""

    rows: np.ndarray  # positions into the store
    true_ecs: list[list[str]]  # ground truth as written, possibly outside the vocabulary
    classes: list[list[int]]  # ground truth as vocabulary indices


def attach_labels(
    features: Features,
    labels: dict[str, tuple[list[str], int | None]],
    class_index: dict[str, int],
    name: str,
    training: bool,
) -> LabelledSet:
    """Join archive rows to their labels; training labels must all be in the vocabulary."""
    rows, true_ecs, classes = [], [], []
    missing = short = 0
    for row, entry in enumerate(features.entries):
        if entry not in labels:
            missing += 1
            continue
        ecs, length = labels[entry]
        if training and length is not None and length < MIN_LENGTH:
            short += 1
            continue
        known = [class_index[ec] for ec in ecs if ec in class_index]
        if training and len(known) != len(ecs):
            unknown = sorted(set(ecs) - set(class_index))
            raise SystemExit(f"{name}: {entry} has EC {unknown} outside the {len(class_index)}-class vocabulary")
        rows.append(row)
        true_ecs.append(ecs)
        classes.append(sorted(set(known)))
    if missing:
        print(f"[warn] {name}: {missing} archive entries have no label row and are skipped", flush=True)
    if short:
        print(f"{name}: dropped {short} sequences shorter than {MIN_LENGTH} residues", flush=True)
    if not rows:
        raise SystemExit(f"{name}: no archive entry has a label")
    unlabelled = len(set(labels) - set(features.entries))
    if unlabelled:
        print(f"[warn] {name}: {unlabelled} labelled entries are missing from the archive", flush=True)
    return LabelledSet(np.asarray(rows, dtype=np.int64), true_ecs, classes)


def subset(data: LabelledSet, positions: np.ndarray) -> LabelledSet:
    return LabelledSet(
        data.rows[positions], [data.true_ecs[i] for i in positions], [data.classes[i] for i in positions]
    )


def padded_labels(classes: list[list[int]]) -> Tensor:
    """``(n, max_labels)`` class indices padded with ``-1``, for building multi-hot targets on device."""
    width = max(1, max(len(c) for c in classes))
    table = torch.full((len(classes), width), -1, dtype=torch.long)
    for row, c in enumerate(classes):
        table[row, : len(c)] = torch.tensor(c, dtype=torch.long)
    return table


def multi_hot(table: Tensor) -> Tensor:
    """Multi-hot ``(batch, NUM_CLASSES)`` targets from a padded index table."""
    target = torch.zeros(table.shape[0], NUM_CLASSES + 1, device=table.device)
    target.scatter_(1, table.masked_fill(table < 0, NUM_CLASSES), 1.0)
    return target[:, :NUM_CLASSES]


def fetch(store: Features, rows: Tensor, device: str, local_only: bool) -> tuple[Tensor, Tensor, Tensor]:
    rows = rows.to(store.domains.device)
    domains = store.domains[rows].to(device, non_blocking=True).float()
    masks = store.masks[rows].to(device, non_blocking=True)
    esm2 = store.esm2[rows].to(device, non_blocking=True)
    if local_only:
        domains, masks = domains[:, :LOCAL_TOKENS], masks[:, :LOCAL_TOKENS]
    return domains, masks, esm2


# --------------------------------------------------------------------------------------------
# Metrics -- CLEAN's evaluation: support-weighted precision, recall and F1 over every EC that is
# either true or predicted in the set (sklearn ``average='weighted'``, ``zero_division=0``).
# --------------------------------------------------------------------------------------------


def weighted_prf(true_ecs: list[list[str]], predicted: list[list[str]]) -> dict[str, float]:
    vocabulary = sorted({ec for ecs in true_ecs for ec in ecs} | {ec for ecs in predicted for ec in ecs})
    index = {ec: i for i, ec in enumerate(vocabulary)}
    truth = np.zeros((len(true_ecs), len(vocabulary)), dtype=bool)
    guess = np.zeros_like(truth)
    for row, (t, p) in enumerate(zip(true_ecs, predicted)):
        truth[row, [index[ec] for ec in t]] = True
        guess[row, [index[ec] for ec in p]] = True
    tp = (truth & guess).sum(0).astype(np.float64)
    support = truth.sum(0).astype(np.float64)
    called = guess.sum(0).astype(np.float64)
    precision = np.divide(tp, called, out=np.zeros_like(tp), where=called > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    weights = support / support.sum()
    return {
        "precision": float((precision * weights).sum()),
        "recall": float((recall * weights).sum()),
        "f1": float((f1 * weights).sum()),
    }


def calls_to_ecs(calls: np.ndarray, classes: np.ndarray) -> list[list[str]]:
    return [list(classes[row]) for row in calls]


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def build_model(name: str) -> nn.Module:
    if name == "milbc":
        return AttentionMIL(global_input_dim=0)
    if name == "milbc_plus":
        return AttentionMIL(global_input_dim=ESM_DIM)
    if name == "esm2_bc":
        return ESM2BC()
    if name == "milcl_backbone":
        return MILBCPlusBackbone()
    return JointMILCL()


def classifier_logits(model: nn.Module, name: str, domains: Tensor, masks: Tensor, esm2: Tensor) -> Tensor:
    if name == "milbc":
        return model(domains, masks, None)[0]
    if name in {"milbc_plus", "esm2_bc"}:
        return model(domains, masks, esm2)[0]
    backbone = model if name == "milcl_backbone" else model.backbone
    representation, _ = backbone.encode(domains, masks, esm2)
    return backbone.classify(representation)


@torch.no_grad()
def predict_logits(model, name, store, rows, device, batch_size, amp) -> np.ndarray:
    model.eval()
    out = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
    local_only = name in {"milbc", "milbc_plus", "esm2_bc"}
    for start in range(0, len(rows), batch_size):
        batch = torch.from_numpy(rows[start : start + batch_size])
        d, m, e = fetch(store, batch, device, local_only)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            logits = classifier_logits(model, name, d, m, e)
        out[start : start + len(batch)] = logits.float().cpu().numpy()
    return out


@torch.no_grad()
def project(model: JointMILCL, store, rows, device, batch_size, amp) -> Tensor:
    """Contrastive embeddings of ``rows``, on ``device``."""
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = torch.from_numpy(rows[start : start + batch_size])
        d, m, e = fetch(store, batch, device, local_only=False)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            chunks.append(model(d, m, e).float())
    return torch.cat(chunks)


def ec_centroids(projections: Tensor, classes: list[list[int]]) -> tuple[Tensor, Tensor]:
    """Mean projection of every EC's enzymes (CLEAN cluster centres) and which ECs have any."""
    owner = torch.tensor([row for row, c in enumerate(classes) for _ in c], device=projections.device)
    target = torch.tensor([k for c in classes for k in c], device=projections.device)
    sums = torch.zeros(NUM_CLASSES, projections.shape[1], device=projections.device)
    sums.index_add_(0, target, projections[owner])
    counts = torch.bincount(target, minlength=NUM_CLASSES).to(projections.dtype)
    present = counts > 0
    centroids = torch.full_like(sums, FAR_AWAY)
    centroids[present] = sums[present] / counts[present].unsqueeze(1)
    return centroids, present


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


def evaluate(model, name, store, eval_sets, train_set, classes, args) -> dict[str, dict[str, float]]:
    if not eval_sets:
        return {}
    results = {}
    if name == "milcl_plus":
        train_projection = project(model, store, train_set.rows, args.device, args.eval_batch_size, args.amp)
        centroids, _ = ec_centroids(train_projection, train_set.classes)
        centroids = centroids.cpu().numpy()
    for set_name, (set_store, data) in eval_sets.items():
        if name == "milcl_plus":
            projection = project(model, set_store, data.rows, args.device, args.eval_batch_size, args.amp)
            calls, _ = apply_max_separation(projection.cpu().numpy(), centroids)
        else:
            logits = predict_logits(model, name, set_store, data.rows, args.device, args.eval_batch_size, args.amp)
            calls = call_ec(logits)
        results[set_name] = weighted_prf(data.true_ecs, calls_to_ecs(calls, classes))
    return results


def format_metrics(results: dict[str, dict[str, float]]) -> str:
    return "  ".join(
        f"{name}: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}" for name, m in results.items()
    )


# --------------------------------------------------------------------------------------------
# Binary cross-entropy training: MIL-BC, MIL-BC+ and the MIL-CL+ backbone
# --------------------------------------------------------------------------------------------


def train_bce(model, name, store, train_set, eval_sets, classes, args, history) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    table = padded_labels(train_set.classes).to(args.device)
    rows = torch.from_numpy(train_set.rows)
    local_only = name in {"milbc", "milbc_plus", "esm2_bc"}
    device_type = args.device.split(":")[0]
    steps = math.ceil(len(rows) / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        order = torch.randperm(len(rows))
        total = 0.0
        for step in range(steps):
            positions = order[step * args.batch_size : (step + 1) * args.batch_size]
            d, m, e = fetch(store, rows[positions], args.device, local_only)
            target = multi_hot(table[positions.to(args.device)])
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.amp):
                logits = classifier_logits(model, name, d, m, e)
            loss = loss_fn(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        record = {"epoch": epoch, "loss": total / steps, "seconds": round(time.time() - started, 1)}
        if eval_sets and (epoch % args.eval_every == 0 or epoch == args.epochs):
            record["eval"] = evaluate(model, name, store, eval_sets, train_set, classes, args)
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs or "eval" in record:
            print(
                f"epoch {epoch:4d}/{args.epochs}  loss {record['loss']:.6f}  {record['seconds']:.0f}s  "
                + format_metrics(record.get("eval", {})),
                flush=True,
            )


# --------------------------------------------------------------------------------------------
# Contrastive training: MIL-CL+
# --------------------------------------------------------------------------------------------


def supcon_hard_loss(embeddings: Tensor, temperature: float, n_pos: int) -> Tensor:
    """CLEAN's SupCon-Hard loss on ``(batch, 1 + n_pos + n_neg, dim)``; the anchor comes first.

    For each anchor e with sampled positives P(e) and sampled set A(e) = positives + negatives:
    ``-1/|P(e)| * sum_p log( exp(h_e.h_p / t) / sum_a exp(h_e.h_a / t) )`` on L2-normalised h.
    """
    features = F.normalize(embeddings, dim=-1)
    logits = torch.einsum("bd,bkd->bk", features[:, 0], features[:, 1:]) / temperature
    log_prob = logits[:, :n_pos] - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_prob.mean(dim=1).mean()


class ContrastiveSampler:
    """CLEAN's per-EC anchor, positive and hard-negative sampling, vectorised over a batch.

    Ids are positions in the training set; ``store_rows`` maps them (and augmented copies) back to
    rows of the feature store.
    """

    def __init__(self, train_set: LabelledSet, augment: dict[int, np.ndarray], seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.store_rows = train_set.rows
        self.n = len(train_set.rows)
        # enzyme -> classes, padded
        self.n_classes = np.array([len(c) for c in train_set.classes])
        self.enzyme_classes = np.full((self.n, self.n_classes.max()), -1, dtype=np.int64)
        for i, c in enumerate(train_set.classes):
            self.enzyme_classes[i, : len(c)] = c
        # class -> enzymes, CSR
        members: list[list[int]] = [[] for _ in range(NUM_CLASSES)]
        for i, c in enumerate(train_set.classes):
            for k in c:
                members[k].append(i)
        self.count = np.array([len(m) for m in members])
        self.offset = np.concatenate(([0], np.cumsum(self.count)))
        self.members = np.fromiter((i for m in members for i in m), dtype=np.int64, count=self.offset[-1])
        self.present = np.flatnonzero(self.count > 0)
        # orphan-class enzymes -> store rows of their augmented copies
        self.augment = augment
        self.negatives: np.ndarray | None = None
        self.cumulative: np.ndarray | None = None

    def set_negatives(self, negatives: np.ndarray, weights: np.ndarray) -> None:
        self.negatives = negatives
        self.cumulative = np.cumsum(weights, axis=1)
        self.cumulative[:, -1] = 1.0

    def _member(self, classes: np.ndarray) -> np.ndarray:
        pick = (self.rng.random(classes.shape) * self.count[classes]).astype(np.int64)
        return self.members[self.offset[classes] + pick]

    def _class_of(self, enzymes: np.ndarray) -> np.ndarray:
        """One random class of each enzyme."""
        pick = (self.rng.random(enzymes.shape) * self.n_classes[enzymes]).astype(np.int64)
        return self.enzyme_classes[enzymes, pick]

    def batch(self, anchor_classes: np.ndarray, n_pos: int = 9, n_neg: int = 30) -> np.ndarray:
        anchors = self._member(anchor_classes)
        return np.concatenate(
            (self.store_rows[anchors][:, None], self.positives(anchors, n_pos), self.hard_negatives(anchors, n_neg)),
            axis=1,
        )

    def positives(self, anchors: np.ndarray, n_pos: int) -> np.ndarray:
        """Another enzyme of a random class of the anchor; for an orphan class, a mutated copy."""
        grid = np.repeat(anchors[:, None], n_pos, axis=1)
        classes = self._class_of(grid)
        chosen = self._member(classes)
        clash = (chosen == grid) & (self.count[classes] > 1)
        while clash.any():
            chosen[clash] = self._member(classes[clash])
            clash = (chosen == grid) & (self.count[classes] > 1)
        rows = self.store_rows[chosen]
        for b, p in zip(*np.nonzero(self.count[classes] == 1)):
            copies = self.augment.get(int(anchors[b]))
            if copies is not None and len(copies):
                rows[b, p] = copies[self.rng.integers(len(copies))]
            # otherwise the anchor itself; dropout makes the two views differ
        return rows

    def hard_negatives(self, anchors: np.ndarray, n_neg: int) -> np.ndarray:
        """An enzyme of an EC drawn from the nearest ECs to a random class of the anchor (1/d weights)."""
        assert self.negatives is not None, "call set_negatives() first"
        grid = np.repeat(anchors[:, None], n_neg, axis=1)
        own = self.enzyme_classes[anchors]  # (b, L)

        def draw(cells: np.ndarray) -> np.ndarray:
            source = self._class_of(grid[cells])
            u = self.rng.random(source.shape)
            column = (u[:, None] > self.cumulative[source]).sum(1)
            return self.negatives[source, np.minimum(column, self.negatives.shape[1] - 1)]

        cells = np.ones(grid.shape, dtype=bool)
        classes = np.zeros(grid.shape, dtype=np.int64)
        for _ in range(100):
            classes[cells] = draw(cells)
            cells = (classes[:, :, None] == own[:, None, :]).any(-1)
            if not cells.any():
                break
        while cells.any():  # a class whose every near neighbour belongs to the anchor too
            classes[cells] = self.rng.choice(self.present, size=int(cells.sum()))
            cells = (classes[:, :, None] == own[:, None, :]).any(-1)
        return self.store_rows[self._member(classes)]


@torch.no_grad()
def mine_negatives(model, store, train_set, args) -> tuple[np.ndarray, np.ndarray]:
    """The ``neg_knn`` nearest EC centroids of every EC and CLEAN's ``1/d`` sampling weights."""
    projection = project(model, store, train_set.rows, args.device, args.eval_batch_size, args.amp)
    centroids, present = ec_centroids(projection, train_set.classes)
    index = torch.nonzero(present).squeeze(1)
    distances = torch.cdist(centroids[index], centroids[index])
    distances.fill_diagonal_(float("inf"))
    knn = min(args.neg_knn, len(index) - 1)
    nearest_distance, nearest = distances.topk(knn, dim=1, largest=False)
    weights = 1.0 / nearest_distance.clamp_min(1e-6)
    weights = weights / weights.sum(1, keepdim=True)
    negatives = np.zeros((NUM_CLASSES, knn), dtype=np.int64)
    table = np.zeros((NUM_CLASSES, knn), dtype=np.float64)
    negatives[index.cpu().numpy()] = index[nearest].cpu().numpy()
    table[index.cpu().numpy()] = weights.double().cpu().numpy()
    return negatives, table


def train_contrastive(model, store, train_set, eval_sets, classes, augment, args, history) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = ContrastiveSampler(train_set, augment, args.seed)
    # Every store row (training or augmented) a batch can reference needs its labels for the BCE term.
    row_labels = [[] for _ in range(len(store))]
    for row, c in zip(train_set.rows, train_set.classes):
        row_labels[row] = c
    for position, copies in augment.items():
        for row in copies:
            row_labels[row] = train_set.classes[position]
    table = padded_labels(row_labels).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    device_type = args.device.split(":")[0]
    group = 1 + args.n_pos + args.n_neg
    print(
        f"contrastive training: {len(sampler.present)} EC anchors per epoch, {group} enzymes each, "
        f"{int((sampler.count == 1).sum())} orphan ECs, {len(augment)} with augmented copies",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 or (epoch - 1) % args.mine_every == 0:
            sampler.set_negatives(*mine_negatives(model, store, train_set, args))
        model.train()
        started = time.time()
        totals = {"loss": 0.0, "supcon": 0.0, "bce": 0.0}
        steps = 0
        order = sampler.rng.permutation(sampler.present)
        for start in range(0, len(order), args.batch_size):
            groups = sampler.batch(order[start : start + args.batch_size], args.n_pos, args.n_neg)
            flat = torch.from_numpy(groups.reshape(-1))
            d, m, e = fetch(store, flat, args.device, local_only=False)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.amp):
                representation, _ = model.backbone.encode(d, m, e)
                embeddings = model.projection(representation)
                logits = model.backbone.classify(representation) if args.bce_weight > 0 else None
            supcon = supcon_hard_loss(embeddings.float().view(len(groups), group, -1), args.temperature, args.n_pos)
            loss = supcon
            bce = torch.zeros((), device=args.device)
            if logits is not None:
                bce = loss_fn(logits.float(), multi_hot(table[flat.to(args.device)]))
                loss = supcon + args.bce_weight * bce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += loss.item()
            totals["supcon"] += supcon.item()
            totals["bce"] += bce.item()
            steps += 1
        record = {"epoch": epoch, **{k: v / steps for k, v in totals.items()}}
        record["seconds"] = round(time.time() - started, 1)
        if eval_sets and (epoch % args.eval_every == 0 or epoch == args.epochs):
            record["eval"] = evaluate(model, "milcl_plus", store, eval_sets, train_set, classes, args)
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs or "eval" in record:
            print(
                f"epoch {epoch:4d}/{args.epochs}  loss {record['loss']:.4f} (supcon {record['supcon']:.4f}, "
                f"bce {record['bce']:.6f})  {record['seconds']:.0f}s  " + format_metrics(record.get("eval", {})),
                flush=True,
            )
        if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
            save_contrastive(model, store, train_set, classes, args, epoch)


def save_contrastive(model, store, train_set, classes, args, epoch: int) -> None:
    """``model.pt`` in the payload ``models.load_milcl`` reads, and ``centroids.npy`` beside it."""
    projection = project(model, store, train_set.rows, args.device, args.eval_batch_size, args.amp)
    centroids, present = ec_centroids(projection, train_set.classes)
    if not bool(present.all()):
        print(f"[warn] {int((~present).sum())} ECs have no training enzyme; their centroids are placeholders")
    np.save(args.output_dir / "centroids.npy", centroids.cpu().numpy().astype(np.float32))
    payload = {
        "version": 1,
        "epoch": epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_sha256": class_digest(classes),
        "config": run_config(args),
    }
    torch.save(payload, args.output_dir / "model.pt")


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def class_digest(classes: np.ndarray) -> str:
    return hashlib.sha256("\n".join(classes).encode()).hexdigest()


def run_config(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def fold_split(n: int, num_folds: int, fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    held_out = np.zeros(n, dtype=bool)
    held_out[np.array_split(order, num_folds)[fold]] = True
    return np.flatnonzero(~held_out), np.flatnonzero(held_out)


def match_augmented(train_store: Features, train_set: LabelledSet, augmented: Features, offset: int):
    """Map training positions of orphan-class enzymes to the store rows of their ``<Entry>_<k>`` copies."""
    position_of = {train_store.entries[row]: position for position, row in enumerate(train_set.rows)}
    copies: dict[int, list[int]] = {}
    unmatched = 0
    for row, entry in enumerate(augmented.entries):
        base = entry.rsplit("_", 1)[0]
        if base not in position_of:
            unmatched += 1
            continue
        copies.setdefault(position_of[base], []).append(offset + row)
    if unmatched:
        print(f"[warn] {unmatched} augmented entries match no training enzyme and are ignored", flush=True)
    return {position: np.asarray(rows, dtype=np.int64) for position, rows in copies.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(PAPER_DEFAULTS), help="what to train")
    parser.add_argument("--npzdata", required=True, nargs="+", help="training archive(s) from process.py")
    parser.add_argument("--csvdata", required=True, help="training labels: 'Entry' and 'EC number' columns")
    parser.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"), help="EC vocabulary")
    parser.add_argument("--output-dir", help="where to write the checkpoint (default: ../runs/<model>)")
    parser.add_argument("--eval-set", nargs=2, action="append", default=[], metavar=("NPZ", "CSV"),
                        help="independent test set to score while training; repeatable")
    parser.add_argument("--num-folds", type=int, default=0, help="split the training data into this many folds")
    parser.add_argument("--fold", type=int, default=0, help="fold held out for validation when --num-folds > 1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="enzymes per step (anchors per step for milcl_plus)")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-device", help="where the feature tensors live (default: --device; 'cpu' saves memory)")
    parser.add_argument("--amp", action="store_true", help="bfloat16 autocast for the forward pass")
    parser.add_argument("--eval-every", type=int, help="epochs between evaluations (default: 5, or 50 for milcl_plus)")
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--log-every", type=int, help="epochs between log lines (default: 1, or 10 for milcl_plus)")
    parser.add_argument("--no-cache", action="store_true", help="do not read or write <archive>.compact.npz")

    contrastive = parser.add_argument_group("milcl_plus")
    contrastive.add_argument("--backbone", help="backbone.pth from a milcl_backbone run (required)")
    contrastive.add_argument("--augment-npz", nargs="+", default=[], help="features of orphans.py mutants")
    contrastive.add_argument("--n-pos", type=int, default=9, help="positives per anchor")
    contrastive.add_argument("--n-neg", type=int, default=30, help="hard negatives per anchor")
    contrastive.add_argument("--neg-knn", type=int, default=100, help="nearest ECs negatives are drawn from")
    contrastive.add_argument("--mine-every", type=int, default=100, help="epochs between hard-negative mining")
    contrastive.add_argument("--temperature", type=float, default=0.1, help="SupCon temperature")
    contrastive.add_argument("--bce-weight", type=float, default=1.0, help="weight of the joint classifier BCE term")
    contrastive.add_argument("--save-every", type=int, default=0, help="also checkpoint every N epochs")
    args = parser.parse_args()

    for key, value in PAPER_DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    contrastive_run = args.model == "milcl_plus"
    args.eval_every = args.eval_every or (50 if contrastive_run else 5)
    args.log_every = args.log_every or (10 if contrastive_run else 1)
    args.output_dir = Path(args.output_dir or PACKAGE_ROOT / "runs" / args.model).expanduser().resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {args.device} requested but CUDA is unavailable; using CPU")
        args.device = "cpu"
    args.data_device = args.data_device or args.device
    if args.amp and not args.device.startswith("cuda"):
        args.amp = False
    if args.num_folds == 1 or args.num_folds < 0 or (args.num_folds and not 0 <= args.fold < args.num_folds):
        parser.error("--num-folds must be 0 or at least 2, and --fold in [0, num_folds)")
    if contrastive_run and not args.backbone:
        parser.error("milcl_plus starts from a BCE-trained backbone: train --model milcl_backbone, then pass --backbone")
    if not contrastive_run and args.augment_npz:
        parser.error("--augment-npz only applies to milcl_plus")
    return args


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    classes = np.load(Path(args.classes).expanduser().resolve(), allow_pickle=True).astype(str)
    if len(classes) != NUM_CLASSES:
        raise SystemExit(f"expected {NUM_CLASSES} EC classes, found {len(classes)}")
    class_index = {ec: i for i, ec in enumerate(classes)}
    use_cache = not args.no_cache

    # Training data, with the augmented orphan copies appended to the same store.
    store = load_archives([Path(p).expanduser().resolve() for p in args.npzdata], use_cache)
    labels = read_labels(Path(args.csvdata).expanduser().resolve())
    train_set = attach_labels(store, labels, class_index, "training set", training=True)
    eval_sets: dict[str, tuple[Features, LabelledSet]] = {}
    if args.num_folds:
        keep, held_out = fold_split(len(train_set.rows), args.num_folds, args.fold, args.seed)
        eval_sets[f"fold{args.fold}"] = (store, subset(train_set, held_out))
        train_set = subset(train_set, keep)
    augment: dict[int, np.ndarray] = {}
    if args.augment_npz:
        augmented = load_archives([Path(p).expanduser().resolve() for p in args.augment_npz], use_cache)
        augment = match_augmented(store, train_set, augmented, offset=len(store))
        store = Features(
            store.entries + augmented.entries,
            torch.cat((store.domains, augmented.domains)),
            torch.cat((store.masks, augmented.masks)),
            torch.cat((store.esm2, augmented.esm2)),
        )
    store = store.to(args.data_device)
    for npz, csv_path in args.eval_set:
        npz_path = Path(npz).expanduser().resolve()
        eval_store = load_archive(npz_path, use_cache).to(args.data_device)
        data = attach_labels(eval_store, read_labels(Path(csv_path).expanduser().resolve()), class_index,
                             npz_path.stem, training=False)
        eval_sets[npz_path.stem] = (eval_store, data)

    model = build_model(args.model).to(args.device)
    if args.model == "milcl_plus":
        state = torch.load(Path(args.backbone).expanduser().resolve(), map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(run_config(args), indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": args.model,
                "training_enzymes": len(train_set.rows),
                "eval_sets": {name: len(data.rows) for name, (_, data) in eval_sets.items()},
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": args.device,
                "parameters": sum(p.numel() for p in model.parameters()),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict] = []
    if args.model == "milcl_plus":
        train_contrastive(model, store, train_set, eval_sets, classes, augment, args, history)
        save_contrastive(model, store, train_set, classes, args, args.epochs)
        written = ["model.pt", "centroids.npy"]
    else:
        train_bce(model, args.model, store, train_set, eval_sets, classes, args, history)
        name = "backbone.pth" if args.model == "milcl_backbone" else "model.pth"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output_dir / name)
        written = [name]
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"wrote {', '.join(str(args.output_dir / name) for name in written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
