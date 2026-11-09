#!/usr/bin/env python
"""Partial fine-tuning of ESMC-600M for EC prediction, warm-started from a frozen-feature head.

    CUDA_VISIBLE_DEVICES=0 envs/esmc/bin/python code/finetune_plm.py --protocol id50 \\
        --init-head runs/feat/id50_esmc600_last --output-dir runs/feat/id50_ft_esmc600

The lower ``--freeze`` transformer blocks of ESMC-600M stay frozen (run without gradients); the
upper blocks, the final norm and the head train. The head is ``train_feat.FeatureHead`` on the
mean-pooled final hidden state, standardized with the same training statistics as
``esmc-600m:last``, and is initialized from the ``train_feat.py`` run given as ``--init-head``
(trained on the frozen features), so step 0 already reproduces that model. Two learning rates:
``--lr-backbone`` (blocks, cosine-decayed) and ``--lr-head``. bf16 autocast, fp32 master weights,
activation checkpointing over the trainable blocks.

Same data, protocols, decision rule and outputs (``metrics.json``, ``logits.npz``,
``predictions.csv``) as ``train_feat.py``; evaluation after every epoch. Runs in ``envs/esmc``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from train import PACKAGE_ROOT, read_labels
from train_feat import FOLDS, TESTS, Bank, FeatureHead, labelled, multi_hot, score, summary, target_table, write_run


class Tuner(torch.nn.Module):
    def __init__(self, backbone, head: FeatureHead, mean: torch.Tensor, std: torch.Tensor, freeze: int) -> None:
        super().__init__()
        self.backbone, self.head, self.freeze = backbone, head, freeze
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def pooled(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pad = ids != self.backbone.config.pad_token_id
        mask = pad[:, None, None, :]
        stack = self.backbone.transformer
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                x = self.backbone.embed(ids)
                for block in stack.blocks[:self.freeze]:
                    x, _ = block(x, mask)
            x = x.detach()
            for block in stack.blocks[self.freeze:]:
                if self.training:
                    x = checkpoint(lambda h, b=block: b(h, mask)[0], x, use_reentrant=False)
                else:
                    x, _ = block(x, mask)
            x = stack.norm(x)
        position = torch.arange(ids.shape[1], device=ids.device)[None]
        residues = ((position >= 1) & (position <= lengths[:, None])).float()[..., None]
        return (x.float() * residues).sum(1) / residues.sum(1)

    def forward(self, ids, lengths):
        return self.head([(self.pooled(ids, lengths) - self.mean) / self.std])


def batches(entries: list[str], sequences: dict[str, str], budget: int, shuffle: bool, rng: np.random.Generator):
    """Length-bucketed batches under a token budget (padding included), in random bucket order."""
    order = sorted(range(len(entries)), key=lambda i: len(sequences[entries[i]]))
    if shuffle:  # jitter within similar lengths, then shuffle the batches
        order = sorted(order, key=lambda i: len(sequences[entries[i]]) * (1 + 0.1 * rng.random()))
    out, current, longest = [], [], 0
    for i in order:
        n = len(sequences[entries[i]]) + 2
        if current and max(longest, n) * (len(current) + 1) > budget:
            out.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        out.append(current)
    if shuffle:
        rng.shuffle(out)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=["id50", "full"], required=True)
    p.add_argument("--init-head", required=True, help="train_feat.py run on esmc-600m:last (same protocol)")
    p.add_argument("--freeze", type=int, default=24, help="frozen lower blocks (of 36)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--batch-tokens", type=int, default=12000)
    p.add_argument("--eval-tokens", type=int, default=32000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="steps per epoch cap (smoke tests)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    classes = np.load(PACKAGE_ROOT / "checkpoints" / "classes.npy", allow_pickle=True).astype(str)
    class_index = {ec: i for i, ec in enumerate(classes)}
    with open(PACKAGE_ROOT / "features" / "domains" / "all_union.csv") as handle:
        sequences = {r["Entry"]: r["Sequence"].strip() for r in csv.DictReader(handle, delimiter="\t")}

    # Same rows, labels and standardization as the train_feat.py run on esmc-600m:last.
    bank = Bank(["esmc-600m:last"], set(), args.device)
    train_rows, train_ecs, train_classes = labelled(bank, read_labels(PACKAGE_ROOT / "features" / "split100_min5.csv"),
                                                    class_index, training=True)
    eval_sets = {}
    if args.protocol == "id50":
        held = set((FOLDS / "id50_val.txt").read_text().split())
        dropped = set((FOLDS / "id50_exclude.txt").read_text().split())
        names = [bank.entries[r] for r in train_rows]
        val = [i for i, e in enumerate(names) if e in held]
        keep = [i for i, e in enumerate(names) if e not in held and e not in dropped]
        eval_sets["id50"] = (train_rows[val], [train_ecs[i] for i in val])
        train_rows, train_classes = train_rows[keep], [train_classes[i] for i in keep]
    for name in TESTS:
        rows, ecs, _ = labelled(bank, read_labels(PACKAGE_ROOT / "features" / "tests" / f"{name}.csv"), class_index, False)
        eval_sets[name] = (rows, ecs)
    bank.fit(torch.from_numpy(train_rows).to(args.device))
    mean, std = bank.mean[0], bank.std[0]
    del bank.blocks
    torch.cuda.empty_cache()

    init = Path(args.init_head).expanduser().resolve()
    head_config = json.loads((init / "config.json").read_text())
    if head_config["features"] != ["esmc-600m:last"] or head_config["protocol"] != args.protocol:
        raise SystemExit(f"{init}: expected an esmc-600m:last head trained with --protocol {args.protocol}")
    head = FeatureHead([1152], len(classes), head_config["arch"], head_config["proj"], head_config["hidden"],
                       head_config["layers"], head_config["dropout"], head_config["input_dropout"], 0.0)
    head.load_state_dict(torch.load(init / "model.pth"))

    from esm.models.esmc import EsmcModel, EsmcTokenizer

    backbone = EsmcModel.from_pretrained("biohub/ESMC-600M", dtype=torch.float32)
    tokenizer = EsmcTokenizer.from_pretrained("biohub/ESMC-600M")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = [q for block in backbone.transformer.blocks[args.freeze:] for q in block.parameters()]
    trainable += list(backbone.transformer.norm.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    model = Tuner(backbone, head, mean, std, args.freeze).to(args.device)

    train_entries = [bank.entries[r] for r in train_rows]
    table = target_table(train_classes, args.device)

    def encode(entries: list[str]):
        enc = tokenizer([sequences[e] for e in entries], padding="longest", return_tensors="pt")
        lengths = torch.tensor([len(sequences[e]) for e in entries])
        return enc["input_ids"].to(args.device), lengths.to(args.device)

    @torch.no_grad()
    def predict(rows: np.ndarray) -> np.ndarray:
        model.eval()
        entries = [bank.entries[r] for r in rows]
        logits = np.zeros((len(entries), len(classes)), dtype=np.float32)
        for batch in batches(entries, sequences, args.eval_tokens, False, rng):
            logits[batch] = model(*encode([entries[i] for i in batch])).float().cpu().numpy()
        return logits

    def evaluate() -> tuple[dict, dict]:
        logits = {n: predict(r) for n, (r, _) in eval_sets.items()}
        return {n: score(logits[n], eval_sets[n][1], classes) for n in logits}, logits

    optimizer = torch.optim.AdamW([{"params": trainable, "lr": args.lr_backbone, "weight_decay": 0.01},
                                   {"params": list(head.parameters()), "lr": args.lr_head, "weight_decay": 0.05}])
    steps_per_epoch = len(batches(train_entries, sequences, args.batch_tokens, True, np.random.default_rng(0)))
    total = steps_per_epoch * args.epochs
    warm = max(1, total // 50)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    config = dict(vars(args), features=["esmc-600m:finetune"], arch="ft-mlp", loss="bce",
                  hidden=head_config["hidden"], layers=head_config["layers"], training_enzymes=len(train_rows),
                  trainable=sum(q.numel() for q in trainable) + sum(q.numel() for q in head.parameters()))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)

    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        plan = batches(train_entries, sequences, args.batch_tokens, True, rng)[:args.max_steps or None]
        for b, batch in enumerate(plan):
            ids, lengths = encode([train_entries[i] for i in batch])
            target = multi_hot(table[torch.tensor(batch, device=args.device)], len(classes))
            loss = F.binary_cross_entropy_with_logits(model(ids, lengths).float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running += loss.item()
            if b % 500 == 0:
                print(f"  epoch {epoch} step {b}/{len(plan)}  loss {running / (b + 1):.6f}  "
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
        metrics, logits = evaluate()
        history.append({"epoch": epoch, "loss": running / len(plan), "minutes": (time.time() - started) / 60,
                        "eval": metrics})
        print(f"epoch {epoch}  loss {history[-1]['loss']:.6f}  {summary(metrics)}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        write_run(out, config, metrics, logits, eval_sets, bank, classes)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if "backbone.transformer.blocks" not in k
                or int(k.split(".")[3]) >= args.freeze}, out / "model.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
