#!/usr/bin/env python
"""Per-domain mean-pooled ESM2 embeddings: the domain features of the domain-attention model.

    python domain_esm.py --csvdata ../features/domains/all_union.csv \\
        --dct ../features/domains/all-dct.npz --output ../features/domains/domain_esm.npz

Domains are DCTdomain's (``dctfast.py`` / ``make_db.py`` output, the ``dom`` strings), chosen exactly
as MIL-BC's local tokens are: a protein with several domains uses those domains and not the
whole-protein entry DCTdomain appends; a single-domain protein uses its one domain; at most 16.
A domain string is 1-indexed ``beg-end`` segments joined by commas (discontinuous domains), and
segments past the sequence end are dropped, as DCTdomain does.

For each domain, ``r_m`` is the mean of the layer-33 residue representations of
``esm2_t33_650M_UR50D`` over the residues of the domain -- the model and layer of the global
embedding in ``process.py``, run on the full sequence (all split100 and test sequences fit in
ESM2's 1,022 residues), so a domain spanning the whole protein reproduces the global embedding.

``--dct`` may also be the ``<output>.partial`` directory of a ``dctfast.py`` run still in progress;
proteins already done are skipped, so rerunning picks up where it stopped. Output (``np.savez``):
``entries`` (n,), ``offsets`` (n + 1,) into ``domains`` (total, 1280) float16, and ``spans`` (total,)
the domain strings.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

MAX_DOMAINS = 16
ESM_LAYER = 33


def domain_lists(path: Path) -> dict[str, list[str]]:
    """``entry -> DCTdomain domain strings`` from a ``*-dct.npz`` or a ``.partial`` chunk directory."""
    out: dict[str, list[str]] = {}
    if path.is_dir():
        for chunk in sorted(path.glob("chunk_*.npz")):
            with np.load(chunk, allow_pickle=True) as data:
                for pid, doms in zip(data["sid"], data["dom"]):
                    out[str(pid)] = [str(d) for d in doms]
        return out
    with np.load(path, allow_pickle=True) as data:
        sid, idx, dom = data["sid"], data["idx"], data["dom"]
    for position, pid in enumerate(sid):
        out[str(pid)] = [str(d) for d in dom[idx[position]:idx[position + 1]]]
    return out


def model_domains(doms: list[str]) -> list[str]:
    """MIL-BC's choice of domains: drop the appended whole-protein entry of a multi-domain protein."""
    chosen = doms[:-1] if len(doms) > 1 else doms
    return chosen[:MAX_DOMAINS]


def residue_index(span: str, length: int) -> np.ndarray:
    """0-based residue positions of a 1-indexed ``beg-end[,beg-end...]`` domain string."""
    positions = []
    for segment in span.split(","):
        beg, end = (int(x) for x in segment.split("-"))
        if beg > length:
            continue
        positions.append(np.arange(beg - 1, min(end, length)))
    return np.concatenate(positions) if positions else np.zeros(0, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvdata", required=True, help="Entry and Sequence columns")
    parser.add_argument("--dct", required=True, help="dctfast.py output (*-dct.npz) or its .partial directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-tokens", type=int, default=12000, help="ESM2 tokens per batch, padding included")
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    with open(args.csvdata) as handle:
        sequences = {r["Entry"]: r["Sequence"].strip() for r in csv.DictReader(handle, delimiter="\t")}
    doms = domain_lists(Path(args.dct).expanduser().resolve())
    done = set()
    for chunk in sorted(partial.glob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=False) as data:
            done.update(str(e) for e in data["entries"])
    todo = sorted((e for e in doms if e in sequences and e not in done), key=lambda e: len(sequences[e]))
    print(f"{len(sequences)} sequences, {len(doms)} with domains, {len(done)} already pooled, {len(todo)} to go",
          flush=True)

    if todo:
        import esm

        torch.backends.cuda.matmul.allow_tf32 = False  # fp32, as process.py's global embedding
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(args.device)
        converter = alphabet.get_batch_converter()
        buffer = {"entries": [], "counts": [], "domains": [], "spans": []}
        started, finished = time.time(), 0

        def flush_chunk() -> None:
            if not buffer["entries"]:
                return
            index = len(list(partial.glob("chunk_*.npz")))
            temporary = partial / f".tmp_{index:05d}.npz"
            np.savez(temporary, entries=np.array(buffer["entries"]), counts=np.array(buffer["counts"]),
                     domains=np.concatenate(buffer["domains"]).astype(np.float16), spans=np.array(buffer["spans"]))
            temporary.replace(partial / f"chunk_{index:05d}.npz")
            for value in buffer.values():
                value.clear()

        def run(batch: list[str]) -> None:
            nonlocal finished
            _, _, tokens = converter([(e, sequences[e]) for e in batch])
            with torch.inference_mode():
                reps = model(tokens.to(args.device), repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
            for row, entry in enumerate(batch):
                seq_len = len(sequences[entry])
                residues = reps[row, 1:seq_len + 1].float()
                spans = [s for s in model_domains(doms[entry]) if len(residue_index(s, seq_len))]
                if not spans:  # every domain past the sequence end: fall back to the whole protein
                    spans = [f"1-{seq_len}"]
                pooled = torch.stack([residues[torch.from_numpy(residue_index(s, seq_len)).to(residues.device)].mean(0)
                                      for s in spans])
                buffer["entries"].append(entry)
                buffer["counts"].append(len(spans))
                buffer["domains"].append(pooled.cpu().numpy())
                buffer["spans"].extend(spans)
            finished += len(batch)
            if len(buffer["entries"]) >= args.checkpoint_every:
                flush_chunk()
                rate = finished / (time.time() - started)
                print(f"  {len(done) + finished}/{len(doms)}  {rate:.1f} seq/s  "
                      f"eta {(len(todo) - finished) / rate / 3600:.2f} h", flush=True)

        batch: list[str] = []
        for entry in todo:
            longest = max(len(sequences[e]) for e in [*batch, entry]) + 2
            if batch and longest * (len(batch) + 1) > args.batch_tokens:
                run(batch)
                batch = []
            batch.append(entry)
        if batch:
            run(batch)
        flush_chunk()

    # Assemble every chunk, in csv order, into the output archive. Each NpzFile key access re-reads
    # the whole array, so read every array of a chunk exactly once and keep only the concatenation
    # (indexing data["domains"] per protein once pinned a full copy per protein: >250 GB, OOM).
    names, counts, domains, spans = [], [], [], []
    for chunk in sorted(partial.glob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=False) as data:
            names.extend(str(e) for e in data["entries"])
            counts.append(data["counts"])
            domains.append(data["domains"])
            spans.append(data["spans"])
    counts = np.concatenate(counts)
    domains = np.concatenate(domains)
    spans = np.concatenate(spans)
    starts = np.concatenate(([0], np.cumsum(counts)))
    where = {e: i for i, e in enumerate(names)}
    order = [e for e in sequences if e in where]
    pick = np.array([where[e] for e in order], dtype=np.int64)
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in pick])
    out_counts = counts[pick]
    np.savez(output, entries=np.array(order), offsets=np.concatenate(([0], np.cumsum(out_counts))),
             domains=domains[rows], spans=spans[rows])
    counts = out_counts
    print(f"wrote {output}: {len(order)} proteins, {counts.sum()} domains ({counts.mean():.2f} each, max {counts.max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
