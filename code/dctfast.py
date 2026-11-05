#!/usr/bin/env python
"""Fast, resumable DCTdomain fingerprinting for large training sets such as split100.

    python dctfast.py --csvdata ../dataset/split100.csv --output ../features/split100-dct.npz
    python process.py --csvdata shard.csv --dct-npz ../features/split100-dct.npz

It computes the same fingerprints as DCTdomain's ``make_db.py``, which ``process.py`` runs, by
calling DCTdomain's own code: ESM-2 150M embeddings of layers 15 and 21 plus the contact map
(``embedding.Batch`` / ``Embedding.embed_seq``, splitting sequences longer than ``--maxlen`` into
overlapping windows exactly as DCTdomain does), then ``Fingerprint.reccut(2.6)`` and
``Fingerprint.quantize([3, 80, 3, 80])``. The output is the ``*-dct.npz`` layout ``make_db.py``
writes (``sid``, ``idx``, ``dom``, ``dct``), which ``process.py --dct-npz`` reads.

What changes is orchestration only:

* GPU batches are filled up to a ``batch * length^2`` budget instead of 500 residues in total
  (ESM's contact head holds every layer's attention maps, so memory grows with L^2);
* device-to-host copies go through one pinned buffer (see ``HostBuffer``);
* RecCut and quantisation run in one persistent process pool instead of a new pool per 24 proteins;
* finished proteins are checkpointed every ``--checkpoint-every`` sequences, so a crash -- on WSL2
  the CUDA driver intermittently fails device-to-host copies with a sticky "out of memory" -- loses
  at most one chunk. Rerun the same command to resume.

Sequences shorter than five residues are skipped: DCTdomain cannot quantise them either.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DCTDOMAIN_SRC = Path(__file__).resolve().parent.parent / "DCTdomain" / "src"
sys.path.insert(0, str(DCTDOMAIN_SRC))

LAYERS = [15, 21]
OVERLAP = 200  # DCTdomain's window overlap for sequences longer than maxlen
MIN_LENGTH = 5


def _init_worker(workdir: str) -> None:
    os.chdir(workdir)  # RecCut reads and writes <pid>.ce in the working directory


def _fingerprint(item):
    """RecCut + iDCT quantisation of one protein, in a pool worker."""
    from fingerprint import Fingerprint

    pid, seq, embed, contacts = item
    fp = Fingerprint(pid=pid, seq=seq, embed=embed, contacts=contacts)
    fp.reccut(2.6)
    fp.quantize([3, 80, 3, 80])
    return pid, list(fp.domains), np.array([fp.quants[d] for d in fp.domains], dtype=np.int8)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from process import read_sequences

    return read_sequences(path)


def batches(pairs, maxlen: int, budget: int, max_batch: int):
    """Long sequences alone (DCTdomain windows them); short ones grouped by ``batch * L^2``."""
    batch: list[tuple[str, str]] = []
    for pid, seq in pairs:
        if len(seq) > maxlen:
            yield [(pid, seq)]
            continue
        longest = max([len(seq)] + [len(s) for _, s in batch]) + 2
        if batch and (longest * longest * (len(batch) + 1) > budget or len(batch) >= max_batch):
            yield batch
            batch = []
        batch.append((pid, seq))
    if batch:
        yield batch


class HostBuffer:
    """Device-to-host copies through one reusable pinned buffer.

    On WSL2 the CUDA driver intermittently fails pageable device-to-host copies with a sticky
    CUDA_ERROR_OUT_OF_MEMORY; DCTdomain makes several small pageable ``.cpu()`` copies per protein.
    Copying each batch output once into pinned memory and slicing on the host gives the same
    numbers through the path that does not fail.
    """

    def __init__(self) -> None:
        self.buffer = None

    def get(self, tensor):
        import torch

        tensor = tensor.detach().float().contiguous()
        if self.buffer is None or self.buffer.numel() < tensor.numel():
            self.buffer = torch.empty(max(tensor.numel(), 1 << 26), dtype=torch.float32, pin_memory=True)
        view = self.buffer[: tensor.numel()].view(tensor.shape)
        view.copy_(tensor)
        return view.numpy().copy()


def esm_forward(model, device, batch):
    import torch

    _, _, tokens = model.esm_tokenizer(batch)
    lengths = (tokens != model.alphabet.padding_idx).sum(1)
    with torch.no_grad():
        out = model.esm_encoder(tokens.to(device), repr_layers=LAYERS, return_contacts=True)
    return out, lengths


def embed(batch, model, device, maxlen, host: HostBuffer):
    """DCTdomain's embeddings and contact maps (``Batch.embed_parallel`` / ``Embedding.embed_seq``)."""
    if len(batch) > 1 or len(batch[0][1]) <= maxlen:
        # embed_parallel: one padded forward pass, each protein sliced to its own length. A single
        # short protein gives the same tensors as embed_single, which runs it unpadded.
        out, lengths = esm_forward(model, device, batch)
        contacts = host.get(out["contacts"])
        layers = {layer: host.get(out["representations"][layer]) for layer in LAYERS}
        items = []
        for i, (pid, seq) in enumerate(batch):
            n = int(lengths[i]) - 2
            embed_ = {layer: layers[layer][i, 1 : n + 1] for layer in LAYERS}
            items.append((pid, seq, embed_, contacts[i, :n, :n]))
        return items

    # embed_seq for one long protein: overlapping windows, averaged embeddings, merged contacts.
    from embedding import Embedding

    pid, seq = batch[0]
    emb = Embedding(pid=pid, seq=seq)
    edata: dict = {}
    for i, window in enumerate(emb.split_seq(maxlen, OVERLAP)):
        out, _ = esm_forward(model, device, [(pid, window)])
        embs = {layer: out["representations"][layer][0][1:-1] for layer in LAYERS}
        ct = out["contacts"][0]
        if not edata:
            edata = embs
            edata["ct"] = ct
            continue
        for layer, e in embs.items():
            edata[layer][-OVERLAP:] = (edata[layer][-OVERLAP:] + e[:OVERLAP]) / 2
            edata[layer] = torch_cat(edata[layer], e[OVERLAP:])
        edata["ct"] = emb.combine_contacts(edata["ct"], ct, maxlen - OVERLAP, i)
    return [(pid, seq, {layer: host.get(edata[layer]) for layer in LAYERS}, host.get(edata["ct"]))]


def torch_cat(a, b):
    import torch

    return torch.cat((a, b), axis=0)


def load_done(partial: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    done = {}
    for chunk in sorted(partial.glob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=True) as data:
            for pid, doms, dct in zip(data["sid"], data["dom"], data["dct"]):
                done[str(pid)] = (list(doms), dct)
    return done


def save_chunk(partial: Path, results: list) -> None:
    index = len(list(partial.glob("chunk_*.npz")))
    target = partial / f"chunk_{index:05d}.npz"
    sid = np.array([r[0] for r in results])
    dom = np.empty(len(results), dtype=object)
    dom[:] = [r[1] for r in results]
    dct = np.empty(len(results), dtype=object)
    dct[:] = [r[2] for r in results]
    temporary = partial / f".tmp_{index:05d}.npz"
    np.savez(temporary, sid=sid, dom=dom, dct=dct)
    temporary.replace(target)


def write_output(path: Path, pairs, done) -> None:
    """``make_db.py``'s ``*-dct.npz`` layout: one ``idx`` offset per protein into ``dom``/``dct``."""
    sids, idxs, doms, fps = [], [], [], []
    offset = 0
    for pid, _ in pairs:
        if pid not in done:
            continue
        domains, quants = done[pid]
        sids.append(pid)
        idxs.append(offset)
        doms.extend(domains)
        fps.extend(quants)
        offset += len(domains)
    idxs.append(offset)
    np.savez(path, sid=np.array(sids), idx=np.array(idxs), dom=np.array(doms), dct=np.array(fps, dtype=np.int8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvdata", required=True, help="table with 'Entry' and 'Sequence' columns")
    parser.add_argument("--output", required=True, help="the *-dct.npz to write")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpus", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--maxlen", type=int, default=500, help="DCTdomain window length (keep 500)")
    parser.add_argument("--budget", type=int, default=250_000, help="max batch * (length+2)^2 per GPU batch")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--workdir", default="/dev/shm", help="scratch for RecCut's per-protein files")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(Path(args.csvdata).expanduser().resolve())
    skipped = [pid for pid, seq in pairs if len(seq) < MIN_LENGTH]
    pairs = [(pid, seq) for pid, seq in pairs if len(seq) >= MIN_LENGTH]
    done = load_done(partial)
    todo = sorted((p for p in pairs if p[0] not in done), key=lambda p: len(p[1]))
    print(
        f"{len(pairs)} sequences ({len(skipped)} shorter than {MIN_LENGTH} skipped), "
        f"{len(done)} already done, {len(todo)} to go",
        flush=True,
    )

    if todo:
        workdir = tempfile.mkdtemp(prefix="dctfast-", dir=args.workdir)
        # Start the pool before CUDA is initialised; workers never touch the GPU.
        pool = mp.get_context("spawn").Pool(args.cpus, initializer=_init_worker, initargs=(workdir,))
        import torch
        from embedding import Model

        device = torch.device(args.device)
        model = Model()
        model.to_device(device)

        host = HostBuffer()
        pending, results = [], []
        started, finished = time.time(), 0
        max_pending = args.cpus * 4

        def collect(block: bool) -> None:
            nonlocal finished
            while pending and (block or pending[0].ready() or len(pending) > max_pending):
                results.append(pending.pop(0).get())
                finished += 1
                if len(results) >= args.checkpoint_every:
                    save_chunk(partial, results)
                    results.clear()
                    rate = finished / (time.time() - started)
                    print(
                        f"  {len(done) + finished}/{len(pairs)}  {rate:.1f} seq/s  "
                        f"eta {(len(todo) - finished) / rate / 3600:.1f} h",
                        flush=True,
                    )

        try:
            for batch in batches(todo, args.maxlen, args.budget, args.max_batch):
                for item in embed(batch, model, device, args.maxlen, host):
                    pending.append(pool.apply_async(_fingerprint, (item,)))
                collect(block=False)
            collect(block=True)
        finally:
            # Keep whatever finished, even when the GPU fails mid-run; a rerun resumes after it.
            if results:
                save_chunk(partial, results)
            pool.terminate()
            pool.join()
        done = load_done(partial)

    write_output(output, pairs, done)
    print(f"wrote {output}: {len(done)} proteins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
