#!/usr/bin/env python
"""Per-protein pooled embeddings from protein language models (and ESMC SAE features).

Runs in the ``envs/esmc`` environment (Biohub ``esm`` + ``transformers``), not the repo ``.venv``
(fair-esm also installs a package named ``esm``)::

    CUDA_VISIBLE_DEVICES=1 envs/esmc/bin/python code/extract_plm.py --model esmc-600m \\
        --csvdata features/domains/all_union.csv --out-dir features/plm/esmc-600m

Every feature is a mean over the residues of the protein (special tokens excluded), except the
``*_max`` SAE features, a max over residues. Features written, by ``--model``:

- ``esmc-600m`` (bf16 backbone, fp32 SAE): ``last`` (final post-norm hidden state, 1152), ``mid`` (hidden state of layer 27,
  the SAE layer, 1152), ``sae_mean`` / ``sae_max`` (16384, the TopK-64 feature magnitudes of
  ``ESMC-600M-sae-layer27-k64-codebook16384``).
- ``esmc-6b``: ``last`` (2560), ``mid`` (layer 60), ``sae_mean`` / ``sae_max`` (layer-60 SAE, 16384).
- ``prott5``: ``last`` (ProtT5-XL-U50 encoder, 1024), run in half precision as recommended.
- ``prostt5``: ``last`` (ProstT5 encoder, 1024; ProtT5 fine-tuned to translate between sequence and
  Foldseek 3Di structure tokens, a structure-aware embedding), ``<AA2fold>`` prefix, half precision.
  Also writes ``3di.fasta``: the per-residue 3Di states predicted by ProstT5's CNN head. Weights are
  rebuilt from Foldseek's GGUF copy by ``prostt5_from_gguf.py`` (huggingface.co is unreachable here).
- ``esm2-3b``: ``last`` (layer 36, 2560), ``mid`` (layer 24), half precision.

Output: ``<out-dir>/entries.txt`` (csv order) and one ``<feature>.npy`` (n, dim) float16 each.
Work is checkpointed into ``<out-dir>/partial/chunk_*.npz``; rerunning resumes.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

HF = "biohub"
SPECS = {
    "esmc-600m": dict(repo=f"{HF}/ESMC-600M", sae=f"{HF}/ESMC-600M-sae-layer27-k64-codebook16384", mid=27),
    "esmc-6b": dict(repo=f"{HF}/ESMC-6B", sae=f"{HF}/ESMC-6B-sae-layer60-k64-codebook16384", mid=60),
    "prott5": dict(repo="Rostlab/prot_t5_xl_uniref50"),
    "prostt5": dict(repo=str(Path.home() / ".local/opt/prostt5_hf"), tokenizer="Rostlab/prot_t5_xl_uniref50"),
    "esm2-3b": dict(repo="facebook/esm2_t36_3B_UR50D", mid=24),
}


class Extractor:
    """Wraps one model: ``__call__(sequences) -> {feature: (batch, dim) float32 numpy}``."""

    def __init__(self, name: str, device: str) -> None:
        self.name, self.device, spec = name, device, SPECS[name]
        self.mid = spec.get("mid")
        if name.startswith("esmc"):
            from esm.models.esmc import EsmcModel, EsmcSaeModel, EsmcTokenizer

            dtype = torch.bfloat16  # 4x faster than fp32 on the A6000; pooled features agree closely
            self.model = EsmcModel.from_pretrained(spec["repo"], dtype=dtype).to(device).eval()
            self.tokenizer = EsmcTokenizer.from_pretrained(spec["repo"])
            sae = EsmcSaeModel.from_pretrained(spec["sae"], device=device)
            self.sae = sae.layers[str(self.mid)].float().eval()
        elif name in ("prott5", "prostt5"):
            from transformers import T5EncoderModel, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(spec.get("tokenizer", spec["repo"]), do_lower_case=False,
                                                         legacy=True)
            self.model = T5EncoderModel.from_pretrained(spec["repo"], torch_dtype=torch.float16).to(device).eval()
            if name == "prostt5":
                from prostt5_from_gguf import CNN

                self.cnn = CNN()
                self.cnn.load_state_dict(torch.load(Path(spec["repo"]) / "cnn.pt"))
                self.cnn = self.cnn.to(device).eval()
        else:
            from transformers import AutoTokenizer, EsmModel

            self.tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
            self.model = EsmModel.from_pretrained(spec["repo"], add_pooling_layer=False,
                                                  torch_dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def __call__(self, sequences: list[str]) -> dict[str, np.ndarray]:
        lengths = torch.tensor([len(s) for s in sequences], device=self.device)
        if self.name in ("prott5", "prostt5"):
            spaced = [" ".join(s.translate(str.maketrans("UZOB", "XXXX"))) for s in sequences]
            enc = self.tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt").to(self.device)
            if self.name == "prostt5":  # amino-acid input is announced by the <AA2fold> prefix token (id 149)
                from prostt5_from_gguf import AA2FOLD, THREE_DI

                ones = torch.ones_like(enc["input_ids"][:, :1])
                enc = {"input_ids": torch.cat([ones * AA2FOLD, enc["input_ids"]], 1),
                       "attention_mask": torch.cat([ones, enc["attention_mask"]], 1)}
            hidden = self.model(**enc).last_hidden_state.float()
            residues = self._residue_mask(lengths, hidden.shape[1], offset=int(self.name == "prostt5"))
            features = {"last": self._mean(hidden, residues)}
            if self.name == "prostt5":
                states = self.cnn(hidden[:, 1:]).argmax(-1).cpu().numpy()  # padding is cut per protein below
                features["3di"] = np.array(["".join(THREE_DI[i] for i in row[:n]) for row, n in
                                            zip(states, lengths.tolist())])
            return features
        enc = self.tokenizer(sequences, padding="longest", return_tensors="pt").to(self.device)
        residues = self._residue_mask(lengths, enc["input_ids"].shape[1], offset=1)  # <cls>/<bos> first
        if self.name == "esm2-3b":
            out = self.model(**enc, output_hidden_states=True)
            return {"last": self._mean(out.last_hidden_state.float(), residues),
                    "mid": self._mean(out.hidden_states[self.mid].float(), residues)}
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True,
                         compute_sae=False)
        mid = out.hidden_states[self.mid].float()
        features = {"last": self._mean(out.last_hidden_state.float(), residues), "mid": self._mean(mid, residues)}
        codes = self.sae(mid[residues]).feature_magnitudes  # (residues in batch, codebook)
        owner = residues.nonzero()[:, 0]
        total = torch.zeros(len(sequences), codes.shape[1], device=self.device).index_add_(0, owner, codes)
        peak = torch.zeros_like(total).index_reduce_(0, owner, codes, "amax", include_self=True)
        features["sae_mean"] = (total / lengths[:, None]).cpu().numpy()
        features["sae_max"] = peak.cpu().numpy()
        return features

    @staticmethod
    def _residue_mask(lengths: torch.Tensor, width: int, offset: int) -> torch.Tensor:
        position = torch.arange(width, device=lengths.device)[None]
        return (position >= offset) & (position < offset + lengths[:, None])

    @staticmethod
    def _mean(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        m = mask.to(hidden.dtype)[..., None]
        return ((hidden * m).sum(1) / m.sum(1)).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(SPECS))
    parser.add_argument("--csvdata", required=True, help="tab-separated Entry and Sequence columns")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-tokens", type=int, default=16000, help="tokens per batch, padding included")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--limit", type=int, help="first N proteins only (smoke test)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    partial = out_dir / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    with open(args.csvdata) as handle:
        rows = [(r["Entry"], r["Sequence"].strip()) for r in csv.DictReader(handle, delimiter="\t")]
    if args.limit:
        rows = rows[:args.limit]
    sequences = dict(rows)
    done: set[str] = set()
    for chunk in sorted(partial.glob("chunk_*.npz")):
        with np.load(chunk) as data:
            done.update(str(e) for e in data["entries"])
    todo = sorted((e for e, _ in rows if e not in done), key=lambda e: len(sequences[e]))
    print(f"{len(rows)} proteins, {len(done)} done, {len(todo)} to go", flush=True)

    if todo:
        extractor = Extractor(args.model, args.device)
        buffer: dict[str, list] = {"entries": []}
        started, finished = time.time(), 0

        def flush() -> None:
            if not buffer["entries"]:
                return
            index = len(list(partial.glob("chunk_*.npz")))
            tmp = partial / f".tmp_{index:05d}.npz"
            np.savez(tmp, entries=np.array(buffer["entries"]),
                     **{k: np.concatenate(v) if k == "3di" else np.concatenate(v).astype(np.float16)
                        for k, v in buffer.items() if k != "entries"})
            tmp.replace(partial / f"chunk_{index:05d}.npz")
            for value in buffer.values():
                value.clear()

        def run(batch: list[str]) -> None:
            nonlocal finished
            for key, value in extractor([sequences[e] for e in batch]).items():
                buffer.setdefault(key, []).append(value)
            buffer["entries"].extend(batch)
            finished += len(batch)
            if len(buffer["entries"]) >= args.checkpoint_every:
                flush()
                rate = finished / (time.time() - started)
                # Sorted by length, so the remaining work is longer than average: a lower bound.
                print(f"  {len(done) + finished}/{len(rows)}  {rate:.1f} seq/s", flush=True)

        batch: list[str] = []
        for entry in todo:
            longest = max(len(sequences[e]) for e in [*batch, entry]) + 2
            if batch and longest * (len(batch) + 1) > args.batch_tokens:
                run(batch)
                batch = []
            batch.append(entry)
        if batch:
            run(batch)
        flush()

    # Assemble in csv order, one array per feature, reading each chunk once.
    chunks = sorted(partial.glob("chunk_*.npz"))
    with np.load(chunks[0]) as data:
        keys = [k for k in data.files if k != "entries"]
    position = {e: i for i, (e, _) in enumerate(rows)}
    if "3di" in keys:  # variable-length strings: one FASTA instead of an array
        keys.remove("3di")
        states = {}
        for chunk in chunks:
            with np.load(chunk) as data:
                states.update(zip((str(e) for e in data["entries"]), (str(t) for t in data["3di"])))
        assert len(states) == len(rows)
        (out_dir / "3di.fasta").write_text("".join(f">{e}\n{states[e]}\n" for e, _ in rows))
    for key in keys:
        target = np.lib.format.open_memmap(out_dir / f".{key}.npy", mode="w+", dtype=np.float16,
                                           shape=(len(rows), np.load(chunks[0])[key].shape[1]))
        filled = 0
        for chunk in chunks:
            with np.load(chunk) as data:
                index = np.array([position[str(e)] for e in data["entries"]])
                target[index] = data[key]
                filled += len(index)
        assert filled == len(rows), (key, filled, len(rows))
        target.flush()
        del target
        os.replace(out_dir / f".{key}.npy", out_dir / f"{key}.npy")
    (out_dir / "entries.txt").write_text("\n".join(e for e, _ in rows) + "\n")
    print(f"wrote {out_dir}: {len(rows)} proteins, features {keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
