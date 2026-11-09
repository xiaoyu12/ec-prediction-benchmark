#!/usr/bin/env python
"""Rebuild ProstT5 (encoder + 3Di CNN head) as a Hugging Face model from Foldseek's GGUF weights.

    envs/esmc/bin/python code/prostt5_from_gguf.py \\
        --gguf ~/.local/opt/foldseek_db/prostt5_weights/prostt5-f16.gguf --out ~/.local/opt/prostt5_hf

``foldseek databases ProstT5`` serves the ProstT5 weights (Heinzinger et al. 2024) from Foldseek's own
mirror as one GGUF file: the T5 encoder (ProtT5-XL shape: 24 blocks, d_model 1024, d_ff 16384, 32
heads of 128, ReLU feed-forward, 150-token vocabulary) plus the two-layer CNN that predicts 3Di
states per residue. We use it because huggingface.co (``Rostlab/ProstT5``) is unreachable from this
host. The encoder is written as a ``T5EncoderModel`` directory (config from the cached ProtT5-XL
config, vocab 150) and the CNN as ``cnn.pt``. Token ids match ProtT5's (``<pad>``=0, ``</s>``=1,
amino acids 3-27); ``<AA2fold>`` is 149.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

AA2FOLD = 149
THREE_DI = "ACDEFGHIKLMNPQRSTVWY"  # CNN class order


class CNN(torch.nn.Module):
    """ProstT5's 3Di head: Conv2d(1024, 32, (7, 1)) - ReLU - Conv2d(32, 20, (7, 1)) over residues."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Conv2d(1024, 32, (7, 1), padding=(3, 0)), torch.nn.ReLU(),
                                              torch.nn.Dropout(0.0), torch.nn.Conv2d(32, 20, (7, 1), padding=(3, 0)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # (B, L, 1024) -> (B, L, 20)
        return self.classifier(hidden.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from transformers import T5Config, T5EncoderModel

    reader = GGUFReader(str(Path(args.gguf).expanduser()))
    w = {t.name: torch.from_numpy(np.array(t.data)) for t in reader.tensors}
    config = T5Config.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    config.vocab_size = 150
    model = T5EncoderModel(config)
    state = {"shared.weight": w["token_embd.weight"], "encoder.final_layer_norm.weight": w["enc.output_norm.weight"],
             "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": w["enc.blk.0.attn_rel_b.weight"]}
    for i in range(config.num_layers):
        g, h = f"enc.blk.{i}.", f"encoder.block.{i}.layer."
        state.update({h + "0.SelfAttention.q.weight": w[g + "attn_q.weight"],
                      h + "0.SelfAttention.k.weight": w[g + "attn_k.weight"],
                      h + "0.SelfAttention.v.weight": w[g + "attn_v.weight"],
                      h + "0.SelfAttention.o.weight": w[g + "attn_o.weight"],
                      h + "0.layer_norm.weight": w[g + "attn_norm.weight"],
                      h + "1.DenseReluDense.wi.weight": w[g + "ffn_up.weight"],
                      h + "1.DenseReluDense.wo.weight": w[g + "ffn_down.weight"],
                      h + "1.layer_norm.weight": w[g + "ffn_norm.weight"]})
    state["encoder.embed_tokens.weight"] = state["shared.weight"]
    expected = model.state_dict()
    for k, v in state.items():
        assert tuple(v.shape) == tuple(expected[k].shape), (k, v.shape, expected[k].shape)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    out = Path(args.out).expanduser()
    model.half().save_pretrained(out)

    cnn = CNN()
    cnn.load_state_dict({"classifier.0.weight": w["classifier.0.weight"].reshape(32, 1024, 7, 1).float(),
                         "classifier.0.bias": w["classifier.0.bias"].float(),
                         "classifier.3.weight": w["classifier.3.weight"].reshape(20, 32, 7, 1).float(),
                         "classifier.3.bias": w["classifier.3.bias"].float()})
    torch.save(cnn.state_dict(), out / "cnn.pt")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
