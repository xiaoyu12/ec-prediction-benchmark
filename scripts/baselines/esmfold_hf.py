"""ESMFold v1 in transformers' EsmForProteinFolding, loaded from Meta's original checkpoints
(esmfold_3B_v1.pt for the folding trunk, esm2_t36_3B_UR50D.pt for the language model), no openfold needed."""
import re
from pathlib import Path
import esm, torch
from transformers import EsmConfig, EsmForProteinFolding

def build(device="cuda"):
    fold = torch.load(Path.home() / "workspace/baselines/download/esmfold/esmfold_3B_v1.pt", map_location="cpu", weights_only=False)
    lm = torch.load(Path.home() / ".cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt", map_location="cpu", weights_only=False)["model"]
    t = fold["cfg"]["model"]["trunk"]
    vocab = list(esm.data.Alphabet.from_architecture("ESM-1b").all_toks)
    cfg = EsmConfig(vocab_size=33, hidden_size=2560, num_hidden_layers=36, num_attention_heads=40, intermediate_size=10240,
                    position_embedding_type="rotary", token_dropout=True, emb_layer_norm_before=False, layer_norm_eps=1e-5,
                    max_position_embeddings=1026, pad_token_id=1, mask_token_id=32, is_folding_model=True, vocab_list=vocab,
                    esmfold_config={"fp16_esm": True, "embed_aa": True, "bypass_lm": False, "lddt_head_hid_dim": 128,
                                    "trunk": {k: (dict(v) if hasattr(v, "keys") else v) for k, v in t.items() if k not in ("_name", "structure_module")}})  # HF structure-module defaults = ESMFold v1
    model = EsmForProteinFolding(cfg)
    sd = {k: v for k, v in fold["model"].items() if not k.startswith("positional_encoding")}
    for k, v in lm.items():
        k = k.replace("encoder.sentence_encoder.", "").replace("encoder.", "")
        m = re.match(r"layers\.(\d+)\.(.*)", k)
        if k == "embed_tokens.weight": sd["esm.embeddings.word_embeddings.weight"] = v
        elif k.startswith("emb_layer_norm_after"): sd["esm.encoder." + k] = v
        elif m:
            i, r = m.groups(); p = f"esm.encoder.layer.{i}."
            for a, b in [("self_attn.q_proj", "attention.self.query"), ("self_attn.k_proj", "attention.self.key"),
                         ("self_attn.v_proj", "attention.self.value"), ("self_attn.out_proj", "attention.output.dense"),
                         ("self_attn.rot_emb", "attention.self.rotary_embeddings"), ("self_attn_layer_norm", "attention.LayerNorm"),
                         ("fc1", "intermediate.dense"), ("fc2", "output.dense"), ("final_layer_norm", "LayerNorm")]:
                if r.startswith(a): sd[p + r.replace(a, b, 1)] = v; break
    res = model.load_state_dict(sd, strict=False)
    return model, res

if __name__ == "__main__":
    m, res = build("cpu")
    print("missing", [k for k in res.missing_keys][:40], len(res.missing_keys))
    print("unexpected", res.unexpected_keys[:40], len(res.unexpected_keys))
