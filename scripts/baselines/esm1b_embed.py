"""ESM-1b mean embeddings (CLEAN format: {'mean_representations': {33: tensor}}) for the six test sets.

Residues beyond 1022 are truncated, as in CLEAN's esm/scripts/extract.py.
    python esm1b_embed.py <out_dir> <test.csv> ...
"""
import csv, sys
from pathlib import Path
import esm, torch

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
seqs = {}
for f in sys.argv[2:]:
    with open(f) as h:
        for r in csv.DictReader(h, delimiter="\t"):
            seqs.setdefault(r["Entry"], r["Sequence"])
todo = [(k, s) for k, s in seqs.items() if not (out / f"{k}.pt").exists()]
print(len(seqs), "sequences,", len(todo), "to embed", flush=True)
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.eval().cuda()
conv = alphabet.get_batch_converter()
todo.sort(key=lambda x: len(x[1]))
i = 0
with torch.no_grad():
    while i < len(todo):
        batch, toks = [], 0
        while i < len(todo) and (toks + min(len(todo[i][1]), 1022)) <= 8000:
            batch.append((todo[i][0], todo[i][1][:1022])); toks += min(len(todo[i][1]), 1022); i += 1
        if not batch:
            batch = [(todo[i][0], todo[i][1][:1022])]; i += 1
        _, _, t = conv(batch)
        rep = model(t.cuda(), repr_layers=[33])["representations"][33]
        for j, (k, s) in enumerate(batch):
            torch.save({"label": k, "mean_representations": {33: rep[j, 1:len(s) + 1].mean(0).clone().cpu()}}, out / f"{k}.pt")
        if i % 500 < len(batch):
            print(i, flush=True)
print("done", flush=True)
