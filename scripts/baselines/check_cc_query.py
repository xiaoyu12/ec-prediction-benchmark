"""Compare our CLEAN-Contact query representations with the released precomputed ones (New-392 and the Price entries in esm_data.zip)."""
import sys, zipfile
from pathlib import Path
import torch
B = Path.home() / "workspace/baselines"
src = dict(l.split("\t") for l in (B / "cc_query/source.tsv").read_text().split("\n") if l)
z = zipfile.ZipFile(B / "download/cleancontact/esm_data.zip")
seq, stru = [], []
for k, how in src.items():
    n = f"esm_data/{k}.pt"
    try: z.getinfo(n)
    except KeyError: continue
    ref = torch.load(z.open(n)); ours = torch.load(B / f"cc_query/{k}.pt")
    cs = torch.nn.functional.cosine_similarity
    seq.append(float(cs(ref[:2560], ours[:2560], 0))); stru.append((float(cs(ref[2560:], ours[2560:], 0)), how))
print("n", len(seq), "seq cos min/median", min(seq), sorted(seq)[len(seq)//2])
for h in ("afdb", "esmfold", "esm2_contacts"):
    v = sorted(c for c, w in stru if w == h)
    if v: print(h, len(v), "struct cos min/p10/median", v[0], v[len(v)//10], v[len(v)//2])
