"""Predict structures with ESMFold v1 for queries without an AFDB structure; writes AF-<id>-F1-model_v6.pdb-named files
into <out_dir> so cleancontact_embed.py can pick them up (prefix 'ESMF-' instead of 'AF-' to keep them distinct).
    python esmfold_predict.py <ids.txt> <out_dir>
"""
import csv, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from esmfold_hf import build
ids = Path(sys.argv[1]).read_text().split(); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
seqs = {}
for t in ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]:
    for r in csv.DictReader(open(Path(__file__).resolve().parents[2] / "dataset" / f"{t}.csv"), delimiter="\t"):
        seqs.setdefault(r["Entry"], r["Sequence"])
model, _ = build()
model = model.eval().cuda(); model.esm = model.esm.half()
model.trunk.set_chunk_size(64)
log = open(out / "plddt.tsv", "a")
for k in sorted(ids, key=lambda k: len(seqs[k])):
    f = out / f"ESMF-{k}.pdb"
    if f.exists():
        continue
    with torch.no_grad():
        o = model.infer([seqs[k]])
    f.write_text(model.output_to_pdb(o)[0])
    pl = float(o["plddt"][0].float().mean())
    log.write(f"{k}\t{len(seqs[k])}\t{pl:.3f}\n"); log.flush()
    print(k, len(seqs[k]), pl, flush=True)
    del o; torch.cuda.empty_cache()
