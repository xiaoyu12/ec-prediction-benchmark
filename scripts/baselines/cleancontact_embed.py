"""CLEAN-Contact query representations: ESM-2 3B mean (layer 36, 2560-d, residues truncated at 1022 as in
esm/scripts/extract.py) concatenated with ResNet50 (IMAGENET1K_V2, fc removed, 2048-d) of the C-beta < 8 A
contact map, exactly as extract_structure_representation.py / merge_sequence_structure_emb do.

    python cleancontact_embed.py <ids.txt> <pdb_dir> <out_dir> [--esmfold DIR] [--fallback esm2]

Structure source per query: AF-<id>-F1-model_v6.pdb in <pdb_dir> (local AFDB Swiss-Prot v6), else
ESMF-<id>.pdb in the --esmfold dir (ESMFold v1 prediction). Queries without either get, with --fallback esm2, the ESM-2 3B predicted contact map (probability > 0.5; ESM's contact head is
trained on the same C-beta < 8 A definition). Writes <out_dir>/<id>.pt (4608-d) and source.tsv.
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision, esm
from biotite.structure.io.pdb import PDBFile, get_structure
sys.path.insert(0, str(Path.home() / "workspace/baselines/CLEAN-Contact"))
from extract_structure_representation import contacts_from_pdb

ids = Path(sys.argv[1]).read_text().split(); pdb_dir = Path(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
fallback = "--fallback" in sys.argv
esmf = Path(sys.argv[sys.argv.index("--esmfold") + 1]) if "--esmfold" in sys.argv else None
seqs = {}
for t in ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]:
    for r in csv.DictReader(open(Path(__file__).resolve().parents[2] / "dataset" / f"{t}.csv"), delimiter="\t"):
        seqs.setdefault(r["Entry"], r["Sequence"])
dev = "cuda"
lm, alphabet = esm.pretrained.esm2_t36_3B_UR50D(); lm = lm.eval().to(dev); conv = alphabet.get_batch_converter()
cnn = torchvision.models.get_model("resnet50", weights="IMAGENET1K_V2"); cnn.fc = nn.Identity(); cnn = cnn.eval().to(dev)
done = {l.split("\t")[0] for l in open(out / "source.tsv")} if (out / "source.tsv").exists() else set()
src = open(out / "source.tsv", "a")
with torch.no_grad():
    for k in ids:
        if k in done:
            continue
        s = seqs[k][:1022]
        pdb = pdb_dir / f"AF-{k}-F1-model_v6.pdb"
        if not pdb.exists() and esmf is not None and (esmf / f"ESMF-{k}.pdb").exists():
            pdb = esmf / f"ESMF-{k}.pdb"
        _, _, t = conv([(k, s)])
        o = lm(t.to(dev), repr_layers=[36], return_contacts=fallback and not pdb.exists())
        seq_emb = o["representations"][36][0, 1:len(s) + 1].mean(0).float().cpu()
        if pdb.exists():
            cmap = contacts_from_pdb(get_structure(PDBFile.read(pdb))[0]); how = "afdb" if pdb.name.startswith("AF-") else "esmfold"
        elif fallback:
            cmap = (o["contacts"][0].float().cpu().numpy() > 0.5).astype(np.int64); how = "esm2_contacts"
        else:
            src.write(f"{k}\tmissing\n"); continue
        img = torch.tensor(np.array([cmap, cmap, cmap])).unsqueeze(0).to(dev, dtype=torch.float32)
        stru = cnn(img).cpu().squeeze(0)
        torch.save(torch.cat([seq_emb, stru], 0), out / f"{k}.pt")
        src.write(f"{k}\t{how}\n"); src.flush()
        del o, img, stru; torch.cuda.empty_cache()
print("done")
