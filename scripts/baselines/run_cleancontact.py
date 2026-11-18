"""CLEAN-Contact (released 'addition' model, LayerNormNet1(512, 256)) max-separation inference on the six test sets.

Same computation as CLEAN-Contact's infer_maxsep, but the split100_reduced training representations are read
straight from the released esm_data.zip (no 238k-file unzip) and their model embeddings are cached.
    python run_cleancontact.py <query_emb_dir> <out_dir>
"""
import csv, sys, zipfile
from pathlib import Path
import pandas as pd, torch
B = Path.home() / "workspace/baselines"; CC = B / "CLEAN-Contact"
sys.path.insert(0, str(CC))
from src.CLEAN.model import LayerNormNet1
from src.CLEAN.utils import get_ec_id_dict, seed_everything
from src.CLEAN.distance_map import get_dist_map_test
from src.CLEAN.evaluate import write_max_sep_choices

TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
D = Path(__file__).resolve().parents[2] / "dataset"
qdir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dev, dt = torch.device("cuda:0"), torch.float32
model = LayerNormNet1(512, 256, dev, dt)
model.load_state_dict(torch.load(CC / "data/model/split100_reduced_resnet50_esm2_2560_addition_triplet.pth", map_location=dev))
model.eval()
_, ec_id_train = get_ec_id_dict(str(CC / "data/split100_reduced.csv"))
cache = B / "cleancontact_train_model_emb.pt"
if cache.exists():
    emb_train = torch.load(cache, map_location=dev)
else:
    order = [i for ec in ec_id_train for i in ec_id_train[ec]]  # esm_embedding's order
    need = set(order); raw = {}
    with zipfile.ZipFile(B / "download/cleancontact/esm_data.zip") as z:
        for n in z.namelist():
            k = n.rsplit("/", 1)[-1][:-3]
            if n.endswith(".pt") and k in need:
                with z.open(n) as h:
                    raw[k] = torch.load(h)
    print("loaded", len(raw), "of", len(need), flush=True)
    with torch.no_grad():
        emb_train = torch.cat([model(torch.stack([raw[i] for i in order[s:s + 8192]]).to(dev, dt))
                               for s in range(0, len(order), 8192)])
    torch.save(emb_train.cpu(), cache)
with torch.no_grad():
    for t in TESTS:
        id_ec_test, _ = get_ec_id_dict(str(D / f"{t}.csv"))
        emb_test = model(torch.stack([torch.load(qdir / f"{i}.pt") for i in id_ec_test]).to(dev, dt))
        eval_df = pd.DataFrame.from_dict(get_dist_map_test(emb_train, emb_test, ec_id_train, id_ec_test, dev, dt))
        seed_everything()
        write_max_sep_choices(eval_df, str(out / t))
        with open(out / f"{t}_maxsep.csv") as h, open(out / f"{t}.csv", "w", newline="") as o:
            w = csv.writer(o); w.writerow(["Entry", "EC number"])
            for row in csv.reader(h):
                w.writerow([row[0], ";".join(c.split("/")[0].replace("EC:", "") for c in row[1:])])
        print(t, "done", flush=True)
