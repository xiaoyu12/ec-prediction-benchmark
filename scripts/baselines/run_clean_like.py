"""Run CLEAN-family max-separation inference (CLEAN or EnzHier) on the six test sets.

    python run_clean_like.py clean|enzhier <out_dir>
Writes <out_dir>/<test>_maxsep.csv (tool format) and <out_dir>/<test>.csv (Entry,EC number; ';'-joined).
"""
import csv, os, shutil, sys
from pathlib import Path

TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
DATA = Path(__file__).resolve().parents[2] / "dataset"
B = Path.home() / "workspace/baselines"
tool, out = sys.argv[1], Path(sys.argv[2]).resolve()
app = B / ("CLEAN/app" if tool == "clean" else "EnzHier")
os.chdir(app)
sys.path.insert(0, str(app / "src") if tool == "clean" else str(app))
esm_dir = app / "data/esm_data"
esm_dir.mkdir(parents=True, exist_ok=True)
for f in (B / "esm1b_data").glob("*.pt"):  # test embeddings; split100 ones are in place already (EnzHier)
    t = esm_dir / f.name
    if not t.exists():
        t.symlink_to(f)
if tool == "clean":
    from CLEAN.infer import infer_maxsep
    run = lambda t: infer_maxsep("split100", t, report_metrics=False, pretrained=True)
else:
    sys.argv = sys.argv[:1]
    import importlib.util
    spec = importlib.util.spec_from_file_location("ei", app / "EnzHier_infer.py"); ei = importlib.util.module_from_spec(spec); spec.loader.exec_module(ei)
    run = lambda t: ei.infer_maxsep2("split100", t, report_metrics=False, pretrained=False, model_name="split100_triplet_withEC_7000")
    if not (app / "data/split100.csv").exists():
        shutil.copy(DATA / "split100.csv", app / "data/split100.csv")
out.mkdir(parents=True, exist_ok=True)
for t in TESTS:
    name = f"bench_{t}"
    shutil.copy(DATA / f"{t}.csv", app / f"data/{name}.csv")
    run(name)
    shutil.copy(app / f"results/{name}_maxsep.csv", out / f"{t}_maxsep.csv")
    with open(out / f"{t}_maxsep.csv") as h, open(out / f"{t}.csv", "w", newline="") as o:
        w = csv.writer(o); w.writerow(["Entry", "EC number"])
        for row in csv.reader(h):
            w.writerow([row[0], ";".join(c.split("/")[0].replace("EC:", "") for c in row[1:])])
    print(t, "done", flush=True)
