import sys, numpy as np, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # scripts/debug/ -> repo root
sys.path.insert(0, str(REPO / "code"))
from pathlib import Path
from train import load_archive, read_labels, attach_labels, padded_labels, multi_hot, fetch, classifier_logits, build_model
F = REPO / "features"
classes = np.load(REPO / "checkpoints/classes.npy", allow_pickle=True)
idx = {str(c): i for i, c in enumerate(classes)}
labels = read_labels(F / "split100_min5.csv")
store = load_archive(F / "shards/split100_03.npz")
ds = attach_labels(store, labels, idx, "s03", True)
rows = torch.from_numpy(ds.rows[:8192])
target = multi_hot(padded_labels(ds.classes[:8192]))
for name, ckpt in [("milbc", str(REPO / "checkpoints/milbc/model_v1.pth")), ("milbc", str(REPO / "checkpoints/retrained/milbc/model.pth"))]:
    m = build_model(name); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval().cuda()
    losses, hits = [], 0
    with torch.no_grad():
        for s in range(0, len(rows), 512):
            d, mk, e = fetch(store, rows[s:s+512], "cuda", True)
            lg = classifier_logits(m, name, d, mk, e).float().cpu()
            t = target[s:s+512, :lg.shape[1]].float()
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(lg, t).item())
            hits += (t.gather(1, lg.argmax(1, keepdim=True)) > 0).sum().item()
    print(ckpt, "train BCE %.2e  top1 %.3f" % (np.mean(losses), hits / len(rows)))
