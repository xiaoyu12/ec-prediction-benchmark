"""Split the released-vs-retrained F1 gap into ranking (top-1 / oracle-k) and calling (call_ec)."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from train import load_archive, read_labels, attach_labels, fetch, build_model, weighted_prf
from infer import call_ec
R = Path(__file__).resolve().parents[2]; F = R / 'features'
classes = np.load(R / 'checkpoints/classes.npy', allow_pickle=True).astype(str); idx = {c: i for i, c in enumerate(classes)}
P = {'rel': ('milbc', 'checkpoints/milbc/model_v1.pth'), 'ret': ('milbc', 'checkpoints/retrained/milbc/model.pth'),
     'relp': ('milbc_plus', 'checkpoints/milbc_plus/model_v1.pth'), 'retp': ('milbc_plus', 'checkpoints/retrained/milbc_plus/model.pth'),
     'esm': ('esm2_bc', 'checkpoints/retrained/esm2_bc/model.pth')}
models = {}
for k, (name, p) in P.items():
    m = build_model(name); m.load_state_dict(torch.load(R / p, map_location='cpu')); models[k] = (name, m.eval().cuda())

def f1(true, calls):
    return weighted_prf(true, [list(classes[c]) for c in calls])['f1']

rows = []
for n in ['new', 'price', 'uniprot-all', 'uniprot-multi', 'test70', 'test50']:
    st = load_archive(F / f'tests/{n}.npz'); ds = attach_labels(st, read_labels(F / f'tests/{n}.csv'), idx, n, False)
    d, mk, e = fetch(st, torch.from_numpy(ds.rows), 'cuda', True)
    in_vocab = np.array([len(c) > 0 for c in ds.classes])
    for k, (name, m) in models.items():
        with torch.no_grad(): lg = m(d, mk, None if name == 'milbc' else e)[0].float().cpu()
        order = lg.argsort(1, descending=True).numpy()
        top1 = order[:, 0]
        acc = np.mean([t in c for t, c in zip(top1, ds.classes)])
        topk = [order[i, :max(1, len(ds.true_ecs[i]))] for i in range(len(order))]
        called = call_ec(lg.numpy())
        ncall = called.sum(1)
        rows.append((n, k, acc, f1(ds.true_ecs, [[t] for t in top1]), f1(ds.true_ecs, topk), f1(ds.true_ecs, [np.flatnonzero(r) for r in called]),
                     ncall.mean(), (lg.numpy() >= 0).any(1).mean()))
    print(f'{n}: {len(ds.rows)} enzymes, {in_vocab.mean():.3f} with an in-vocabulary label, {np.mean([len(t) for t in ds.true_ecs]):.2f} labels each', flush=True)
print(f"\n{'set':14s}{'model':6s}{'top1 acc':>9s}{'F1 top1':>9s}{'F1 oracle-k':>12s}{'F1 call_ec':>11s}{'#calls':>8s}{'any>=0':>8s}")
for r in rows:
    print(f'{r[0]:14s}{r[1]:6s}{r[2]:9.3f}{r[3]:9.3f}{r[4]:12.3f}{r[5]:11.3f}{r[6]:8.2f}{r[7]:8.2f}')
