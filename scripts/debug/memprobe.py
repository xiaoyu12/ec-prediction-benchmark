"""Do released checkpoints look like they trained on test enzymes? Compare per-enzyme top logits."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from train import load_archive, read_labels, attach_labels, fetch, build_model
R = Path(__file__).resolve().parents[2]; F = R / 'features'
classes = np.load(R / 'checkpoints/classes.npy', allow_pickle=True).astype(str); idx = {c: i for i, c in enumerate(classes)}
P = {'rel': ('milbc', 'checkpoints/milbc/model_v1.pth'), 'ret': ('milbc', 'checkpoints/retrained/milbc/model.pth'),
     'relp': ('milbc_plus', 'checkpoints/milbc_plus/model_v1.pth'), 'retp': ('milbc_plus', 'checkpoints/retrained/milbc_plus/model.pth'),
     'esm': ('esm2_bc', 'checkpoints/retrained/esm2_bc/model.pth')}
models = {}
for k, (name, p) in P.items():
    m = build_model(name); m.load_state_dict(torch.load(R / p, map_location='cpu')); models[k] = (name, m.eval().cuda())
print('per set: % enzymes whose top-1 is correct AND top logit > 10 (train-like confidence); then mean top logit')
for n in ['new', 'price', 'uniprot-all', 'test70', 'test50']:
    st = load_archive(F / f'tests/{n}.npz'); ds = attach_labels(st, read_labels(F / f'tests/{n}.csv'), idx, n, False)
    d, mk, e = fetch(st, torch.from_numpy(ds.rows), 'cuda', True)
    out = []
    for k, (name, m) in models.items():
        with torch.no_grad(): lg = m(d, mk, None if name == 'milbc' else e)[0].float().cpu()
        top, arg = lg.max(1)
        ok = np.array([a in c for a, c in zip(arg.numpy(), ds.classes)])
        out.append(f'{k}: {100 * (ok & (top.numpy() > 10)).mean():4.1f}% / {top.mean():5.2f}')
    print(f'{n:12s}', '   '.join(out))
