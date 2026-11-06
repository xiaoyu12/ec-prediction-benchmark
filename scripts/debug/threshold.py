"""How call_ec's threshold and the support-weighted F1 interact: free false positives and threshold shifts."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from train import load_archive, read_labels, attach_labels, fetch, build_model, weighted_prf
R = Path(__file__).resolve().parents[2]; F = R / 'features'
classes = np.load(R / 'checkpoints/classes.npy', allow_pickle=True).astype(str); idx = {c: i for i, c in enumerate(classes)}
P = {'rel': ('milbc', 'checkpoints/milbc/model_v1.pth'), 'ret': ('milbc', 'checkpoints/retrained/milbc/model.pth'),
     'relp': ('milbc_plus', 'checkpoints/milbc_plus/model_v1.pth'), 'retp': ('milbc_plus', 'checkpoints/retrained/milbc_plus/model.pth'),
     'esm': ('esm2_bc', 'checkpoints/retrained/esm2_bc/model.pth')}

def call(logits, t):  # call_ec with the 0 threshold moved to t
    s = torch.from_numpy(logits); c = s >= t
    none = ~c.any(1)
    if none.any(): c[none] = s[none] >= s[none].max(1, keepdim=True).values
    over = c.sum(1) > 5
    if over.any(): c[over] = s[over] >= s[over].topk(5, 1).values[:, -1:]
    return c.numpy()

sets = {}
for n in ['new', 'price', 'uniprot-all', 'uniprot-multi', 'test70', 'test50']:
    st = load_archive(F / f'tests/{n}.npz'); sets[n] = (st, attach_labels(st, read_labels(F / f'tests/{n}.csv'), idx, n, False))
logits = {}
for k, (name, p) in P.items():
    m = build_model(name); m.load_state_dict(torch.load(R / p, map_location='cpu')); m.eval().cuda()
    for n, (st, ds) in sets.items():
        d, mk, e = fetch(st, torch.from_numpy(ds.rows), 'cuda', True)
        with torch.no_grad(): logits[k, n] = m(d, mk, None if name == 'milbc' else e)[0].float().cpu().numpy()

print('A. extra calls (beyond the top-1) at threshold 0: share that are correct / wrong-but-free (EC absent from the set truth) / wrong-and-costly')
for k in P:
    tot = np.zeros(3)
    for n, (st, ds) in sets.items():
        truth_ecs = {ec for t in ds.true_ecs for ec in t}
        c = call(logits[k, n], 0.0); top = logits[k, n].argmax(1)
        for i in range(len(c)):
            for j in np.flatnonzero(c[i]):
                if j == top[i]: continue
                ec = classes[j]
                tot += [ec in ds.true_ecs[i], ec not in ds.true_ecs[i] and ec not in truth_ecs, ec not in ds.true_ecs[i] and ec in truth_ecs]
    print(f'  {k:5s} {int(tot.sum()):5d} extra calls: {tot[0]/tot.sum():.2f} correct, {tot[1]/tot.sum():.2f} free, {tot[2]/tot.sum():.2f} costly')

print('\nB. F1 (and mean #calls on test70) as the call threshold moves; t=0 is call_ec')
names = list(sets)
print(f"{'model':6s}{'t':>5s}  " + ''.join(f'{n[:9]:>10s}' for n in names) + '   #calls(test70)')
for k in P:
    for t in [0.0, -1.0, -2.0, -3.0, -4.0]:
        f = [weighted_prf(ds.true_ecs, [list(classes[np.flatnonzero(r)]) for r in call(logits[k, n], t)])['f1'] for n, (_, ds) in sets.items()]
        print(f'{k:6s}{t:5.0f}  ' + ''.join(f'{x:10.3f}' for x in f) + f'   {call(logits[k, "test70"], t).sum(1).mean():.2f}')
