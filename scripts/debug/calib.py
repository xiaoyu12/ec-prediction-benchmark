"""Calibration of every checkpoint on one test set: calls/enzyme, enzymes with any logit >= 0, second-best logit."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from train import load_archive, read_labels, attach_labels, fetch, build_model, weighted_prf
from infer import call_ec
R = Path(__file__).resolve().parents[2]; F = R / 'features'
classes = np.load(R / 'checkpoints/classes.npy', allow_pickle=True).astype(str); idx = {c: i for i, c in enumerate(classes)}
test = sys.argv[1] if len(sys.argv) > 1 else 'test70'
st = load_archive(F / f'tests/{test}.npz'); ds = attach_labels(st, read_labels(F / f'tests/{test}.csv'), idx, test, False)
d, mk, e = fetch(st, torch.from_numpy(ds.rows), 'cuda', True)
ckpts = {'released': R / 'checkpoints/milbc/model_v1.pth', 'released_plus': R / 'checkpoints/milbc_plus/model_v1.pth',
         'retrained': R / 'checkpoints/retrained/milbc/model.pth', 'retrained_plus': R / 'checkpoints/retrained/milbc_plus/model.pth'}
for path in sorted((R / 'runs').glob('*/*/model.pth')):
    ckpts[f'{path.parent.parent.name}/{path.parent.name}'] = path
print(f"{test}: {'checkpoint':34s}{'init*':>6s}{'calls':>7s}{'any>=0':>8s}{'2nd logit':>10s}{'top logit':>10s}{'top1':>7s}{'F1':>7s}")
for tag, path in ckpts.items():
    sd = torch.load(path, map_location='cpu')
    if 'instance_encoder.0.weight' not in sd: continue
    name = 'milbc_plus' if sd['classifier.0.weight'].shape[1] > 480 else 'milbc'
    m = build_model(name); m.load_state_dict(sd); m.eval().cuda()
    with torch.no_grad(): lg = m(d, mk, None if name == 'milbc' else e)[0].float().cpu()
    top2 = lg.topk(2, 1).values
    ok = np.mean([a in c for a, c in zip(lg.argmax(1).numpy(), ds.classes)])
    calls = call_ec(lg.numpy())
    f1 = weighted_prf(ds.true_ecs, [list(classes[np.flatnonzero(r)]) for r in calls])['f1']
    std = sd['attention_V.0.weight'].std().item()  # ~0.065 after kaiming_normal init, ~0.035-0.045 after default
    print(f"{'':{len(test)+2}s}{tag[:34]:34s}{std:6.3f}{calls.sum(1).mean():7.2f}{(lg >= 0).any(1).float().mean():8.2f}"
          f"{top2[:, 1].mean():10.2f}{top2[:, 0].mean():10.2f}{ok:7.3f}{f1:7.3f}")
