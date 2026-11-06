import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code'))
__import__('os').chdir(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code')  # paths below are relative to code/
from train import load_archives, read_labels, attach_labels, fetch, build_model
F=Path('../features'); S=Path(sys.argv[1])
classes=np.load('../checkpoints/classes.npy',allow_pickle=True); idx={str(c):i for i,c in enumerate(classes)}
store=load_archives([F/f'shards/split100_{i:02d}.npz' for i in range(16)])
ds=attach_labels(store, read_labels(F/'split100_min5.csv'), idx, 'train', True)
P={'rel':'../checkpoints/milbc/model_v1.pth','ret':'../checkpoints/retrained/milbc/model.pth','relp':'../checkpoints/milbc_plus/model_v1.pth','retp':'../checkpoints/retrained/milbc_plus/model.pth','esm':'../checkpoints/retrained/esm2_bc/model.pth'}
rows=torch.from_numpy(ds.rows); out={}
for n,p in P.items():
    name={'relp':'milbc_plus','retp':'milbc_plus','esm':'esm2_bc'}.get(n,'milbc')
    m=build_model(name); m.load_state_dict(torch.load(p,map_location='cpu')); m.eval().cuda()
    am=[]
    with torch.no_grad():
        for s in range(0,len(rows),4096):
            d,mk,e=fetch(store,rows[s:s+4096],'cuda',True)
            am.append(m(d,mk,None if name=='milbc' else e)[0].argmax(1).cpu().numpy())
    out[n]=np.concatenate(am)
np.savez(S/'train_argmax.npz', entries=np.array([store.entries[r] for r in ds.rows]), **out)
wrong={n:np.array([a not in c for a,c in zip(v,ds.classes)]) for n,v in out.items()}
for n in out: print(n,'wrong',wrong[n].sum())
for a,b in [('rel','relp'),('ret','retp'),('rel','ret'),('relp','retp'),('rel','esm'),('ret','esm')]:
    both=wrong[a]&wrong[b]; same=(out[a]==out[b])&both
    print(f'{a:4s}&{b:4s}: both wrong {both.sum():5d}, same wrong EC {same.sum():5d}')
