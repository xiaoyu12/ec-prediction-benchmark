import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code'))
__import__('os').chdir(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code')  # paths below are relative to code/
from train import load_archives, read_labels, attach_labels, fetch, build_model
F=Path('../features')
classes=np.load('../checkpoints/classes.npy',allow_pickle=True); idx={str(c):i for i,c in enumerate(classes)}
store=load_archives([F/f'shards/split100_{i:02d}.npz' for i in range(16)])
ds=attach_labels(store, read_labels(F/'split100_min5.csv'), idx, 'train', True)
cnt=np.zeros(5242,int)
for c in ds.classes:
    for k in c: cnt[k]+=1
# per-enzyme frequency = min count over its labels
efreq=np.array([min(cnt[k] for k in c) for c in ds.classes])
bins=[1,2,3,6,11,31,101,1001,10**6]
P={'released':'../checkpoints/milbc/model_v1.pth','retrained':'../checkpoints/retrained/milbc/model.pth','rel_plus':'../checkpoints/milbc_plus/model_v1.pth','ret_plus':'../checkpoints/retrained/milbc_plus/model.pth'}
rows=torch.from_numpy(ds.rows)
print('class-count bins:',bins[:-1]); print('enzymes per bin', [int(((efreq>=a)&(efreq<b)).sum()) for a,b in zip(bins,bins[1:])])
for n,p in P.items():
    name='milbc_plus' if 'plus' in n else 'milbc'
    m=build_model(name); sd=torch.load(p,map_location='cpu'); m.load_state_dict(sd); m.eval().cuda()
    hit=np.zeros(len(rows),bool)
    with torch.no_grad():
        for s in range(0,len(rows),4096):
            d,mk,e=fetch(store,rows[s:s+4096],'cuda',True)
            am=m(d,mk,e if name=='milbc_plus' else None)[0].argmax(1).cpu().numpy()
            for j,a in enumerate(am): hit[s+j]= a in ds.classes[s+j]
    W=sd['classifier.8.weight'].norm(dim=1).numpy()
    print(f'{n:10s} train top1 by bin', ' '.join('%.3f'%hit[(efreq>=a)&(efreq<b)].mean() for a,b in zip(bins,bins[1:])), ' overall %.4f'%hit.mean())
    print(f'{"":10s} out-row norm by bin', ' '.join('%.2f'%W[(cnt>=a)&(cnt<b)].mean() for a,b in zip(bins,bins[1:])), ' corr(log cnt, norm) %.2f'%np.corrcoef(np.log(cnt[cnt>0]),W[cnt>0])[0,1])
