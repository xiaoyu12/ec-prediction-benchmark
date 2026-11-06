import sys, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code'))
__import__('os').chdir(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code')  # paths below are relative to code/
from train import load_archive, read_labels, attach_labels, padded_labels, multi_hot, fetch, build_model
F='../features'
classes=np.load('../checkpoints/classes.npy',allow_pickle=True); idx={str(c):i for i,c in enumerate(classes)}
store=load_archive(__import__('pathlib').Path(F+'/shards/split100_03.npz'))
ds=attach_labels(store, read_labels(__import__('pathlib').Path(F+'/split100_min5.csv')), idx, 's03', True)
rows=torch.from_numpy(ds.rows[:8192]); T=multi_hot(padded_labels(ds.classes[:8192]))
P={'released':'../checkpoints/milbc/model_v1.pth','retrained':'../checkpoints/retrained/milbc/model.pth','rel_plus':'../checkpoints/milbc_plus/model_v1.pth','ret_plus':'../checkpoints/retrained/milbc_plus/model.pth'}
for n,p in P.items():
    name='milbc_plus' if 'plus' in n else 'milbc'
    m=build_model(name); sd=torch.load(p,map_location='cpu'); m.load_state_dict(sd); m.eval().cuda()
    print(n,'bias mean %.5f'%sd['classifier.8.bias'].mean().item())
    L=[]
    with torch.no_grad():
        for s in range(0,len(rows),1024):
            d,mk,e=fetch(store,rows[s:s+1024],'cuda',True)
            L.append(m(d,mk,e if name=='milbc_plus' else None)[0].float().cpu())
    L=torch.cat(L); pos=L[T>0]; neg=L[T==0]
    rowmean=L.mean(1); lse=torch.logsumexp(L,1)
    sm=torch.softmax(L,1)[T>0]
    print('  pos logit mean %.2f  neg logit mean %.2f std %.2f  max-neg per row mean %.2f'%(pos.mean(),neg.mean(),neg.std(),L.masked_fill(T>0,-1e9).max(1).values.mean()))
    print('  row-mean logit %.2f +- %.2f   logsumexp %.2f +- %.2f   softmax prob at true %.4f'%(rowmean.mean(),rowmean.std(),lse.mean(),lse.std(),sm.mean()))
