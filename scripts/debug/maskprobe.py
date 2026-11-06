import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code'))
__import__('os').chdir(__import__('pathlib').Path(__file__).resolve().parents[2] / 'code')  # paths below are relative to code/
from train import load_archive, read_labels, attach_labels, fetch, build_model, weighted_prf, calls_to_ecs
from infer import call_ec
F=Path('../features')
classes=np.load('../checkpoints/classes.npy',allow_pickle=True).astype(str); idx={c:i for i,c in enumerate(classes)}
sets={}
for n in ['new','price','uniprot-all','uniprot-multi','test70','test50']:
    st=load_archive(F/f'tests/{n}.npz'); sets[n]=(st,attach_labels(st,read_labels(F/f'tests/{n}.csv'),idx,n,False))
P=dict(a.split('=') for a in sys.argv[1:])
for tag,p in P.items():
    name='milbc_plus' if 'plus' in tag else 'milbc'
    m=build_model(name); m.load_state_dict(torch.load(p,map_location='cpu')); m.eval().cuda()
    for mode in ['mask','nomask']:
        r=[]
        for n,(st,ds) in sets.items():
            d,mk,e=fetch(st,torch.from_numpy(ds.rows),'cuda',True)
            with torch.no_grad(): lg=m(d, mk if mode=='mask' else None, e if name=='milbc_plus' else None)[0].float().cpu().numpy()
            r.append(weighted_prf(ds.true_ecs, calls_to_ecs(call_ec(lg),classes))['f1'])
        print(f'{tag:14s} {mode:7s}', ' '.join('%.3f'%x for x in r))
