"""Compare benchmark PDB coordinates locally and after rigid-body alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def read_ca(path):
    rows=[r for r in path.read_text().splitlines() if r.startswith('ATOM  ') and r[12:16].strip()=='CA']
    ids=[(r[21:27],r[17:20]) for r in rows]
    xyz=np.array([[float(r[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for r in rows])
    confidence=np.array([float(r[60:66]) for r in rows])
    return ids,xyz,confidence

def compare(a,b):
    ia,x,pa=read_ca(a);ib,y,pb=read_ca(b)
    if ia!=ib or not ia:raise ValueError(f'Residue identity mismatch: {a}, {b}')
    x0=x-x.mean(0);y0=y-y.mean(0)
    u,s,v=np.linalg.svd(x0.T@y0)
    d=np.eye(3);d[-1,-1]=np.linalg.det(u@v)
    rotated=x0@u@d@v
    da=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)
    db=np.linalg.norm(y[:,None,:]-y[None,:,:],axis=-1)
    mask=np.triu((da<15)&(da>0),1);error=np.abs(da[mask]-db[mask])
    return {'identical_pdb_bytes':a.read_bytes()==b.read_bytes(),'residues':len(x),
            'ca_rmsd_angstrom':float(np.sqrt(np.mean(np.sum((rotated-y0)**2,axis=1)))),
            'ca_lddt':float(np.mean([np.mean(error<t) for t in [.5,1.,2.,4.]])),
            'mean_absolute_plddt_difference':float(np.mean(np.abs(pa-pb))),
            'reference':str(a),'candidate':str(b)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[]
    for case in sorted(args.reference.iterdir()):
        if not case.is_dir() or case.name.startswith('embedding_'):continue
        for rep in [1,2]:
            for a in sorted((case/f'run{rep}'/'prediction').rglob('*.pdb')):
                b=args.candidate/a.relative_to(args.reference)
                if not b.exists():raise ValueError(f'Missing candidate: {b}')
                rows.append({'comparison':'paired','case':case.name,'replicate':rep,**compare(a,b)})
        for root,label in [(args.reference,'original_repeat'),(args.candidate,'candidate_repeat')]:
            for a in sorted((root/case.name/'run1'/'prediction').rglob('*.pdb')):
                b=root/case.name/'run2'/'prediction'/a.relative_to(root/case.name/'run1'/'prediction')
                rows.append({'comparison':label,'case':case.name,**compare(a,b)})
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    paired=[r for r in rows if r['comparison']=='paired']
    unrelaxed=[r for r in paired if r['reference'].endswith('_unrelaxed.pdb')]
    print(json.dumps({'comparisons':len(rows),'paired':len(paired),
                      'unrelaxed_exact':sum(r['identical_pdb_bytes'] for r in unrelaxed),
                      'unrelaxed_count':len(unrelaxed),'minimum_ca_lddt':min(r['ca_lddt'] for r in paired),
                      'maximum_ca_rmsd_angstrom':max(r['ca_rmsd_angstrom'] for r in paired)},indent=2))

if __name__=='__main__':main()
