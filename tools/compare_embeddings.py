"""Check tensor agreement for the separately timed ESM embedding stages."""
import argparse
import json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmarks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for protein in ['short','medium']:
        ref=list((args.benchmarks/'original'/f'embedding_{protein}'/'run1'/'embeddings').rglob('*.pt'))
        assert len(ref)==1
        a=torch.load(ref[0],map_location='cpu',weights_only=True)['representations'][33]
        for condition in ['original','rebuilt','modern']:
            for rep in [1,2]:
                paths=list((args.benchmarks/condition/f'embedding_{protein}'/f'run{rep}'/'embeddings').rglob('*.pt'))
                assert len(paths)==1
                b=torch.load(paths[0],map_location='cpu',weights_only=True)['representations'][33]
                if a.shape!=b.shape or not torch.isfinite(b).all():raise ValueError('Invalid embeddings')
                rows.append({'protein':protein,'condition':condition,'replicate':rep,'shape':list(a.shape),
                             'identical_tensors':torch.equal(a,b),'max_absolute_difference':float((a-b).abs().max())})
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'compared':len(rows),'identical':sum(r['identical_tensors'] for r in rows)}))

if __name__=='__main__':main()
