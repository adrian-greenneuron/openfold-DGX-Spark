"""Repeat the eight published OpenFold cases with frozen inputs and resources."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

def system_memory():
    d={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
    return d['MemTotal']-d['MemAvailable']

def validate_pdb(path,expected):
    rows=[line for line in path.read_text().splitlines() if line.startswith('ATOM  ') and line[12:16].strip()=='CA']
    if len(rows)!=expected:raise ValueError(f'{path}: expected {expected} residues, found {len(rows)}')
    coords=[[float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]] for line in rows]
    if not all(math.isfinite(x) for r in coords for x in r):raise ValueError(f'Nonfinite coordinates: {path}')
    return {'file':path.name,'residues':len(rows),'mean_plddt':sum(float(r[60:66]) for r in rows)/len(rows),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def execute(image,tools,frozen,out,task,protein,relax,embedding,rep):
    ident=json.loads((frozen/protein/'identity.json').read_text())
    dest=out/f'run{rep}'
    dest.mkdir(parents=True,exist_ok=False)
    entry='/opt/openfold/scripts/precompute_embeddings.py' if task=='embedding' else '/opt/openfold/run_pretrained_openfold.py'
    cmd=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--shm-size=64g',
         '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
         '-e','PYTHONHASHSEED=0','--mount',f'type=bind,src={tools},dst=/measurement,readonly',
         '--mount',f'type=bind,src={frozen},dst=/frozen,readonly',
         '--mount',f'type=bind,src={dest},dst=/output']
    if embedding:cmd+=['--mount',f'type=bind,src={embedding},dst=/embeddings,readonly']
    cmd += [image,'python3','/measurement/measure_cli.py','--metrics','/output/resource_metrics.json','--entrypoint',entry]
    if task=='embedding':
        cmd += [f'/frozen/{protein}/fasta','/output/embeddings','--use_local_esm','/frozen/esm']
    else:
        mode=task
        preset='model_1_ptm' if mode=='standard' else 'seq_model_esm1b_ptm'
        weights='/opt/openfold/openfold_params/finetuning_ptm_2.pt' if mode=='standard' else '/opt/openfold/openfold_soloseq_params/seq_model_esm1b_ptm.pt'
        cmd += [f'/frozen/{protein}/fasta','/frozen/templates','--output_dir','/output/prediction',
                '--config_preset',preset,'--openfold_checkpoint_path',weights,
                '--use_precomputed_alignments',f'/frozen/{protein}/alignments' if mode=='standard' else '/embeddings',
                '--data_random_seed','42','--model_device','cuda:0']
        if not relax:cmd+=['--skip_relaxation']
    (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    start=time.perf_counter();baseline=system_memory();samples=[(0.,baseline)];done=threading.Event()
    def monitor():
        while not done.wait(.1):samples.append((time.perf_counter()-start,system_memory()))
    t=threading.Thread(target=monitor,daemon=True);t.start()
    try:
        with (dest/'inference.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    finally:
        wall=time.perf_counter()-start;done.set();t.join();samples.append((wall,system_memory()))
        with (dest/'system_memory.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['elapsed_seconds','system_nonavailable_bytes']);w.writerows(samples)
    if r.returncode:raise RuntimeError(f'{task}/{protein}: exit={r.returncode}; see {dest}/inference.log')
    metrics=json.loads((dest/'resource_metrics.json').read_text())
    if not metrics['completed'] or metrics['peak_cuda_allocated_bytes']<=0:raise RuntimeError(f'Incomplete GPU measurement: {dest}')
    outputs=[];timings=[]
    if task!='embedding':
        pdbs=list((dest/'prediction').rglob('*.pdb'))
        for suffix in ['_unrelaxed.pdb']+(['_relaxed.pdb'] if relax else []):
            matching=[p for p in pdbs if p.name.endswith(suffix)]
            if len(matching)!=1:raise RuntimeError(f'Missing/duplicate output {suffix}: {dest}')
            outputs.append(validate_pdb(matching[0],ident['residues']))
        timings=[json.loads(p.read_text()) for p in (dest/'prediction').rglob('timings.json')]
    elif not list((dest/'embeddings').rglob('*.pt')):raise RuntimeError(f'Missing embeddings: {dest}')
    row={'warmup':rep==0,'replicate':rep,'task':task,'protein':protein,'residues':ident['residues'],
         'relaxation':relax,'wall_seconds':wall,'system_nonavailable_baseline_bytes':baseline,
         'sampled_system_nonavailable_peak_bytes':max(x[1] for x in samples),
         'sampled_system_increase_peak_bytes':max(x[1] for x in samples)-baseline,
         'internal_timings':timings,'outputs':outputs,**metrics}
    (dest/'measurement.json').write_text(json.dumps(row,indent=2)+'\n')
    # Writable outputs from root containers remain accessible to the invoking user.
    print(json.dumps({'task':task,'protein':protein,'relax':relax,'replicate':rep,'wall_seconds':wall}),flush=True)
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tools',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--canonical-embeddings',type=Path)
    args=p.parse_args();args.frozen=args.frozen.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for protein in ['short','medium']:
        out=args.output/f'embedding_{protein}'
        for rep in range(3):rows.append(execute(args.image,args.tools,args.frozen,out,'embedding',protein,False,None,rep))
        canonical=(args.canonical_embeddings or args.output)/f'embedding_{protein}'/'run1'/'embeddings'
        for task in ['standard','soloseq']:
            for relax in [False,True]:
                out=args.output/f'{protein}_{task}_{"relax" if relax else "no_relax"}'
                for rep in range(3):
                    rows.append(execute(args.image,args.tools,args.frozen,out,task,protein,relax,canonical if task=='soloseq' else None,rep))
                    (args.output/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
