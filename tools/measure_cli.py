"""Measure an otherwise unchanged OpenFold inference/embedding command."""
import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--entrypoint',required=True)
    args,cli=p.parse_known_args()
    sys.path.insert(0,'/opt/openfold')
    import torch
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    # Seed embedding preparation too; inference also receives its upstream seed flag.
    torch.manual_seed(43)
    completed=False
    sys.argv=[args.entrypoint,*cli]
    try:
        runpy.run_path(args.entrypoint,run_name='__main__')
        completed=True
    except SystemExit as e:
        if e.code not in (None,0):raise
        completed=True
    finally:
        torch.cuda.synchronize()
        peak=Path('/sys/fs/cgroup/memory.peak')
        result={'completed':completed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'peak_main_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'container_memory_peak_bytes':int(peak.read_text()) if peak.exists() else None,
                'torch':torch.__version__,'cuda':torch.version.cuda,
                'allocator':os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}
        args.metrics.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
