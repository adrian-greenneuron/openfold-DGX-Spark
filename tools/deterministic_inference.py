"""Opt-in deterministic model-forward verification, excluding relaxation.

GPU relaxation was not byte-repeatable even with fixed preparation seeds and
OpenMM deterministic forces in the tested images. This wrapper therefore
requires --skip_relaxation and makes no repeatability claim for minimisation.
Normal run_pretrained_openfold.py calls are unaffected.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import random
import runpy
import sys
if '--skip_relaxation' not in sys.argv and '--help' not in sys.argv:
    raise SystemExit('Verification requires --skip_relaxation; GPU relaxation is not guaranteed repeatable.')
random.seed(42)
sys.path.insert(0,'/opt/openfold')
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(43)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
sys.argv=['/opt/openfold/run_pretrained_openfold.py',*sys.argv[1:]]
runpy.run_path(sys.argv[0],run_name='__main__')
