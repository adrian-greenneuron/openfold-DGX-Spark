# =============================================================================
# OpenFold Dockerfile Optimized for NVIDIA DGX Spark (ARM64 / Blackwell)
# =============================================================================
# Using NGC PyTorch base image:
# - Pre-installed PyTorch, CUDA, cuDNN, nccl
# - Optimized for NVIDIA hardware (ARM64 + Blackwell sm_121 support)
# - Significantly faster build time
# =============================================================================

# Use the latest stable NGC PyTorch image (25.01-py3) matching OpenFold3 example
# This guarantees CUDA 12.8 / Blackwel sm_121 optimizations
FROM nvcr.io/nvidia/pytorch:25.01-py3

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
# OpenFold needs hmmer, kalign, and alignment tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    git \
    hmmer \
    kalign \
    aria2 \
    pdb2pqr \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Dependencies
# -----------------------------------------------------------------------------
# Libraries not in the standard NGC image
RUN pip install --no-cache-dir \
    biopython \
    ml-collections \
    pyyaml \
    requests \
    tqdm \
    pytorch-lightning \
    dm-tree \
    modelcif \
    wandb \
    biotite

# -----------------------------------------------------------------------------
# DeepSpeed (Patched for Blackwell)
# -----------------------------------------------------------------------------
# Pin to 0.15.4 and patch builder.py to fix Blackwell sm_121 detection
COPY patch_ds.py /opt/patch_ds.py
RUN pip install --no-cache-dir deepspeed==0.15.4 && python3 /opt/patch_ds.py

# -----------------------------------------------------------------------------
# Flash Attention
# -----------------------------------------------------------------------------
# NGC containers often have flash-attn. If not, or if we need specific version:
# For now, let's try to install it. The NGC image has ninja/packaging pre-installed.
# We limit to sm_90 (Hopper) and sm_100/120 (Blackwell) to speed up if compiled.
# If pre-installed, this step completes instantly.
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install flash-attn --no-build-isolation

# -----------------------------------------------------------------------------
# Triton Nightly (Required for sm_121 / Blackwell)
# -----------------------------------------------------------------------------
# NGC container's Triton is too old for Blackwell. Install compatible nightly.
RUN pip install --pre triton --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# -----------------------------------------------------------------------------
# CUTLASS & OpenFold Setup
# -----------------------------------------------------------------------------
WORKDIR /opt
# Clone CUTLASS (required for DeepSpeed evoformer attention on Blackwell)
RUN git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1
ENV CUTLASS_PATH=/opt/cutlass
# Clone OpenFold
RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold

# -----------------------------------------------------------------------------
# OpenMM & pdbfixer (ARM64 via Conda)
# -----------------------------------------------------------------------------
# OpenMM builds for ARM64 are only reliably available via conda-forge.
# We install Miniconda, create an env, and point the system Python to it.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Install openmm and pdbfixer into a clean env matching the system python version (3.12)
# We use --override-channels to STRICTLY use conda-forge and avoid Anaconda ToS errors
RUN /opt/conda/bin/conda create -n openmm -y --override-channels -c conda-forge python=3.12 openmm pdbfixer

# Remove numpy and scipy from conda environment to avoid conflicts with NGC's optimized versions
# OpenMM will use the system numpy/scipy instead
RUN rm -rf /opt/conda/envs/openmm/lib/python3.12/site-packages/numpy* \
    && rm -rf /opt/conda/envs/openmm/lib/python3.12/site-packages/scipy*

# Add Conda libraries to System Python path
# This allows the optimized NGC PyTorch (system python) to import module from Conda
ENV PYTHONPATH="/opt/conda/envs/openmm/lib/python3.12/site-packages"
# Add Conda shared libraries to LD_LIBRARY_PATH (needed for OpenMM C++ backend)
ENV LD_LIBRARY_PATH="/opt/conda/envs/openmm/lib:$LD_LIBRARY_PATH"

WORKDIR /opt/openfold
# RUN python3 /opt/patch_openfold.py # Skipping patch, using real install
# Install in editable mode or standard
# We use --no-build-isolation because OpenFold setup requires torch, which is in the system env
# but hidden by pip's build isolation.
# IMPORTANT: Force Blackwell (sm_120) for CUDA kernel compilation
# Patch setup.py to add Blackwell compute capability since there's no GPU at build time
RUN sed -i "s/compute_capabilities = set(\[/compute_capabilities = set([(12, 0),/" setup.py
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install --no-build-isolation .

# Install awscli for downloading model weights
RUN pip install --no-cache-dir awscli

# -----------------------------------------------------------------------------
# Code Fixes
# -----------------------------------------------------------------------------
# Fix SyntaxWarning: invalid escape sequence '\W' in script_utils.py
RUN sed -i "s/re.split('\\\\W| \\\\|'/re.split(r'\\\\W| \\\\|'/g" /opt/openfold/openfold/utils/script_utils.py

# Pre-create cache directories to silence Triton warnings
RUN mkdir -p /root/.triton/autotune

# -----------------------------------------------------------------------------
# Model Weights
# -----------------------------------------------------------------------------
# Download OpenFold model parameters from AWS S3 (public bucket, no auth required)
RUN bash /opt/openfold/scripts/download_openfold_params.sh /opt/openfold

# -----------------------------------------------------------------------------
# Validation & Runtime
# -----------------------------------------------------------------------------
# Create a test to verify all imports work
RUN python3 -c "import openfold; import torch; import openmm; import pdbfixer; from openfold.np.relax import relax; print(f'OpenFold on PyTorch {torch.__version__}, OpenMM {openmm.version.short_version}')"

WORKDIR /opt/openfold
CMD ["/bin/bash"]
