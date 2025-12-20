# =============================================================================
# OpenFold Dockerfile Optimized for NVIDIA DGX Spark (ARM64 / Blackwell)
# =============================================================================
# Using NGC PyTorch base image:
# - Pre-installed PyTorch 2.10, CUDA 13.0, cuDNN, nccl
# - Optimized for NVIDIA hardware (ARM64 + Blackwell sm_120/sm_121 support)
# - Includes Flash Attention 2.7.4, Triton 3.5.0, CUTLASS 4.0.0
# - Significantly faster build time
# =============================================================================
# hadolint ignore=DL3006
FROM nvcr.io/nvidia/pytorch:25.11-py3

# -----------------------------------------------------------------------------
# Build Configuration
# -----------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="12.0" \
    CUTLASS_PATH=/opt/pytorch/ao/third_party/cutlass \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
# OpenFold needs hmmer, kalign, and alignment tools
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    hmmer \
    kalign \
    aria2 \
    pdb2pqr \
    openbabel \
    cmake \
    doxygen \
    swig \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Dependencies
# -----------------------------------------------------------------------------
# hadolint ignore=DL3013
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
    biotite \
    cuda-python

# -----------------------------------------------------------------------------
# DeepSpeed (Patched for Blackwell)
# -----------------------------------------------------------------------------
# Pin to 0.15.4 and patch builder.py to fix Blackwell sm_121 detection
# DeepSpeed is NOT included in NGC 25.11 - we must install and patch it
COPY patch_ds.py /opt/patch_ds.py
RUN pip install --no-cache-dir deepspeed==0.15.4 \
    && python3 /opt/patch_ds.py \
    && rm /opt/patch_ds.py

# -----------------------------------------------------------------------------
# NGC Built-in Components (No Installation Required)
# -----------------------------------------------------------------------------
# NGC 25.11 includes:
# - Flash Attention 2.7.4 (pre-installed)
# - Triton 3.5.0 with CUDA 13 and Blackwell support (pre-installed)
# - CUTLASS 4.0.0 at /opt/pytorch/ao/third_party/cutlass (CUTLASS_PATH set above)
#
# Do NOT install Triton nightly - it bundles CUDA 12.8 PTXAS causing performance regression.
# Do NOT install flash-attn - NGC version is optimized for this container.

# -----------------------------------------------------------------------------
# Clone OpenFold
# -----------------------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold \
    && rm -rf /opt/openfold/.git

# -----------------------------------------------------------------------------
# OpenMM (Source Build for Blackwell/sm_120 Support)
# -----------------------------------------------------------------------------
# Conda binaries are incompatible with Blackwell driver (PTX error).
# We must build from source linking against the local CUDA toolkit.
WORKDIR /tmp
# hadolint ignore=DL3003,SC2046
RUN git clone https://github.com/openmm/openmm.git \
    && cd openmm \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    && make -j"$(nproc)" install \
    && cd python && OPENMM_INCLUDE_PATH=/usr/local/include OPENMM_LIB_PATH=/usr/local/lib python3 setup.py install \
    && cd /tmp && rm -rf openmm

# Install pdbfixer from source
# hadolint ignore=DL3013
RUN pip install --no-cache-dir git+https://github.com/openmm/pdbfixer.git

# -----------------------------------------------------------------------------
# OpenFold Installation
# -----------------------------------------------------------------------------
WORKDIR /opt/openfold

# Patch setup.py to add Blackwell compute capability since there's no GPU at build time
# Fix missing stereo_chemical_props.txt (required for relaxation)
# Install OpenFold (--no-build-isolation required for torch access)
# Install awscli for downloading model weights
# hadolint ignore=DL3013
RUN sed -i "s/compute_capabilities = set(\[/compute_capabilities = set([(12, 0),/" setup.py \
    && mkdir -p openfold/resources \
    && wget -q --progress=dot:giga https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
    -O openfold/resources/stereo_chemical_props.txt \
    && pip install --no-cache-dir --no-build-isolation . \
    && pip install --no-cache-dir awscli

# -----------------------------------------------------------------------------
# Code Fixes for CUDA 13 Compatibility
# -----------------------------------------------------------------------------
# Fix SyntaxWarning: invalid escape sequence '\W' in script_utils.py
# Fix cuda-python 13.x import structure change (cuda.cudart -> cuda.bindings.runtime)
# Pre-create cache directories to silence Triton warnings
RUN sed -i "s/re.split('\\\\W| \\\\|'/re.split(r'\\\\W| \\\\|'/g" /opt/openfold/openfold/utils/script_utils.py \
    && sed -i 's/import cuda.cudart as cudart/from cuda.bindings import runtime as cudart/g' /opt/openfold/openfold/utils/tensorrt_lazy_compiler.py \
    && mkdir -p /root/.triton/autotune

# -----------------------------------------------------------------------------
# Model Weights
# -----------------------------------------------------------------------------
# OPTIONAL: Comment out the following lines to build a lighter image without embedded weights.
# You will need to mount weights at runtime if you skip this.
# Download OpenFold model parameters from AWS S3 (public bucket, no auth required)
# Install fair-esm (required for SoloSeq) and download SoloSeq parameters
# Pre-download ESM-1b weights for SoloSeq (bakes ~2.5GB model into image)
# hadolint ignore=DL3013
RUN bash /opt/openfold/scripts/download_openfold_params.sh /opt/openfold \
    && pip install --no-cache-dir fair-esm \
    && bash /opt/openfold/scripts/download_openfold_soloseq_params.sh /opt/openfold \
    && python3 -c "import esm; esm.pretrained.esm1b_t33_650M_UR50S()"

# -----------------------------------------------------------------------------
# AlphaFold Multimer Weights (OPTIONAL - adds ~1.5GB)
# -----------------------------------------------------------------------------
# Uncomment the following to enable multimer inference (protein complexes).
# NOTE: Multimer also requires precomputed alignments or massive databases (~2.5TB).
# RUN bash /opt/openfold/scripts/download_alphafold_params.sh /opt/openfold

# -----------------------------------------------------------------------------
# Example Templates
# -----------------------------------------------------------------------------
# OPTIONAL: Comment out to skip downloading example templates (saves space/time)
# Download mmCIF templates required for the monomer example (6KWC)
# This allows the example to run out-of-the-box without mounting external templates
COPY scripts/download_example_templates.sh /opt/openfold/scripts/
RUN bash /opt/openfold/scripts/download_example_templates.sh

# -----------------------------------------------------------------------------
# Validation & Runtime
# -----------------------------------------------------------------------------
# Create a test to verify all imports work
RUN python3 -c "import openfold; import torch; import openmm; import pdbfixer; from openfold.np.relax import relax; print(f'OpenFold on PyTorch {torch.__version__}, CUDA {torch.version.cuda}, OpenMM {openmm.version.short_version}')"

WORKDIR /opt/openfold
CMD ["/bin/bash"]
