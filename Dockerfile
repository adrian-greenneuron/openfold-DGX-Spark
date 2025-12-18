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
# -----------------------------------------------------------------------------
# OpenMM (Source Build for Blackwell/sm_120 Support)
# -----------------------------------------------------------------------------
# Conda binaries are incompatible with Blackwell driver (PTX error).
# We must build from source linking against the local CUDA toolkit.

# Install build dependencies
RUN apt-get update && apt-get install -y git cmake doxygen swig wget && \
    rm -rf /var/lib/apt/lists/*

# Build OpenMM from source
WORKDIR /tmp
RUN git clone https://github.com/openmm/openmm.git && \
    cd openmm && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda && \
    make -j$(nproc) install && \
    cd python && OPENMM_INCLUDE_PATH=/usr/local/include OPENMM_LIB_PATH=/usr/local/lib python3 setup.py install && \
    cd ../.. && rm -rf openmm

# Install pdbfixer from source
RUN pip install git+https://github.com/openmm/pdbfixer.git

# Install other dependencies
RUN pip install biopython dm-tree ml-collections scipy

# Remove LD_LIBRARY_PATH hack as we installed to system locations


WORKDIR /opt/openfold
# RUN python3 /opt/patch_openfold.py # Skipping patch, using real install
# Install in editable mode or standard
# We use --no-build-isolation because OpenFold setup requires torch, which is in the system env
# but hidden by pip's build isolation.
# IMPORTANT: Force Blackwell (sm_120) for CUDA kernel compilation
# Patch setup.py to add Blackwell compute capability since there's no GPU at build time
RUN sed -i "s/compute_capabilities = set(\[/compute_capabilities = set([(12, 0),/" setup.py
ENV TORCH_CUDA_ARCH_LIST="12.0"
# Fix missing stereo_chemical_props.txt (required for relaxation)
# OpenFold expects this file in openfold/resources but it might be missing from the pip install
RUN mkdir -p openfold/resources && \
    wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
    -O openfold/resources/stereo_chemical_props.txt

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
# OPTIONAL: Comment out the following lines to build a lighter image without embedded weights.
# You will need to mount weights at runtime if you skip this.
# Download OpenFold model parameters from AWS S3 (public bucket, no auth required)
RUN bash /opt/openfold/scripts/download_openfold_params.sh /opt/openfold

# Install fair-esm (required for SoloSeq) and download SoloSeq parameters
RUN pip install --no-cache-dir fair-esm && \
    bash /opt/openfold/scripts/download_openfold_soloseq_params.sh /opt/openfold

# Pre-download ESM-1b weights for SoloSeq (bakes ~2.5GB model into image)
# OPTIONAL: Comment this out to save ~2.5GB if you do not need SoloSeq baked in
RUN python3 -c "import esm; esm.pretrained.esm1b_t33_650M_UR50S()"

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
RUN python3 -c "import openfold; import torch; import openmm; import pdbfixer; from openfold.np.relax import relax; print(f'OpenFold on PyTorch {torch.__version__}, OpenMM {openmm.version.short_version}')"

WORKDIR /opt/openfold
CMD ["/bin/bash"]
