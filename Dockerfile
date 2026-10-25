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
FROM nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21

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
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Dependencies
# -----------------------------------------------------------------------------
COPY runtime-constraints.txt /opt/runtime-constraints.txt
# hadolint ignore=DL3013
RUN pip install --no-cache-dir -c /opt/runtime-constraints.txt \
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

# Current released DeepSpeed; no builder.py patch is applied.
RUN pip install --no-cache-dir -c /opt/runtime-constraints.txt deepspeed==0.19.8

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
    && git -C /opt/openfold checkout --detach be2ec1841f16c966c65ae0e7599ebbadc725757d \
    && rm -rf /opt/openfold/.git

# Published ARM64 wheels replace the historical custom OpenMM/PDBFixer builds.
RUN pip install --no-cache-dir -c /opt/runtime-constraints.txt "openmm[cuda13]==8.6.1" pdbfixer==1.12.0

# -----------------------------------------------------------------------------
# OpenFold Installation
# -----------------------------------------------------------------------------
WORKDIR /opt/openfold

# Upstream setup.py already includes sm_120 for CUDA 13.
# Fix missing stereo_chemical_props.txt (required for relaxation)
# Install OpenFold (--no-build-isolation required for torch access)
# Install awscli for downloading model weights
# hadolint ignore=DL3013
RUN mkdir -p openfold/resources \
    && wget -q --progress=dot:giga https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
    -O openfold/resources/stereo_chemical_props.txt \
    && echo "24510899eeb49167cffedec8fa45363a4d08279c0c637a403b452f7d0ac09451  openfold/resources/stereo_chemical_props.txt" | sha256sum --check - \
    && pip install --no-cache-dir -c /opt/runtime-constraints.txt --no-build-isolation . \
    && pip install --no-cache-dir -c /opt/runtime-constraints.txt awscli

# -----------------------------------------------------------------------------
# Code Fixes for CUDA 13 Compatibility
# -----------------------------------------------------------------------------
# Fix cuda-python 13.x import structure change (cuda.cudart -> cuda.bindings.runtime)
# Pre-create cache directories to silence Triton warnings
RUN sed -i 's/import cuda.cudart as cudart/from cuda.bindings import runtime as cudart/g' /opt/openfold/openfold/utils/tensorrt_lazy_compiler.py \
    && mkdir -p /root/.triton/autotune

# -----------------------------------------------------------------------------
# Model Weights
# -----------------------------------------------------------------------------
# OPTIONAL: Comment out the following lines to build a lighter image without embedded weights.
# You will need to mount weights at runtime if you skip this.
# Download OpenFold model parameters from AWS S3 (public bucket, no auth required)
# Install fair-esm (required for SoloSeq) and download SoloSeq parameters
# Pre-download the ESM-1b checkpoint (~7.83 GB on disk).
# hadolint ignore=DL3013
RUN bash /opt/openfold/scripts/download_openfold_params.sh /opt/openfold \
    && pip install --no-cache-dir -c /opt/runtime-constraints.txt fair-esm \
    && bash /opt/openfold/scripts/download_openfold_soloseq_params.sh /opt/openfold \
    && python3 -c "import esm; esm.pretrained.esm1b_t33_650M_UR50S()"

# Freeze the ESM code too: upstream precompute_embeddings uses torch.hub's main
# cache. Populating it makes the documented SoloSeq workflow work offline.
RUN git clone https://github.com/facebookresearch/esm.git /root/.cache/torch/hub/facebookresearch_esm_main \
    && git -C /root/.cache/torch/hub/facebookresearch_esm_main checkout --detach 2b369911bb5b4b0dda914521b9475cad1656b2ac
COPY weights.sha256 /opt/weights.sha256
RUN sha256sum --check /opt/weights.sha256

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
COPY example-templates.sha256 /opt/example-templates.sha256
RUN bash /opt/openfold/scripts/download_example_templates.sh \
    && sha256sum --check /opt/example-templates.sha256

# -----------------------------------------------------------------------------
# Validation & Runtime
# -----------------------------------------------------------------------------
# Create a test to verify all imports work
RUN python3 -c "import openfold; import torch; import openmm; import pdbfixer; from openfold.np.relax import relax; print(f'OpenFold on PyTorch {torch.__version__}, CUDA {torch.version.cuda}, OpenMM {openmm.version.short_version}')"

WORKDIR /opt/openfold
CMD ["/bin/bash"]
