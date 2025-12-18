# OpenFold for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository provides a specialized Docker deployment for running **OpenFold** (AlphaFold2) on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

> **Looking for OpenFold3?** See [openfold3-DGX-Spark](https://github.com/adriancarr/openfold3-DGX-Spark)

## Why This Exists

The official OpenFold Docker images are built for x86_64 architecture. DGX Spark uses ARM64 and Blackwell GPUs (sm_121), requiring custom builds with specific compatibility fixes.

This build solves:
- **CUDA Kernel Compatibility**: Compiles `attn_core_inplace_cuda` kernel for sm_120 (Blackwell)
- **DeepSpeed Fixes**: Patches DeepSpeed to correctly parse sm_121 architecture
- **Triton Compatibility**: Uses triton-nightly for Blackwell kernel support
- **ARM64 Dependencies**: Installs OpenMM/pdbfixer via conda-forge for relaxation support

## Quick Start

### 1. Build the Docker Image

```bash
# Clone this repository
git clone https://github.com/adriancarr/openfold-DGX-Spark.git
cd openfold-DGX-Spark

# Build (takes ~7-8 minutes with cached layers)
docker build -t openfold-spark:latest .
```

### 2. Run Inference

```bash
docker run --gpus all --ipc=host --shm-size=64g \
    -v $(pwd)/output:/output \
    openfold-spark:latest \
    python3 run_pretrained_openfold.py \
    /opt/openfold/examples/monomer/fasta_dir \
    /path/to/template_mmcif \
    --output_dir /output \
    --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
    --config_preset model_1_ptm \
    --skip_relaxation \
    --use_precomputed_alignments /opt/openfold/examples/monomer/alignments \
    --model_device cuda:0
```

### 3. Verification

If successful, you should see output like:
```text
INFO:...Running inference for 6KWC_1...
INFO:...Inference time: 35.03052546199979
INFO:...Output written to /output/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb...
```

## Benchmark Results

Benchmarks run on **NVIDIA DGX Spark** (Grace Blackwell GB10, 20 CPU cores, 119GB RAM).

*Benchmark date: 2025-12-18*

| Example | Residues | Inference Time | Total Time |
|---------|----------|----------------|------------|
| 6KWC (monomer) | 185 | **35 seconds** | 44 seconds |

> **Note**: Total time includes Docker container startup, model loading, and template downloading. For batch processing, consider keeping the container running to amortize startup costs.

## Repository Structure

- `Dockerfile`: The complete build recipe with all Blackwell fixes
- `patch_ds.py`: DeepSpeed patch for sm_121 → sm_120 mapping
- `README.md`: This file

## Technical Details

- **Base Image**: `nvcr.io/nvidia/pytorch:25.01-py3`
- **CUDA Version**: 12.8
- **DeepSpeed**: 0.15.4 (pinned, with sm_121 patch)
- **Model Weights**: 10 checkpoints embedded (~3.5 GB)
- **Triton**: Nightly build for sm_121 support
- **OpenMM**: 8.4.0 (via conda-forge, for relaxation)

### Key Fixes Applied

| Fix | Purpose |
|-----|---------|
| `patch_ds.py` | Maps sm_121 → sm_120 for DeepSpeed JIT compilation |
| Triton nightly | Provides sm_121 kernel support |
| CUTLASS v3.6.0 | Required for DeepSpeed evoformer attention |
| setup.py patch | Adds (12, 0) to compute_capabilities for `attn_core_inplace_cuda` |
| conda openmm | ARM64-compatible OpenMM/pdbfixer installation |

## Requirements

- **Hardware**: NVIDIA DGX Spark (Grace Blackwell / GB10)
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Disk Space**: ~30GB for the Docker image
- **Memory**: 64GB+ recommended (set via `--shm-size`)

## Troubleshooting

### "no kernel image is available for execution"
This error occurs when CUDA kernels aren't compiled for Blackwell (sm_120/121). Make sure you're using the pre-built image from this repository.

### "Could not find CIFs in..."
OpenFold requires template mmCIF files for inference. Download them from RCSB or use `--use_single_seq_mode` for template-free inference (with appropriate model weights).

### Warning about GB10 GPU
The message "WARNING: Detected NVIDIA GB10 GPU, which may not yet be supported" is informational and can be safely ignored.

## Resources

- [OpenFold GitHub](https://github.com/aqlaboratory/openfold)
- [OpenFold Documentation](https://openfold.readthedocs.io/)
- [NVIDIA NGC PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

## Credits

- [OpenFold](https://github.com/aqlaboratory/openfold) by AlQuraishi Lab

## License

Apache 2.0 (same as OpenFold)
