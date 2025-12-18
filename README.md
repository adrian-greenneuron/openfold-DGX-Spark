# OpenFold for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository provides a specialized Docker deployment for running **OpenFold** (AlphaFold2) on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

> **Looking for OpenFold3?** See [openfold3-DGX-Spark](https://github.com/adriancarr/openfold3-DGX-Spark)

## Why This Exists

The official OpenFold Docker images are built for x86_64 architecture. DGX Spark uses ARM64 and Blackwell GPUs (sm_121), requiring custom builds with specific compatibility fixes.

This build solves:
- **CUDA Kernel Compatibility**: Compiles `attn_core_inplace_cuda` kernel for sm_120 (Blackwell)
- **DeepSpeed Fixes**: Patches DeepSpeed to correctly parse sm_121 architecture
- **Triton Compatibility**: Uses triton-nightly for Blackwell kernel support
- **OpenMM Source Build**: Builds OpenMM and PDBFixer from source for full Blackwell GPU relaxation support

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
    /opt/openfold/examples/monomer/template_mmcif \
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

### Comparative Performance (Standard vs. SoloSeq)

| Example | Mode | Relaxed? | Inference (s) | Relaxation (s) | Embedding (s) | Total (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Short (2Q2K, 35 res)** | Standard | Yes | 9.74 | 30.98 | - | 50.5 |
| Short | Standard | No | 9.66 | - | - | 19.2 |
| **Short** | **SoloSeq** | **Yes** | **1.06** | **4.62** | **10.8** | **24.5** |
| Short | SoloSeq | No | 1.07 | - | 11.3 | 20.3 |
| **Medium (6KWC, 152 res)** | Standard | Yes | 35.70 | 5.59 | - | 51.2 |
| Medium | Standard | No | 35.19 | - | - | 45.1 |
| **Medium** | **SoloSeq** | **Yes** | **6.59** | **5.60** | **10.7** | **31.0** |
| Medium | SoloSeq | No | 6.59 | - | 11.2 | 26.1 |

**Key Observations:**
- **SoloSeq is 5-9x faster** for inference (1s vs 10s for Short, 6.5s vs 35s for Medium)
- **Relaxation adds ~5-30s** depending on protein size and template complexity
- **ESM embedding generation** takes ~11s (one-time cost, can be batched)

---

### Run in SoloSeq Mode
OpenFold supports a "SoloSeq" mode using **ESM-1b embeddings**, offering a faster alternative that skips MSA generation. The ESM-1b model (~2.5GB) is now **baked into the Docker image**, so no internet is required at runtime.

To run SoloSeq, use the **Split Workflow**:

1.  **Generate Embeddings** (uses `precompute_embeddings.py`):
    ```bash
    # Prepare input
    # mkdir -p my_fasta && echo ">1UBQ" > my_fasta/ubiquitin.fasta ...

    docker run --gpus all --ipc=host --shm-size=64g \
        -v $(pwd)/my_fasta:/fasta_dir \
        -v $(pwd)/embeddings:/embeddings \
        openfold-dgx-spark:latest \
        python3 /opt/openfold/scripts/precompute_embeddings.py \
        /fasta_dir \
        /embeddings
    ```

    > **Tip**: Ensure your FASTA header matches the filename (e.g., `>1UBQ` inside `1UBQ.fasta`) to align with embedding output.

2.  **Run Inference** (uses local embeddings):
    ```bash
    docker run --gpus all --ipc=host --shm-size=64g \
        -v $(pwd)/my_fasta:/fasta_dir \
        -v $(pwd)/embeddings:/embeddings \
        -v $(pwd)/output:/output \
        openfold-dgx-spark:latest \
        python3 run_pretrained_openfold.py \
        /fasta_dir \
        /opt/openfold/examples/monomer/template_mmcif \
        --output_dir /output \
        --config_preset seq_model_esm1b_ptm \
        --openfold_checkpoint_path /opt/openfold/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
        --use_precomputed_alignments /embeddings \
        --model_device cuda:0
    ```


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
- **OpenMM**: Latest Source (Master) (Built for Blackwell relaxation)

### Key Fixes Applied

| Fix | Purpose |
|-----|---------|
| `patch_ds.py` | Maps sm_121 → sm_120 for DeepSpeed JIT compilation |
| Triton nightly | Provides sm_121 kernel support |
| CUTLASS v3.6.0 | Required for DeepSpeed evoformer attention |
| setup.py patch | Adds (12, 0) to compute_capabilities for `attn_core_inplace_cuda` |
| Source OpenMM | Replaces Conda install with source build for Blackwell/ARM64 compatibility |

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
