# OpenFold for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository provides a specialized Docker deployment for running **OpenFold** (AlphaFold2) on the NVIDIA DGX Spark system, powered by **Grace Blackwell (GB10)** GPUs and **ARM64** architecture.

> **Looking for OpenFold3?** See [openfold3-DGX-Spark](https://github.com/adriancarr/openfold3-DGX-Spark)

## Why This Exists

The official OpenFold Docker images are built for x86_64 architecture. DGX Spark uses ARM64 and Blackwell GPUs (sm_120/sm_121), requiring custom builds with specific compatibility fixes.

This build solves:
- **CUDA Kernel Compatibility**: Compiles `attn_core_inplace_cuda` kernel for sm_120 (Blackwell)
- **DeepSpeed Fixes**: Patches DeepSpeed to correctly parse sm_121 architecture
- **OpenMM Source Build**: Builds OpenMM and PDBFixer from source for full Blackwell GPU relaxation support
- **CUDA 13 Support**: Uses NGC 25.11 with native CUDA 13.0, Triton 3.5.0, and Flash Attention 2.7.4

## Quick Start

### 1. Build the Docker Image

```bash
# Clone this repository
git clone https://github.com/adriancarr/openfold-DGX-Spark.git
cd openfold-DGX-Spark

# Checkout CUDA 13 branch
git checkout CUDA13

# Build (takes ~25-30 minutes including model downloads)
docker build -t openfold-spark:cuda13 .
```

### 2. Run Inference

```bash
docker run --gpus all --ipc=host --shm-size=64g \
    -v $(pwd)/output:/output \
    openfold-spark:cuda13 \
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
INFO:...Inference time: 34.48...
INFO:...Output written to /output/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb...
```

## Benchmark Results

Benchmarks run on **NVIDIA DGX Spark** (Grace Blackwell GB10, 20 CPU cores, 119GB unified RAM).

*Benchmark date: 2025-12-20 | CUDA 13.0 | PyTorch 2.10*

### Performance with Memory Usage

| Example | Mode | Relaxed | Inference (s) | Relaxation (s) | Total (s) | Peak Memory (GB) | Memory Used (GB) |
|---------|------|---------|---------------|----------------|-----------|------------------|------------------|
| **Short (35 res)** | Standard | Yes | 9.90 | 19.28 | 39.4 | 10.9 | +5.6 |
| Short | Standard | No | 9.87 | - | 20.7 | 10.7 | +5.4 |
| **Short** | **SoloSeq** | **Yes** | **1.13** | **3.52** | **25.7** | 8.7 | +2.9 |
| Short | SoloSeq | No | 1.10 | - | 21.7 | 8.1 | +2.6 |
| **Medium (152 res)** | Standard | Yes | 34.49 | 4.54 | 49.2 | 14.8 | +9.4 |
| Medium | Standard | No | 34.57 | - | 45.7 | 14.5 | +9.1 |
| **Medium** | **SoloSeq** | **Yes** | **6.56** | **4.48** | **31.3** | 9.0 | +3.2 |
| Medium | SoloSeq | No | 6.58 | - | 26.3 | 8.6 | +3.1 |

> **Note:** ESM Embedding generation uses ~10 GB additional memory (peak ~15.4 GB)

### Memory Summary

| Metric | Standard Mode | SoloSeq Mode |
|--------|---------------|--------------|
| **Short protein memory** | 5.4-5.6 GB | 2.6-2.9 GB |
| **Medium protein memory** | 9.1-9.4 GB | 3.1-3.2 GB |
| **ESM embedding (one-time)** | - | ~10 GB |

**Key Observations:**
- **SoloSeq is 5-9x faster** for inference (1.1s vs 9.9s for Short, 6.6s vs 34.5s for Medium)
- **SoloSeq uses 50-70% less memory** than Standard mode
- **Relaxation adds ~4-19s** depending on protein size and template complexity
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
        openfold-spark:cuda13 \
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
        openfold-spark:cuda13 \
        python3 run_pretrained_openfold.py \
        /fasta_dir \
        /opt/openfold/examples/monomer/template_mmcif \
        --output_dir /output \
        --config_preset seq_model_esm1b_ptm \
        --openfold_checkpoint_path /opt/openfold/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
        --use_precomputed_alignments /embeddings \
        --model_device cuda:0
    ```

---

### Multimer Inference (Protein Complexes)

OpenFold supports **multimer inference** for predicting protein complex structures using AlphaFold-Multimer v2.3 weights.

> **Note**: Multimer benchmarks are not included in this repository because they require either:
> - Precomputed alignments for the target complex, OR
> - The full AlphaFold databases (~2.5TB) for alignment generation

To enable multimer support, uncomment the AlphaFold params download in the Dockerfile and rebuild:
```dockerfile
# In Dockerfile, uncomment:
RUN bash /opt/openfold/scripts/download_alphafold_params.sh /opt/openfold
```

Then run inference with `--config_preset model_1_multimer_v3`. See [OpenFold Multimer Docs](https://openfold.readthedocs.io/en/latest/Multimer_Inference.html).

---

### About Database Requirements

| Mode | Databases Required | Size |
|------|-------------------|------|
| **SoloSeq** (recommended) | ESM-1b model (baked into image) | ~2.5 GB |
| Standard with precomputed alignments | None (alignments provided) | 0 |
| Standard with alignment generation | UniRef90, MGnify, BFD, UniRef30, PDB70 | ~2.5 TB |
| Multimer with alignment generation | Above + UniProt, PDB SeqRes | ~2.5 TB |

> **Why we benchmark with SoloSeq and precomputed alignments**: The full alignment databases are massive (~2.5TB). This Docker image includes the ESM-1b model (~2.5GB) and example data with precomputed alignments, allowing you to run benchmarks immediately without downloading terabytes of data.

> **Note**: Total time includes Docker container startup, model loading, and template downloading. For batch processing, consider keeping the container running to amortize startup costs.

## Repository Structure

- `Dockerfile`: The complete build recipe with all Blackwell fixes
- `patch_ds.py`: DeepSpeed patch for sm_121 → sm_120 mapping
- `scripts/benchmark_full.sh`: Standard benchmark script
- `scripts/benchmark_with_memory.sh`: Benchmark script with memory tracking
- `README.md`: This file

## Technical Details

- **Base Image**: `nvcr.io/nvidia/pytorch:25.11-py3`
- **CUDA Version**: 13.0
- **PyTorch Version**: 2.10
- **Triton**: 3.5.0 (NGC built-in, native Blackwell support)
- **Flash Attention**: 2.7.4 (NGC built-in)
- **CUTLASS**: 4.0.0 (NGC built-in at `/opt/pytorch/ao/third_party/cutlass`)
- **DeepSpeed**: 0.15.4 (pinned, with sm_121 patch)
- **OpenMM**: Latest Source (Master) (Built for Blackwell relaxation)
- **Model Weights**: 10 checkpoints embedded (~3.5 GB)

### Key Fixes Applied

| Fix | Purpose |
|-----|---------|
| `patch_ds.py` | Maps sm_121 → sm_120 for DeepSpeed JIT compilation |
| cuda-python import fix | Updates `cuda.cudart` → `cuda.bindings.runtime` for CUDA 13 |
| CUTLASS_PATH | Points to NGC's built-in CUTLASS 4.0.0 |
| setup.py patch | Adds (12, 0) to compute_capabilities for `attn_core_inplace_cuda` |
| Source OpenMM | Replaces Conda install with source build for Blackwell/ARM64 compatibility |

### NGC 25.11 Built-in Components (No Installation Required)

The NGC 25.11 container includes these components pre-optimized for CUDA 13:
- **Flash Attention 2.7.4** - No pip install needed
- **Triton 3.5.0** - Native Blackwell support, no nightly required
- **CUTLASS 4.0.0** - Available at `/opt/pytorch/ao/third_party/cutlass`

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

### TF32 Deprecation Warning
The warning about `torch.backends.cuda.matmul.allow_tf32` deprecation is expected with PyTorch 2.10 and can be ignored.

## Resources

- [OpenFold GitHub](https://github.com/aqlaboratory/openfold)
- [OpenFold Documentation](https://openfold.readthedocs.io/)
- [NVIDIA NGC PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

## Credits

- [OpenFold](https://github.com/aqlaboratory/openfold) by AlQuraishi Lab

## License

Apache 2.0 (same as OpenFold)
