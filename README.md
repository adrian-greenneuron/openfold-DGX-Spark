# OpenFold for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository contains Dockerfiles optimized for building and running OpenFold on NVIDIA DGX Spark systems, which feature ARM64 architecture (Grace CPU) and Blackwell GPUs (GB10).

## Why This Exists

The official OpenFold Docker images are built for x86_64 architecture. DGX Spark uses ARM64, requiring custom builds with specific optimizations to:

1. **Prevent memory exhaustion** - Default builds spawn too many compilation threads
2. **Target correct GPU architecture** - Blackwell GPUs use `sm_121` compute capability
3. **Handle ARM-specific dependencies** - Some x86-only optimizations must be disabled

## Quick Start

### OpenFold (AlphaFold2-based)

```bash
# Clone OpenFold source
git clone https://github.com/aqlaboratory/openfold.git
cd openfold

# Copy the DGX Spark Dockerfile
curl -O https://raw.githubusercontent.com/adriancarr/openfold-DGX-Spark/main/openfold/Dockerfile.spark

# Build (takes ~45 minutes due to Flash Attention compilation)
docker build -t openfold-spark:cuda13 -f Dockerfile.spark .

# Test
docker run --gpus all --rm openfold-spark:cuda13 python3 -c \
  "import openfold; print('OpenFold ready')"
```

### OpenFold3 (AlphaFold3-based)

```bash
# Clone OpenFold3 source
git clone https://github.com/aqlaboratory/openfold-3.git
cd openfold-3

# Copy the DGX Spark Dockerfile
curl -O https://raw.githubusercontent.com/adriancarr/openfold-DGX-Spark/main/openfold3/Dockerfile.spark
mv Dockerfile.spark docker/

# Build (takes ~15 minutes)
docker build --no-cache -t openfold3-spark:cuda13 -f docker/Dockerfile.spark .

# Test
docker run --gpus all --rm openfold3-spark:cuda13 python3 -c \
  "import openfold3; print('OpenFold3 ready')"
```

## Key Optimizations

| Setting | Value | Reason |
|---------|-------|--------|
| `MAX_JOBS` | 4 | Limits parallel compilation to prevent OOM |
| `TORCH_CUDA_ARCH_LIST` | `9.0;12.1` | Targets Hopper/Blackwell only |
| `DS_BUILD_CPU_ADAM` | 0 | Disables x86-only Intel optimizations |
| `DS_BUILD_CCL_COMM` | 0 | Disables Intel oneCCL (x86-only) |

## System Requirements

- NVIDIA DGX Spark (Grace Blackwell / GB10)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for build
- ~60GB RAM during Flash Attention compilation (OpenFold only)

## Directory Structure

```
├── openfold/
│   └── Dockerfile.spark     # OpenFold (AlphaFold2-based) ARM64 build
├── openfold3/
│   └── Dockerfile.spark     # OpenFold3 (AlphaFold3-based) ARM64 build
└── README.md
```

## Notes

- PyTorch Nightly with CUDA 13.0 is used for Blackwell GPU support
- A warning about compute capability 12.1 may appear - this is expected and can be ignored
- OpenFold3 requires `--no-cache` on first build to ensure correct PyTorch version

## Credits

- [OpenFold](https://github.com/aqlaboratory/openfold) by AlQuraishi Lab
- [OpenFold3](https://github.com/aqlaboratory/openfold-3) by OpenFold Consortium
- Build optimizations developed for DGX Spark deployment

## License

Apache 2.0 (same as OpenFold)
