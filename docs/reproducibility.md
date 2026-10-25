# Building and verifying OpenFold on DGX Spark

The `main` branch preserves the original trained models and inference defaults.
The [official/openfold-spark branch](https://github.com/adrian-greenneuron/openfold-DGX-Spark/tree/official/openfold-spark)
tests a dependency-modernised deployment separately. Both use the same upstream
OpenFold source revision and checkpoints; the development branch is not a newer
trained model.

## Original build identities

| Component | Pinned identity |
|---|---|
| NVIDIA base | PyTorch 25.11, digest `417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21` |
| OpenFold | `be2ec1841f16c966c65ae0e7599ebbadc725757d` |
| OpenMM source | `07b738c560f31f539674528c3a8ca629fc8320ba` |
| PDBFixer source | `94cfa4c0ca551cdc5f13320f9a658efd59f2b881` |
| DeepSpeed | 0.15.4, retaining the original architecture-parsing patch |
| ESM source | `2b369911bb5b4b0dda914521b9475cad1656b2ac` |
| Python distributions | [runtime-constraints.txt](../runtime-constraints.txt) |
| Downloaded parameters and notices | [weights.sha256](../weights.sha256) |
| Example templates | [example-templates.sha256](../example-templates.sha256) |

The original OpenFold source has three deployment adaptations: architecture
selection in `setup.py`, a raw-regex warning fix, and the CUDA 13 Python import
change. The source hashes and the dependency versions were recovered from the
retained working image rather than inferred from the unpinned Dockerfile.

## Build from a fresh clone

```bash
git clone --branch main https://github.com/adrian-greenneuron/openfold-DGX-Spark.git
cd openfold-DGX-Spark
./build.sh --no-cache --test -t rebuild-check
```

Requirements are Docker, NVIDIA Container Toolkit, a working Spark driver and
internet access to the public image, source repositories, package repositories,
weights and templates. No previous OpenFold image, host checkpoint or compiled
kernel is required. Model files are downloaded during the build and checked by
SHA-256. The ESM code and weights are cached inside the image for offline use.

The tested original clean build took about **25 minutes**. Its 140 checked source
files, all 278 recorded distribution versions and 16 downloaded artifacts
matched the retained original. All eight benchmark configurations then ran
successfully, with one warm-up and two measured repetitions each. The modernised
build took about **22 minutes** and also completed all eight configurations.
Times depend on network and machine conditions.

Docker `--no-cache` reruns build steps; it can still reuse the pinned base image.
Ubuntu packages and isolated package-build dependencies remain external inputs.
This is a verified functional rebuild, not a promise of identical image bytes.
After the full builds, additional template/resource checksum assertions and
constraints for already-installed modern CUDA packages were added. The assertions
were checked against both built images; these additions do not change model
settings or binaries.

An expired saved NGC login caused one initial pull to fail. The public image was
then fetched using an isolated empty Docker authentication configuration, leaving
the normal credentials untouched. If you encounter that specific error, renew
your NGC credentials or use a separate Docker configuration for public pulls;
do not remove credentials for unrelated private registries.

## What matched, and what did not

The matched tests fixed FASTA sequences, alignments, templates, model weights,
precision and seeds, with the expandable CUDA allocator set before initialisation.
The actual example inputs contain **73 and 191 residues**. Earlier README labels
of 35 and 152 were incorrect.

All 16 measured unrelaxed PDB outputs in each candidate matched the corresponding
original output byte-for-byte. All 12 separately checked ESM embedding tensors
also matched exactly. This establishes agreement for these fixtures, not general
prediction accuracy or repeatability across hardware/software versions.

GPU relaxation is different: repeated runs in the **original image itself** gave
different relaxed coordinates and timings. For the low-confidence short Standard
prediction (mean pLDDT about 41.6), two relaxed structures differed by about 5.18 Å
Cα RMSD despite identical unrelaxed input. Other cases had much smaller changes.
PDBFixer preparation includes a zero seed, which OpenMM treats as selecting a new
seed; GPU force accumulation and minimisation are additional potential sources
of variability. A focused check fixing preparation seeds and enabling OpenMM
deterministic forces still did not make relaxation byte-repeatable. These checks
do not establish the complete internal cause.

Consequently, an exact rebuild check should compare **unrelaxed** structures.
Relaxed structures require separately reported numerical comparisons. Do not
attribute all differences to a rebuild or interpret two relaxation timings as
proof of a general performance change. Normal inference defaults remain intact.

## Opt-in deterministic model-forward verification

The wrapper fixes Python/NumPy/PyTorch seeds and requests deterministic PyTorch
operations. It deliberately requires `--skip_relaxation`. It applies only to the
verification process, not ordinary OpenFold calls.

```bash
OF_VERIFY_IMAGE="${OF_VERIFY_IMAGE:-openfold-official-spark:2.2.0-251026}"
mkdir -p verification/run1 verification/run2
for RUN in run1 run2; do
  docker run --rm --network none --gpus all --ipc=host --shm-size=64g \
    -e PYTHONHASHSEED=0 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --mount "type=bind,src=$PWD/tools,dst=/verification,readonly" \
    --mount "type=bind,src=$PWD/verification/$RUN,dst=/output" \
    "$OF_VERIFY_IMAGE" \
    python3 /verification/deterministic_inference.py \
    /opt/openfold/examples/monomer/fasta_dir \
    /opt/openfold/examples/monomer/template_mmcif \
    --output_dir /output --config_preset model_1_ptm \
    --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
    --use_precomputed_alignments /opt/openfold/examples/monomer/alignments \
    --data_random_seed 42 --model_device cuda:0 --skip_relaxation
done
cmp verification/run1/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb \
    verification/run2/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb
```

`cmp` exits zero for identical files and fails for missing or differing files.
Use fresh output directories and also inspect the command exit status and
successful prediction log. Comparing empty directories is not a valid check.
The focused wrapper test ran the short example twice in all three images; all
six unrelaxed files were identical. The command above uses the bundled medium
example, whose normal seeded predictions also matched across the images in the
eight-case study. Unsupported deterministic operations should raise errors;
exactness on arbitrary inputs, alternative backends or other GPUs is unverified.

## Development branch changes

The development deployment replaces the custom OpenMM build with published
`openmm[cuda13]==8.6.1` ARM64 wheels, installs released PDBFixer 1.12.0, and uses
DeepSpeed 0.19.8 without the old builder patch. The original redundant architecture
patch is removed because this upstream setup already includes sm_120 with CUDA 13.
The cosmetic regex patch is also removed. The CUDA import adaptation remains
necessary for the pinned CUDA Python interface.

The NVIDIA base, PyTorch, model source, weights and preset settings stay matched.
DeepSpeed is installed but its optional attention flag was not enabled by the
original benchmark commands, so that backend is not qualified by these results.
The official OpenMM installation test passed for Reference, CPU and CUDA forces.
The upstream OpenFold Dockerfile still contains x86_64 installation steps; this
branch is a Spark recipe using upstream source and published dependencies, not an
unmodified upstream Docker image.

See the [full tables and limitations](openfold_benchmarks_251026.md) and
[machine-readable records](data/). No existing images, containers or job data were
deleted during this investigation.
