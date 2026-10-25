# OpenFold on DGX Spark: modernised dependencies

This development branch uses official [OpenFold source](https://github.com/aqlaboratory/openfold)
and published dependencies in place of several historical compatibility workarounds.
The [main branch](https://github.com/adrian-greenneuron/openfold-DGX-Spark/tree/main)
retains the original deployment as the default.

**The trained models and upstream source revision are the same as the original.**
Current upstream is still at the revision used by the retained image. This is a
Spark deployment recipe, not the upstream project's own Docker image.

## What changed

| Component | Original | This branch |
|---|---|---|
| OpenFold source | `be2ec1841f16c966c65ae0e7599ebbadc725757d` | Same |
| NVIDIA base / PyTorch / CUDA | NGC 25.11 / PyTorch 2.10 / CUDA 13.0 | Same pinned base |
| Weights and model presets | Standard and SoloSeq | Same, checksum-verified |
| OpenMM | Custom source build at `07b738c` | Published 8.6.1 ARM64 CUDA 13 wheels |
| PDBFixer | Git source at `94cfa4c` | Published 1.12.0 package |
| DeepSpeed | 0.15.4 with architecture patch | 0.19.8 without the patch |
| OpenFold architecture patch | Present | Removed; upstream handles sm_120 with CUDA 13 |
| Cosmetic regex patch | Present | Removed |
| CUDA Python import adaptation | Present | Still required |

Other application versions remain pinned. Optional DeepSpeed attention,
cuEquivariance and TensorRT acceleration are not enabled by this launcher and
were not evaluated. Most measured cases performed similarly; this branch does
not claim a general speed improvement. GPU relaxation was variable.

## Build from scratch

Requires DGX Spark, Docker, NVIDIA Container Toolkit, a compatible driver and
internet access for public inputs. No old image, host weights or compiled kernels
are required.

```bash
git clone --branch official/openfold-spark --single-branch \
  https://github.com/adrian-greenneuron/openfold-DGX-Spark.git
cd openfold-DGX-Spark
./build.sh --no-cache --test
```

The default image is `openfold-official-spark:2.2.0-251026`. The build downloads and
checks Standard/SoloSeq weights, ESM-1b and example templates. The ESM checkpoint
alone is about 7.83 GB on disk. The tested clean build took about 22 minutes;
network and build times vary.

## Run the example

```bash
./run.sh
```

This predicts the bundled 191-residue example using precomputed alignments, seed
42 and no relaxation. Results go to `output/`. Set `OF_NETWORK=none` for offline
operation. Custom calls forward arguments to the upstream CLI:

```bash
OF_INPUT_DIR=/path/to/inputs OF_OUTPUT_DIR=/path/to/results \
./run.sh predict /input /opt/openfold/examples/monomer/template_mmcif \
  --output_dir /output --config_preset model_1_ptm \
  --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
  --use_precomputed_alignments /input/alignments --model_device cuda:0
```

CLI paths use container locations. Inputs are read-only, outputs writable; new
files may be owned by root. `OF_IMAGE` selects another image tag. Optional
`OF_WEIGHTS_DIR` contains `openfold_params/` and `openfold_soloseq_params/`, mounted
read-only at their normal locations. Embedded parameters are used by default.

## SoloSeq

Place FASTA files under `input/`, with matching filenames and FASTA identifiers:

```bash
./run.sh embed
./run.sh predict /input /opt/openfold/examples/monomer/template_mmcif \
  --output_dir /output --config_preset seq_model_esm1b_ptm \
  --openfold_checkpoint_path /opt/openfold/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
  --use_precomputed_alignments /embeddings --model_device cuda:0 --skip_relaxation
```

Embeddings go to `embeddings/`; `OF_EMBEDDINGS_DIR` selects another directory.
Standard and SoloSeq use different models: their speed difference does not
establish equal accuracy. No large sequence database is needed for these examples.

## Verification and results

All eight cases ran successfully in the original, rebuilt and modernised images,
with one warm-up and two measured repeats per case. All measured unrelaxed PDB
outputs and checked embedding tensors matched exactly. The official OpenMM
installation test passed on Reference, CPU and CUDA platforms.

**GPU relaxation is not guaranteed repeatable**, including in the original image.
The low-confidence short Standard example showed large coordinate and timing
variability. Fixed preparation seeds and deterministic GPU forces did not remove
it. The opt-in verification procedure explicitly skips relaxation; ordinary
inference defaults remain unchanged.

See the [complete runtime/memory tables](docs/openfold_benchmarks_251026.md),
[rebuild and repeatability guide](docs/reproducibility.md), and
[compact records](docs/data/). Other inputs/backends, biological accuracy and
maximum structure size fitting in memory remain untested.
