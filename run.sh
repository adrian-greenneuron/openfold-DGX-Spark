#!/usr/bin/env bash
# Forward to the upstream CLI with explicit host input/output/embedding paths.
set -euo pipefail
OF_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OF_IMAGE="${OF_IMAGE:-openfold-official-spark:2.2.0-251026}"
OF_INPUT_DIR="${OF_INPUT_DIR:-$OF_REPO_DIR/input}"
OF_OUTPUT_DIR="${OF_OUTPUT_DIR:-$OF_REPO_DIR/output}"
OF_EMBEDDINGS_DIR="${OF_EMBEDDINGS_DIR:-$OF_REPO_DIR/embeddings}"
if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: ./run.sh [predict|embed] [upstream CLI arguments]
No arguments: run the bundled Standard monomer example without relaxation.
embed: prepare embeddings from /input into /embeddings.
Host paths: OF_INPUT_DIR, OF_OUTPUT_DIR, OF_EMBEDDINGS_DIR.
Optional OF_WEIGHTS_DIR contains openfold_params/ and openfold_soloseq_params/.
Use OF_IMAGE for another image tag; OF_NETWORK=none requests offline operation.
HELP
  exit 0
fi
OF_MODE=predict
if [[ "${1:-}" == predict || "${1:-}" == embed ]]; then OF_MODE="$1"; shift; fi
mkdir -p "$OF_INPUT_DIR" "$OF_OUTPUT_DIR" "$OF_EMBEDDINGS_DIR"
OF_INPUT_DIR="$(cd -- "$OF_INPUT_DIR" && pwd)"
OF_OUTPUT_DIR="$(cd -- "$OF_OUTPUT_DIR" && pwd)"
OF_EMBEDDINGS_DIR="$(cd -- "$OF_EMBEDDINGS_DIR" && pwd)"
OF_DOCKER_ARGS=(--rm --gpus all --ipc=host --shm-size=64g --network "${OF_NETWORK:-bridge}"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --mount "type=bind,src=$OF_INPUT_DIR,dst=/input,readonly"
  --mount "type=bind,src=$OF_OUTPUT_DIR,dst=/output")
if [[ -n "${OF_WEIGHTS_DIR:-}" ]]; then
  OF_WEIGHTS_DIR="$(cd -- "$OF_WEIGHTS_DIR" && pwd)"
  for OF_KIND in openfold_params openfold_soloseq_params; do
    [[ -d "$OF_WEIGHTS_DIR/$OF_KIND" ]] || { echo "Missing $OF_WEIGHTS_DIR/$OF_KIND" >&2; exit 1; }
    OF_DOCKER_ARGS+=(--mount "type=bind,src=$OF_WEIGHTS_DIR/$OF_KIND,dst=/opt/openfold/$OF_KIND,readonly")
  done
fi
if [[ "$OF_MODE" == embed ]]; then
  OF_DOCKER_ARGS+=(--mount "type=bind,src=$OF_EMBEDDINGS_DIR,dst=/embeddings")
  if [[ $# == 0 ]]; then set -- /input /embeddings --use_local_esm /root/.cache/torch/hub/facebookresearch_esm_main; fi
  exec docker run "${OF_DOCKER_ARGS[@]}" "$OF_IMAGE" python3 /opt/openfold/scripts/precompute_embeddings.py "$@"
else
  OF_DOCKER_ARGS+=(--mount "type=bind,src=$OF_EMBEDDINGS_DIR,dst=/embeddings,readonly")
  if [[ $# == 0 ]]; then
    set -- /opt/openfold/examples/monomer/fasta_dir /opt/openfold/examples/monomer/template_mmcif \
      --output_dir /output --config_preset model_1_ptm \
      --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
      --use_precomputed_alignments /opt/openfold/examples/monomer/alignments \
      --data_random_seed 42 --model_device cuda:0 --skip_relaxation
  fi
  exec docker run "${OF_DOCKER_ARGS[@]}" "$OF_IMAGE" python3 /opt/openfold/run_pretrained_openfold.py "$@"
fi
