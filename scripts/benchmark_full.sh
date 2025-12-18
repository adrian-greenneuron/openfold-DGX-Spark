#!/bin/bash
set -e

# Setup directories
WORK_DIR="/tmp/bench_clean"
# Use docker to clean to avoid permission issues
docker run --rm -v /tmp:/tmp_mount alpine rm -rf /tmp_mount/bench_clean
mkdir -p "$WORK_DIR/fasta_short" "$WORK_DIR/fasta_medium" "$WORK_DIR/output" "$WORK_DIR/embeddings"

echo "Extracting inputs from image..."
# Use a temporary container to extract files
id=$(docker create openfold-dgx-spark:latest)
docker cp $id:/opt/openfold/tests/test_data/short.fasta "$WORK_DIR/fasta_short/2Q2K.fasta"
docker cp $id:/opt/openfold/examples/monomer/fasta_dir/6kwc.fasta "$WORK_DIR/fasta_medium/6KWC_1.fasta"
docker rm -v $id

# Correct headers
# Short Fasta originally >query. We rename to >2q2k_A to match alignment dir.
sed -i 's/>query/>2q2k_A/g' "$WORK_DIR/fasta_short/2Q2K.fasta"
# 6KWC_1.fasta typically has correct header

echo "=================================================="
echo "Starting Benchmarks"
echo "=================================================="

run_bench() {
    NAME=$1
    FASTA_DIR=$2
    MODE=$3 # STD or SOLO
    RELAX_SETTING=$4 # "RELAX" or "NORELAX"

    echo "--------------------------------------------------"
    echo "Benchmarking $NAME ($MODE Mode) - $RELAX_SETTING"
    
    OUT_DIR="$WORK_DIR/output/${NAME}_${MODE}_${RELAX_SETTING}"
    mkdir -p "$OUT_DIR"
    
    RELAX_ARG=""
    if [ "$RELAX_SETTING" == "NORELAX" ]; then
        RELAX_ARG="--skip_relaxation"
    fi

    if [ "$MODE" == "SOLO" ]; then
        # 1. Embeddings
        echo "Generating Embeddings..."
        # Clean specific embedding (use docker to avoid permission issues)
        docker run --rm -v "$WORK_DIR/embeddings":/clean alpine sh -c "rm -rf /clean/*"
        
        start_emb=$(date +%s.%N)
        docker run --gpus all --ipc=host --shm-size=64g \
            -v "$FASTA_DIR":/fasta_dir \
            -v "$WORK_DIR/embeddings":/embeddings \
            openfold-dgx-spark:latest \
            python3 /opt/openfold/scripts/precompute_embeddings.py \
            /fasta_dir /embeddings > /dev/null 2>&1
        end_emb=$(date +%s.%N)
        emb_time=$(echo "$end_emb - $start_emb" | bc)
        echo "Embedding Time: $emb_time s"

        # Correct embedding dir name if needed
        # precompute_embeddings uses filename (2Q2K) but inference uses header (2q2k_A)
        # Note: only for Short, Medium is 6KWC.
        if [ "$NAME" == "Short" ] && [ -d "$WORK_DIR/embeddings/2Q2K" ]; then
             # Rename using docker to avoid permission issues
             docker run --rm -v "$WORK_DIR/embeddings":/clean alpine sh -c "mv /clean/2Q2K /clean/2q2k_A && if [ -f /clean/2q2k_A/2Q2K.pt ]; then mv /clean/2q2k_A/2Q2K.pt /clean/2q2k_A/2q2k_A.pt; fi"
        fi

        # 2. Inference
        echo "Running Inference..."
        start_inf=$(date +%s.%N)
        output=$(docker run --gpus all --ipc=host --shm-size=64g \
            -v "$FASTA_DIR":/fasta_dir \
            -v "$WORK_DIR/embeddings":/embeddings \
            -v "$OUT_DIR":/output \
            openfold-dgx-spark:latest \
            python3 run_pretrained_openfold.py \
            /fasta_dir \
            /opt/openfold/examples/monomer/template_mmcif \
            --output_dir /output \
            --config_preset seq_model_esm1b_ptm \
            --openfold_checkpoint_path /opt/openfold/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
            --use_precomputed_alignments /embeddings \
            --model_device cuda:0 $RELAX_ARG 2>&1) || { echo "Command Failed!"; echo "$output"; exit 1; }
        end_inf=$(date +%s.%N)
        total_wall=$(echo "$end_inf - $start_inf" | bc)
        
        # Parse output for Inference/Relaxation
        infer_time=$(echo "$output" | grep "Inference time:" | tail -n1 | awk '{print $NF}')
        relax_time=$(echo "$output" | grep "Relaxation time:" | tail -n1 | awk '{print $NF}')
        
        echo "Raw Output Snippet:"
        echo "$output" | grep -E "Inference time|Relaxation time"

        # SoloSeq Total = Embedding + Wall Time of Inference (approx)
        full_total=$(echo "$emb_time + $total_wall" | bc)
        echo "RESULT: $NAME | $MODE | Infer: $infer_time | Relax: $relax_time | Wall: $total_wall | Embed: $emb_time | Total(Emb+Wall): $full_total"

    else
        # STD Mode
        echo "Running Standard Inference..."
        # We need a dummy alignment dir? Standard mode usually runs alignments?
        # Wait, if we use --use_precomputed_alignments, we skip alignment step.
        # But for benchmark, do we want to simulate full run? Usually yes.
        # But '2Q2K' inputs?
        # Standard benchmark uses existing alignments if possible to isolate inference performance.
        # OpenFold examples come with 6KWC alignments. 
        # But Short (2Q2K) doesn't have checks.
        
        # If I run WITHOUT precomputed alignments, it will try to run Jackhmmer/HHblits.
        # I don't have large databases mounted (Uniref90/MGnify etc are HUGE).
        # So I CANNOT run full pipeline.
        # I MUST use precomputed alignments for Standard Benchmark too, or it fails.
        # For 6KWC, I have them in /opt/openfold/examples/monomer/alignments.
        # For Short (2Q2K), I DO NOT HAVE THEM.
        
        # Ah. Previous benchmark of Short 2Q2K? (Task 525 etc).
        # Task 525 (Medium) used `--use_precomputed_alignments`.
        # Task 542 (Short) used... wait.
        # Task 542 copied `short_alignments`?
        # "Re-staged the necessary input files for the Short protein."
        # I copied `tests/test_data/short_alignments`!
        
        # So I need to extract `tests/test_data/alignments` for Short.
        
        TEMPLATE_DIR="/opt/openfold/examples/monomer/template_mmcif"
        
        if [ "$NAME" == "Short" ]; then
             # Use standard templates (Rename logic below fixes alignment finding)
             ALIGN_ARG="--use_precomputed_alignments /alignments_short"
        elif [ "$NAME" == "Medium" ]; then
             ALIGN_ARG="--use_precomputed_alignments /opt/openfold/examples/monomer/alignments"
        fi

        start_std=$(date +%s.%N)
        
        # Run command
        # For Short: mount extracted alignments
        RUN_CMD="docker run --gpus all --ipc=host --shm-size=64g \
            -v $FASTA_DIR:/fasta_dir \
            -v $OUT_DIR:/output"
            
        if [ "$NAME" == "Short" ]; then
            RUN_CMD="$RUN_CMD -v $WORK_DIR/alignments_short:/alignments_short"
        fi
        
        RUN_CMD="$RUN_CMD openfold-dgx-spark:latest \
            python3 run_pretrained_openfold.py \
            /fasta_dir \
            /opt/openfold/examples/monomer/template_mmcif \
            --output_dir /output \
            --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
            --config_preset model_1_ptm \
            $ALIGN_ARG \
            --model_device cuda:0 $RELAX_ARG 2>&1"
            
        output=$(eval $RUN_CMD) || { echo "Command Failed!"; echo "$output"; exit 1; }
        end_std=$(date +%s.%N)
        total_time=$(echo "$end_std - $start_std" | bc)

        infer_time=$(echo "$output" | grep "Inference time:" | tail -n1 | awk '{print $NF}')
        relax_time=$(echo "$output" | grep "Relaxation time:" | tail -n1 | awk '{print $NF}')

        echo "Raw Output Snippet:"
        echo "$output" | grep -E "Inference time|Relaxation time"

        echo "RESULT: $NAME | $MODE | Infer: $infer_time | Relax: $relax_time | Total: $total_time"
    fi
}

# Extraction of Short Alignments
echo "Extracting Short Alignments..."
id=$(docker create openfold-dgx-spark:latest)
# The directory is tests/test_data/alignments
docker cp $id:/opt/openfold/tests/test_data/alignments "$WORK_DIR/alignments_short"
docker rm -v $id
# Note: tests/test_data/alignments contains 2q2k_A but missing pdb70_hits.hhr in subdir
# We fix this by copying from root if needed
if [ -f "$WORK_DIR/alignments_short/pdb70_hits.hhr" ]; then
    cp "$WORK_DIR/alignments_short/pdb70_hits.hhr" "$WORK_DIR/alignments_short/2q2k_A/"
fi

# CRITICAL FIX: Remove hmm_output.sto which creates hits with Sum_probs=None
# Mixing None and Float sum_probs causes TypeError in sort
# Keep original pdb70_hits.hhr (don't mock it)
if [ -f "$WORK_DIR/alignments_short/2q2k_A/hmm_output.sto" ]; then
    rm "$WORK_DIR/alignments_short/2q2k_A/hmm_output.sto"
fi

# Run Benchmarks
# Skip Short STD as it is unstable with default templates/alignments
run_bench "Short" "$WORK_DIR/fasta_short" "STD" "RELAX"
run_bench "Short" "$WORK_DIR/fasta_short" "STD" "NORELAX"

run_bench "Short" "$WORK_DIR/fasta_short" "SOLO" "RELAX"
run_bench "Short" "$WORK_DIR/fasta_short" "SOLO" "NORELAX"

run_bench "Medium" "$WORK_DIR/fasta_medium" "STD" "RELAX"
run_bench "Medium" "$WORK_DIR/fasta_medium" "STD" "NORELAX"

run_bench "Medium" "$WORK_DIR/fasta_medium" "SOLO" "RELAX"
run_bench "Medium" "$WORK_DIR/fasta_medium" "SOLO" "NORELAX"
