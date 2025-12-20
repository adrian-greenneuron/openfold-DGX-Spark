#!/bin/bash
set -e

# Setup directories
WORK_DIR="/tmp/bench_mem"
docker run --rm -v /tmp:/tmp_mount alpine rm -rf /tmp_mount/bench_mem 2>/dev/null || true
mkdir -p "$WORK_DIR/fasta_short" "$WORK_DIR/fasta_medium" "$WORK_DIR/output" "$WORK_DIR/embeddings"

echo "Extracting inputs from image..."
id=$(docker create openfold-spark:cuda13)
docker cp $id:/opt/openfold/tests/test_data/short.fasta "$WORK_DIR/fasta_short/2Q2K.fasta"
docker cp $id:/opt/openfold/examples/monomer/fasta_dir/6kwc.fasta "$WORK_DIR/fasta_medium/6KWC_1.fasta"
docker rm -v $id

# Correct headers
sed -i 's/>query/>2q2k_A/g' "$WORK_DIR/fasta_short/2Q2K.fasta"

echo "=================================================="
echo "Starting Benchmarks with Memory Tracking"
echo "=================================================="

# Function to get current memory usage in MB
get_memory_mb() {
    free -m | awk '/^Mem:/ {print $3}'
}

# Function to monitor peak memory during a command
monitor_peak_memory() {
    local peak=0
    local current=0
    while kill -0 $1 2>/dev/null; do
        current=$(get_memory_mb)
        if [ "$current" -gt "$peak" ]; then
            peak=$current
        fi
        sleep 0.5
    done
    echo $peak
}

run_bench() {
    NAME=$1
    FASTA_DIR=$2
    MODE=$3
    RELAX_SETTING=$4

    echo "--------------------------------------------------"
    echo "Benchmarking $NAME ($MODE Mode) - $RELAX_SETTING"
    
    OUT_DIR="$WORK_DIR/output/${NAME}_${MODE}_${RELAX_SETTING}"
    mkdir -p "$OUT_DIR"
    
    RELAX_ARG=""
    if [ "$RELAX_SETTING" == "NORELAX" ]; then
        RELAX_ARG="--skip_relaxation"
    fi

    # Get baseline memory
    BASELINE_MEM=$(get_memory_mb)
    echo "Baseline Memory: ${BASELINE_MEM} MB"

    if [ "$MODE" == "SOLO" ]; then
        # Clean embeddings
        docker run --rm -v "$WORK_DIR/embeddings":/clean alpine sh -c "rm -rf /clean/*"
        
        # 1. Embeddings with memory tracking
        echo "Generating Embeddings..."
        start_emb=$(date +%s.%N)
        docker run --gpus all --ipc=host --shm-size=64g \
            -v "$FASTA_DIR":/fasta_dir \
            -v "$WORK_DIR/embeddings":/embeddings \
            openfold-spark:cuda13 \
            python3 /opt/openfold/scripts/precompute_embeddings.py \
            /fasta_dir /embeddings > /dev/null 2>&1 &
        PID=$!
        PEAK_EMB=$(monitor_peak_memory $PID)
        wait $PID
        end_emb=$(date +%s.%N)
        emb_time=$(echo "$end_emb - $start_emb" | bc)
        EMB_USED=$((PEAK_EMB - BASELINE_MEM))
        echo "Embedding Time: $emb_time s | Peak Memory: ${PEAK_EMB} MB (+${EMB_USED} MB)"

        # Fix embedding names for Short
        if [ "$NAME" == "Short" ] && [ -d "$WORK_DIR/embeddings/2Q2K" ]; then
             docker run --rm -v "$WORK_DIR/embeddings":/clean alpine sh -c "mv /clean/2Q2K /clean/2q2k_A && if [ -f /clean/2q2k_A/2Q2K.pt ]; then mv /clean/2q2k_A/2Q2K.pt /clean/2q2k_A/2q2k_A.pt; fi"
        fi

        # 2. Inference with memory tracking
        echo "Running Inference..."
        BASELINE_MEM=$(get_memory_mb)
        start_inf=$(date +%s.%N)
        docker run --gpus all --ipc=host --shm-size=64g \
            -v "$FASTA_DIR":/fasta_dir \
            -v "$WORK_DIR/embeddings":/embeddings \
            -v "$OUT_DIR":/output \
            openfold-spark:cuda13 \
            python3 run_pretrained_openfold.py \
            /fasta_dir \
            /opt/openfold/examples/monomer/template_mmcif \
            --output_dir /output \
            --config_preset seq_model_esm1b_ptm \
            --openfold_checkpoint_path /opt/openfold/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
            --use_precomputed_alignments /embeddings \
            --model_device cuda:0 $RELAX_ARG > /tmp/bench_output.txt 2>&1 &
        PID=$!
        PEAK_INF=$(monitor_peak_memory $PID)
        wait $PID || { echo "Command Failed!"; cat /tmp/bench_output.txt; exit 1; }
        end_inf=$(date +%s.%N)
        total_wall=$(echo "$end_inf - $start_inf" | bc)
        INF_USED=$((PEAK_INF - BASELINE_MEM))
        
        # Parse output
        infer_time=$(grep "Inference time:" /tmp/bench_output.txt | tail -n1 | awk '{print $NF}')
        relax_time=$(grep "Relaxation time:" /tmp/bench_output.txt | tail -n1 | awk '{print $NF}')
        
        full_total=$(echo "$emb_time + $total_wall" | bc)
        echo "RESULT: $NAME | $MODE | $RELAX_SETTING | Infer: ${infer_time:-N/A} | Relax: ${relax_time:-N/A} | Total: $full_total | Peak Mem: ${PEAK_INF} MB (+${INF_USED} MB)"

    else
        # STD Mode
        echo "Running Standard Inference..."
        
        TEMPLATE_DIR="/opt/openfold/examples/monomer/template_mmcif"
        
        if [ "$NAME" == "Short" ]; then
             ALIGN_ARG="--use_precomputed_alignments /alignments_short"
        elif [ "$NAME" == "Medium" ]; then
             ALIGN_ARG="--use_precomputed_alignments /opt/openfold/examples/monomer/alignments"
        fi

        BASELINE_MEM=$(get_memory_mb)
        start_std=$(date +%s.%N)
        
        if [ "$NAME" == "Short" ]; then
            docker run --gpus all --ipc=host --shm-size=64g \
                -v $FASTA_DIR:/fasta_dir \
                -v $OUT_DIR:/output \
                -v $WORK_DIR/alignments_short:/alignments_short \
                openfold-spark:cuda13 \
                python3 run_pretrained_openfold.py \
                /fasta_dir \
                /opt/openfold/examples/monomer/template_mmcif \
                --output_dir /output \
                --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
                --config_preset model_1_ptm \
                $ALIGN_ARG \
                --model_device cuda:0 $RELAX_ARG > /tmp/bench_output.txt 2>&1 &
        else
            docker run --gpus all --ipc=host --shm-size=64g \
                -v $FASTA_DIR:/fasta_dir \
                -v $OUT_DIR:/output \
                openfold-spark:cuda13 \
                python3 run_pretrained_openfold.py \
                /fasta_dir \
                /opt/openfold/examples/monomer/template_mmcif \
                --output_dir /output \
                --openfold_checkpoint_path /opt/openfold/openfold_params/finetuning_ptm_2.pt \
                --config_preset model_1_ptm \
                $ALIGN_ARG \
                --model_device cuda:0 $RELAX_ARG > /tmp/bench_output.txt 2>&1 &
        fi
        
        PID=$!
        PEAK_MEM=$(monitor_peak_memory $PID)
        wait $PID || { echo "Command Failed!"; cat /tmp/bench_output.txt; exit 1; }
        end_std=$(date +%s.%N)
        total_time=$(echo "$end_std - $start_std" | bc)
        MEM_USED=$((PEAK_MEM - BASELINE_MEM))

        infer_time=$(grep "Inference time:" /tmp/bench_output.txt | tail -n1 | awk '{print $NF}')
        relax_time=$(grep "Relaxation time:" /tmp/bench_output.txt | tail -n1 | awk '{print $NF}')

        echo "RESULT: $NAME | $MODE | $RELAX_SETTING | Infer: ${infer_time:-N/A} | Relax: ${relax_time:-N/A} | Total: $total_time | Peak Mem: ${PEAK_MEM} MB (+${MEM_USED} MB)"
    fi
}

# Extraction of Short Alignments
echo "Extracting Short Alignments..."
id=$(docker create openfold-spark:cuda13)
docker cp $id:/opt/openfold/tests/test_data/alignments "$WORK_DIR/alignments_short"
docker rm -v $id

if [ -f "$WORK_DIR/alignments_short/pdb70_hits.hhr" ]; then
    cp "$WORK_DIR/alignments_short/pdb70_hits.hhr" "$WORK_DIR/alignments_short/2q2k_A/"
fi

if [ -f "$WORK_DIR/alignments_short/2q2k_A/hmm_output.sto" ]; then
    rm "$WORK_DIR/alignments_short/2q2k_A/hmm_output.sto"
fi

echo ""
echo "System Info:"
echo "  Total RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Run Benchmarks
run_bench "Short" "$WORK_DIR/fasta_short" "STD" "RELAX"
run_bench "Short" "$WORK_DIR/fasta_short" "STD" "NORELAX"

run_bench "Short" "$WORK_DIR/fasta_short" "SOLO" "RELAX"
run_bench "Short" "$WORK_DIR/fasta_short" "SOLO" "NORELAX"

run_bench "Medium" "$WORK_DIR/fasta_medium" "STD" "RELAX"
run_bench "Medium" "$WORK_DIR/fasta_medium" "STD" "NORELAX"

run_bench "Medium" "$WORK_DIR/fasta_medium" "SOLO" "RELAX"
run_bench "Medium" "$WORK_DIR/fasta_medium" "SOLO" "NORELAX"

echo ""
echo "=================================================="
echo "Benchmark Complete!"
echo "=================================================="
