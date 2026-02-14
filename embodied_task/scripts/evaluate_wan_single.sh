#!/bin/bash
set -e

echo "[INFO] Starting multi-GPU evaluation for VPP_wan (RobotWin) ..."

# Initialize conda and activate environment
eval "$(conda shell.bash hook)"
conda activate worldarena_embodied


# Set cache and HF mirror
export XDG_CACHE_HOME=./cache
export TMPDIR=./tmp
export HF_HOME=./huggingface
export HF_ENDPOINT=https://hf-mirror.com

# ================== 配置区 ==================
DATASET_PATH="The absolute path to your dataset." #like .//robotwin2.0_clean50_firstinstr_10
METADATA_FILE="./robotwin_metadata.json"

# Data parameters
HEIGHT=480
WIDTH=640
OBS_SEQ_LEN=1
ACT_SEQ_LEN=45

# Evaluation config
ACTION_MODEL_PATH="./models/wan_adjust_bottle/adjust_bottle.pt"
OUTPUT_PATH="your output folder path" #like ./output/wan_adjust_bottle

# ===== Multi-GPU Settings =====
TOTAL_CHUNKS=2540                     # dataset len
GPUS=(0 1 2 3 4 5 6 7)                #  GPU 0 1 2 3 4 5 6 7
NUM_GPUS=${#GPUS[@]}
LOG_DIR="$OUTPUT_PATH/logs"
# =============================

PER_GPU=$(( (TOTAL_CHUNKS + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total chunks: $TOTAL_CHUNKS"
echo "Using GPUs: ${GPUS[*]}"
echo "Chunks per GPU: $PER_GPU"
echo "Output directory: $OUTPUT_PATH"
echo "Log directory: $LOG_DIR"

mkdir -p "$LOG_DIR"

# 
PIDS=()
cleanup() {
    echo "🛑 Caught interrupt. Killing all child processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait " $ {PIDS[@]}" 2>/dev/null || true
    echo "✅ All child processes terminated."
    exit 1
}
trap cleanup SIGINT SIGTERM
# ===================

# Launch processes for each GPU
for i in "${!GPUS[@]}"; do
    START=$(( i * PER_GPU ))
    END=$(( START + PER_GPU ))
    if [ $END -gt $TOTAL_CHUNKS ]; then
        END=$TOTAL_CHUNKS
    fi

    GPU=${GPUS[$i]}
    echo "Launching GPU $GPU: chunks [$START, $END)"

    CUDA_VISIBLE_DEVICES=$GPU \
    python policy_evaluation/wan_single_chunk.py \
        --action_model_path "$ACTION_MODEL_PATH" \
        --output_path "$OUTPUT_PATH" \
        --dataset_path "$DATASET_PATH" \
        --metadata_file "$METADATA_FILE" \
        --width $WIDTH \
        --height $HEIGHT \
        --obs_seq_len $OBS_SEQ_LEN \
        --act_seq_len $ACT_SEQ_LEN \
        --is_i2v True \
        --start_index $START \
        --end_index $END \
        > "$LOG_DIR/gpu ${GPU}.log" 2>&1 &

    PIDS+=($!)
done

echo "All jobs launched. Logs in  $LOG_DIR/"

wait "${PIDS[@]}"
echo "✅ All done!"