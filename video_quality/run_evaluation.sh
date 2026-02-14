#!/bin/bash
set -euo pipefail

# Usage: run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH]
# METRIC_LIST example: "image_quality,photometric_smoothness,action_following"

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
CONFIG_PATH=${5:-"./config/config.yaml"}
if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH]"
    exit 1
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena
export PATH="your absolute path:$PATH"

# Parse metrics
CLEAN_METRICS=$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')
METRIC_ARRAY=($CLEAN_METRICS)
echo ">>> Input metrics: $RAW_METRICS"
echo ">>> Formatted for evaluate.py: ${METRIC_ARRAY[*]}"

DATA_DIR="./data"
CONFIG_DIR="./config"
OUTPUT_DIR="./output"
OUTPUT_DIR_ACTION="./output_action_following"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"

# Split metrics
EVAL_METRICS=()
RUN_ACTION=false
for metric in "${METRIC_ARRAY[@]}"; do
    if [ "$metric" == "action_following" ]; then
        RUN_ACTION=true
    else
        EVAL_METRICS+=("$metric")
    fi
done

# Standard metrics
if [ ${#EVAL_METRICS[@]} -gt 0 ]; then
    echo ">>> Running Preprocessing for standard metrics..."
    python preprocess_datasets.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR"
    echo ">>> Running processing (Resize & Detection)..."
    python ./processing/video_resize.py --config_path "$CONFIG_PATH"
    python ./processing/detection_tracking.py --config_path "$CONFIG_PATH"

    echo ">>> Starting Standard Evaluation: ${EVAL_METRICS[*]}"
    python evaluate.py --dimension ${EVAL_METRICS[@]} --config "$CONFIG_PATH" --overwrite
fi


echo ">>> ✅ All evaluations finished"


