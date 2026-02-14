#!/bin/bash
set -euo pipefail

# Usage: run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH]

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
CONFIG_PATH=${4:-"./config/config.yaml"}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ]; then
  echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH]"
  exit 1
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena
export PATH="your absolute path:$PATH"

DATA_DIR="./data_action_following"
CONFIG_DIR="./config"
OUTPUT_DIR_ACTION="./output_action_following"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR_ACTION"

echo ">>> Running action_following preprocessing..."
python preprocess_datasets_diversity.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR"

echo ">>> Running action_following evaluation..."
python evaluate.py --dimension "action_following" --config "$CONFIG_PATH" --overwrite || echo ">>> [WARNING] evaluate.py (action_following) returned non-zero code"

echo ">>> ✅ Action Following Script Finished"
