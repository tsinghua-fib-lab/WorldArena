#!/bin/bash
set -e  # 

eval "$(conda shell.bash hook)"
conda activate worldarena_embodied

export XDG_CACHE_HOME=./cache
export TMPDIR=./tmp
export HF_HOME=./huggingface
export HF_ENDPOINT=https://hf-mirror.com

# ==================  ==================
DATASET_PATH="The absolute path to your dataset."
OUTPUT_DIR="your output folder path" # save logs
Metadata_DIR="./robotwin_metadata.json"

HEIGHT=480
WIDTH=640
OBS_SEQ_LEN=1          
ACT_SEQ_LEN=45        

NUM_WORKERS=4
# ============================================

echo "🚀 Starting Step 1: Data Preprocessing (data_process)"
echo "   Dataset path: $DATASET_PATH"
echo "   Metadata:     $Metadata_DIR"
echo "   Output dir:   $OUTPUT_DIR"
echo "   Resolution:   ${WIDTH}x${HEIGHT} @ ${ACT_SEQ_LEN} chunk"
echo

# 
mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7

python ./step1_prepare_latent_wan.py \
    --task data_process \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_DIR" \
    --height $HEIGHT \
    --width $WIDTH \
    --obs_seq_len $OBS_SEQ_LEN \
    --act_seq_len $ACT_SEQ_LEN \
    --metadata_file $Metadata_DIR \
    --dataloader_num_workers $NUM_WORKERS \
    --size "640*480"
echo "✅ Step 1: Data preprocessing completed!"