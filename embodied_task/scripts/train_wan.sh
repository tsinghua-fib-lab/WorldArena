#!/bin/bash
set -e  # 

echo "[INFO] Starting training script for VPP (wan) ..."

echo "[INFO] Initializing conda and activating environment ..."
eval "$(conda shell.bash hook)"
conda activate worldarena_embodied

# check Python and flash_attn
echo "[DEBUG] Current Python: $(which python)"
python -c "import sys; print(f'[DEBUG] Python executable: {sys.executable}')"
python -c "import flash_attn; print('[DEBUG] flash_attn imported successfully! Version:', flash_attn.__version__ if hasattr(flash_attn, '__version__') else 'unknown')"

echo "[INFO] Setting cache and temporary directories ..."
export XDG_CACHE_HOME=./cache
export TMPDIR=./tmp
export HF_HOME=./huggingface
export HF_ENDPOINT=https://hf-mirror.com

echo "[INFO] Setting CUDA environment variables ..."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  #,1,2,3
export TOKENIZERS_PARALLELISM=true
export TORCH_DISTRIBUTED_DEBUG=DETAIL


DATASET_PATH="The absolute path to your dataset."
OUTPUT_DIR="your output folder path"

echo "[INFO] Dataset path: $DATASET_PATH"
echo "[INFO] Output directory: $OUTPUT_DIR"

# launch training with accelerate
# attention change allow_task in step1prepare_latent_wan.py to adjust_bottle or click_bell
echo "[INFO] Launching training with accelerate ..."
python -m accelerate.commands.accelerate_cli launch \
    ./step2_train_action_robotwin_wan.py \
    --root_data_dir "$DATASET_PATH" \
    --out_dir "$OUTPUT_DIR"

echo "[INFO] Training script finished successfully."