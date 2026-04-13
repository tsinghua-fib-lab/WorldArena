#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <variant: 10data|20data|30data|50data|fulldata> <local_dir>"
  exit 1
fi

VARIANT="$1"
LOCAL_DIR="$2"

huggingface-cli download WorldArena/WorldArena \
  "${VARIANT}/model.safetensors" \
  "${VARIANT}/metadata.pt" \
  "${VARIANT}/assets/robotwin_clean/norm_stats.json" \
  --repo-type model \
  --local-dir "${LOCAL_DIR}"
