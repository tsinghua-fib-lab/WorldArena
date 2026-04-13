#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_ROOT="$(dirname "$SCRIPT_DIR")"

LOCAL_DIR="${1:-$TEMPLATE_ROOT}"

echo "Downloading dataset.tar.gz to ${LOCAL_DIR} ..."
huggingface-cli download WorldArena/WorldArena_Robotwin2.0 \
  dataset.tar.gz \
  --repo-type dataset \
  --local-dir "${LOCAL_DIR}"

echo "Extracting dataset.tar.gz ..."
tar -xzf "${LOCAL_DIR}/dataset.tar.gz" -C "${LOCAL_DIR}"

echo "Done. Dataset extracted to ${LOCAL_DIR}/dataset/"
