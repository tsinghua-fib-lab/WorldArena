#!/bin/bash
set -euo pipefail

cd your absolute path
source /root/miniconda3/bin/activate
conda activate WorldArena_JEPA
python batch.py --real_dir your absolute path --gen_dir your absolute path

