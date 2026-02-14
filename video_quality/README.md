## WorldArena Video Quality Evaluation Environment and Usage Guide

Scope: standard metrics, VLM interaction quality/perspectivity/instruction following, JEPA similarity.
Environment split: `WorldArena` (base/action following), `WorldArena_VLM` (VLM evaluation), `WorldArena_JEPA` (JEPA similarity).

### 1. Prerequisites
- OS: Linux, CUDA 12.8 (aligned with torch cu128).
- Python: 3.10.
- GPU: sufficient for SAM/RAFT/CLIP/VLM inference.

### 2. Base Environment `WorldArena`
Used for standard metrics and action following.
```bash
cd WorldArena
conda create -y -n WorldArena python=3.10
conda activate WorldArena
pip install -r video_quality/requirements.txt
# Optional: pip install jupyter notebook jupyterlab
```

### 3. VLM Environment `WorldArena_VLM`
Used for Interaction Quality / Perspectivity / Instruction Following.
```bash
cd WorldArena
conda create -y -n WorldArena_VLM python=3.10
conda activate WorldArena_VLM
# Core GPU stack (cu128 with torch 2.9.1)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
# VLM dependencies
pip install -r video_quality/requirements_worldarena_vlm.txt
# Required libs not listed in the slim deps
pip install PyYAML==6.0.3 Pillow==10.0.0
# If you need local wheels (flash-attn etc.), uncomment and set paths in requirements_worldarena_vlm.txt
```

### 4. JEPA Environment `WorldArena_JEPA`
Used for JEPA similarity.
```bash
cd WorldArena
conda create -y -n WorldArena_JEPA python=3.10
conda activate WorldArena_JEPA
pip install -r requirements_jedi.txt
# Download weights to video_quality/JEDi/pretrained_models/
mkdir -p video_quality/JEDi/pretrained_models
cd video_quality/JEDi/pretrained_models
wget -O vith16.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
wget -O ssv2-probe.pth.tar https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
```

### 5. Data and Naming Rules
- Prepare `summary.json` (example):
```json
[
  {
    "gt_path": "/path/to/gt_video/episode40.mp4",
    "image": "/path/to/gt_frames/episode40.png",
    "prompt": [
      "In a fixed robotic workspace, generate a rigid, physically consistent..."
    ]
  },
    ...
]
```
- Generated video directory must be named `modelname_sort`; put only videos inside, named `{taskname}_episode_{xx}.mp4`; no subfolders allowed.

### 6. External Weights / Paths
Configure local weights and I/O paths in [config](config/config.yaml) (do not change model_name: test):
### 7. Run Evaluation

For the first two evaluations, directly use the generated video directory and summary_json; JEPA requires a GT video directory (only .mp4 files following naming rules, no nesting).
- VLM metrics (interaction quality, perspectivity, instruction following) (requires `WorldArena_VLM` env):
```bash
bash video_quality/run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> 
```
- JEPA similarity (requires `WorldArena_JEPA` env):
```bash
bash video_quality/run_evaluation_JEPA.sh
```
The following metrics first run a format preprocessing step (the bash already includes it), producing the structure under data_action_following configured in config:
```
data_action_following
  ├── gt_dataset/
  │      ├── {task_name}/
  │      │   ├── episode_{x}/
  │      │   │   ├── prompt/
  │      │   │   │   ├── init_frame.png
  │      │   │   │   └── prompt.txt
  │      │   │   └── video/
  │      │   │       ├── frame_00000.jpg
  │      │   │       ├── ...
  │      │   │       └── frame_0000n.jpg
  │      │   ├── episode_{x+1}/
  │      │   └── ...
  │      ├── {task_name1}/
  │      └── ...
  ├── generated_dataset/
  │      ├── {task_name}/
  │      │   ├── episode_{x}/
  │      │   │   ├── 1/
  │      │   │   │   └── video/
  │      │   │   │       ├── frame_00000.jpg
  │      │   │   │       ├── ...
  │      │   │   │       └── frame_0000n.jpg
  │      │   │   ├── 2/
  │      │   │   └── 3/
  │      │   ├── episode_{x+1}/
  │      │   └── ...
  │      ├── {task_name1}/
  │      └── ...
  │

```
For action following, episode_{x} must contain subfolders 1, 2, 3; other metrics only need subfolder 1.

Videos in subfolders 2 and 3 can be created by modifying the original prompt to guide two different actions; you can call an LLM or write manually. Sample LLM prompt:

Use these two instructions to generate two new action videos. If the action-guided video lacks a modifiable prompt, consider using other actions from the same task to achieve different actions.

Finally, place the two new-action videos into corresponding directories (all must be MP4, no nested files, names match generated video names). Name the three directories `modelname_sort` `modelname_1_sort` `modelname_2_sort`:

- action following (requires `WorldArena` env):
```bash
bash video_quality/run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON>
# If using a specific split, run preprocess_datasets_diversity.py first, or ensure config data_action_following path exists
```
- Other metrics (requires `WorldArena` env):
```bash
bash video_quality/run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> 
```

- Metric aggregation (requires `WorldArena` env):
```bash
python video_quality/csv_results/aggregate_results.py --model_name <MODEL_NAME> --base_dir . --csv_name aggregated_results.csv
```
You can view all metric results under the csv_results directory.
