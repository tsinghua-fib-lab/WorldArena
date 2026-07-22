# WorldArena Evaluation Guideline

This document describes how to evaluate your model on the WorldArena benchmark locally。

Note:The WorldArena Challenge has officially concluded, with the submission deadline being June 30, 2026. The ground truth (GT) and evaluation resources have been completely released and are now publicly available. You may conduct local evaluations by following the provided documentation. For leaderboard updates, please pay attention to [WorldArena Challenge 2.0](http://iros2026challenge.world-arena.ai/).
---

## Contents

- [Track 1: Video Quality](#track-1-video-quality)
- [Track 2: Functional Performance](#track-2-functional-performance)
  - [Task 1: Data Engine](#task-1-data-engine)
  - [Task 2: Policy Evaluator](#task-2-policy-evaluator)

---

## Track 1: Video Quality

**Pipeline:** [video_quality/README.md](../video_quality/README.md)

### 1. Data preparation

Download the test datasets from the official Hugging Face dataset:

- [WorldArena_Robotwin2.0](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0)

| Folder | Purpose |
|--------|---------|
| `test_dataset` | Evaluation set for the **leaderboard**. |


#### Inference requirements

For each episode in the test set, generate a video from the provided initial frame (`first_frame`) and text instruction (`instruction`) or actions (`data/_traj_data`) with gt videos(`video`).

| Item | Requirement |
|------|-------------|
| Resolution | **640×480** or higher (recommended) |
| Length | **Text-driven:** fixed **121** frames.<br>**Action-driven:** rollout according to the provided action sequence length, and align the generated video length with the corresponding GT trajectory length. |
| Frame rate | **24** fps |

### 2. Compute full score


---

## Track 2: Functional Performance

### Task 1: Data Engine

**Pipeline:** [DATA_ENGINE.md](../embodied_task/worldarena_track2/docs/DATA_ENGINE.md)

#### 1. Data preparation

Download the official dataset from Hugging Face:

- [WorldArena_Robotwin2.0](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0)




---

### Task 2: Policy Evaluator

Track 2 evaluates **world models** in closed loop with a fixed policy. You provide the world model; we provide the policy, dataset, and evaluation pipeline.

**Details:** [DETAILS.md](../embodied_task/worldarena_track2/docs/DETAILS.md) (dataset format, action space, bridge, rollout internals)

**Full pipeline:**[Policy_eval.md](../embodied_task/policy_eval_release_bundle/Policy_eval.md)

#### 1. Environment setup

**1a. Policy environment ([openpi](https://github.com/Physical-Intelligence/openpi))**

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**1b. Policy checkpoints (5 variants)**

```bash
huggingface-cli download WorldArena/WorldArena \
  --repo-type model --local-dir ./policy_ckpt
```

This downloads `10data/`, `20data/`, `30data/`, `50data/`, `fulldata/` — each with `model.safetensors`, `metadata.pt`, and norm stats.

**1c. Dataset (500 episodes, ≈21 MB)**

```bash
bash scripts/download_dataset.sh
# or manually:
wget https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0/resolve/main/dataset.tar.gz
tar -xzf dataset.tar.gz
```

#### 2. Write your adapter

Add an adapter under [embodied_task/worldarena_track2/src/worldarena_track2_template/adapters/](../embodied_task/worldarena_track2/src/worldarena_track2_template/adapters/). Examples:

| File | Use when |
|------|----------|
| `example_joint14.py` | WM trained on **joint angles** (no bridge) |
| `example_endpose14.py` | WM trained on **end-effector poses** (kNN bridge) |

Implement `build_command()` — it returns the shell command to run your world-model rollout script.

> **Action space:** `joint14` → `bridge_mode = "passthrough"` · `endpose14` → `bridge_mode = "task_knn"` (see [DETAILS.md — bridge](../embodied_task/worldarena_track2/docs/DETAILS.md)).

#### 3. Run generation (5 policies × 500 episodes)

```bash
for variant in 10data 20data 30data 50data fulldata; do
  python scripts/run_generation.py \
    --wm <your_wm> \
    --dataset-root ./dataset \
    --output-dir ./output/${variant} \
    --policy-variant ${variant} \
    --max-episode-index 500
done
```

Output: **2500** videos (5 folders × 500 episodes each).


#### 4. Run evaluation locally (5 policies × 500 episodes)

see [Policy_eval.md](../embodied_task/worldarena_track2/docs/Policy_eval.md)
