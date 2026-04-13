# Technical Details

> This document provides in-depth technical details for Track 2.

---

## Dataset Format

The benchmark dataset is hosted on HuggingFace:
<https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0>

### Structure

```text
dataset/
├── actions/fixed_scene_task/
│   ├── episode1.npy              # (T, 14) joint14 actions
│   ├── episode2.npy
│   └── ...                       # 500 episodes total
├── images/fixed_scene_task/
│   ├── episode1.png              # First-frame RGB image (320×240)
│   └── ...
├── instructions/fixed_scene_task/
│   ├── episode1.json             # {"instruction": "..."}
│   └── ...
└── states/fixed_scene_task/
    ├── episode1.npy              # (T, 16) endpose16 states
    └── ...
```

- **500 episodes** (episode1 – episode500), all under `fixed_scene_task/`.
- No video files — only first-frame PNGs in `images/`.

### State & Action Semantics

| File | Dim | Space | Layout |
|------|-----|-------|--------|
| `actions/` | 14D | **joint14** | `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]` |
| `states/` | 16D | **endpose16** | `[left_xyz(3), left_quat_xyzw(4), left_grip(1), right_xyz(3), right_quat_xyzw(4), right_grip(1)]` |

---

## World Model Action Space & Bridge

This is the **most critical section** for correctly integrating a world model.

### Two WM Training Modes

| Mode | WM Training Data | Bridge Needed? |
|------|-----------------|----------------|
| **joint14** | Joint angles from `actions/*.npy` | No — policy output feeds directly to WM |
| **endpose14** | End-effector poses from `states/*.npy` (qw dropped) | **Yes** — must bridge joint14 → endpose14 |

**endpose14** layout: `[left_xyz(3), left_qxyz(3), left_grip(1), right_xyz(3), right_qxyz(3), right_grip(1)]`
(derived from 16D states by dropping `qw` at dims 6 and 14)

### Why Bridge Is Needed

The policy outputs **joint14** (joint angles). If your WM was trained on **endpose14** (Cartesian poses), directly feeding joint14 to the WM causes a **semantic mismatch** — the WM receives values in the wrong coordinate space, leading to hallucinated / drifting video.

### Bridge Modes

The closed-loop rollout scripts support two bridge modes via `--bridge_mode`:

| `--bridge_mode` | Behavior |
|-----------------|----------|
| `passthrough` | No conversion. Use for **joint14**-trained WMs. |
| `task_knn` | kNN lookup on paired (joint14, endpose14) data from the dataset. Use for **endpose14**-trained WMs. |

### Bridge Implementation

The bridge module is at `src/worldarena_track2_template/action_bridge.py`.

For `task_knn` mode, the bridge:
1. Loads paired `(actions/*.npy, states/*.npy)` for the current task
2. Builds a kNN index mapping joint14 → endpose14
3. For each policy action, finds k nearest joint14 neighbors and returns the inverse-distance-weighted average of their corresponding endpose14 states

Additional CLI parameters:
- `--bridge_knn_k` (default 5): number of nearest neighbors
- `--bridge_sample_stride` (default 1): subsampling stride for the bank
- `--bridge_max_points` (default 0, unlimited): max points in the bank

---

## Policy Model Details

### Policy Interface

```text
policy(image_rgb, state_14d, instruction) -> action_chunk (N, 14)
```

- **Input:** current RGB frame, 14D robot state, task instruction text
- **Output:** horizon-length action sequence in **joint14** space

The policy (Pi0) is trained on **14D joint angles**: `[left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]`.

### Recommended Deployment

```text
Process A: FastAPI policy server  (openpi / jax environment)
Process B: World model rollout    (native PyTorch environment)
Communication: HTTP /reset + /infer
```

This keeps `jax/openpi` dependencies out of the world-model environment.

---

## Closed-Loop Rollout

### Loop Structure

Each rollout iteration:

1. Policy infers an action chunk from `(current_frame, current_state, instruction)`
2. Bridge converts joint14 actions → endpose14 (if needed)
3. Assemble WM input: `[current_wm_state, future_bridged_actions...]`
4. WM generates video frames conditioned on current frame + actions
5. Update state:
   - `current_frame` ← last generated frame
   - `current_policy_state` ← last joint14 action (for policy)
   - `current_wm_state` ← last bridged endpose14 (for WM)

### Target Video Length

The default target video length is based on the GT action trajectory length:

```text
target_len = ceil(action_length × length_scale / down_sample)
```

- `action_length` = `actions/episodeK.npy` shape[0] (number of action frames)
- `length_scale` = scaling factor (default 1.2)
- `down_sample` = frame skip factor (default 1; DiffSynth uses 3)

For example, an episode with 143 action frames, `length_scale=1.2`, `down_sample=3`:
`ceil(143 × 1.2 / 3) = 58` frames.

---

## Action Queue & Temporal Alignment

The policy outputs **50 actions at 50 fps** per call.  Different WMs consume actions at
different rates depending on their chunk size and `down_sample` factor:

1. **Downsample**: `effective_actions = policy_actions[::down_sample]`
2. **Queue**: push all effective actions into a FIFO queue
3. **WM consumes** `future_per_chunk` actions per forward call
   (WM total input = `[current_state] + future_per_chunk` actions)
4. When `queue < future_per_chunk` → discard remainder, call policy again

**Example** (`down_sample=1`, WM chunk needs 16 future actions per call):

```text
policy → 50 actions → queue
WM chunk 1: pop 16 → 34 left
WM chunk 2: pop 16 → 18 left
WM chunk 3: pop 16 →  2 left  →  2 < 16  →  discard, replan
```

**Example** (`down_sample=2`, WM chunk needs 16 future):

```text
policy → 50 actions → [::2] → 25 in queue
WM chunk 1: pop 16 →  9 left  →  9 < 16  →  discard, replan
```

---

## Directory Layout

```text
worldarena-track2-template/
├── README.md                          # Quick-start guide
├── docs/
│   └── DETAILS.md                     # This file — technical details
├── examples/
│   ├── model_README.template.md       # Submission metadata template
│   └── submission_tree.txt            # Expected archive structure
├── scripts/
│   ├── download_dataset.sh            # Dataset download helper
│   ├── download_policy_checkpoint.sh  # Policy checkpoint download
│   ├── package_submission.py          # Submission packaging
│   └── run_generation.py              # Unified rollout launcher
└── src/worldarena_track2_template/
    ├── action_bridge.py               # joint14 <-> endpose14 bridge
    ├── contracts.py                   # Policy/WM protocol definitions
    ├── dataset.py                     # Dataset loading utilities
    ├── length.py                      # Target length computation
    ├── packaging.py                   # Archive building
    └── adapters/                      # Per-WM adapter configs
        ├── base.py
        └── <your_wm>.py              # One adapter per WM type
```
