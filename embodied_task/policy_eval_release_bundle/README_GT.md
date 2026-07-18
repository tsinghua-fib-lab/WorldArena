# WorldArena Track 2 Policy Evaluation GT

This package contains the ground-truth reference data used by the Track 2 VLM policy evaluator.

## Contents

```text
GT/
  <task_name>/
    videos/episode40.mp4 ... episode49.mp4
    instructions/episode40.json ... episode49.json
gt_manifest.csv
README_GT.md
```

The package contains 50 tasks, 500 GT videos, and 500 instruction JSON files.

## Important: evaluation index mapping

Submission videos are usually named with a 1-based global index, such as `episode1.mp4` or `fixed_scene_task_episode1.mp4`. The GT files keep their original task-local names, such as `adjust_bottle/videos/episode40.mp4`.

The evaluator maps submission index to GT using folder-major order:

1. Sort task folders alphabetically.
2. Inside each task, sort `episode*.mp4` by episode number.
3. Assign `eval_index` from 1 to 500.

Use `gt_manifest.csv` as the canonical mapping from `eval_index` to `task_name`, `episode_name`, instruction, GT video path, and instruction JSON path.

Example:

```text
eval_index=1   -> GT/adjust_bottle/videos/episode40.mp4
eval_index=500 -> GT/turn_switch/videos/episode49.mp4
```

## Relationship to the rollout dataset

This GT package is different from the Track 2 rollout dataset used for generating submissions. The rollout dataset contains first-frame images, robot states, and action trajectories under `fixed_scene_task/episode1..episode500`. This package contains reference videos and instructions for VLM judging.

## Expected use

```bash
python scripts/vlm_policy_evaluator.py \
  --gt-root ./worldarena_track2_policy_eval_gt/GT \
  --submission-root ./your_model_eval \
  --models "your_model_10data your_model_20data your_model_30data your_model_50data your_model_fulldata" \
  --policy-template "{model}/fixed_scene_task_episode{index}_*.mp4" \
  --gt-order folder-major \
  --dry-run
```
