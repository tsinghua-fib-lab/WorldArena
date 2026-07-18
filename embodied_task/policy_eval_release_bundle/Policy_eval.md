# Track 2 Policy Evaluation with a VLM Judge

This document describes how to run the open-source VLM-based policy evaluation for WorldArena Track 2 submissions.

The evaluator compares each submitted policy rollout video against a ground-truth reference video and asks an OpenAI-compatible vision-language model judge to return a binary success label.

## Files to release

The policy evaluation release has two parts:

1. Code in this repository:
   - `embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py`
   - `embodied_task/worldarena_track2/scripts/calculate_policy_pearson_r.py`
   - this document, `embodied_task/worldarena_track2/docs/Policy_eval.md`
2. Ground-truth package hosted as a dataset artifact:
   - `worldarena_track2_policy_eval_gt.tar.gz`

The GT package is required for public reproducibility. It is different from the rollout dataset used to generate submissions.

## Install dependencies

```bash
pip install opencv-python requests
```

The evaluator uses an OpenAI-compatible chat completions endpoint with image input blocks. API keys are read only from environment variables.

## Download and unpack GT

Download `worldarena_track2_policy_eval_gt.tar.gz` from the WorldArena dataset release page and unpack it:

```bash
tar -xzf worldarena_track2_policy_eval_gt.tar.gz
```

Expected structure:

```text
worldarena_track2_policy_eval_gt/
├── GT/
│   ├── adjust_bottle/
│   │   ├── videos/episode40.mp4 ... episode49.mp4
│   │   └── instructions/episode40.json ... episode49.json
│   └── ...
├── gt_manifest.csv
└── README_GT.md
```

The package contains 50 tasks, 500 reference videos, and 500 instruction JSON files.

## Evaluation index mapping

Submissions are evaluated with a 1-based global `eval_index` from 1 to 500. GT files keep their task-local names, such as `adjust_bottle/videos/episode40.mp4`.

The default official order is `folder-major`:

1. Sort task folders alphabetically.
2. Inside each task, sort `episode*.mp4` by episode number.
3. Assign `eval_index` from 1 to 500.

Use `gt_manifest.csv` as the canonical mapping. For example:

```text
eval_index=1   -> GT/adjust_bottle/videos/episode40.mp4
eval_index=500 -> GT/turn_switch/videos/episode49.mp4
```

## Submission layout

A submitted archive usually has five model folders:

```text
your_model_eval/
├── your_model_10data/
├── your_model_20data/
├── your_model_30data/
├── your_model_50data/
├── your_model_fulldata/
└── model_README.md
```

The evaluator is flexible about video names through `--policy-template`. Supported template variables are:

- `{model}`: model folder name
- `{index}`: 1-based `eval_index`
- `{task}`: GT task name
- `{episode}`: GT episode name such as `episode40`
- `{episode_number}`: GT episode number such as `40`

Glob characters such as `*` are supported in the template.

Common examples:

```bash
--policy-template "{model}/episode{index}.mp4"
--policy-template "{model}/fixed_scene_task_episode{index}.mp4"
--policy-template "{model}/fixed_scene_task_episode{index}_*.mp4"
--policy-template "{model}/fixed_scene_task/episode{index}.mp4"
```

## Dry run

Always run a dry run first to verify the mapping before calling the VLM API:

```bash
python embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py \
  --gt-root ./worldarena_track2_policy_eval_gt/GT \
  --submission-root ./your_model_eval \
  --models "your_model_10data your_model_20data your_model_30data your_model_50data your_model_fulldata" \
  --policy-template "{model}/fixed_scene_task_episode{index}_*.mp4" \
  --gt-order folder-major \
  --dry-run
```

The dry run prints the first resolved mappings and does not call the VLM API.

## Run VLM evaluation

Example with SiliconFlow / Qwen:

```bash
export VLM_API_KEY=YOUR_API_KEY
export VLM_API_URL=https://api.siliconflow.cn/v1/chat/completions
export VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct

python embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py \
  --gt-root ./worldarena_track2_policy_eval_gt/GT \
  --submission-root ./your_model_eval \
  --models "your_model_10data your_model_20data your_model_30data your_model_50data your_model_fulldata" \
  --policy-template "{model}/fixed_scene_task_episode{index}_*.mp4" \
  --gt-order folder-major \
  --api-url "$VLM_API_URL" \
  --api-key-env VLM_API_KEY \
  --vlm-model "$VLM_MODEL" \
  --checkpoint-json ./your_model_policy_eval.json \
  --output-csv ./your_model_policy_eval.csv \
  --run-name your_model_policy_eval
```

The JSON checkpoint is updated after each item, so interrupted runs can be resumed with the same command.

## Outputs

The evaluator writes:

- JSON checkpoint with all raw VLM responses
- CSV summary with one row per `(eval_index, policy_model)` pair

Important fields:

- `vlm_answer`: `1` for success, `0` for failure, `-1` for invalid/error
- `thinking`: VLM judge rationale
- `policy_video`: resolved submitted video path
- `gt_video`: matched GT reference video path
- `error`: non-empty if this row failed before obtaining a valid VLM answer

## Compute Pearson R

After evaluation, compute Pearson correlation against the default simulator scores:

```bash
python embodied_task/worldarena_track2/scripts/calculate_policy_pearson_r.py \
  --eval ./your_model_policy_eval.json
```

The default simulator scores are:

```text
10data:   28.60
20data:   34.58
30data:   37.78
50data:   43.52
fulldata: 46.80
```

If you use a custom simulator score file:

```bash
python embodied_task/worldarena_track2/scripts/calculate_policy_pearson_r.py \
  --eval ./your_model_policy_eval.json \
  --sim-scores ./simulator_scores.csv
```

## Notes for maintainers

Do not commit API keys or provider-specific secrets. Keep API keys in environment variables only. Public releases should include the GT package and `gt_manifest.csv`; otherwise third-party users cannot reproduce the VLM policy evaluation.
