# VLM Policy Evaluator

A script for evaluating robot policy videos with an OpenAI-compatible vision-language model API.
The evaluation script is [vlm_policy_evaluator.py](embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py)

## Default VLM Model

The default VLM model is:

```text
Qwen/Qwen3-VL-32B-Instruct
```

## Expected Data Layout

Ground-truth videos should be organized as:

```text
GT_ROOT/
  task_name/
    videos/
      episode1.mp4
    instructions/
      episode1.json
```

Submission videos can use any layout supported by `--policy-template`, for example:

```text
SUBMISSION_ROOT/
  policy_a/
    episode1.mp4
  policy_b/
    episode1.mp4
```

## Usage

```bash
export VLM_API_KEY="your_api_key"

python vlm_policy_evaluator.py \
  --gt-root /path/to/GT_ROOT \
  --submission-root /path/to/SUBMISSION_ROOT \
  --models "policy_a policy_b" \
  --policy-template "{model}/episode{index}.mp4" \
  --gt-order folder-major \
  --api-url https://api.example.com/v1/chat/completions \
  --run-name evaluation_run
```

For a different submission filename pattern, change `--policy-template`, for example:

```bash
--policy-template "{model}/fixed_scene_task_episode{index}.mp4"
```

## Dry Run

Use `--dry-run` to verify path mapping without calling the API:

```bash
python vlm_policy_evaluator.py \
  --gt-root /path/to/GT_ROOT \
  --submission-root /path/to/SUBMISSION_ROOT \
  --models "policy_a" \
  --policy-template "{model}/episode{index}.mp4" \
  --dry-run
```

## Outputs

By default, the script writes:

- `RUN_NAME.json`: full checkpoint and detailed responses.
- `RUN_NAME.csv`: compact tabular results.

You can override these paths with `--checkpoint-json` and `--output-csv`.
