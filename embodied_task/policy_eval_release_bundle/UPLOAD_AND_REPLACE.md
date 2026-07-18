# Upload and Replace Checklist

This bundle prepares the public release for WorldArena Track 2 policy evaluation.

## 1. Replace files on GitHub

Repository page:

- https://github.com/tsinghua-fib-lab/WorldArena

Replace this documentation file:

- GitHub file: `embodied_task/worldarena_track2/docs/Policy_eval.md`
- Local replacement: `policy_eval_release_bundle/Policy_eval.md`
- Web edit page: https://github.com/tsinghua-fib-lab/WorldArena/edit/main/embodied_task/worldarena_track2/docs/Policy_eval.md

Replace or add these evaluator scripts if the public repository does not already contain the latest versions:

- GitHub file: `embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py`
- Local replacement: `policy_eval_release_bundle/vlm_policy_evaluator.py`
- Web edit page: https://github.com/tsinghua-fib-lab/WorldArena/edit/main/embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py

- GitHub file: `embodied_task/worldarena_track2/scripts/calculate_policy_pearson_r.py`
- Local replacement: `policy_eval_release_bundle/calculate_policy_pearson_r.py`
- Web edit page: https://github.com/tsinghua-fib-lab/WorldArena/edit/main/embodied_task/worldarena_track2/scripts/calculate_policy_pearson_r.py

Important code change to keep: `vlm_policy_evaluator.py` supports glob policy templates, for example `{model}/fixed_scene_task_episode{index}_*.mp4`.

## 2. Upload GT package to Hugging Face

Recommended dataset repo, matching the existing Track 2 dataset docs:

- https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0/tree/main

Upload these files:

- `worldarena_track2_policy_eval_gt.tar.gz`
- optional but recommended: `worldarena_track2_policy_eval_gt.sha256`

Suggested upload path in the dataset repo root:

```text
worldarena_track2_policy_eval_gt.tar.gz
worldarena_track2_policy_eval_gt.sha256
```

After upload, the direct download URL should look like:

```text
https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0/resolve/main/worldarena_track2_policy_eval_gt.tar.gz
```

If you prefer to keep policy-eval GT separate from the rollout dataset, create a new dataset repo such as:

- https://huggingface.co/datasets/WorldArena/WorldArena_Track2_Policy_Eval_GT

Then update `Policy_eval.md` to point to that repo instead.

## 3. What is inside the GT package

The archive should extract to:

```text
worldarena_track2_policy_eval_gt/
├── GT/
├── gt_manifest.csv
└── README_GT.md
```

The GT package contains:

- 50 task folders
- 500 reference videos
- 500 instruction JSON files
- `gt_manifest.csv`, the canonical `eval_index` mapping

## 4. Suggested GitHub commit message

```text
Add public Track 2 VLM policy evaluation docs and GT manifest support
```

## 5. After publishing

Check these commands from a clean checkout:

```bash
wget https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0/resolve/main/worldarena_track2_policy_eval_gt.tar.gz
tar -xzf worldarena_track2_policy_eval_gt.tar.gz
python embodied_task/worldarena_track2/scripts/vlm_policy_evaluator.py --gt-root ./worldarena_track2_policy_eval_gt/GT --submission-root ./your_model_eval --models "your_model_10data your_model_20data your_model_30data your_model_50data your_model_fulldata" --policy-template "{model}/fixed_scene_task_episode{index}_*.mp4" --gt-order folder-major --dry-run
```
