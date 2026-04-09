# Track 2: Embodied Task

## Task 1: Embodied Data Engine

---

## 1. Data Preparation

Download the official datasets from Hugging Face:

- [WorldArena_Robotwin2.0](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0)

---

## 2. Submission Format

Package your generated data and model metadata into a single archive (e.g. `.zip`, `.tar`). Name the archive:

```text
{Your_Model_Name}_eval_track2
```

### 2.1 Archive layout

Organize **image–action** outputs as follows:

```text
result/
├── model_README.md               # Model documentation
├── adjust_bottle/                # Task name
│   ├── episode_0/                # Episode ID
│   │   ├── label/
│   │   │   ├── frame_0.json      # Keys: image_path, actions (length 40, joint14), state, instruction
│   │   │   └── ...
│   │   └── image/
│   │       ├── frame_0.png
│   │       └── ...
│   └── ...
├── click_bell/
│   ├── episode_0/
│   │   ├── label/
│   │   │   ├── frame_0.json
│   │   │   └── ...
│   │   └── image/
│   │       ├── frame_0.png
│   │       └── ...
│   └── ...
└── ...
```

### 2.2 Model documentation (`model_README.md`)

Include a `model_README.md` (or `.txt`) at the root of `result/` with at least:


| Field                   | Required | Notes        |
| ----------------------- | -------- | ------------ |
| Model name              | Yes      |              |
| GitHub repository       | No       |              |
| Release year            | Yes      |              |
| Open-source status      | Yes      | `Yes` / `No` |
| Brief description       | No       |              |
| Contact / communication | No       |              |
| Other notes             | No       |              |


---

## 3. Submission Process

### Step 1 — Package

Place all generated data and `model_README.md` inside `{Your_Model_Name}_eval_track2` before compressing.

### Step 2 — Email

Send the archive to **[WorldArena1@outlook.com](mailto:WorldArena1@outlook.com)** with:

- **Subject:** `{Your_Model_Name}_evaluation_track2`
- **Attachment:** `{Your_Model_Name}_eval_track2.zip` (or equivalent archive format)

---

## 4. Evaluation Timeline & Leaderboard

- **Leaderboard:** Submissions are evaluated and reflected on the [WorldArena Leaderboard](https://huggingface.co/spaces/WorldArena/WorldArena) within **3–4 business days**.
- **Notification:** You will receive a confirmation email when evaluation finishes. Thank you for your patience and contribution.

---

## 5. Official Evaluation Pipeline

We evaluate submissions with the **official π₀.₅ policy** on **five RoboTwin 2.0 subtasks**: `adjust_bottle`, `click_bell`, `blocks_ranking_rgb`, `open_laptop`, `pick_dual_bottles`.

### 5.1 Environment setup

1. **Simulator:** Install and configure RoboTwin 2.0 per the [official documentation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) (**NVIDIA RTX-series GPU** is recommended).
2. **Policy:** Set up the π₀.₅ policy environment following the official **[pi05](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file)** instructions.

### 5.2 Training configuration

We fine-tune π₀.₅ on your generated data with the official pretrained checkpoint. Reference `TrainConfig`:

```python
TrainConfig(
    name="pi05_aloha_robotwin_mulitask_clean_wowrist_data_genie",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=40),
    data=LeRobotAlohaDataConfig(
        repo_id={/path/merged/data},
        adapt_to_pi=False,
        only_base_image=True,
        repack_transforms=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {
                    "cam_high": "observation.images.cam_high",
                    # "cam_left_wrist": "observation.images.cam_left_wrist",
                    # "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            })
        ]),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
    freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
    batch_size=64,
    num_workers=8,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "openpi/openpi-assets/checkpoints/pi05_base_torch/params"
    ),
    pytorch_weight_path="openpi/openpi-assets/checkpoints/pi05_base_torch",
    num_train_steps=10000,
)
```

**Note:** For this track we use **only the head (high) camera** as visual input; **wrist cameras are disabled**.

### 5.3 Reporting metrics

Follow the official **RoboTwin 2.0** evaluation protocol. Under the **clean** setting, run **100 trials per task** and report the **final success rate** per task.