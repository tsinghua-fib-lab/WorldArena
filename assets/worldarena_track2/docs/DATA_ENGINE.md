# Evaluation Pipeline

> This document provides evaluation pipeline for Track 2: data engine.
> For a quick-start guide, see the main [README](../README.md).

---
We evaluate submissions with the **official π₀.₅ policy** on **five RoboTwin 2.0 subtasks**: `adjust_bottle`, `click_bell`, `blocks_ranking_rgb`, `open_laptop`, `pick_dual_bottles`.

### Environment setup

1. **Simulator:** Install and configure RoboTwin 2.0 per the [official documentation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) (**NVIDIA RTX-series GPU** is recommended).
2. **Policy:** Set up the π₀.₅ policy environment following the official **[pi05](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file)** instructions.

### Training configuration

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

### Reporting metrics

Follow the official **RoboTwin 2.0** evaluation protocol. Under the **clean** setting, run **100 trials per task** and report the **final success rate** per task.