from __future__ import annotations

from .base import AdapterBase, CommonRunArgs


class ExampleJoint14Adapter(AdapterBase):
    """Example adapter for a WM trained on joint14 action space.

    joint14 WMs accept policy output directly — no bridge needed.
    Set bridge_mode='passthrough'.
    """
    wm_name = "example_joint14"
    wm_action_space = "joint14"
    default_bridge_mode = "passthrough"

    def build_command(self, args: CommonRunArgs):
        # TODO: replace placeholders with your actual paths
        return [
            "python", "<WM_ROOT>/scripts/rollout_closedloop.py",
            "--ckpt_path", "<CKPT_PATH>",
            "--val10_root", str(args.dataset_root),
            "--output_dir", str(args.output_dir),
            "--bridge_mode", self.default_bridge_mode,
            "--task_list", args.task_list,
            "--max_episode_index", str(args.max_episode_index),
            "--length_scale", str(args.length_scale),
            "--down_sample", str(args.down_sample),
            "--record_policy_actions",
        ]
