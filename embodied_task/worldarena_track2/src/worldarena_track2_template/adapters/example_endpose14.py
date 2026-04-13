from __future__ import annotations

from .base import AdapterBase, CommonRunArgs


class ExampleEndpose14Adapter(AdapterBase):
    """Example adapter for a WM trained on endpose14 action space.

    endpose14 WMs require bridging from policy joint14 output.
    Set bridge_mode='task_knn'.
    """
    wm_name = "example_endpose14"
    wm_action_space = "endpose14"
    default_bridge_mode = "task_knn"

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
