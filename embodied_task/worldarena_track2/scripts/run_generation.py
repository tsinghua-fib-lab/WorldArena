#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_ROOT = SCRIPT_DIR.parent
SRC = TEMPLATE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldarena_track2_template.adapters import (
    ExampleJoint14Adapter, ExampleEndpose14Adapter,
)
from worldarena_track2_template.adapters.base import CommonRunArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Track 2 generation runner")
    parser.add_argument("--wm", choices=["example_joint14", "example_endpose14"],
                        required=True, help="WM adapter name. Add your own adapter here.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-list", default="adjust_bottle")
    parser.add_argument("--max-episode-index", type=int, default=1)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted dataset directory (e.g., ./dataset).",
    )
    parser.add_argument("--val10-root", dest="dataset_root", help=argparse.SUPPRESS)
    parser.add_argument("--length-scale", type=float, default=1.2)
    parser.add_argument("--down-sample", type=int, default=1)
    parser.add_argument("--policy-model-name", default="robotwin_all_clean_wnorm_wowrist_10radiodata")
    parser.add_argument("--policy-checkpoint-id", default="10000")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# Register your own adapters here
_ADAPTERS = {
    "example_joint14": ExampleJoint14Adapter,
    "example_endpose14": ExampleEndpose14Adapter,
}


def make_adapter(name: str):
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(_ADAPTERS)}")
    return _ADAPTERS[name]()


def main() -> int:
    args = parse_args()
    adapter = make_adapter(args.wm)
    common = CommonRunArgs(
        output_dir=Path(args.output_dir),
        task_list=args.task_list,
        max_episode_index=args.max_episode_index,
        dataset_root=Path(args.dataset_root),
        length_scale=args.length_scale,
        down_sample=args.down_sample,
        policy_model_name=args.policy_model_name,
        policy_checkpoint_id=args.policy_checkpoint_id,
    )
    cmd = [str(x) for x in adapter.build_command(common)]
    print("Command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONUNBUFFERED"] = "1"
    env.update(adapter.build_env())
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
