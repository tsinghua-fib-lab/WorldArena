import os
import re
from pathlib import Path

import h5py
import numpy as np


def collect_episodes(root_dir):
    episodes = []
    root = Path(root_dir)
    # collect all task directories
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        demo_dir = task_dir / "aloha-agilex_clean_50" / "aloha-agilex_clean_50"
        video_dir = demo_dir / "video"
        data_dir = demo_dir / "data"  #
        if not (video_dir.exists() and data_dir.exists()):
            continue

        def _get_ep_num(path):
            name = path.name
            match = re.search(r"episode(\d+)", name)
            return int(match.group(1)) if match else -1

        hdf5_files = list(data_dir.glob("episode*.hdf5"))
        hdf5_files.sort(key=_get_ep_num)
        hdf5_files = hdf5_files[:40]  #

        for hdf5_path in hdf5_files:
            ep_name = hdf5_path.stem  # e.g., "episode0"
            video_path = video_dir / f"{ep_name}.mp4"
            if not video_path.exists():
                continue
            episodes.append(
                {
                    "task": task_dir.name,
                    "ep_name": ep_name,
                    "hdf5_path": str(hdf5_path),
                    "video_path": str(video_path),
                    "rel_hdf5_path": str(hdf5_path.relative_to(root)),
                }
            )
    return episodes


def extract_and_save_states_actions(input_root, output_root, max_episodes_per_task=40):
    input_root = Path(input_root)
    output_root = Path(output_root)
    episodes = collect_episodes(str(input_root))

    print(f"Found {len(episodes)} episodes to process.")

    for ep_info in episodes:
        task_name = ep_info["task"]
        ep_name = ep_info["ep_name"]
        hdf5_path = Path(ep_info["hdf5_path"])

        #
        out_state_dir = (
            output_root
            / task_name
            / "aloha-agilex_clean_50"
            / "aloha-agilex_clean_50"
            / "states"
        )
        out_action_dir = (
            output_root
            / task_name
            / "aloha-agilex_clean_50"
            / "aloha-agilex_clean_50"
            / "actions"
        )
        out_state_dir.mkdir(parents=True, exist_ok=True)
        out_action_dir.mkdir(parents=True, exist_ok=True)

        state_file = out_state_dir / f"{ep_name}.npy"
        action_file = out_action_dir / f"{ep_name}.npy"

        #
        if state_file.exists() and action_file.exists():
            print(f"Skipping {ep_info['rel_hdf5_path']} (already processed)")
            continue

        try:
            with h5py.File(hdf5_path, "r") as f:
                # === Extract state from /endpose ===
                right_pose = f["/endpose/right_endpose"][()]  # (T, 7)
                left_pose = f["/endpose/left_endpose"][()]  # (T, 7)
                right_gripper = f["/endpose/right_gripper"][()]  # (T,)
                left_gripper = f["/endpose/left_gripper"][()]  # (T,)

                T = right_pose.shape[0]
                #
                state = np.concatenate(
                    [
                        right_pose,
                        right_gripper[:, None],
                        left_pose,
                        left_gripper[:, None],
                    ],
                    axis=1,
                )  # (T, 14)

                # === Extract action from /joint_action/vector ===
                action = f["/joint_action/vector"][()]  # (T, 14)

                #
                assert state.shape[0] == action.shape[0] == T, (
                    f"T mismatch in {hdf5_path}"
                )

                #
                np.save(state_file, state.astype(np.float32))
                np.save(action_file, action.astype(np.float32))

                print(
                    f"Saved: {state_file} (shape={state.shape}), {action_file} (shape={action.shape})"
                )

        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root", type=str, required=True, help="Root dir of original dataset"
    )
    parser.add_argument(
        "--output_root", type=str, required=True, help="Root dir to save states/actions"
    )
    args = parser.parse_args()

    extract_and_save_states_actions(
        input_root=args.input_root, output_root=args.output_root
    )
