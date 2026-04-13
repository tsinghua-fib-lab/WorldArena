from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import EpisodeSpec


# ── Episode discovery ──────────────────────────────────────────────────────

def _is_track2_format(dataset_root: Path) -> bool:
    """Track 2 format: actions/task/*.npy + images/task/*.png."""
    return (dataset_root / "actions").is_dir()


def _is_track1_format(dataset_root: Path) -> bool:
    """Track 1 format: first_frame/task/*.png + data/task/*.hdf5."""
    return (dataset_root / "first_frame").is_dir()


def build_episode_specs(dataset_root: Path, prompt_variant: str = "base") -> list[EpisodeSpec]:
    """Auto-detect dataset format and build episode specs."""
    if _is_track2_format(dataset_root):
        return _build_specs_track2(dataset_root)
    if _is_track1_format(dataset_root):
        return _build_specs_track1(dataset_root, prompt_variant)
    raise FileNotFoundError(
        f"Cannot detect dataset format in {dataset_root}. "
        "Expected either actions/ (Track 2) or first_frame/ (Track 1) directory."
    )


def _build_specs_track2(dataset_root: Path) -> list[EpisodeSpec]:
    """Track 2: actions/task/*.npy, images/task/*.png, instructions/task/*.json, states/task/*.npy."""
    actions_root = dataset_root / "actions"
    images_root = dataset_root / "images"
    instructions_root = dataset_root / "instructions"

    specs: list[EpisodeSpec] = []
    for task_dir in sorted(actions_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for action_file in sorted(task_dir.glob("episode*.npy")):
            episode_stem = action_file.stem
            image_path = images_root / task_name / f"{episode_stem}.png"
            instruction_path = instructions_root / task_name / f"{episode_stem}.json"
            if not instruction_path.exists():
                continue
            specs.append(
                EpisodeSpec(
                    episode_id=episode_stem,
                    task_name=task_name,
                    first_frame_path=image_path if image_path.exists() else None,
                    instruction_path=instruction_path,
                    instruction_variant="base",
                    trajectory_hdf5_path=None,
                )
            )
    return specs


PROMPT_VARIANTS = {
    "base": "instructions",
    "instruction_1": "instructions_1",
    "instruction_2": "instructions_2",
}


def _build_specs_track1(dataset_root: Path, prompt_variant: str = "base") -> list[EpisodeSpec]:
    """Track 1: first_frame/task/*.png, data/task/*.hdf5, instructions/task/*.json."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant}")

    first_frame_root = dataset_root / "first_frame"
    instruction_root = dataset_root / PROMPT_VARIANTS[prompt_variant]
    data_root = dataset_root / "data"

    specs: list[EpisodeSpec] = []
    for frame_path in sorted(first_frame_root.rglob("*.png")):
        if frame_path.name.startswith("."):
            continue
        rel = frame_path.relative_to(first_frame_root)
        task_name = rel.parent.as_posix()
        episode_stem = frame_path.stem
        instruction_path = instruction_root / rel.with_suffix(".json")
        trajectory_hdf5_path = data_root / rel.with_suffix(".hdf5")
        if not instruction_path.exists():
            raise FileNotFoundError(f"Missing instruction file for {frame_path}: {instruction_path}")
        if not trajectory_hdf5_path.exists():
            raise FileNotFoundError(f"Missing trajectory file for {frame_path}: {trajectory_hdf5_path}")
        specs.append(
            EpisodeSpec(
                episode_id=episode_stem,
                task_name=task_name,
                first_frame_path=frame_path,
                instruction_path=instruction_path,
                instruction_variant=prompt_variant,
                trajectory_hdf5_path=trajectory_hdf5_path,
            )
        )
    return specs


# ── Data loading helpers ───────────────────────────────────────────────────

def load_instruction_text(instruction_path: Path) -> str:
    with open(instruction_path, "r") as f:
        payload = json.load(f)
    instruction = payload.get("instruction", "")
    if not isinstance(instruction, str):
        raise TypeError(f"Expected string instruction in {instruction_path}, got {type(instruction)!r}")
    return instruction


def load_joint14_actions_npy(npy_path: Path) -> np.ndarray:
    """Load (T, 14) joint14 actions from .npy file (Track 2 format)."""
    return np.load(npy_path).astype(np.float32)


def load_joint14_actions(hdf5_path: Path) -> np.ndarray:
    """Load (T, 14) joint14 actions from .hdf5 file (Track 1 format)."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f["joint_action/vector"], dtype=np.float32)


def load_initial_joint14(hdf5_path: Path) -> np.ndarray:
    actions = load_joint14_actions(hdf5_path)
    if actions.shape[0] == 0:
        raise ValueError(f"Empty joint_action/vector in {hdf5_path}")
    return actions[0]


def load_endpose16(hdf5_path: Path) -> np.ndarray:
    """Load (T, 16) endpose16 states from .hdf5 file (Track 1 format)."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        left_endpose = np.asarray(f["endpose/left_endpose"], dtype=np.float32)
        left_gripper = np.asarray(f["endpose/left_gripper"], dtype=np.float32)[:, None]
        right_endpose = np.asarray(f["endpose/right_endpose"], dtype=np.float32)
        right_gripper = np.asarray(f["endpose/right_gripper"], dtype=np.float32)[:, None]
    return np.concatenate([left_endpose, left_gripper, right_endpose, right_gripper], axis=1)


def load_states_npy(npy_path: Path) -> np.ndarray:
    """Load (T, 16) endpose16 states from .npy file (Track 2 format)."""
    return np.load(npy_path).astype(np.float32)


def load_initial_endpose16(hdf5_path: Path) -> np.ndarray:
    endpose = load_endpose16(hdf5_path)
    if endpose.shape[0] == 0:
        raise ValueError(f"Empty endpose sequence in {hdf5_path}")
    return endpose[0]
