from __future__ import annotations

import math

from pathlib import Path


def downsampled_length(total_frames: int, down_sample: int) -> int:
    if down_sample <= 0:
        raise ValueError(f"down_sample must be positive, got {down_sample}")
    return len(range(0, total_frames, down_sample))


def target_length(reference_length: int, length_scale: float = 1.2, down_sample: int = 1) -> int:
    ds_len = downsampled_length(reference_length, down_sample)
    return int(math.ceil(ds_len * length_scale))


def trajectory_length(joint14_actions) -> int:
    if getattr(joint14_actions, "ndim", None) != 2:
        raise ValueError(f"Expected 2D joint14 action array, got shape {getattr(joint14_actions, 'shape', None)}")
    return int(joint14_actions.shape[0])


def target_length_from_joint14(joint14_actions, length_scale: float = 1.2, down_sample: int = 1) -> int:
    return target_length(trajectory_length(joint14_actions), length_scale=length_scale, down_sample=down_sample)


def target_length_from_hdf5(hdf5_path: Path, length_scale: float = 1.2, down_sample: int = 1) -> int:
    from .dataset import load_joint14_actions

    return target_length_from_joint14(load_joint14_actions(hdf5_path), length_scale=length_scale, down_sample=down_sample)
