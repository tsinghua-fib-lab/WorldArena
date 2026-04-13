from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    task_name: str = ""
    first_frame_path: Path | None = None
    instruction_path: Path | None = None
    instruction_variant: str = "base"
    trajectory_hdf5_path: Path | None = None

    @property
    def action_path(self) -> Path | None:
        return self.trajectory_hdf5_path


class PolicyProtocol(Protocol):
    def reset(self) -> None:
        ...

    def infer_actions(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        ...


class WorldModelProtocol(Protocol):
    wm_name: str
    wm_chunk_size: int
