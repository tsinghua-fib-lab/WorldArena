from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CommonRunArgs:
    output_dir: Path
    task_list: str
    max_episode_index: int
    dataset_root: Path
    length_scale: float
    down_sample: int
    policy_model_name: str
    policy_checkpoint_id: str


class AdapterBase:
    wm_name: str

    def build_command(self, args: CommonRunArgs) -> Sequence[str]:
        raise NotImplementedError

    def build_env(self) -> dict[str, str]:
        """Optional: return extra environment variables for the subprocess."""
        return {}
