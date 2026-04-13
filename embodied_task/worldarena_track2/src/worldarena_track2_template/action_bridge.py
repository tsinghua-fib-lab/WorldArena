#!/usr/bin/env python3
"""
Action-space bridge: joint14 <-> endpose14.

Two WM action-space modes exist in WorldArena Track 2:

  joint14   – [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
              WMs trained on joint angles.
              Policy output can be fed DIRECTLY to the WM.

  endpose14 – [left_xyz(3), left_qxyz(3), left_grip(1),
               right_xyz(3), right_qxyz(3), right_grip(1)]
              WMs trained on end-effector poses with qw dropped.
              Policy output (joint14) must be BRIDGED to endpose14 first.

The bridge uses a kNN lookup on paired (joint14, endpose14) data from the
evaluation dataset itself.  Each episode provides:
  - actions/episodeK.npy  -> (T, 14) joint14
  - states/episodeK.npy   -> (T, 16) endpose16  (xyz + quat_xyzw + grip) x 2

This module is self-contained (numpy only) and can be dropped into any
closedloop rollout script.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Conversion helpers ──────────────────────────────────────────────────────

def state16_to_endpose14(state: np.ndarray) -> np.ndarray:
    """Drop qw (dims 6, 14) from 16D endpose -> 14D endpose."""
    return np.concatenate(
        [state[..., 0:6], state[..., 7:8], state[..., 8:14], state[..., 15:16]],
        axis=-1,
    ).astype(np.float32)


def state16_to_policy14(state: np.ndarray) -> np.ndarray:
    """First 14 dims of 16D state -> policy input (legacy compat)."""
    return state[..., :14].astype(np.float32)


# ── kNN Bridge ──────────────────────────────────────────────────────────────

class JointEndposeBridge:
    """
    Maps joint14 policy actions to endpose14 WM actions via kNN on paired data.

    Usage::

        bridge = JointEndposeBridge(mode="task_knn")
        # task_episodes: list of dicts with "action_path" and "state_path"
        wm_actions = bridge.joint14_to_endpose14(
            task_name, task_episodes, joint_actions, last_gripper=0.0
        )

    Two modes:
      - "task_knn" : inverse-distance-weighted kNN on per-task paired data
      - "passthrough" : return joint14 as-is (for joint14-trained WMs)
    """

    def __init__(
        self,
        mode: str = "task_knn",
        k: int = 5,
        sample_stride: int = 1,
        max_points: int = 0,
    ):
        self.mode = mode
        self.k = max(1, k)
        self.sample_stride = max(1, sample_stride)
        self.max_points = max(0, max_points)
        self._task_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ── public API ──

    def joint14_to_endpose14(
        self,
        task_name: str,
        task_episodes: list[dict[str, Any]],
        actions: np.ndarray,
        last_gripper: float = 0.0,
    ) -> np.ndarray:
        """
        Convert (N, 14) joint actions to (N, 14) endpose actions.

        For mode="passthrough", returns actions unchanged.
        For mode="task_knn", uses per-task paired data for kNN lookup.
        """
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = np.asarray(actions, dtype=np.float32)

        if self.mode == "passthrough":
            return actions.copy()

        try:
            if task_name not in self._task_cache:
                self._build_task_bank(task_name, task_episodes)
            joint_bank, wm_bank, joint_sq = self._task_cache[task_name]
            k = min(self.k, len(joint_bank))
            out = np.empty((actions.shape[0], wm_bank.shape[1]), dtype=np.float32)
            for i, row in enumerate(actions):
                dist = joint_sq - 2.0 * (joint_bank @ row) + float(np.dot(row, row))
                if k < len(dist):
                    idx = np.argpartition(dist, k - 1)[:k]
                else:
                    idx = np.arange(len(dist))
                local_dist = np.maximum(dist[idx], 1e-8)
                weights = 1.0 / local_dist
                weights /= np.sum(weights)
                out[i] = np.sum(wm_bank[idx] * weights[:, None], axis=0)
            return out
        except Exception as exc:
            log.warning("Bridge fallback to passthrough for %s: %s", task_name, exc)
            return actions.copy()

    # ── internals ──

    def _build_task_bank(self, task_name: str, task_episodes: list[dict[str, Any]]) -> None:
        joint_chunks: list[np.ndarray] = []
        wm_chunks: list[np.ndarray] = []
        for ep in task_episodes:
            state_path = ep.get("state_path")
            action_path = ep.get("action_path")
            if not state_path or not action_path:
                continue
            try:
                joints = np.load(action_path).astype(np.float32)
                states = np.load(state_path).astype(np.float32)
            except Exception as exc:
                log.warning("Bridge skip %s/%s: %s", task_name, ep.get("episode", "?"), exc)
                continue
            if joints.ndim != 2 or joints.shape[-1] != 14:
                continue
            if states.ndim != 2:
                continue
            if states.shape[-1] == 16:
                wm_states = state16_to_endpose14(states)
            elif states.shape[-1] == 14:
                wm_states = states.astype(np.float32)
            else:
                continue
            n = min(len(joints), len(wm_states))
            if n <= 0:
                continue
            joint_chunks.append(joints[:n:self.sample_stride])
            wm_chunks.append(wm_states[:n:self.sample_stride])

        if not joint_chunks:
            raise RuntimeError(f"No valid bridge pairs found for task {task_name}")

        joint_bank = np.concatenate(joint_chunks, axis=0).astype(np.float32, copy=False)
        wm_bank = np.concatenate(wm_chunks, axis=0).astype(np.float32, copy=False)

        if self.max_points > 0 and len(joint_bank) > self.max_points:
            keep = np.linspace(0, len(joint_bank) - 1, num=self.max_points, dtype=np.int64)
            joint_bank = joint_bank[keep]
            wm_bank = wm_bank[keep]

        joint_sq = np.sum(joint_bank * joint_bank, axis=1)
        self._task_cache[task_name] = (joint_bank, wm_bank, joint_sq)
        log.info("Bridge[%s]: %d pairs, mode=%s, k=%d", task_name, len(joint_bank), self.mode, self.k)
