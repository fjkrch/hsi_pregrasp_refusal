"""Deterministic pre-grasp trigger helpers."""

from __future__ import annotations

import torch


def detect_pregrasp(
    close_commanded: torch.Tensor,
    gripper_width: torch.Tensor,
    ee_object_distance: torch.Tensor,
    *,
    previous_close_commanded: torch.Tensor | None = None,
    open_width_threshold: float = 0.04,
    near_distance_threshold: float = 0.08,
) -> torch.Tensor:
    """Return a boolean mask for the gripper-closure commitment point.

    The trigger is intentionally simple and deterministic, matching the paper proposal:
    close is commanded while the gripper is still open and the end effector is near the object.
    """
    pregrasp = (
        close_commanded.bool()
        & (gripper_width > open_width_threshold)
        & (ee_object_distance < near_distance_threshold)
    )
    if previous_close_commanded is not None:
        pregrasp = pregrasp & ~previous_close_commanded.bool()
    return pregrasp
