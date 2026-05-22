"""Simulation-only analysis helpers for language and visual proxy baselines."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

import numpy as np

from .features import LANGUAGE_TARGETS, PREGRASP_FEATURE_COLUMNS
from .metrics import compute_refusal_metrics


ORACLE_GEOMETRY_COLUMNS = [
    "ee_object_distance",
    "ee_object_lateral_error",
    "ee_object_height_error",
    "object_height",
    "gripper_width",
    "action_delta_pos_norm",
    "action_delta_rot_distance",
    "close_commanded",
    "sm_state",
    "sm_wait_time",
]


def instruction_for_target(target: str, fixed_task: str) -> str:
    """Return an instruction string for a sampled language target."""
    if target == "default":
        return fixed_task
    if target not in LANGUAGE_TARGETS:
        valid = ", ".join(LANGUAGE_TARGETS)
        raise ValueError(f"Unknown language target {target!r}. Valid targets: {valid}")
    return f"pick up the {target} cube"


def sample_language_target(rng: random.Random, *, mode: str, default_probability: float) -> str:
    """Sample the language target used by simulation-only wrong-object runs."""
    if mode == "fixed":
        return "default"
    if mode != "multi_object_default_policy":
        raise ValueError(f"Unsupported language mode {mode!r}")
    if rng.random() < default_probability:
        return "default"
    return rng.choice([target for target in LANGUAGE_TARGETS if target != "default"])


def wrong_object_mask(rows: Iterable[Mapping[str, object]]) -> np.ndarray:
    """Return rows where accepting the default policy violates a non-default language command."""
    mask = []
    for row in rows:
        target = str(row.get("language_target", "") or "")
        reason = str(row.get("label_reason", "") or "")
        mask.append((target != "" and target != "default") or reason.startswith("wrong_object"))
    return np.asarray(mask, dtype=bool)


def failure_type(row: Mapping[str, object]) -> str:
    """Assign a coarse simulation failure bucket from existing CSV metadata."""
    target = str(row.get("language_target", "") or "")
    reason = str(row.get("label_reason", "") or "").lower()
    variant = str(row.get("variant", "") or "").lower()
    success = str(row.get("grasp_success", "") or "")
    if (target and target != "default") or reason.startswith("wrong_object"):
        return "wrong_object"
    if "occlusion" in reason or "occlusion" in variant or "clutter" in reason or "clutter" in variant:
        return "occlusion_clutter"
    if "shift" in reason or "shift" in variant:
        return "shift"
    if success in {"1", "1.0", "true", "True"}:
        return "success"
    return "geometric_approach"


def failure_type_masks(rows: list[Mapping[str, object]]) -> dict[str, np.ndarray]:
    """Return aggregate and available failure-type masks for result splits."""
    labels = np.asarray([failure_type(row) for row in rows], dtype=object)
    masks = {"aggregate": np.ones(labels.shape[0], dtype=bool)}
    for name in ["wrong_object", "geometric_approach", "occlusion_clutter", "shift", "success"]:
        mask = labels == name
        if bool(mask.any()):
            masks[name] = mask
    return masks


def _row_float(row: Mapping[str, object], column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _robust_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = np.nanpercentile(values, [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def estimated_geometry_proxy_scores(rows: list[Mapping[str, object]]) -> np.ndarray:
    """Score failure risk using only stored image-summary statistics.

    This is an image-summary-derived proxy, not true geometry. It rewards centered,
    textured, high-contrast object evidence in the stored camera summaries and treats
    weak evidence as higher pre-grasp risk.
    """
    raw_safety = []
    for row in rows:
        per_camera = []
        for camera in ["camera1", "camera2", "camera3"]:
            center = _row_float(row, f"{camera}_center_mean")
            edge = _row_float(row, f"{camera}_edge_mean")
            center_std = _row_float(row, f"{camera}_center_std")
            red = _row_float(row, f"{camera}_red_mean")
            green = _row_float(row, f"{camera}_green_mean")
            blue = _row_float(row, f"{camera}_blue_mean")
            color_separation = max(red, green, blue) - min(red, green, blue)
            center_contrast = abs(center - _row_float(row, f"{camera}_rgb_mean"))
            per_camera.append(0.35 * center_contrast + 0.30 * edge + 0.20 * center_std + 0.15 * color_separation)
        raw_safety.append(float(np.mean(per_camera)))
    safety = _robust_minmax(np.asarray(raw_safety, dtype=np.float64))
    return (1.0 - safety).astype(np.float64)


def oracle_geometry_scores(rows: list[Mapping[str, object]]) -> np.ndarray:
    """Score failure risk from simulator-state geometry as an upper-bound baseline."""
    distance = np.asarray([_row_float(row, "ee_object_distance") for row in rows], dtype=np.float64)
    lateral = np.asarray([_row_float(row, "ee_object_lateral_error") for row in rows], dtype=np.float64)
    height = np.asarray([abs(_row_float(row, "ee_object_height_error")) for row in rows], dtype=np.float64)
    action = np.asarray([_row_float(row, "action_delta_pos_norm") for row in rows], dtype=np.float64)
    score = 0.35 * _robust_minmax(distance) + 0.30 * _robust_minmax(lateral)
    score += 0.20 * _robust_minmax(height) + 0.15 * _robust_minmax(action)
    return score.astype(np.float64)


def metrics_by_failure_type(
    accepted: np.ndarray,
    success: np.ndarray,
    rows: list[Mapping[str, object]],
    wrong_object: np.ndarray | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Compute refusal metrics for aggregate and each failure bucket present."""
    accepted = np.asarray(accepted, dtype=bool).reshape(-1)
    success = np.asarray(success, dtype=bool).reshape(-1)
    wrong = wrong_object_mask(rows) if wrong_object is None else np.asarray(wrong_object, dtype=bool).reshape(-1)
    scores = np.where(accepted, 0.0, 1.0)
    result = {}
    for name, mask in failure_type_masks(rows).items():
        result[name] = compute_refusal_metrics(
            scores[mask],
            success[mask],
            threshold=0.5,
            wrong_object=wrong[mask],
        ).as_dict()
    return result


def has_oracle_geometry_columns(columns: Iterable[str]) -> bool:
    """Return whether a method uses true simulator geometry columns."""
    return bool(set(columns) & set(PREGRASP_FEATURE_COLUMNS))
