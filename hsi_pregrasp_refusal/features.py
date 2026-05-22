"""Feature definitions for pre-grasp refusal experiments."""

from __future__ import annotations

from collections.abc import Mapping


PREGRASP_FEATURE_COLUMNS = [
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

CAMERA_FEATURE_NAMES = [
    "rgb_mean",
    "rgb_std",
    "red_mean",
    "green_mean",
    "blue_mean",
    "center_mean",
    "center_std",
    "edge_mean",
]

SMOLVLA_CAMERA_KEYS = ["camera1", "camera2", "camera3"]

VISUAL_FEATURE_COLUMNS = [
    f"{camera}_{feature}" for camera in SMOLVLA_CAMERA_KEYS for feature in CAMERA_FEATURE_NAMES
]

VLA_ACTION_FEATURE_COLUMNS = [
    *[f"vla_action_mean_{idx}" for idx in range(6)],
    *[f"vla_action_std_{idx}" for idx in range(6)],
    *[f"vla_action_var_{idx}" for idx in range(6)],
    "vla_action_var_mean",
    "vla_action_std_norm",
    "vla_action_range_norm",
    "vla_action_entropy_proxy",
]

LANGUAGE_TARGETS = ["default", "blue", "red", "green", "yellow"]

LANGUAGE_FEATURE_COLUMNS = [f"language_target_{target}" for target in LANGUAGE_TARGETS]

ALL_FEATURE_COLUMNS = [
    *PREGRASP_FEATURE_COLUMNS,
    *VISUAL_FEATURE_COLUMNS,
    *VLA_ACTION_FEATURE_COLUMNS,
]

FEATURE_GROUPS = {
    "robot_state": PREGRASP_FEATURE_COLUMNS,
    "visual": VISUAL_FEATURE_COLUMNS,
    "language": LANGUAGE_FEATURE_COLUMNS,
    "vla_action_uncertainty": VLA_ACTION_FEATURE_COLUMNS,
    "vla_only": VLA_ACTION_FEATURE_COLUMNS,
    "visual_language": [*VISUAL_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
    "robot_visual": [*PREGRASP_FEATURE_COLUMNS, *VISUAL_FEATURE_COLUMNS],
    "robot_visual_language": [*PREGRASP_FEATURE_COLUMNS, *VISUAL_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
    "full": ALL_FEATURE_COLUMNS,
    "full_language": [*ALL_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
}


def resolve_feature_columns(
    *, feature_group: str | None = None, feature_columns: list[str] | None = None
) -> list[str] | None:
    """Resolve an optional feature group and explicit column list."""
    if feature_columns:
        return feature_columns
    if feature_group is None:
        return None
    try:
        return list(FEATURE_GROUPS[feature_group])
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_GROUPS))
        raise ValueError(f"Unknown feature group {feature_group!r}. Valid groups: {valid}") from exc


def feature_vector(row: Mapping[str, object], feature_columns: list[str] | None = None) -> list[float]:
    """Convert a CSV row-like object to a numeric feature vector."""
    columns = PREGRASP_FEATURE_COLUMNS if feature_columns is None else feature_columns
    return [float(row[column]) for column in columns]


def language_feature_row(target: str) -> dict[str, float]:
    """Return one-hot language target features for a task command."""
    if target not in LANGUAGE_TARGETS:
        valid = ", ".join(LANGUAGE_TARGETS)
        raise ValueError(f"Unknown language target {target!r}. Valid targets: {valid}")
    return {f"language_target_{name}": float(name == target) for name in LANGUAGE_TARGETS}
