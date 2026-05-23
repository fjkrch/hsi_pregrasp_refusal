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

VISUAL_PROXY_FEATURE_COLUMNS = [
    "visual_proxy_center_contrast_mean",
    "visual_proxy_center_contrast_max",
    "visual_proxy_center_std_mean",
    "visual_proxy_edge_mean",
    "visual_proxy_color_separation_mean",
    "visual_proxy_camera_rgb_disagreement",
    "visual_proxy_center_mean_disagreement",
    "visual_proxy_evidence_score",
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
    "visual_proxy": VISUAL_PROXY_FEATURE_COLUMNS,
    "language": LANGUAGE_FEATURE_COLUMNS,
    "vla_action_uncertainty": VLA_ACTION_FEATURE_COLUMNS,
    "vla_only": VLA_ACTION_FEATURE_COLUMNS,
    "visual_language": [*VISUAL_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
    "visual_proxy_language": [*VISUAL_PROXY_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
    "robot_visual": [*PREGRASP_FEATURE_COLUMNS, *VISUAL_FEATURE_COLUMNS],
    "robot_visual_proxy": [*PREGRASP_FEATURE_COLUMNS, *VISUAL_PROXY_FEATURE_COLUMNS],
    "robot_visual_language": [*PREGRASP_FEATURE_COLUMNS, *VISUAL_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
    "robot_visual_proxy_language": [
        *PREGRASP_FEATURE_COLUMNS,
        *VISUAL_PROXY_FEATURE_COLUMNS,
        *LANGUAGE_FEATURE_COLUMNS,
    ],
    "full": ALL_FEATURE_COLUMNS,
    "full_language": [*ALL_FEATURE_COLUMNS, *LANGUAGE_FEATURE_COLUMNS],
    "full_proxy_language": [
        *PREGRASP_FEATURE_COLUMNS,
        *VISUAL_PROXY_FEATURE_COLUMNS,
        *VLA_ACTION_FEATURE_COLUMNS,
        *LANGUAGE_FEATURE_COLUMNS,
    ],
}

COMPUTED_FEATURE_COLUMNS = set(VISUAL_PROXY_FEATURE_COLUMNS)


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
    visual_proxy_row: dict[str, float] | None = None
    values = []
    for column in columns:
        if column in row and row[column] != "":
            values.append(float(row[column]))
            continue
        if column in COMPUTED_FEATURE_COLUMNS:
            if visual_proxy_row is None:
                visual_proxy_row = visual_proxy_feature_row(row)
            values.append(float(visual_proxy_row[column]))
            continue
        raise KeyError(column)
    return values


def language_feature_row(target: str) -> dict[str, float]:
    """Return one-hot language target features for a task command."""
    if target not in LANGUAGE_TARGETS:
        valid = ", ".join(LANGUAGE_TARGETS)
        raise ValueError(f"Unknown language target {target!r}. Valid targets: {valid}")
    return {f"language_target_{name}": float(name == target) for name in LANGUAGE_TARGETS}


def _row_float(row: Mapping[str, object], column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def visual_proxy_feature_row(row: Mapping[str, object]) -> dict[str, float]:
    """Return cheap visual proxy features from stored camera summary columns.

    These columns intentionally avoid simulator pose and raw image tensors. They
    provide a small learned path for CSV-only runs when the full visual summary
    feature set is too wide or raw camera frames are unavailable.
    """
    center_contrasts = []
    center_stds = []
    edge_means = []
    color_separations = []
    rgb_means = []
    center_means = []
    for camera in SMOLVLA_CAMERA_KEYS:
        rgb_mean = _row_float(row, f"{camera}_rgb_mean")
        center_mean = _row_float(row, f"{camera}_center_mean")
        red = _row_float(row, f"{camera}_red_mean")
        green = _row_float(row, f"{camera}_green_mean")
        blue = _row_float(row, f"{camera}_blue_mean")
        center_contrasts.append(abs(center_mean - rgb_mean))
        center_stds.append(_row_float(row, f"{camera}_center_std"))
        edge_means.append(_row_float(row, f"{camera}_edge_mean"))
        color_separations.append(max(red, green, blue) - min(red, green, blue))
        rgb_means.append(rgb_mean)
        center_means.append(center_mean)

    center_contrast = sum(center_contrasts) / len(center_contrasts)
    center_std = sum(center_stds) / len(center_stds)
    edge_mean = sum(edge_means) / len(edge_means)
    color_separation = sum(color_separations) / len(color_separations)
    evidence = 0.35 * center_contrast + 0.25 * center_std + 0.25 * edge_mean + 0.15 * color_separation
    return {
        "visual_proxy_center_contrast_mean": center_contrast,
        "visual_proxy_center_contrast_max": max(center_contrasts),
        "visual_proxy_center_std_mean": center_std,
        "visual_proxy_edge_mean": edge_mean,
        "visual_proxy_color_separation_mean": color_separation,
        "visual_proxy_camera_rgb_disagreement": max(rgb_means) - min(rgb_means),
        "visual_proxy_center_mean_disagreement": max(center_means) - min(center_means),
        "visual_proxy_evidence_score": evidence,
    }
