"""CSV loading helpers for pre-grasp event datasets."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import (
    ALL_FEATURE_COLUMNS,
    COMPUTED_FEATURE_COLUMNS,
    PREGRASP_FEATURE_COLUMNS,
    feature_vector,
)


EXCLUDED_COLUMNS = {
    "event_id",
    "env_id",
    "step",
    "time_s",
    "label_horizon_s",
    "label_reason",
    "close_accepted",
    "close_was_refused",
    "grasp_success",
    "decision",
    "risk",
    "threshold",
    "forced_accept",
    "refusals_before_event",
    "vla_samples",
    "vla_inference_ms",
    "variant",
    "language_mode",
    "language_instruction",
    "language_target",
    "failure_type",
    "geometry_good_pregrasp",
}


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def infer_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    """Infer numeric feature columns from CSV rows."""
    if not rows:
        raise ValueError("Cannot infer feature columns from an empty CSV.")
    if all(column in rows[0] for column in ALL_FEATURE_COLUMNS):
        return list(ALL_FEATURE_COLUMNS)
    available_defaults = [column for column in PREGRASP_FEATURE_COLUMNS if column in rows[0]]
    if available_defaults:
        return available_defaults
    return [
        column
        for column, value in rows[0].items()
        if column not in EXCLUDED_COLUMNS and value != "" and _is_float(value)
    ]


def load_event_csv(
    path: str | Path,
    *,
    feature_columns: list[str] | None = None,
    label_column: str = "grasp_success",
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    """Load a pre-grasp event CSV.

    Returns:
        features, success labels, feature column names, raw rows.
    """
    path = Path(path)
    with path.open("r", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    columns = infer_feature_columns(rows) if feature_columns is None else feature_columns
    missing = [
        column
        for column in columns
        if column not in rows[0] and column not in COMPUTED_FEATURE_COLUMNS
    ]
    if label_column not in rows[0]:
        missing.append(label_column)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    try:
        features = np.asarray([feature_vector(row, columns) for row in rows], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"Missing required column in {path}: {exc.args[0]}") from exc
    labels = np.asarray([bool(int(float(row[label_column]))) for row in rows], dtype=bool)
    return features, labels, columns, rows
