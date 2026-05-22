from __future__ import annotations

import numpy as np
import torch

from hsi_pregrasp_refusal.calibration import calibrate_threshold
from hsi_pregrasp_refusal.data import infer_feature_columns
from hsi_pregrasp_refusal.features import (
    ALL_FEATURE_COLUMNS,
    FEATURE_GROUPS,
    LANGUAGE_FEATURE_COLUMNS,
    language_feature_row,
    resolve_feature_columns,
)
from hsi_pregrasp_refusal.metrics import compute_refusal_metrics
from hsi_pregrasp_refusal.sim_analysis import (
    estimated_geometry_proxy_scores,
    failure_type,
    metrics_by_failure_type,
    sample_language_target,
    wrong_object_mask,
)
from hsi_pregrasp_refusal.vision import visual_feature_row
from hsi_pregrasp_refusal.vla import build_smolvla_state, summarize_actions


def test_metrics_false_accept_risk_and_rates():
    scores = np.asarray([0.05, 0.10, 0.20, 0.90])
    success = np.asarray([True, False, True, False])
    wrong_object = np.asarray([False, True, False, True])

    metrics = compute_refusal_metrics(scores, success, threshold=0.20, wrong_object=wrong_object)
    metrics_dict = metrics.as_dict()

    assert metrics.accepted_closures == 3
    assert metrics.refused_closures == 1
    assert metrics.failed_accepted_closures == 1
    assert metrics.wrong_object_accepted_closures == 1
    assert metrics.false_accept_risk == 1 / 3
    assert metrics.wrong_object_false_accept_rate == 1 / 3
    assert metrics.acceptance_rate == 0.75
    assert metrics_dict["accepted_success"] == metrics.accepted_grasp_success


def test_calibration_picks_largest_valid_threshold():
    scores = np.asarray([0.05, 0.10, 0.20, 0.90])
    success = np.asarray([True, True, False, False])

    result = calibrate_threshold(scores, success, target_false_accept_risk=0.0)

    assert result.threshold == 0.10
    assert result.accepted_calibration_events == 2
    assert result.empirical_false_accept_risk == 0.0


def test_feature_group_resolution_and_vla_csv_inference():
    assert resolve_feature_columns(feature_group="robot_state") == FEATURE_GROUPS["robot_state"]
    assert resolve_feature_columns(feature_group="visual_language") == FEATURE_GROUPS["visual_language"]
    rows = [{column: "0.1" for column in ALL_FEATURE_COLUMNS}]
    rows[0]["grasp_success"] = "1"

    assert infer_feature_columns(rows) == ALL_FEATURE_COLUMNS


def test_language_feature_row_is_one_hot():
    row = language_feature_row("red")

    assert set(row) == set(LANGUAGE_FEATURE_COLUMNS)
    assert row["language_target_red"] == 1.0
    assert row["language_target_default"] == 0.0


def test_language_sampling_and_wrong_object_masks_are_default_policy_compatible():
    import random

    rng = random.Random(3)
    assert sample_language_target(rng, mode="fixed", default_probability=0.0) == "default"
    rows = [
        {"language_target": "default", "label_reason": "horizon"},
        {"language_target": "red", "label_reason": "wrong_object_default_cube_lifted"},
    ]

    assert wrong_object_mask(rows).tolist() == [False, True]
    assert failure_type(rows[1]) == "wrong_object"


def test_estimated_geometry_proxy_uses_visual_summaries_only():
    weak = {}
    strong = {}
    for camera in ["camera1", "camera2", "camera3"]:
        weak.update(
            {
                f"{camera}_center_mean": "0.50",
                f"{camera}_rgb_mean": "0.50",
                f"{camera}_edge_mean": "0.00",
                f"{camera}_center_std": "0.00",
                f"{camera}_red_mean": "0.50",
                f"{camera}_green_mean": "0.50",
                f"{camera}_blue_mean": "0.50",
            }
        )
        strong.update(
            {
                f"{camera}_center_mean": "0.85",
                f"{camera}_rgb_mean": "0.45",
                f"{camera}_edge_mean": "0.40",
                f"{camera}_center_std": "0.35",
                f"{camera}_red_mean": "0.90",
                f"{camera}_green_mean": "0.20",
                f"{camera}_blue_mean": "0.15",
            }
        )

    scores = estimated_geometry_proxy_scores([weak, strong])

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_metrics_by_failure_type_splits_wrong_object_and_geometric_rows():
    rows = [
        {"language_target": "default", "variant": "single", "grasp_success": "0", "label_reason": "horizon"},
        {
            "language_target": "blue",
            "variant": "wrong_object",
            "grasp_success": "0",
            "label_reason": "wrong_object_default_cube_lifted",
        },
        {"language_target": "default", "variant": "single", "grasp_success": "1", "label_reason": "horizon"},
    ]
    accepted = np.asarray([True, True, False])
    success = np.asarray([False, False, True])

    metrics = metrics_by_failure_type(accepted, success, rows)

    assert metrics["aggregate"]["accepted_closures"] == 2
    assert metrics["wrong_object"]["wrong_object_accepted_closures"] == 1
    assert metrics["geometric_approach"]["failed_accepted_closures"] == 1


def test_visual_feature_row_has_all_three_camera_summaries():
    image = torch.zeros(256, 256, 3, dtype=torch.uint8)
    image[..., 0] = 255
    row = visual_feature_row({"camera1": image, "camera2": image, "camera3": image})

    assert row["camera1_red_mean"] == 1.0
    assert row["camera2_green_mean"] == 0.0
    assert "camera3_edge_mean" in row


def test_vla_state_and_action_uncertainty_summary():
    state = build_smolvla_state(
        torch.tensor([0.5, 0.1, 0.2]),
        torch.tensor([0.4, 0.0, 0.1]),
        torch.tensor([0.55, 0.1, 0.25]),
    )
    actions = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    summary = summarize_actions(actions)

    assert torch.allclose(state, torch.tensor([0.1, 0.1, 0.1, 0.05, 0.0, 0.05]))
    assert summary.action_mean[0] == 0.5
    assert summary.action_variance[0] == 0.25
    assert summary.feature_row()["vla_action_var_mean"] > 0.0
