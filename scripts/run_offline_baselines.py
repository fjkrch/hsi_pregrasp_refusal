"""Run offline pre-grasp refusal baselines on event CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.calibration import calibrate_threshold  # noqa: E402
from hsi_pregrasp_refusal.data import load_event_csv  # noqa: E402
from hsi_pregrasp_refusal.metrics import compute_refusal_metrics  # noqa: E402


def _split_indices(count: int, seed: int, train_fraction: float, calibration_fraction: float):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    train_end = max(1, int(round(count * train_fraction)))
    calibration_end = min(count - 1, train_end + max(1, int(round(count * calibration_fraction))))
    return indices[:train_end], indices[train_end:calibration_end], indices[calibration_end:]


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _random_matched_metrics(success: np.ndarray, acceptance_rate: float, *, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(repeats):
        accepted = rng.random(success.shape[0]) < acceptance_rate
        accepted_count = int(accepted.sum())
        accepted_success = int((accepted & success).sum())
        accepted_fail = int((accepted & ~success).sum())
        total = int(success.shape[0])
        metrics.append(
            {
                "total_events": total,
                "accepted_closures": accepted_count,
                "refused_closures": total - accepted_count,
                "successful_accepted_closures": accepted_success,
                "failed_accepted_closures": accepted_fail,
                "false_accept_risk": _safe_div(accepted_fail, accepted_count),
                "accepted_grasp_success": _safe_div(accepted_success, accepted_count),
                "acceptance_rate": accepted_count / total if total else 0.0,
                "refusal_rate": 1.0 - accepted_count / total if total else 0.0,
            }
        )
    return {
        key: float(np.mean([item[key] for item in metrics if item[key] is not None]))
        for key in metrics[0]
        if not isinstance(metrics[0][key], str)
    }


def _calibrated_score_result(scores: np.ndarray, success: np.ndarray, ids, eval_scores, eval_success, args):
    calibration = calibrate_threshold(
        scores[ids["calibration"]],
        success[ids["calibration"]],
        target_false_accept_risk=args.target_false_accept_risk,
        min_acceptance_rate=args.min_acceptance_rate,
        risk_estimator=args.risk_estimator,
    )
    return {
        "threshold": calibration.as_dict(),
        "train": compute_refusal_metrics(scores[ids["train"]], success[ids["train"]], calibration.threshold).as_dict(),
        "calibration": compute_refusal_metrics(
            scores[ids["calibration"]], success[ids["calibration"]], calibration.threshold
        ).as_dict(),
        "test": compute_refusal_metrics(scores[ids["test"]], success[ids["test"]], calibration.threshold).as_dict(),
        "eval": compute_refusal_metrics(eval_scores, eval_success, calibration.threshold).as_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run offline pre-grasp refusal baselines.")
    parser.add_argument("--input", required=True, help="Training/calibration/test CSV.")
    parser.add_argument("--eval-input", required=True, help="Independent evaluation CSV.")
    parser.add_argument("--output", default="logs/hsi_pregrasp/offline_baselines.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--calibration_fraction", type=float, default=0.20)
    parser.add_argument("--target_false_accept_risk", type=float, default=0.10)
    parser.add_argument("--min_acceptance_rate", type=float, default=0.0)
    parser.add_argument("--risk_estimator", choices=["empirical", "add_one"], default="empirical")
    parser.add_argument("--matched_acceptance_rate", type=float, default=0.835)
    parser.add_argument("--random_repeats", type=int, default=1000)
    args = parser.parse_args()

    _, success, _, rows = load_event_csv(args.input)
    _, eval_success, _, eval_rows = load_event_csv(args.eval_input)
    train_ids, calibration_ids, test_ids = _split_indices(
        len(success), args.seed, args.train_fraction, args.calibration_fraction
    )
    ids = {"train": train_ids, "calibration": calibration_ids, "test": test_ids}

    all_zero_scores = np.zeros(len(success), dtype=np.float32)
    eval_zero_scores = np.zeros(len(eval_success), dtype=np.float32)
    results = {
        "always_close": {
            "test": compute_refusal_metrics(all_zero_scores[test_ids], success[test_ids], threshold=0.0).as_dict(),
            "eval": compute_refusal_metrics(eval_zero_scores, eval_success, threshold=0.0).as_dict(),
        },
        "matched_random_refusal": {
            "acceptance_rate_target": args.matched_acceptance_rate,
            "test": _random_matched_metrics(
                success[test_ids], args.matched_acceptance_rate, repeats=args.random_repeats, seed=args.seed
            ),
            "eval": _random_matched_metrics(
                eval_success, args.matched_acceptance_rate, repeats=args.random_repeats, seed=args.seed + 1
            ),
        },
    }

    scalar_baselines = {
        "distance_only": "ee_object_distance",
        "lateral_error_only": "ee_object_lateral_error",
        "height_error_only": "ee_object_height_error",
        "action_delta_pos_only": "action_delta_pos_norm",
        "vla_action_variance_only": "vla_action_var_mean",
        "vla_entropy_proxy_only": "vla_action_entropy_proxy",
        "vla_std_norm_only": "vla_action_std_norm",
        "table_camera_center_mean_only": "camera1_center_mean",
        "wrist_camera_center_mean_only": "camera2_center_mean",
        "overhead_camera_center_mean_only": "camera3_center_mean",
    }
    for name, column in scalar_baselines.items():
        if column not in rows[0] or column not in eval_rows[0]:
            continue
        scores = np.asarray([float(row[column]) for row in rows], dtype=np.float32)
        eval_scores = np.asarray([float(row[column]) for row in eval_rows], dtype=np.float32)
        results[name] = _calibrated_score_result(scores, success, ids, eval_scores, eval_success, args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
