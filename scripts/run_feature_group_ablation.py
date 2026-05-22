"""Train/evaluate named feature-group refusal heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.features import FEATURE_GROUPS  # noqa: E402


def _run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Run feature-group ablations for VLA/camera pre-grasp datasets.")
    parser.add_argument("--input", required=True, help="Training/calibration/test CSV.")
    parser.add_argument("--eval-input", required=True, help="Independent evaluation CSV.")
    parser.add_argument("--output", default="logs/hsi_pregrasp/vla/feature_group_ablations.json")
    parser.add_argument("--checkpoint-dir", default="logs/hsi_pregrasp/vla/checkpoints")
    parser.add_argument(
        "--feature-groups",
        default="robot_state,visual,vla_action_uncertainty,robot_visual,full",
        help="Comma-separated feature groups.",
    )
    parser.add_argument("--target_false_accept_risk", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_script = PROJECT_ROOT / "scripts" / "train_refusal_head.py"
    eval_script = PROJECT_ROOT / "scripts" / "evaluate_refusal.py"
    groups = [group.strip() for group in args.feature_groups.split(",") if group.strip()]
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature groups {unknown}; valid groups: {sorted(FEATURE_GROUPS)}")

    results = {}
    for group in groups:
        checkpoint = checkpoint_dir / f"{group}_refusal_head.pt"
        train_summary = _run_json(
            [
                sys.executable,
                str(train_script),
                "--input",
                args.input,
                "--output",
                str(checkpoint),
                "--feature_group",
                group,
                "--target_false_accept_risk",
                str(args.target_false_accept_risk),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ]
        )
        eval_summary = _run_json(
            [
                sys.executable,
                str(eval_script),
                "--input",
                args.eval_input,
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
            ]
        )
        results[group] = {"train": train_summary, "eval": eval_summary}

    output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
