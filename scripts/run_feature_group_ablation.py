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
from hsi_pregrasp_refusal.sim_analysis import has_oracle_geometry_columns  # noqa: E402


def _run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _format(value: object) -> str:
    if value is None:
        return "`n/a`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Feature Group Ablation",
        "",
        f"- Train input: `{summary['input']}`",
        f"- Eval input: `{summary['eval_input']}`",
        f"- Target false-accept risk: `{summary['target_false_accept_risk']:.4f}`",
        "",
        "| Feature group | Features | Oracle geometry? | Eval FAR | Accepted success | Acceptance | Refusal |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for group, result in summary["results"].items():
        columns = result["train"]["feature_columns"]
        metrics = result["eval"]["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{group}`",
                    _format(len(columns)),
                    "`yes`" if has_oracle_geometry_columns(columns) else "`no`",
                    _format(metrics["false_accept_risk"]),
                    _format(metrics["accepted_success"]),
                    _format(metrics["acceptance_rate"]),
                    _format(metrics["refusal_rate"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run feature-group ablations for VLA/camera pre-grasp datasets.")
    parser.add_argument("--input", required=True, help="Training/calibration/test CSV.")
    parser.add_argument("--eval-input", required=True, help="Independent evaluation CSV.")
    parser.add_argument("--output", default="logs/hsi_pregrasp/vla/feature_group_ablations.json")
    parser.add_argument("--output-md", default=None, help="Optional markdown table path.")
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

    summary = {
        "input": args.input,
        "eval_input": args.eval_input,
        "target_false_accept_risk": float(args.target_false_accept_risk),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2))
    output_md = None
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(output_md, summary)
    print(json.dumps({"output": str(output), "output_md": None if output_md is None else str(output_md), **summary}, indent=2))


if __name__ == "__main__":
    main()
