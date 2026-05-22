"""Render compact markdown tables from HSI result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value):
    if value is None:
        return "`n/a`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


def main():
    parser = argparse.ArgumentParser(description="Generate markdown result tables for README/paper notes.")
    parser.add_argument("--feature-groups", default=None, help="Feature-group ablation JSON.")
    parser.add_argument("--online", nargs="*", default=[], help="Online summary JSON files.")
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    sections = []
    if args.feature_groups:
        data = json.loads(Path(args.feature_groups).read_text())
        lines = [
            "| Feature Group | False-Accept Risk | Accepted Success | Acceptance Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, result in data.items():
            metrics = result["eval"]["metrics"]
            lines.append(
                f"| {name} | {_fmt(metrics['false_accept_risk'])} | "
                f"{_fmt(metrics['accepted_grasp_success'])} | {_fmt(metrics['acceptance_rate'])} |"
            )
        sections.append("\n".join(lines))

    if args.online:
        lines = [
            "| Online Summary | Episodes | Success | False-Accept Risk | Acceptance | Refusal | Mean VLA ms/Event |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for path in args.online:
            data = json.loads(Path(path).read_text())
            lines.append(
                f"| {Path(path).stem} | {_fmt(data.get('completed_episodes'))} | "
                f"{_fmt(data.get('task_success_after_reapproach'))} | {_fmt(data.get('false_accept_risk'))} | "
                f"{_fmt(data.get('acceptance_rate'))} | {_fmt(data.get('refusal_rate'))} | "
                f"{_fmt(data.get('mean_vla_inference_ms_per_pregrasp'))} |"
            )
        sections.append("\n".join(lines))

    markdown = "\n\n".join(sections)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
