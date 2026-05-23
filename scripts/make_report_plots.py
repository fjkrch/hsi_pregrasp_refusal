"""Generate report-ready plots for the HSI VLA simulation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _metric(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_groups(vla_dir: Path, output_dir: Path) -> Path:
    data = _load_json(vla_dir / "feature_group_ablations_main600.json")
    results = data.get("results", data)
    groups = list(results)
    false_accept = [_metric(results[group]["eval"]["metrics"]["false_accept_risk"]) for group in groups]
    acceptance = [_metric(results[group]["eval"]["metrics"]["acceptance_rate"]) for group in groups]

    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - width / 2, false_accept, width, label="False-accept risk", color="#B94E48")
    ax.bar(x + width / 2, acceptance, width, label="Acceptance rate", color="#4D7EA8")
    ax.axhline(0.10, color="#2F2F2F", linewidth=1, linestyle="--", label="0.10 target")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("Feature-Group Holdout Results")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.legend()
    path = output_dir / "feature_groups_main600.png"
    _save(fig, path)
    return path


def plot_robustness(vla_dir: Path, output_dir: Path) -> Path:
    groups = ["robot_state", "visual", "robot_visual", "full"]
    shifts = [
        "partial_occlusion100",
        "lighting100_full",
        "clutter100",
        "camera_shift100",
        "object_shift100",
        "approach003_100",
    ]
    values = []
    for group in groups:
        row = []
        for shift in shifts:
            data = _load_json(vla_dir / f"eval_{group}_main600_{shift}.json")
            row.append(_metric(data["metrics"]["false_accept_risk"]))
        values.append(row)
    array = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(array, vmin=0.0, vmax=0.35, cmap="YlOrRd")
    ax.set_title("False-Accept Risk Across Robustness Shifts")
    ax.set_xticks(np.arange(len(shifts)))
    ax.set_xticklabels(shifts, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            label = "n/a" if np.isnan(array[y, x]) else f"{array[y, x]:.2f}"
            ax.text(x, y, label, ha="center", va="center", color="#1F1F1F", fontsize=8)
    fig.colorbar(image, ax=ax, label="False-accept risk")
    path = output_dir / "robustness_false_accept_main600.png"
    _save(fig, path)
    return path


def plot_online(vla_dir: Path, output_dir: Path) -> Path:
    shifts = ["partial_occlusion", "clutter", "camera_shift", "object_shift", "approach003"]
    always = []
    refusal = []
    for shift in shifts:
        always_data = _load_json(vla_dir / f"online_robust_{shift}_always_close_summary.json")
        refusal_data = _load_json(vla_dir / f"online_robust_{shift}_robot_visual_summary.json")
        always.append(_metric(always_data["false_accept_risk"]))
        refusal.append(_metric(refusal_data["false_accept_risk"]))

    x = np.arange(len(shifts))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - width / 2, always, width, label="Always close", color="#8A8A8A")
    ax.bar(x + width / 2, refusal, width, label="Robot+visual refusal", color="#4D7EA8")
    ax.axhline(0.10, color="#2F2F2F", linewidth=1, linestyle="--", label="0.10 target")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("False-accept risk")
    ax.set_title("Online Robustness Smoke Tests")
    ax.set_xticks(x)
    ax.set_xticklabels(shifts, rotation=20, ha="right")
    ax.legend()
    path = output_dir / "online_robust_false_accept_main600.png"
    _save(fig, path)
    return path


def plot_threshold_tuning(vla_dir: Path, output_dir: Path) -> Path:
    data = _load_json(vla_dir / "threshold_tuning_robot_visual_main600.json")
    names = list(data)
    default_values = [_metric(data[name]["default_metrics"]["false_accept_risk"]) for name in names]
    tuned_values = [_metric(data[name]["tuned_metrics"]["false_accept_risk"]) for name in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, default_values, width, label="Checkpoint threshold", color="#B94E48")
    ax.bar(x + width / 2, tuned_values, width, label="Diagnostic tuned threshold", color="#5B9E6D")
    ax.axhline(0.10, color="#2F2F2F", linewidth=1, linestyle="--", label="0.10 target")
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("False-accept risk")
    ax.set_title("Robot+Visual Threshold Tuning Diagnostic")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    path = output_dir / "threshold_tuning_robot_visual_main600.png"
    _save(fig, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Make report plots for VLA simulation results.")
    parser.add_argument("--vla-dir", default="logs/hsi_pregrasp/vla")
    parser.add_argument("--output-dir", default="logs/hsi_pregrasp/vla/report_assets")
    args = parser.parse_args()

    vla_dir = Path(args.vla_dir)
    output_dir = Path(args.output_dir)
    paths = [
        plot_feature_groups(vla_dir, output_dir),
        plot_robustness(vla_dir, output_dir),
        plot_online(vla_dir, output_dir),
        plot_threshold_tuning(vla_dir, output_dir),
    ]
    manifest = output_dir / "report_assets.md"
    manifest.write_text("\n".join(f"- `{path}`" for path in paths) + "\n")
    print(manifest.read_text())


if __name__ == "__main__":
    main()
