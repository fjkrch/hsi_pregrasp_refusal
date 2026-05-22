"""Run simulation-only proxy baselines on pre-grasp event CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.data import load_event_csv  # noqa: E402
from hsi_pregrasp_refusal.metrics import compute_refusal_metrics  # noqa: E402
from hsi_pregrasp_refusal.sim_analysis import (  # noqa: E402
    estimated_geometry_proxy_scores,
    metrics_by_failure_type,
    oracle_geometry_scores,
    wrong_object_mask,
)


def _accepted_by_rate(scores: np.ndarray, acceptance_rate: float) -> tuple[np.ndarray, float | None]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    total = int(scores.shape[0])
    accepted_count = int(round(float(np.clip(acceptance_rate, 0.0, 1.0)) * total))
    accepted_count = max(0, min(total, accepted_count))
    accepted = np.zeros(total, dtype=bool)
    if accepted_count == 0:
        return accepted, None
    order = np.argsort(scores)
    accepted[order[:accepted_count]] = True
    return accepted, float(scores[order[accepted_count - 1]])


def _metrics_from_mask(
    *,
    name: str,
    kind: str,
    accepted: np.ndarray,
    success: np.ndarray,
    wrong_object: np.ndarray,
    rows: list[dict[str, str]],
    threshold: float | None = None,
    uses_oracle_geometry: bool = False,
    notes: str = "",
) -> dict:
    scores = np.where(accepted, 0.0, 1.0)
    return {
        "name": name,
        "kind": kind,
        "threshold": threshold,
        "uses_oracle_geometry": uses_oracle_geometry,
        "notes": notes,
        "metrics": compute_refusal_metrics(scores, success, threshold=0.5, wrong_object=wrong_object).as_dict(),
        "failure_type_metrics": metrics_by_failure_type(accepted, success, rows, wrong_object),
    }


def _format(value: object) -> str:
    if value is None:
        return "`n/a`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Simulation Proxy Baselines",
        "",
        f"- Input: `{summary['input']}`",
        f"- Events: `{summary['total_events']}`",
        f"- Target acceptance: `{summary['target_acceptance_rate']:.4f}`",
        "",
        "| Method | Oracle geometry? | FAR | Accepted success | Acceptance | Wrong-object FAR | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in summary["results"]:
        metrics = result["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result['name']}`",
                    "`yes`" if result["uses_oracle_geometry"] else "`no`",
                    _format(metrics["false_accept_risk"]),
                    _format(metrics["accepted_success"]),
                    _format(metrics["acceptance_rate"]),
                    _format(metrics["wrong_object_false_accept_rate"]),
                    result["notes"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure-Type Splits",
            "",
            "| Method | Failure type | Events | Accepted | FAR | Wrong-object FAR | Acceptance |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in summary["results"]:
        for failure_name, metrics in result["failure_type_metrics"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{result['name']}`",
                        f"`{failure_name}`",
                        _format(metrics["total_events"]),
                        _format(metrics["accepted_closures"]),
                        _format(metrics["false_accept_risk"]),
                        _format(metrics["wrong_object_false_accept_rate"]),
                        _format(metrics["acceptance_rate"]),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Evaluation CSV.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--target-acceptance-rate",
        type=float,
        default=1.0,
        help="Acceptance rate for matched proxy/oracle rows. Use a checkpoint acceptance rate for fair comparisons.",
    )
    args = parser.parse_args()

    _, success, _, rows = load_event_csv(args.input, feature_columns=[])
    success = success.astype(bool)
    wrong_object = wrong_object_mask(rows)
    target_acceptance = float(np.clip(args.target_acceptance_rate, 0.0, 1.0))

    results = [
        _metrics_from_mask(
            name="always_close",
            kind="baseline",
            accepted=np.ones_like(success, dtype=bool),
            success=success,
            wrong_object=wrong_object,
            rows=rows,
            notes="Accepts every pre-grasp event.",
        )
    ]

    proxy_scores = estimated_geometry_proxy_scores(rows)
    proxy_accepted, proxy_threshold = _accepted_by_rate(proxy_scores, target_acceptance)
    results.append(
        _metrics_from_mask(
            name="estimated_geometry_proxy",
            kind="image_summary_proxy",
            accepted=proxy_accepted,
            success=success,
            wrong_object=wrong_object,
            rows=rows,
            threshold=proxy_threshold,
            notes="Uses stored camera summary statistics only; this is a proxy, not true simulator pose.",
        )
    )

    oracle_scores = oracle_geometry_scores(rows)
    oracle_accepted, oracle_threshold = _accepted_by_rate(oracle_scores, target_acceptance)
    results.append(
        _metrics_from_mask(
            name="oracle_geometry_upper_bound",
            kind="oracle_geometry",
            accepted=oracle_accepted,
            success=success,
            wrong_object=wrong_object,
            rows=rows,
            threshold=oracle_threshold,
            uses_oracle_geometry=True,
            notes="Uses simulator-state geometry columns; keep as an upper bound.",
        )
    )

    summary = {
        "input": args.input,
        "total_events": int(success.shape[0]),
        "successes": int(success.sum()),
        "failures": int((~success).sum()),
        "wrong_object_events": int(wrong_object.sum()),
        "target_acceptance_rate": target_acceptance,
        "results": results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(output_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
