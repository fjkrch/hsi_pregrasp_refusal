"""Matched-acceptance refusal comparisons with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.data import load_event_csv  # noqa: E402
from hsi_pregrasp_refusal.metrics import compute_refusal_metrics  # noqa: E402
from hsi_pregrasp_refusal.model import RefusalHead  # noqa: E402
from hsi_pregrasp_refusal.sim_analysis import (  # noqa: E402
    ORACLE_GEOMETRY_COLUMNS,
    estimated_geometry_proxy_scores,
    metrics_by_failure_type,
    oracle_geometry_scores,
    wrong_object_mask,
)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_scores(input_path: str | Path, checkpoint_path: str | Path, device: torch.device) -> tuple[np.ndarray, dict]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    feature_columns = list(checkpoint["feature_columns"])
    features, _, _, _ = load_event_csv(input_path, feature_columns=feature_columns)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(checkpoint["feature_std"], dtype=np.float32), 1e-6)
    features_std = (features - mean) / std

    model = RefusalHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(int(value) for value in checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(features_std, dtype=torch.float32, device=device)
        scores = model.predict_failure_probability(tensor).detach().cpu().numpy()
    return scores.astype(np.float64), checkpoint


def _parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=PATH.")
    return name, path


def _parse_scalar(value: str) -> tuple[str, str, str]:
    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=COLUMN:low or NAME=COLUMN:high.")
    name, spec = value.split("=", 1)
    column, direction = spec.rsplit(":", 1)
    if direction not in {"low", "high"}:
        raise argparse.ArgumentTypeError("Scalar direction must be 'low' or 'high'.")
    if not name or not column:
        raise argparse.ArgumentTypeError("Expected non-empty NAME and COLUMN.")
    return name, column, direction


def _parse_proxy(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=estimated_geometry_proxy or NAME=oracle_geometry_proxy.")
    name, proxy = value.split("=", 1)
    valid = {"estimated_geometry_proxy", "oracle_geometry_proxy"}
    if not name or proxy not in valid:
        raise argparse.ArgumentTypeError(f"Expected non-empty NAME and proxy in {sorted(valid)}.")
    return name, proxy


def _accepted_by_rate(scores: np.ndarray, acceptance_rate: float, *, low_is_safe: bool) -> tuple[np.ndarray, float | None]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    total = scores.shape[0]
    accepted_count = int(round(float(np.clip(acceptance_rate, 0.0, 1.0)) * total))
    accepted_count = max(0, min(total, accepted_count))
    accepted = np.zeros(total, dtype=bool)
    if accepted_count == 0:
        return accepted, None
    order = np.argsort(scores if low_is_safe else -scores)
    accepted[order[:accepted_count]] = True
    threshold_idx = order[accepted_count - 1]
    return accepted, float(scores[threshold_idx])


def _metrics_from_mask(accepted: np.ndarray, success: np.ndarray, wrong_object: np.ndarray | None = None) -> dict:
    accepted = np.asarray(accepted, dtype=bool).reshape(-1)
    success = np.asarray(success, dtype=bool).reshape(-1)
    scores = np.where(accepted, 0.0, 1.0)
    metrics = compute_refusal_metrics(scores, success, threshold=0.5, wrong_object=wrong_object).as_dict()
    return metrics


def _metric_value(
    accepted: np.ndarray,
    success: np.ndarray,
    wrong_object: np.ndarray | None,
    metric: str,
) -> float:
    metrics = _metrics_from_mask(accepted, success, wrong_object)
    value = metrics[metric]
    return float("nan") if value is None else float(value)


def _bootstrap_ci(
    accepted: np.ndarray,
    success: np.ndarray,
    wrong_object: np.ndarray | None,
    *,
    metric: str,
    bootstrap: int,
    rng: np.random.Generator,
    random_acceptance_rate: float | None = None,
) -> list[float | None]:
    values: list[float] = []
    total = success.shape[0]
    wrong_object_mask = None if wrong_object is None else np.asarray(wrong_object, dtype=bool).reshape(-1)
    for _ in range(bootstrap):
        indices = rng.integers(0, total, size=total)
        sampled_success = success[indices]
        sampled_wrong_object = None if wrong_object_mask is None else wrong_object_mask[indices]
        if random_acceptance_rate is None:
            sampled_accepted = accepted[indices]
        else:
            sampled_accepted = np.zeros(total, dtype=bool)
            accepted_count = int(round(random_acceptance_rate * total))
            if accepted_count > 0:
                chosen = rng.choice(total, size=min(accepted_count, total), replace=False)
                sampled_accepted[chosen] = True
        values.append(_metric_value(sampled_accepted, sampled_success, sampled_wrong_object, metric))
    array = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return [None, None]
    lo, hi = np.nanpercentile(array, [2.5, 97.5])
    return [float(lo), float(hi)]


def _summarize(
    *,
    name: str,
    kind: str,
    accepted: np.ndarray,
    success: np.ndarray,
    wrong_object: np.ndarray,
    rows: list[dict[str, str]],
    bootstrap: int,
    rng: np.random.Generator,
    acceptance_mode: str,
    threshold: float | None = None,
    columns: list[str] | None = None,
    uses_oracle_geometry: bool | None = None,
    random_acceptance_rate: float | None = None,
) -> dict:
    metrics = _metrics_from_mask(accepted, success, wrong_object)
    columns = [] if columns is None else columns
    uses_oracle = bool(set(columns) & set(ORACLE_GEOMETRY_COLUMNS))
    if uses_oracle_geometry is not None:
        uses_oracle = uses_oracle_geometry
    result = {
        "name": name,
        "kind": kind,
        "acceptance_mode": acceptance_mode,
        "threshold": threshold,
        "uses_oracle_geometry": uses_oracle,
        "feature_columns": columns,
        "metrics": metrics,
        "failure_type_metrics": metrics_by_failure_type(accepted, success, rows, wrong_object),
        "ci95": {},
    }
    for metric in ["false_accept_risk", "accepted_success", "acceptance_rate", "wrong_object_false_accept_rate"]:
        result["ci95"][metric] = _bootstrap_ci(
            accepted,
            success,
            wrong_object,
            metric=metric,
            bootstrap=bootstrap,
            rng=rng,
            random_acceptance_rate=random_acceptance_rate,
        )
    result["ci95"]["accepted_grasp_success"] = result["ci95"]["accepted_success"]
    return result


def _summarize_matched_random(
    *,
    success: np.ndarray,
    wrong_object: np.ndarray,
    rows: list[dict[str, str]],
    acceptance_rate: float,
    repeats: int,
    rng: np.random.Generator,
) -> dict:
    success = np.asarray(success, dtype=bool).reshape(-1)
    total = int(success.shape[0])
    accepted_count = int(round(float(np.clip(acceptance_rate, 0.0, 1.0)) * total))
    accepted_count = max(0, min(total, accepted_count))
    false_accept_values: list[float] = []
    accepted_success_values: list[float] = []
    wrong_object_false_accept_values: list[float] = []
    failed_counts: list[int] = []
    success_counts: list[int] = []
    wrong_object_counts: list[int] = []
    failure_type_values: dict[str, dict[str, list[float]]] = {}
    for _ in range(max(1, repeats)):
        accepted = np.zeros(total, dtype=bool)
        if accepted_count > 0:
            accepted[rng.choice(total, size=accepted_count, replace=False)] = True
        metrics = _metrics_from_mask(accepted, success, wrong_object)
        for failure_name, failure_metrics in metrics_by_failure_type(accepted, success, rows, wrong_object).items():
            metric_values = failure_type_values.setdefault(failure_name, {})
            for key, value in failure_metrics.items():
                if value is None:
                    metric_values.setdefault(key, []).append(float("nan"))
                else:
                    metric_values.setdefault(key, []).append(float(value))
        false_accept_values.append(float("nan") if metrics["false_accept_risk"] is None else metrics["false_accept_risk"])
        accepted_success_values.append(
            float("nan") if metrics["accepted_success"] is None else metrics["accepted_success"]
        )
        wrong_object_false_accept_values.append(
            float("nan")
            if metrics["wrong_object_false_accept_rate"] is None
            else metrics["wrong_object_false_accept_rate"]
        )
        failed_counts.append(int(metrics["failed_accepted_closures"]))
        success_counts.append(int(metrics["successful_accepted_closures"]))
        wrong_object_counts.append(int(metrics["wrong_object_accepted_closures"]))

    false_accept = float(np.nanmean(false_accept_values)) if accepted_count else None
    accepted_success = float(np.nanmean(accepted_success_values)) if accepted_count else None
    wrong_object_false_accept = float(np.nanmean(wrong_object_false_accept_values)) if accepted_count else None
    failed_accepted = float(np.mean(failed_counts)) if accepted_count else 0.0
    successful_accepted = float(np.mean(success_counts)) if accepted_count else 0.0
    wrong_object_accepted = float(np.mean(wrong_object_counts)) if accepted_count else 0.0
    failure_type_metrics = {}
    for failure_name, metric_values in failure_type_values.items():
        failure_type_metrics[failure_name] = {}
        for key, values in metric_values.items():
            array = np.asarray(values, dtype=np.float64)
            if np.all(np.isnan(array)):
                failure_type_metrics[failure_name][key] = None
            else:
                value = float(np.nanmean(array))
                failure_type_metrics[failure_name][key] = int(round(value)) if key == "total_events" else value
    return {
        "name": "matched_random",
        "kind": "baseline",
        "acceptance_mode": "matched target acceptance",
        "threshold": None,
        "uses_oracle_geometry": False,
        "feature_columns": [],
        "metrics": {
            "total_events": total,
            "accepted_closures": accepted_count,
            "refused_closures": total - accepted_count,
            "successful_accepted_closures": successful_accepted,
            "failed_accepted_closures": failed_accepted,
            "wrong_object_accepted_closures": wrong_object_accepted,
            "false_accept_risk": false_accept,
            "accepted_grasp_success": accepted_success,
            "accepted_success": accepted_success,
            "wrong_object_false_accept_rate": wrong_object_false_accept,
            "acceptance_rate": float(accepted_count / total) if total else 0.0,
            "refusal_rate": float(1.0 - accepted_count / total) if total else 0.0,
        },
        "ci95": {
            "false_accept_risk": _percentile_ci(false_accept_values),
            "accepted_success": _percentile_ci(accepted_success_values),
            "accepted_grasp_success": _percentile_ci(accepted_success_values),
            "wrong_object_false_accept_rate": _percentile_ci(wrong_object_false_accept_values),
            "acceptance_rate": [
                float(accepted_count / total) if total else 0.0,
                float(accepted_count / total) if total else 0.0,
            ],
        },
        "failure_type_metrics": failure_type_metrics,
    }


def _percentile_ci(values: list[float]) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return [None, None]
    lo, hi = np.nanpercentile(array, [2.5, 97.5])
    return [float(lo), float(hi)]


def _format_ci(value: float | None, ci: list[float | None]) -> str:
    if value is None:
        return "n/a"
    if ci[0] is None or ci[1] is None:
        return f"`{value:.4f}`"
    return f"`{value:.4f}` [`{ci[0]:.4f}`, `{ci[1]:.4f}`]"


def _format_count(value: float | int) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.1f}"
    return str(int(value))


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "`n/a`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        f"# {summary['title']}",
        "",
        f"- Input: `{summary['input']}`",
        f"- Events: `{summary['total_events']}`",
        f"- Wrong-object command events: `{summary['wrong_object_events']}`",
        f"- Target acceptance: `{summary['target_acceptance_rate']:.4f}`",
        f"- Bootstrap resamples: `{summary['bootstrap']}`",
        "",
        "| Method | Kind | Oracle geometry? | Acceptance mode | FAR, 95% CI | Accepted success, 95% CI | Acceptance, 95% CI | Wrong-object FAR, 95% CI | Accepted / failed / wrong-object accepted |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        metrics = result["metrics"]
        ci = result["ci95"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result['name']}`",
                    result["kind"],
                    "`yes`" if result["uses_oracle_geometry"] else "`no`",
                    result["acceptance_mode"],
                    _format_ci(metrics["false_accept_risk"], ci["false_accept_risk"]),
                    _format_ci(metrics["accepted_success"], ci["accepted_success"]),
                    _format_ci(metrics["acceptance_rate"], ci["acceptance_rate"]),
                    _format_ci(
                        metrics["wrong_object_false_accept_rate"],
                        ci["wrong_object_false_accept_rate"],
                    ),
                    (
                        f"`{_format_count(metrics['accepted_closures'])}` / "
                        f"`{_format_count(metrics['failed_accepted_closures'])}` / "
                        f"`{_format_count(metrics['wrong_object_accepted_closures'])}`"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure-Type Splits",
            "",
            "| Method | Failure type | Events | Accepted | FAR | Accepted success | Wrong-object FAR | Acceptance |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in summary["results"]:
        for failure_name, metrics in result.get("failure_type_metrics", {}).items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{result['name']}`",
                        f"`{failure_name}`",
                        _format_value(metrics["total_events"]),
                        _format_value(metrics["accepted_closures"]),
                        _format_value(metrics["false_accept_risk"]),
                        _format_value(metrics["accepted_success"]),
                        _format_value(metrics["wrong_object_false_accept_rate"]),
                        _format_value(metrics["acceptance_rate"]),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Evaluation CSV.")
    parser.add_argument("--title", default="Matched-Acceptance Refusal Comparison")
    parser.add_argument(
        "--reference-checkpoint",
        type=_parse_named_path,
        required=True,
        help="Reference default checkpoint as NAME=PATH; its default acceptance becomes the target.",
    )
    parser.add_argument(
        "--checkpoint",
        type=_parse_named_path,
        action="append",
        default=[],
        help="Additional checkpoint scored at the target acceptance, as NAME=PATH.",
    )
    parser.add_argument(
        "--scalar",
        type=_parse_scalar,
        action="append",
        default=[],
        help="Scalar baseline scored at the target acceptance, as NAME=COLUMN:low or NAME=COLUMN:high.",
    )
    parser.add_argument(
        "--proxy",
        type=_parse_proxy,
        action="append",
        default=[],
        help=(
            "CSV-only proxy baseline scored at target acceptance. Supported specs: "
            "NAME=estimated_geometry_proxy or NAME=oracle_geometry_proxy."
        ),
    )
    parser.add_argument("--include-always-close", action="store_true", help="Include unselective always-close row.")
    parser.add_argument("--include-matched-random", action="store_true", help="Include matched random row.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    _, success, _, rows = load_event_csv(args.input, feature_columns=[])
    success = success.astype(bool)
    wrong_object = wrong_object_mask(rows)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    reference_name, reference_path = args.reference_checkpoint
    reference_scores, reference_checkpoint = _checkpoint_scores(args.input, reference_path, device)
    reference_threshold = float(reference_checkpoint["threshold"])
    reference_accepted = reference_scores <= reference_threshold
    target_acceptance = float(reference_accepted.mean())
    results = [
        _summarize(
            name=reference_name,
            kind="checkpoint",
            accepted=reference_accepted,
            success=success,
            wrong_object=wrong_object,
            rows=rows,
            bootstrap=args.bootstrap,
            rng=rng,
            acceptance_mode="checkpoint threshold",
            threshold=reference_threshold,
            columns=list(reference_checkpoint["feature_columns"]),
        )
    ]

    if args.include_always_close:
        results.append(
            _summarize(
                name="always_close",
                kind="baseline",
                accepted=np.ones_like(success, dtype=bool),
                success=success,
                wrong_object=wrong_object,
                rows=rows,
                bootstrap=args.bootstrap,
                rng=rng,
                acceptance_mode="accept all",
            )
        )

    if args.include_matched_random:
        results.append(
            _summarize_matched_random(
                success=success,
                wrong_object=wrong_object,
                rows=rows,
                acceptance_rate=target_acceptance,
                repeats=args.bootstrap,
                rng=rng,
            )
        )

    for name, path in args.checkpoint:
        scores, checkpoint = _checkpoint_scores(args.input, path, device)
        accepted, threshold = _accepted_by_rate(scores, target_acceptance, low_is_safe=True)
        results.append(
            _summarize(
                name=name,
                kind="checkpoint",
                accepted=accepted,
                success=success,
                wrong_object=wrong_object,
                rows=rows,
                bootstrap=args.bootstrap,
                rng=rng,
                acceptance_mode="matched target acceptance",
                threshold=threshold,
                columns=list(checkpoint["feature_columns"]),
            )
        )

    row_columns = rows[0].keys()
    for name, column, direction in args.scalar:
        if column not in row_columns:
            raise ValueError(f"Column {column!r} not found in {args.input}")
        scores = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        accepted, threshold = _accepted_by_rate(scores, target_acceptance, low_is_safe=direction == "low")
        results.append(
            _summarize(
                name=name,
                kind=f"scalar:{column}:{direction}",
                accepted=accepted,
                success=success,
                wrong_object=wrong_object,
                rows=rows,
                bootstrap=args.bootstrap,
                rng=rng,
                acceptance_mode="matched target acceptance",
                threshold=threshold,
                columns=[column],
            )
        )

    for name, proxy in args.proxy:
        if proxy == "estimated_geometry_proxy":
            scores = estimated_geometry_proxy_scores(rows)
            kind = "proxy:image_summary_geometry"
            uses_oracle = False
            columns = []
        else:
            scores = oracle_geometry_scores(rows)
            kind = "proxy:oracle_geometry"
            uses_oracle = True
            columns = list(ORACLE_GEOMETRY_COLUMNS)
        accepted, threshold = _accepted_by_rate(scores, target_acceptance, low_is_safe=True)
        results.append(
            _summarize(
                name=name,
                kind=kind,
                accepted=accepted,
                success=success,
                wrong_object=wrong_object,
                rows=rows,
                bootstrap=args.bootstrap,
                rng=rng,
                acceptance_mode="matched target acceptance",
                threshold=threshold,
                columns=columns,
                uses_oracle_geometry=uses_oracle,
            )
        )

    summary = {
        "title": args.title,
        "input": args.input,
        "total_events": int(success.shape[0]),
        "successes": int(success.sum()),
        "failures": int((~success).sum()),
        "wrong_object_events": int(wrong_object.sum()),
        "reference_checkpoint": {"name": reference_name, "path": reference_path},
        "target_acceptance_rate": target_acceptance,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "results": results,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    _write_markdown(output_md, summary)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
