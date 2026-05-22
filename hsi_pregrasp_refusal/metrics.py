"""Metrics for selective close/refuse evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


@dataclass(frozen=True)
class RefusalMetrics:
    """Core metrics from the proposal."""

    total_events: int
    accepted_closures: int
    refused_closures: int
    successful_accepted_closures: int
    failed_accepted_closures: int
    wrong_object_accepted_closures: int
    false_accept_risk: float | None
    accepted_grasp_success: float | None
    wrong_object_false_accept_rate: float | None
    acceptance_rate: float
    refusal_rate: float

    def as_dict(self) -> dict[str, float | int | None]:
        metrics = asdict(self)
        metrics["accepted_success"] = self.accepted_grasp_success
        return metrics


def compute_refusal_metrics(
    scores: np.ndarray,
    grasp_success: np.ndarray,
    threshold: float,
    *,
    wrong_object: np.ndarray | None = None,
) -> RefusalMetrics:
    """Compute close/refuse metrics for a failure-risk score.

    Args:
        scores: Predicted probability or score for ``close will fail``. Lower scores are accepted.
        grasp_success: Boolean or 0/1 labels where true means the closure succeeded.
        threshold: Accept close when ``score <= threshold``.
        wrong_object: Optional boolean mask where true means accepting the close would execute the
            wrong-object command. This is used for language-conditioned multi-object evaluations.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    success = np.asarray(grasp_success).astype(bool).reshape(-1)
    if scores.shape[0] != success.shape[0]:
        raise ValueError(f"scores and labels must have the same length, got {scores.shape[0]} and {success.shape[0]}")
    if wrong_object is None:
        wrong_object_mask = np.zeros_like(success, dtype=bool)
    else:
        wrong_object_mask = np.asarray(wrong_object).astype(bool).reshape(-1)
        if wrong_object_mask.shape[0] != success.shape[0]:
            raise ValueError(
                "wrong_object and labels must have the same length, "
                f"got {wrong_object_mask.shape[0]} and {success.shape[0]}"
            )

    accepted = scores <= threshold
    accepted_count = int(accepted.sum())
    total = int(scores.shape[0])
    refused_count = total - accepted_count
    accepted_success = int((accepted & success).sum())
    accepted_fail = int((accepted & ~success).sum())
    accepted_wrong_object = int((accepted & wrong_object_mask).sum())

    return RefusalMetrics(
        total_events=total,
        accepted_closures=accepted_count,
        refused_closures=refused_count,
        successful_accepted_closures=accepted_success,
        failed_accepted_closures=accepted_fail,
        wrong_object_accepted_closures=accepted_wrong_object,
        false_accept_risk=_safe_div(accepted_fail, accepted_count),
        accepted_grasp_success=_safe_div(accepted_success, accepted_count),
        wrong_object_false_accept_rate=_safe_div(accepted_wrong_object, accepted_count),
        acceptance_rate=float(accepted_count / total) if total else 0.0,
        refusal_rate=float(refused_count / total) if total else 0.0,
    )
