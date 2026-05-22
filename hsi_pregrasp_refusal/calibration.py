"""Threshold calibration for close/refuse decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    """Selected refusal threshold and calibration-set diagnostics."""

    threshold: float
    target_false_accept_risk: float
    calibration_events: int
    accepted_calibration_events: int
    empirical_false_accept_risk: float | None
    acceptance_rate: float
    risk_estimator: str

    def as_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def _risk_estimate(failures: int, accepted: int, estimator: str) -> float | None:
    if accepted == 0:
        return None
    if estimator == "empirical":
        return failures / accepted
    if estimator == "add_one":
        return (failures + 1) / (accepted + 1)
    raise ValueError(f"Unknown risk estimator: {estimator}")


def calibrate_threshold(
    scores: np.ndarray,
    grasp_success: np.ndarray,
    *,
    target_false_accept_risk: float = 0.10,
    min_acceptance_rate: float = 0.0,
    risk_estimator: str = "empirical",
) -> ThresholdResult:
    """Choose the largest threshold whose calibration false-accept risk is within target.

    This is an empirical selective-risk threshold. It is useful for simulator and pilot experiments, but it should not be
    described as a distribution-free guarantee unless the paper later adds a formal conformal risk-control procedure.
    """
    if not 0.0 <= target_false_accept_risk <= 1.0:
        raise ValueError("target_false_accept_risk must be in [0, 1].")
    if not 0.0 <= min_acceptance_rate <= 1.0:
        raise ValueError("min_acceptance_rate must be in [0, 1].")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    success = np.asarray(grasp_success).astype(bool).reshape(-1)
    if scores.shape[0] != success.shape[0]:
        raise ValueError(f"scores and labels must have the same length, got {scores.shape[0]} and {success.shape[0]}")
    if scores.shape[0] == 0:
        raise ValueError("Cannot calibrate threshold with an empty calibration set.")

    best: tuple[float, int, float | None, float] | None = None
    for threshold in np.unique(scores):
        accepted = scores <= threshold
        accepted_count = int(accepted.sum())
        acceptance_rate = accepted_count / scores.shape[0]
        if acceptance_rate < min_acceptance_rate:
            continue
        failures = int((accepted & ~success).sum())
        risk = _risk_estimate(failures, accepted_count, risk_estimator)
        if risk is not None and risk <= target_false_accept_risk:
            best = (float(threshold), accepted_count, risk, acceptance_rate)

    if best is None:
        threshold = float(np.nextafter(scores.min(), -np.inf))
        return ThresholdResult(
            threshold=threshold,
            target_false_accept_risk=float(target_false_accept_risk),
            calibration_events=int(scores.shape[0]),
            accepted_calibration_events=0,
            empirical_false_accept_risk=None,
            acceptance_rate=0.0,
            risk_estimator=risk_estimator,
        )

    threshold, accepted_count, risk, acceptance_rate = best
    return ThresholdResult(
        threshold=threshold,
        target_false_accept_risk=float(target_false_accept_risk),
        calibration_events=int(scores.shape[0]),
        accepted_calibration_events=accepted_count,
        empirical_false_accept_risk=risk,
        acceptance_rate=float(acceptance_rate),
        risk_estimator=risk_estimator,
    )
