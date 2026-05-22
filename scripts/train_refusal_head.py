"""Train and calibrate a small pre-grasp refusal head from event CSV data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hsi_pregrasp_refusal.calibration import calibrate_threshold  # noqa: E402
from hsi_pregrasp_refusal.data import load_event_csv  # noqa: E402
from hsi_pregrasp_refusal.features import FEATURE_GROUPS, resolve_feature_columns  # noqa: E402
from hsi_pregrasp_refusal.metrics import compute_refusal_metrics  # noqa: E402
from hsi_pregrasp_refusal.model import RefusalHead  # noqa: E402


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_feature_columns(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_indices(count: int, seed: int, train_fraction: float, calibration_fraction: float):
    if count < 3:
        raise ValueError("Need at least 3 events for train/calibration/test splitting.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    train_end = max(1, int(round(count * train_fraction)))
    calibration_end = min(count - 1, train_end + max(1, int(round(count * calibration_fraction))))
    if train_end >= calibration_end:
        calibration_end = min(count - 1, train_end + 1)
    return indices[:train_end], indices[train_end:calibration_end], indices[calibration_end:]


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / np.maximum(std, 1e-6)


@torch.inference_mode()
def _score(model: RefusalHead, features: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return model.predict_failure_probability(tensor).detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Train the HSI pre-grasp refusal head.")
    parser.add_argument("--input", type=str, required=True, help="Input pre-grasp event CSV.")
    parser.add_argument("--output", type=str, default="logs/hsi_pregrasp/refusal_head.pt", help="Output checkpoint.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden_dims", type=str, default="64,32")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature columns. Defaults to the full pre-grasp feature set.",
    )
    parser.add_argument(
        "--feature_group",
        choices=sorted(FEATURE_GROUPS),
        default=None,
        help="Named feature group. Ignored when --features is provided.",
    )
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--calibration_fraction", type=float, default=0.20)
    parser.add_argument("--target_false_accept_risk", type=float, default=0.10)
    parser.add_argument("--min_acceptance_rate", type=float, default=0.0)
    parser.add_argument("--risk_estimator", choices=["empirical", "add_one"], default="empirical")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    selected_columns = resolve_feature_columns(
        feature_group=args.feature_group,
        feature_columns=_parse_feature_columns(args.features),
    )
    features, success, feature_columns, _ = load_event_csv(args.input, feature_columns=selected_columns)
    train_ids, calibration_ids, test_ids = _split_indices(
        len(success), args.seed, args.train_fraction, args.calibration_fraction
    )

    mean = features[train_ids].mean(axis=0)
    std = features[train_ids].std(axis=0)
    features_std = _standardize(features, mean, std)

    device = torch.device(args.device)
    model = RefusalHead(
        input_dim=features.shape[1],
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    train_x = torch.as_tensor(features_std[train_ids], dtype=torch.float32, device=device)
    train_y_failure = torch.as_tensor((~success[train_ids]).astype(np.float32), dtype=torch.float32, device=device)
    positive_count = float(train_y_failure.sum().detach().cpu().item())
    negative_count = float(train_y_failure.numel() - positive_count)
    pos_weight = torch.tensor([negative_count / positive_count], device=device) if positive_count > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    for _epoch in range(args.epochs):
        order = torch.randperm(train_x.shape[0], generator=generator, device=device)
        for start in range(0, train_x.shape[0], args.batch_size):
            batch_ids = order[start : start + args.batch_size]
            logits = model(train_x[batch_ids])
            loss = criterion(logits, train_y_failure[batch_ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    calibration_scores = _score(model, features_std[calibration_ids], device)
    threshold_result = calibrate_threshold(
        calibration_scores,
        success[calibration_ids],
        target_false_accept_risk=args.target_false_accept_risk,
        min_acceptance_rate=args.min_acceptance_rate,
        risk_estimator=args.risk_estimator,
    )

    split_metrics = {}
    for split_name, ids in [("train", train_ids), ("calibration", calibration_ids), ("test", test_ids)]:
        split_scores = _score(model, features_std[ids], device)
        split_metrics[split_name] = compute_refusal_metrics(
            split_scores, success[ids], threshold_result.threshold
        ).as_dict()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1],
        "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
        "dropout": args.dropout,
        "feature_columns": feature_columns,
        "feature_group": args.feature_group,
        "feature_mean": mean.astype(float).tolist(),
        "feature_std": std.astype(float).tolist(),
        "threshold": threshold_result.threshold,
        "threshold_result": threshold_result.as_dict(),
        "metrics": split_metrics,
        "label": "failure_risk",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    summary = {
        "checkpoint": str(output_path),
        "feature_group": args.feature_group,
        "feature_columns": feature_columns,
        "threshold": threshold_result.as_dict(),
        "metrics": split_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
