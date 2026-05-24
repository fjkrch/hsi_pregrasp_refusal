"""Evaluate a trained refusal head on a pre-grasp event CSV."""

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
from hsi_pregrasp_refusal.model import RefusalHead, TargetAwareRefusalHead  # noqa: E402


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an HSI pre-grasp refusal checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="Input pre-grasp event CSV.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Refusal head checkpoint from train_refusal_head.py.")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    feature_columns = list(checkpoint["feature_columns"])
    features, success, _, _ = load_event_csv(args.input, feature_columns=feature_columns)

    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(checkpoint["feature_std"], dtype=np.float32), 1e-6)
    features_std = (features - mean) / std

    model_type = checkpoint.get("model_type", "refusal_head")
    hidden_dims = tuple(int(v) for v in checkpoint["hidden_dims"])
    dropout = float(checkpoint["dropout"])
    if model_type == "target_aware":
        aux_heads = tuple(checkpoint.get("aux_heads", []))
        model: RefusalHead | TargetAwareRefusalHead = TargetAwareRefusalHead(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=hidden_dims,
            dropout=dropout,
            aux_heads=aux_heads,
        ).to(device)
    else:
        model = RefusalHead(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        tensor = torch.as_tensor(features_std, dtype=torch.float32, device=device)
        scores = model.predict_failure_probability(tensor).detach().cpu().numpy()

    threshold = float(checkpoint["threshold"] if args.threshold is None else args.threshold)
    metrics = compute_refusal_metrics(scores, success, threshold)
    print(
        json.dumps(
            {
                "input": args.input,
                "checkpoint": args.checkpoint,
                "threshold": threshold,
                "metrics": metrics.as_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
