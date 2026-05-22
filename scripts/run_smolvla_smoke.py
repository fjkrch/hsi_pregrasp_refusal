"""Smoke-test the staged SmolVLA checkpoint and report simple action-uncertainty features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description="Run a SmolVLA smoke test with dummy camera/state inputs.")
    parser.add_argument("--checkpoint", default="logs/hsi_pregrasp/vla/smolvla_base")
    parser.add_argument("--task", default="pick up the cube")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="logs/hsi_pregrasp/vla/smolvla_smoke_summary.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, local_files_only=True).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    frame = {
        "observation.state": torch.zeros(6),
        "observation.images.camera1": torch.zeros(3, 256, 256),
        "observation.images.camera2": torch.zeros(3, 256, 256),
        "observation.images.camera3": torch.zeros(3, 256, 256),
        "task": args.task,
    }
    batch = preprocess(frame)

    actions = []
    with torch.inference_mode():
        for _ in range(args.num_samples):
            policy.reset()
            action = policy.select_action(batch)
            actions.append(postprocess(action).detach().cpu())
    actions_tensor = torch.cat(actions, dim=0)
    action_mean = actions_tensor.mean(dim=0)
    action_std = actions_tensor.std(dim=0, unbiased=False)
    action_variance = actions_tensor.var(dim=0, unbiased=False)

    summary = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "device": str(device),
        "num_samples": args.num_samples,
        "input_features": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
            for key, value in batch.items()
            if torch.is_tensor(value)
        },
        "output_action_shape": list(actions_tensor.shape),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "action_variance": action_variance.tolist(),
        "action_variance_mean": float(action_variance.mean().item()),
        "action_std_norm": float(torch.linalg.norm(action_std).item()),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
