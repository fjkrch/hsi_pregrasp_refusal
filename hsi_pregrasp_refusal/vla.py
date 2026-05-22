"""VLA asset and adapter helpers.

This module keeps the VLA integration narrow and explicit. The full SmolVLA runtime depends on the external LeRobot
package, but the project can still stage/check the checkpoint and record exactly what is missing before running VLA
features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import torch

from .features import VLA_ACTION_FEATURE_COLUMNS
from .vision import resize_chw, rgb_hwc_to_chw_float


DEFAULT_SMOLVLA_REPO_ID = "lerobot/smolvla_base"


@dataclass(frozen=True)
class VLAAssetStatus:
    """Status of a staged VLA checkpoint."""

    repo_id: str
    local_dir: str
    config_exists: bool
    weights_exist: bool
    preprocessor_exists: bool
    postprocessor_exists: bool
    lerobot_installed: bool
    ready_for_runtime: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def check_smolvla_assets(local_dir: str | Path, repo_id: str = DEFAULT_SMOLVLA_REPO_ID) -> VLAAssetStatus:
    """Check whether SmolVLA assets and runtime dependencies are available."""
    local_dir = Path(local_dir)
    config_exists = (local_dir / "config.json").exists()
    weights_exist = (local_dir / "model.safetensors").exists()
    preprocessor_exists = (local_dir / "policy_preprocessor.json").exists()
    postprocessor_exists = (local_dir / "policy_postprocessor.json").exists()
    lerobot_installed = find_spec("lerobot") is not None
    return VLAAssetStatus(
        repo_id=repo_id,
        local_dir=str(local_dir),
        config_exists=config_exists,
        weights_exist=weights_exist,
        preprocessor_exists=preprocessor_exists,
        postprocessor_exists=postprocessor_exists,
        lerobot_installed=lerobot_installed,
        ready_for_runtime=all(
            [config_exists, weights_exist, preprocessor_exists, postprocessor_exists, lerobot_installed]
        ),
    )


@dataclass(frozen=True)
class VLAActionSummary:
    """Compact summary of repeated VLA action samples."""

    action_mean: list[float]
    action_std: list[float]
    action_variance: list[float]
    action_variance_mean: float
    action_std_norm: float
    action_range_norm: float
    action_entropy_proxy: float
    inference_ms: float

    def feature_row(self) -> dict[str, float]:
        row: dict[str, float] = {}
        for idx, value in enumerate(self.action_mean):
            row[f"vla_action_mean_{idx}"] = float(value)
        for idx, value in enumerate(self.action_std):
            row[f"vla_action_std_{idx}"] = float(value)
        for idx, value in enumerate(self.action_variance):
            row[f"vla_action_var_{idx}"] = float(value)
        row["vla_action_var_mean"] = float(self.action_variance_mean)
        row["vla_action_std_norm"] = float(self.action_std_norm)
        row["vla_action_range_norm"] = float(self.action_range_norm)
        row["vla_action_entropy_proxy"] = float(self.action_entropy_proxy)
        return row


def zero_vla_feature_row() -> dict[str, float]:
    """Return a zero-filled VLA action feature row."""
    return {column: 0.0 for column in VLA_ACTION_FEATURE_COLUMNS}


def build_smolvla_state(tcp_position: torch.Tensor, object_position: torch.Tensor, action_position: torch.Tensor) -> torch.Tensor:
    """Build the 6-D simulator proxy state used for SmolVLA inference."""
    tcp_position = tcp_position.detach().float()
    object_position = object_position.detach().float()
    action_position = action_position.detach().float()
    ee_object_delta = tcp_position - object_position
    action_delta = action_position - tcp_position
    return torch.cat([ee_object_delta[:3], action_delta[:3]], dim=0)


def summarize_actions(actions: torch.Tensor, inference_ms: float = 0.0) -> VLAActionSummary:
    """Summarize repeated continuous action samples."""
    actions = actions.detach().float().cpu()
    if actions.ndim != 2:
        raise ValueError(f"Expected action samples with shape (N, D), got {tuple(actions.shape)}")
    variance = actions.var(dim=0, unbiased=False)
    std = actions.std(dim=0, unbiased=False)
    action_range = actions.max(dim=0).values - actions.min(dim=0).values
    entropy_proxy = 0.5 * torch.log(2.0 * torch.pi * torch.e * torch.clamp(variance, min=1e-8)).sum()
    return VLAActionSummary(
        action_mean=actions.mean(dim=0).tolist(),
        action_std=std.tolist(),
        action_variance=variance.tolist(),
        action_variance_mean=float(variance.mean().item()),
        action_std_norm=float(torch.linalg.norm(std).item()),
        action_range_norm=float(torch.linalg.norm(action_range).item()),
        action_entropy_proxy=float(entropy_proxy.item()),
        inference_ms=float(inference_ms),
    )


class SmolVLAAdapter:
    """Small runtime wrapper around LeRobot's SmolVLA policy."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        task: str = "pick up the cube",
        image_size: tuple[int, int] = (256, 256),
    ):
        if find_spec("lerobot") is None:
            raise RuntimeError("LeRobot is not installed in this environment; cannot run SmolVLA.")
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        self.task = task
        self.image_size = image_size
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint, local_files_only=True).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def _frame(self, images: dict[str, torch.Tensor], state: torch.Tensor) -> dict[str, torch.Tensor | str]:
        frame: dict[str, torch.Tensor | str] = {
            "observation.state": state.detach().cpu().float(),
            "task": self.task,
        }
        for key in ["camera1", "camera2", "camera3"]:
            image = resize_chw(rgb_hwc_to_chw_float(images[key]), self.image_size)
            frame[f"observation.images.{key}"] = image.detach().cpu()
        return frame

    @torch.inference_mode()
    def sample_actions(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        *,
        num_samples: int = 4,
    ) -> VLAActionSummary:
        start = perf_counter()
        batch = self.preprocess(self._frame(images, state))
        actions = []
        for _ in range(num_samples):
            self.policy.reset()
            action = self.policy.select_action(batch)
            actions.append(self.postprocess(action).detach().cpu())
        action_tensor = torch.cat(actions, dim=0)
        return summarize_actions(action_tensor, inference_ms=(perf_counter() - start) * 1000.0)
