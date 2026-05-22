"""Visual feature helpers for camera-enabled simulator events."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .features import CAMERA_FEATURE_NAMES, SMOLVLA_CAMERA_KEYS, VISUAL_FEATURE_COLUMNS


def rgb_hwc_to_chw_float(image: torch.Tensor) -> torch.Tensor:
    """Convert one IsaacLab RGB image from HWC uint8/float to CHW float in [0, 1]."""
    if image.ndim != 3:
        raise ValueError(f"Expected one HWC or CHW image, got shape {tuple(image.shape)}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        chw = image[:3]
    else:
        chw = image[..., :3].permute(2, 0, 1)
    chw = chw.contiguous().float()
    if chw.max() > 1.5:
        chw = chw / 255.0
    return chw.clamp(0.0, 1.0)


def resize_chw(image: torch.Tensor, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Resize one CHW image if needed."""
    if tuple(image.shape[-2:]) == size:
        return image
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def summarize_chw_image(image: torch.Tensor) -> dict[str, float]:
    """Compute compact visual evidence features from one CHW image."""
    image = rgb_hwc_to_chw_float(image) if image.ndim == 3 else image.float()
    _, height, width = image.shape
    h0, h1 = height // 3, (2 * height) // 3
    w0, w1 = width // 3, (2 * width) // 3
    center = image[:, h0:h1, w0:w1]
    dx = torch.abs(image[:, :, 1:] - image[:, :, :-1]).mean()
    dy = torch.abs(image[:, 1:, :] - image[:, :-1, :]).mean()
    return {
        "rgb_mean": float(image.mean().item()),
        "rgb_std": float(image.std(unbiased=False).item()),
        "red_mean": float(image[0].mean().item()),
        "green_mean": float(image[1].mean().item()),
        "blue_mean": float(image[2].mean().item()),
        "center_mean": float(center.mean().item()),
        "center_std": float(center.std(unbiased=False).item()),
        "edge_mean": float(((dx + dy) * 0.5).item()),
    }


def visual_feature_row(images: dict[str, torch.Tensor]) -> dict[str, float]:
    """Flatten per-camera image summaries into CSV-ready features."""
    row: dict[str, float] = {}
    for camera in SMOLVLA_CAMERA_KEYS:
        summary = summarize_chw_image(images[camera])
        for feature in CAMERA_FEATURE_NAMES:
            row[f"{camera}_{feature}"] = summary[feature]
    return row


def zero_visual_feature_row() -> dict[str, float]:
    """Return a zero-filled visual feature row."""
    return {column: 0.0 for column in VISUAL_FEATURE_COLUMNS}
