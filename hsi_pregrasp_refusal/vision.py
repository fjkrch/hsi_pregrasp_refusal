"""Visual feature helpers for camera-enabled simulator events."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .features import (
    CAMERA_FEATURE_NAMES,
    CLIP_EMBED_DIM,
    CLIP_GLOBAL_FEATURE_COLUMNS,
    CLIP_PER_CAM_FEATURE_COLUMNS,
    DINO_EMBED_DIM,
    DINO_GLOBAL_FEATURE_COLUMNS,
    DINO_PER_CAM_FEATURE_COLUMNS,
    SMOLVLA_CAMERA_KEYS,
    VISUAL_FEATURE_COLUMNS,
)


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


class LearnedEmbeddingExtractor:
    """Extract frozen DINOv2 or CLIP embeddings from camera images.

    Loads model lazily on first call so the import cost is only paid when
    embeddings are actually requested (``--embedding_model`` flag in the
    collector).

    Supports:
        ``"dinov2"``  – ``facebook/dinov2-small`` (384-dim CLS token)
        ``"clip"``    – ``openai/clip-vit-base-patch32`` (512-dim pooler output)

    Per-camera embeddings and a cross-camera global mean are stored in the
    returned feature row so callers can use either or both in feature groups.
    """

    _HF_IDS: dict[str, str] = {
        "dinov2": "facebook/dinov2-small",
        "clip": "openai/clip-vit-base-patch32",
    }
    _EMBED_DIMS: dict[str, int] = {
        "dinov2": DINO_EMBED_DIM,
        "clip": CLIP_EMBED_DIM,
    }

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        if model_name not in self._HF_IDS:
            valid = ", ".join(sorted(self._HF_IDS))
            raise ValueError(f"Unknown embedding model {model_name!r}. Valid: {valid}")
        self.model_name = model_name
        self.embed_dim = self._EMBED_DIMS[model_name]
        self.device = torch.device(device)
        self._model: object | None = None
        self._processor: object | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        hf_id = self._HF_IDS[self.model_name]
        if self.model_name == "dinov2":
            from transformers import AutoImageProcessor, Dinov2Model  # noqa: PLC0415

            self._processor = AutoImageProcessor.from_pretrained(hf_id)
            model = Dinov2Model.from_pretrained(hf_id)
        else:  # clip
            from transformers import CLIPImageProcessor, CLIPVisionModel  # noqa: PLC0415

            self._processor = CLIPImageProcessor.from_pretrained(hf_id)
            model = CLIPVisionModel.from_pretrained(hf_id)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model

    @torch.inference_mode()
    def _embed_one(self, image: torch.Tensor) -> torch.Tensor:
        """Embed one CHW float [0,1] image; returns a 1-D tensor of shape (embed_dim,)."""
        self._ensure_loaded()
        chw = rgb_hwc_to_chw_float(image)
        # Processor expects HWC uint8 numpy array.
        hwc_np = (chw.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        inputs = self._processor(images=hwc_np, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        if self.model_name == "dinov2":
            # CLS token at position 0
            return outputs.last_hidden_state[:, 0, :].squeeze(0)
        else:
            # Global image pooler output
            return outputs.pooler_output.squeeze(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embedding_feature_row(self, images: dict[str, torch.Tensor]) -> dict[str, float]:
        """Extract per-camera embeddings plus global mean → flat CSV feature dict.

        Keys follow the naming defined in ``features.py``:
        * ``"{model}_{cam}_dim{i}"``    – per-camera embedding component
        * ``"{model}_global_dim{i}"``   – mean over all cameras
        """
        cam_embeddings: list[torch.Tensor] = []
        row: dict[str, float] = {}
        for camera in SMOLVLA_CAMERA_KEYS:
            emb = self._embed_one(images[camera])  # (embed_dim,)
            for i, val in enumerate(emb.tolist()):
                row[f"{self.model_name}_{camera}_dim{i}"] = val
            cam_embeddings.append(emb)

        global_mean = torch.stack(cam_embeddings).mean(dim=0)
        for i, val in enumerate(global_mean.tolist()):
            row[f"{self.model_name}_global_dim{i}"] = val
        return row

    def zero_embedding_row(self) -> dict[str, float]:
        """Return a zero-filled embedding row (same keys, all zeros)."""
        row: dict[str, float] = {}
        for camera in SMOLVLA_CAMERA_KEYS:
            for i in range(self.embed_dim):
                row[f"{self.model_name}_{camera}_dim{i}"] = 0.0
        for i in range(self.embed_dim):
            row[f"{self.model_name}_global_dim{i}"] = 0.0
        return row

    @property
    def per_cam_feature_columns(self) -> list[str]:
        if self.model_name == "dinov2":
            return DINO_PER_CAM_FEATURE_COLUMNS
        return CLIP_PER_CAM_FEATURE_COLUMNS

    @property
    def global_feature_columns(self) -> list[str]:
        if self.model_name == "dinov2":
            return DINO_GLOBAL_FEATURE_COLUMNS
        return CLIP_GLOBAL_FEATURE_COLUMNS
