"""Small refusal heads for pre-grasp risk scoring."""

from __future__ import annotations

import torch
from torch import nn


class RefusalHead(nn.Module):
    """MLP that predicts ``p(close will fail)`` from pre-grasp features."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for failure risk."""
        return self.net(features).squeeze(-1)

    @torch.inference_mode()
    def predict_failure_probability(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``p(close will fail)``."""
        return torch.sigmoid(self(features))


def _recommended_hidden_dims(input_dim: int) -> tuple[int, ...]:
    """Scale default hidden dims with input dimension to reduce overfitting."""
    if input_dim <= 64:
        return (64, 32)
    if input_dim <= 256:
        return (128, 64)
    if input_dim <= 512:
        return (256, 128, 64)
    # Large learned-embedding inputs (DINO/CLIP)
    return (512, 256, 128)


class TargetAwareRefusalHead(nn.Module):
    """Pre-grasp refusal head with optional auxiliary failure-type heads.

    Phase 5 model: a shared encoder feeds a main binary close-failure head
    plus optional auxiliary heads that predict specific failure causes
    (wrong object, occlusion/clutter, geometric approach error).

    Auxiliary heads force the shared representation to encode non-geometric
    failure causes rather than collapsing onto distance features. They are
    trained only when per-event failure-type labels are available.

    Args:
        input_dim: Width of the input feature vector.
        hidden_dims: Widths of the shared encoder layers. Defaults scale
            automatically with ``input_dim`` when ``None``.
        dropout: Dropout probability applied after each hidden layer.
        aux_heads: Names of auxiliary binary heads to add. Supported values:
            ``"wrong_object"``, ``"occlusion"``, ``"geometric"``.
    """

    SUPPORTED_AUX = frozenset({"wrong_object", "occlusion", "geometric"})

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | None = None,
        dropout: float = 0.2,
        aux_heads: tuple[str, ...] = ("wrong_object", "occlusion"),
    ) -> None:
        super().__init__()
        unknown = set(aux_heads) - self.SUPPORTED_AUX
        if unknown:
            raise ValueError(f"Unknown aux heads {unknown}. Supported: {sorted(self.SUPPORTED_AUX)}")

        resolved = hidden_dims if hidden_dims is not None else _recommended_hidden_dims(input_dim)

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in resolved:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.main_head = nn.Linear(last_dim, 1)
        self.aux_head_names: tuple[str, ...] = tuple(aux_heads)
        self.aux_heads = nn.ModuleDict(
            {name: nn.Linear(last_dim, 1) for name in aux_heads}
        )
        self._hidden_dims = resolved

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (main_logit, {name: aux_logit}) for the full batch."""
        shared = self.encoder(features)
        main_logit = self.main_head(shared).squeeze(-1)
        aux_logits = {
            name: head(shared).squeeze(-1) for name, head in self.aux_heads.items()
        }
        return main_logit, aux_logits

    def forward_main(self, features: torch.Tensor) -> torch.Tensor:
        """Return only the main failure-risk logit (used at inference time)."""
        main_logit, _ = self.forward(features)
        return main_logit

    @torch.inference_mode()
    def predict_failure_probability(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``p(close will fail)`` — compatible with ``RefusalHead`` interface."""
        return torch.sigmoid(self.forward_main(features))

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self._hidden_dims
