"""Small refusal head for pre-grasp risk scoring."""

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
