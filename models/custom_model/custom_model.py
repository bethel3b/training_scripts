"""Custom model for custom dataset."""

import torch
import torch.nn as nn


class CustomModel(nn.Module):
    """Custom model for custom dataset."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.layer = nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layer(x)
