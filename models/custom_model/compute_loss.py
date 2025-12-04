"""loss utils for custom model"""

from typing import Any

import torch


def compute_loss(model_output: Any, batch_data: Any = None) -> dict[str, torch.Tensor]:
    """Compute loss.

    Args:
        model_output (Any): The output of the model.
        batch_data (Any): The batch data of the model.

    Returns:
        dict[str, torch.Tensor]: The step loss.
    """
    # Extract the logtis from the model output
    logits: torch.Tensor = None

    # Extract ground truth labels from batch data
    gt_labels: torch.Tensor = None

    # Loss
    losses: dict[str, torch.Tensor] = None

    # Total loss
    total_loss = None

    # append pocket_losses and affinity_losses
    step_loss: dict[str, torch.Tensor] = {**losses, "total_loss": total_loss}

    return step_loss
