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
    logits: torch.Tensor = model_output.squeeze()

    # Extract ground truth labels from batch data
    gt_labels: torch.Tensor = batch_data["labels"].float().to(logits.device)

    # Loss
    losses: dict[str, torch.Tensor] = {
        "ce_loss": torch.nn.functional.cross_entropy(logits, gt_labels),
        "mse_loss": torch.nn.functional.mse_loss(logits, gt_labels),
    }

    # Total loss
    total_loss: torch.Tensor = torch.tensor(0.0, device=logits.device)
    for key, value in losses.items():
        total_loss += value

    losses["total_loss"] = total_loss

    return losses
