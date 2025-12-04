"""metrics utils for custom model"""

from typing import Any

import torch


def compute_metrics(
    model_output: Any, batch_data: Any = None
) -> dict[str, torch.Tensor]:
    """Compute metrics.

    Args:
        model_output (Any): The output of the model.
        batch_data (Any): The batch data of the model.

    Returns:
        dict[str, torch.Tensor]: The step metrics.
    """
    logits: torch.Tensor = None

    # Extract ground truth labels from batch data
    gt_labels: torch.Tensor = None

    # Metrics
    metrics: dict[str, torch.Tensor] = None

    step_metrics: dict[str, torch.Tensor] = {**metrics}
    return step_metrics
