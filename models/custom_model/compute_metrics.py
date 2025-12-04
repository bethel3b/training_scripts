"""metrics utils for custom model"""

from typing import Any

import torch
import torchmetrics


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
    logits: torch.Tensor = model_output.squeeze()

    # Extract ground truth labels from batch data
    gt_labels: torch.Tensor = batch_data["labels"].float()

    # Metrics
    metrics: dict[str, torch.Tensor] = {
        "accuracy": torchmetrics.functional.accuracy(logits, gt_labels, task="binary"),
        "precision": torchmetrics.functional.precision(
            logits, gt_labels, task="binary"
        ),
        "recall": torchmetrics.functional.recall(logits, gt_labels, task="binary"),
        "f1_score": torchmetrics.functional.f1_score(logits, gt_labels, task="binary"),
    }

    return metrics
