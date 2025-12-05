"""Test modules for the custom model."""

import time
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from models.custom_model.compute_loss import compute_loss
from models.custom_model.compute_metrics import compute_metrics
from models.custom_model.custom_dataset import (
    CustomDataset,
    custom_evaluation_dataloader,
)
from models.custom_model.custom_model import CustomModel
from utils.utils import elapsed_time


def get_batch() -> HeteroData:
    """Get the batch."""
    start_time = time.time()
    # Example usage
    print("=" * 100)
    print("\nLoading dataset...")
    dataset = CustomDataset(input_path="data/sample_train_data.pt")

    # Test dataloader batching
    print("\nBatch from DataLoader:")
    loader = custom_evaluation_dataloader(dataset, batch_size=8)
    batch = next(iter(loader))
    print(batch)

    time_taken = elapsed_time(start_time=start_time)
    print(f"Time taken: {time_taken}\n")
    return batch


def run_model(batch: Any) -> HeteroData:
    """Test the custom model."""
    with torch.no_grad():
        model = CustomModel()
        model.eval()

        output = model(x=batch["inputs"])
    print(f"Output Shape: {output.shape}")
    return output


def run_loss_and_metrics(output: Any, batch: Any) -> None:
    """Test the affinity loss and metrics."""
    loss_dict = compute_loss(model_output=output, batch_data=batch)
    df_loss = pd.DataFrame(loss_dict, index=[0])
    print("\nLoss:")
    print(df_loss)

    metrics_dict = compute_metrics(model_output=output, batch_data=batch)
    df_metrics = pd.DataFrame(metrics_dict, index=[0])

    print("\nMetrics:")
    print(df_metrics)


if __name__ == "__main__":
    batch = get_batch()
    output = run_model(batch)
    run_loss_and_metrics(output=output, batch=batch)
