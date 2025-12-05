"""CustomDataset class."""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """A PyTorch dataset wrapper for custom dataset."""

    def __init__(self, input_path: str) -> None:
        """Initialize the dataset."""
        logger.info(f"Loading dataset from {input_path}...")
        self.dataset = torch.load(input_path)
        self.inputs = self.dataset["inputs"]
        self.labels = self.dataset["labels"]
        logger.info(f"Loaded dataset with size: {len(self.inputs)} samples!")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        """Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            dict[str, list[int]]: The item at the given index.
        """
        return {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx],
        }


def custom_training_dataloader(
    train_set: CustomDataset, valid_set: CustomDataset, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Initialize PyTorch DataLoader for custom training dataset."""
    logger.info("Initializing Dataloader...")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def custom_evaluation_dataloader(
    test_set: CustomDataset, batch_size: int
) -> DataLoader:
    """Initialize PyTorch DataLoader for custom evaluation dataset."""
    logger.info("Initializing Dataloader...\n")

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader


def custom_input_fn(batch_data: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    """Get input data from batch data."""
    return batch_data["inputs"].to(device)


def custom_input_fn_acc(batch_data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get input data from batch data."""
    return batch_data["inputs"]
