"""CustomDataset class."""

import logging

from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """A PyTorch dataset wrapper for custom dataset."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        pass

    def __len__(self) -> int:
        """Return the length of the dataset."""
        pass

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        """Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            dict[str, list[int]]: The item at the given index.
        """
        pass


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
    logger.info("Initializing Dataloader...")

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader
