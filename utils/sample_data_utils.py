"""Sample data utilities."""

import logging
import os

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(data_path: str) -> None:
    """Create sample data."""
    logger.info("Creating sample data...")
    # Create sample data
    sample_data = {
        "inputs": torch.randn(100, 10),
        "labels": torch.randint(0, 2, (100,)),
    }

    # save sample data to disk
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save(sample_data, data_path)

    logger.info(f"Sample data saved to {data_path}")


if __name__ == "__main__":
    create_sample_data(data_path="data/sample_train_data.pt")
    create_sample_data(data_path="data/sample_valid_data.pt")
    create_sample_data(data_path="data/sample_test_data.pt")
