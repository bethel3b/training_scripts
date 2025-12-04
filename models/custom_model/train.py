"""Training script."""

import argparse
import logging
import time

import torch
import yaml
from torch import nn, optim

from models.custom_model.custom_dataset import CustomDataset, custom_training_dataloader
from models.custom_model.custom_model import CustomModel
from utils.run_utils import log_parameter_counts, training_output_dir
from utils.trainer import Trainer
from utils.utils import elapsed_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Training:
    """Run orchestrator for Custom Model."""

    def __init__(self, config: dict, save_results: bool = False) -> None:
        """Initialize the Run orchestrator."""
        self.save_results: bool = save_results
        self.config = config

    def _load_configs(self) -> None:
        """Load and set all hyperparameters."""
        logger.info("Loading Hyperparameters...")

        # Data directory
        data_dir_config = self.config["data_dir"]
        self.input_dir: str = data_dir_config["input_dir"]

        # Models
        self.model_config = self.config["model"]
        self.num_layers: int = self.model_config["num_layers"]

        self.batch_size: int = self.config["batch_size"]
        # Setup device (GPU if available, else CPU)
        cuda = self.config["cuda"]
        self.device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"

        # Training hyperparams
        training_config = self.config["training"]
        self.total_epochs: int = training_config["total_epochs"]
        self.lr: float = float(training_config["lr"])
        # Training directory
        training_dir_config = self.config["training_dir"]
        self.output_dir: str = training_dir_config["output_dir"]

    def _load_training_dataset(self) -> tuple[CustomDataset, CustomDataset]:
        """Load training and validation datasets."""
        logger.info("Loading Datasets...")
        train_set = CustomDataset()
        valid_set = CustomDataset()

        logger.info(f"Train set: {len(train_set):,} examples")
        logger.info(f"Validation set: {len(valid_set):,} examples")
        return train_set, valid_set

    def _initialize_model(
        self,
    ) -> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler, dict[str, nn.Module]]:
        """Initialize Model, Optimizer, and LR Scheduler."""
        logger.info("Initializing Model...")
        model: nn.Module = CustomModel()
        log_parameter_counts(model)

        logger.info("Initializing Optimizer...")
        optimizer: optim.Optimizer = optim.AdamW(params=model.parameters(), lr=self.lr)

        logger.info("Initializing LR Scheduler...")
        scheduler: optim.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10, gamma=0.1
        )
        return model, optimizer, scheduler

    def run(self) -> None:
        """Execute the complete training pipeline."""

        logger.info("Training Pipeline Starting...")

        # Initialize hyperparameters
        self._load_configs()

        # Initialize output directories
        checkpoint_dir, tensorboard_dir, result_dir = None, None, None
        if self.save_results:
            checkpoint_dir, tensorboard_dir, result_dir = training_output_dir(
                output_dir=self.output_dir
            )

        # Initialize datasets
        train_set, valid_set = self._load_training_dataset()

        # Initialize dataloaders
        train_loader, val_loader = custom_training_dataloader(
            train_set=train_set, valid_set=valid_set, batch_size=self.batch_size
        )

        # Initialize model, optimizer, scheduler, criterion
        model, optimizer, scheduler = self._initialize_model()

        # Create trainer instance with all components
        trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            total_epochs=self.total_epochs,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=tensorboard_dir,
            result_dir=result_dir,
        )
        # Execute training loop
        trainer.train_and_validate()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for training script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - config (str): Path to YAML configuration file
            - model_type (str): Model architecture variant identifier
            - training_dataset (str): Dataset identifier from config
    """
    parser = argparse.ArgumentParser(
        description="Train Transformer model for neural machine translation."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="models/custom_model/config.yaml",
        help="Path to YAML configuration file with hyperparameters and settings.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """Entry point for training script."""
    start_time = time.time()

    # Parse arguments and load configuration
    args = get_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Initialize and execute training
    training = Training(config=config)
    training.run()
    logger.info(f"Time taken: {elapsed_time(start_time)}")


if __name__ == "__main__":
    main()
