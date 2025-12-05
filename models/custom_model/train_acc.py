"""Training script with accelerator."""

import argparse
import logging
import time

import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch import nn, optim

from models.custom_model.compute_loss import compute_loss
from models.custom_model.compute_metrics import compute_metrics
from models.custom_model.custom_dataset import (
    CustomDataset,
    custom_input_fn_acc,
    custom_training_dataloader,
)
from models.custom_model.custom_model import CustomModel
from scripts.trainer_acc import Trainer
from utils.dir_utils import training_output_dir
from utils.model_utils import log_parameter_counts
from utils.utils import elapsed_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Training:
    """Run orchestrator for Custom Model."""

    def __init__(self, config: dict, save_results: bool = True) -> None:
        """Initialize the Run orchestrator."""
        self.save_results: bool = save_results
        self.config = config

    def _setup_accelerator(self) -> None:
        """Set up the accelerator."""
        accelerator_config = self.config["accelerator"]
        self.output_dir: str = accelerator_config["project_dir"]
        log_with: str = accelerator_config["log_with"]
        mixed_precision: str = accelerator_config["mixed_precision"]
        gradient_accumulation_steps: int = accelerator_config[
            "gradient_accumulation_steps"
        ]
        ddp_find_unused_parameters: bool = accelerator_config[
            "ddp_find_unused_parameters"
        ]

        self.accelerator: Accelerator = Accelerator(
            mixed_precision=mixed_precision,
            project_dir=self.output_dir,
            log_with=log_with,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=ddp_find_unused_parameters
                )
            ],
        )
        self._log_accelerator_info()

    def _log_accelerator_info(self) -> None:
        """Log accelerator information."""
        if self.accelerator.is_local_main_process:
            logger.info("=" * 100)
            logger.info("Accelerator Information:")
            logger.info(f"Device: {self.accelerator.device}")
            logger.info(f"Num Processes: {self.accelerator.num_processes}")
            logger.info(f"Distributed training: {self.accelerator.distributed_type}")
            logger.info(f"Mixed Precision: {self.accelerator.mixed_precision}")
            logger.info(
                f"Gradient Accumulation Steps: {self.accelerator.gradient_accumulation_steps}"
            )
            logger.info(f"Project Dir: {self.accelerator.project_dir}")
            logger.info(f"Log With: {self.accelerator.log_with}")

            logger.info("=" * 100 + "\n")

    def _load_configs(self) -> None:
        """Load and set all Configurations."""
        if self.accelerator.is_local_main_process:
            logger.info("Loading Configurations...\n")

        self.batch_size: int = self.config["batch_size"]

        # Training hyperparams
        training_config = self.config["training"]
        self.total_epochs: int = training_config["total_epochs"]
        self.lr: float = float(training_config["lr"])

        # Data directory
        data_dir_config = self.config["data_dir"]
        self.input_dir: dict[str, str] = data_dir_config["input_dir"]

    def _load_datasets(self) -> tuple[CustomDataset, CustomDataset]:
        """Load training and validation datasets."""
        if self.accelerator.is_local_main_process:
            logger.info("Loading Datasets...")

        train_set = CustomDataset(input_path=self.input_dir["train_path"])
        valid_set = CustomDataset(input_path=self.input_dir["valid_path"])

        logger.info(f"Train set: {len(train_set):,} examples")
        logger.info(f"Validation set: {len(valid_set):,} examples\n")
        return train_set, valid_set

    def _initialize_model(
        self,
    ) -> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler, dict[str, nn.Module]]:
        """Initialize Model, Optimizer, and LR Scheduler."""
        if self.accelerator.is_local_main_process:
            logger.info("Initializing Model...\n")
        model: nn.Module = CustomModel()
        log_parameter_counts(model)

        if self.accelerator.is_local_main_process:
            logger.info("Initializing Optimizer...\n")
        optimizer: optim.Optimizer = optim.AdamW(params=model.parameters(), lr=self.lr)

        if self.accelerator.is_local_main_process:
            logger.info("Initializing LR Scheduler...\n")
        scheduler: optim.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10, gamma=0.1
        )
        return model, optimizer, scheduler

    def run(self) -> None:
        """Execute the complete training pipeline."""
        # Setup accelerator
        self._setup_accelerator()

        if self.accelerator.is_local_main_process:
            logger.info("=" * 100)
            logger.info("Training Pipeline Starting...")
            logger.info("=" * 100 + "\n")

        # Load configurations
        self._load_configs()

        # Initialize output directories
        checkpoint_dir = None
        if self.save_results:
            checkpoint_dir, _, _ = training_output_dir(output_dir=self.output_dir)

        # Load datasets
        train_set, valid_set = self._load_datasets()

        # Load dataloaders
        train_loader, val_loader = custom_training_dataloader(
            train_set=train_set, valid_set=valid_set, batch_size=self.batch_size
        )

        # Initialize model, optimizer, scheduler
        model, optimizer, scheduler = self._initialize_model()

        # Prepare model for training
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        # Create trainer instance with all components
        trainer = Trainer(
            accelerator=self.accelerator,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            total_epochs=self.total_epochs,
            checkpoint_dir=checkpoint_dir,
            input_fn=custom_input_fn_acc,
            loss_fn=compute_loss,
            metrics_fn=compute_metrics,
        )
        # Execute training loop
        trainer.train_and_validate()

        if self.accelerator.is_local_main_process:
            logger.info("=" * 100)
            logger.info("Training Pipeline Completed...")
            logger.info("=" * 100 + "\n")


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

    if training.accelerator.is_local_main_process:
        logger.info(f"Time taken: {elapsed_time(start_time)}")


if __name__ == "__main__":
    main()
