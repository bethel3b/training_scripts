"""Training script."""

import argparse
import logging
import time

import torch
import yaml
from torch import nn, optim

from models.custom_model.compute_loss import compute_loss
from models.custom_model.compute_metrics import compute_metrics
from models.custom_model.custom_dataset import (
    CustomDataset,
    custom_evaluation_dataloader,
    custom_input_fn,
)
from models.custom_model.custom_model import CustomModel
from scripts.trainer import Trainer
from utils.dir_utils import evaluation_output_dir
from utils.model_utils import log_parameter_counts
from utils.utils import elapsed_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Evaluation:
    """Run orchestrator for Custom Model Evaluation."""

    def __init__(self, config: dict, save_results: bool = False) -> None:
        """Initialize the Run orchestrator."""
        self.save_results: bool = save_results
        self.config = config

    def _load_configs(self) -> None:
        """Load and set all Configurations."""
        logger.info("Loading Configurations...\n")

        # Data directory
        data_dir_config = self.config["data_dir"]
        self.input_dir: dict[str, str] = data_dir_config["input_dir"]

        # Models
        self.model_config = self.config["model"]
        self.num_layers: int = self.model_config["num_layers"]

        self.batch_size: int = self.config["batch_size"]
        # Setup device (GPU if available, else CPU)
        cuda = self.config["cuda"]
        self.device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"

        eval_dir = self.config["eval_dir"]
        self.eval_output_dir: str = eval_dir["eval_output_dir"]
        self.best_model_path: str = eval_dir["best_model_path"]

    def _load_datasets(self) -> tuple[CustomDataset, CustomDataset]:
        """Load evaluation dataset."""
        logger.info("Loading Evaluation Dataset...\n")
        test_set = CustomDataset(input_path=self.input_dir["test_path"])
        return test_set

    def _initialize_model(
        self,
    ) -> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler, dict[str, nn.Module]]:
        """Initialize Binding Affinity Prediction Model."""
        logger.info("Initializing Model...\n")
        model: nn.Module = CustomModel()
        log_parameter_counts(model)
        return model

    def run(self) -> None:
        """Execute the complete evaluation pipeline."""
        logger.info("=" * 100)
        logger.info("Evaluation Pipeline Starting...")
        logger.info("=" * 100 + "\n")
        # Initialize hyperparameters
        self._load_configs()

        # Initialize output directories
        tensorboard_dir, result_dir = None, None
        if self.save_results:
            tensorboard_dir, result_dir = evaluation_output_dir(
                output_dir=self.eval_output_dir
            )

        # Initialize datasets
        test_set = self._load_datasets()

        # Initialize dataloaders
        test_loader = custom_evaluation_dataloader(
            test_set=test_set, batch_size=self.batch_size
        )

        # Initialize model
        model = self._initialize_model()

        # Create trainer instance with all components
        trainer = Trainer(
            test_loader=test_loader,
            model=model,
            device=self.device,
            tensorboard_dir=tensorboard_dir,
            result_dir=result_dir,
            input_fn=custom_input_fn,
            loss_fn=compute_loss,
            metrics_fn=compute_metrics,
        )
        trainer.evaluate()


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
    evaluation = Evaluation(config=config)
    evaluation.run()
    logger.info(f"Time taken: {elapsed_time(start_time)}")


if __name__ == "__main__":
    main()
