"""Directory utilities for training and evaluation."""

import logging
import os
from datetime import datetime

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def training_output_dir(output_dir: str) -> tuple[str, str, str]:
    """Create and save directories for checkpoints, tensorboard, and training
    results."""
    logger.info("Loading Directories...")
    # Create timestamp for unique run identification
    date_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")

    # Build hierarchical directory structure
    result_dir: str = os.path.join(output_dir, date_time_str, "training_results")

    # Define specific output directories
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    tensorboard_dir = os.path.join(result_dir, "tensorboard")

    # Create all directories
    for directories in [checkpoint_dir, tensorboard_dir, result_dir]:
        os.makedirs(directories, exist_ok=True)

    logger.info(
        f"Output Directories: \n"
        f"\tCheckpoint Dir: {checkpoint_dir}\n"
        f"\tTensorboard Dir: {tensorboard_dir}\n"
        f"\tTraining Results Dir: {result_dir}\n"
    )
    return checkpoint_dir, tensorboard_dir, result_dir


def evaluation_output_dir(output_dir: str) -> tuple[str, str]:
    """Create and save directories for checkpoints, tensorboard, and training
    results."""
    logger.info("Loading Directories...")
    # Create timestamp for unique run identification
    date_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")

    # Build hierarchical directory structure
    result_dir: str = os.path.join(output_dir, "evaluation_results", date_time_str)

    # Define specific output directories
    tensorboard_dir = os.path.join(result_dir, "tensorboard")

    # Create all directories
    for directories in [tensorboard_dir, result_dir]:
        os.makedirs(directories, exist_ok=True)

    logger.info(
        f"Output Directories: \n"
        f"\tTensorboard Dir: {tensorboard_dir}\n"
        f"\tTraining Results Dir: {result_dir}\n"
    )
    return tensorboard_dir, result_dir


def dump_config(config: dict, result_dir: str, checkpoint_dir: str) -> None:
    """Dump configurations to file."""
    # Dump configurations to file
    config_path = os.path.join(result_dir, "config.yaml")
    config_to_append = {
        "eval_dir": {
            "eval_output_dir": result_dir,
            "best_model_path": os.path.join(checkpoint_dir, "best_model.pth"),
        }
    }
    new_config = {**config, **config_to_append}

    with open(config_path, "w") as file:
        yaml.dump(new_config, file)
