import argparse
import time

import yaml

from models.custom_model.eval import Evaluation
from utils.utils import elapsed_time


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
