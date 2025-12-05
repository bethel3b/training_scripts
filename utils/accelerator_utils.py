"""Accelerator utilities for training and evaluation."""

import logging
import os

import torch
from accelerate import Accelerator
from torch import nn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_checkpoint(
    accelerator: Accelerator, checkpoint_dir: str, epoch: int, model: nn.Module
) -> None:
    """Save model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    if accelerator.is_local_main_process:
        logger.info(f"Checkpoint saved: {checkpoint_path}\n")


def save_final_model(
    accelerator: Accelerator, checkpoint_dir: str, model: nn.Module
) -> None:
    """Save the final trained model."""
    model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    if accelerator.is_local_main_process:
        logger.info(f"Final model saved: {model_path}")


def save_best_model(
    accelerator: Accelerator,
    checkpoint_dir: str,
    best_model: nn.Module,
    best_epoch: int,
) -> None:
    """Save the best trained model."""
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(best_model.state_dict(), model_path)
    if accelerator.is_local_main_process:
        logger.info(f"Best model (Epoch {best_epoch}) saved: {model_path}")
