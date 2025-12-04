"""Utility functions for training."""

import logging
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_parameter_counts(model: nn.Module) -> None:
    """Log the number of parameters in the model."""
    # Calculate and log parameter counts
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")


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


def visulaize(
    epoch: int,
    epoch_loss: dict[str, float],
    epoch_metrics: dict[str, float],
    mode: str,
    writer: SummaryWriter,
    result_dir: str,
    tensorboard_dir: str,
    lr_scheduler: optim.lr_scheduler = None,
) -> None:
    """Visualize the training results and save to tensorboard and csv."""
    # Get learning rate
    lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else None
    log_result(epoch_loss=epoch_loss, epoch_metrics=epoch_metrics, mode=mode, lr=lr)

    if tensorboard_dir:
        write_to_tensorboard(
            epoch=epoch,
            epoch_loss=epoch_loss,
            epoch_metrics=epoch_metrics,
            writer=writer,
            mode=mode,
            lr=lr,
        )
    if result_dir:
        write_to_csv(
            epoch=epoch,
            epoch_loss=epoch_loss,
            epoch_metrics=epoch_metrics,
            result_dir=result_dir,
            mode=mode,
            lr=lr,
        )


def save_checkpoint(checkpoint_dir: str, epoch: int, model: nn.Module) -> None:
    """Save model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def save_final_model(checkpoint_dir: str, model: nn.Module) -> None:
    """Save the final trained model."""
    model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Final model saved: {model_path}")


def save_best_model(
    checkpoint_dir: str, best_model: nn.Module, best_epoch: int
) -> None:
    """Save the best trained model."""
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(best_model.state_dict(), model_path)
    logger.info(f"Best model (Epoch {best_epoch}) saved: {model_path}")


def write_to_tensorboard(
    epoch: int,
    epoch_loss: dict[str, float],
    epoch_metrics: dict[str, float] | None,
    writer: SummaryWriter,
    mode: str,
    grad_norm: float | None = None,
    lr: float = None,
    is_step: bool = False,
) -> None:
    """Write the result to tensorboard."""
    if is_step:
        mode = f"{mode} (Step)"

    # Write loss to tensorboard with hierarchical grouping
    for key, value in epoch_loss.items():
        writer.add_scalar(f"{mode} Loss/{key}", value, epoch)

    # Write metrics to tensorboard with hierarchical grouping
    if epoch_metrics is not None:
        for key, value in epoch_metrics.items():
            writer.add_scalar(f"{mode} Metrics/{key}", value, epoch)

    if "Train" in mode:
        # Write learning rate to tensorboard
        if lr is not None:
            writer.add_scalar(f"Learning_Rate", lr, epoch)
        # Write gradient norm to tensorboard
        if grad_norm is not None:
            writer.add_scalar(f"Gradient_Norm", grad_norm, epoch)


def log_result(
    epoch_loss: dict[str, float],
    epoch_metrics: dict[str, float] | None,
    mode: str,
    lr: float,
) -> None:
    """Log the training results for the current epoch."""
    log_string = f"{mode} | "

    for key, value in epoch_loss.items():
        string = f"{key}: {value:.4f} | "
        log_string += string

    if epoch_metrics is not None:
        for key, value in epoch_metrics.items():
            string = f"{key}: {value:.4f} | "
            log_string += string

    if mode == "Train":
        log_string += f"lr: {lr:.2e}"

    logger.info(log_string)


def write_to_csv(
    epoch: int,
    epoch_loss: dict[str, float],
    epoch_metrics: dict[str, float] | None,
    result_dir: str,
    mode: str,
    lr: float,
) -> None:
    """Write the result to csv."""
    output_path = os.path.join(result_dir, f"{mode.lower()}_result.csv")
    row_data = {"Epoch": [epoch]}
    for key, value in epoch_loss.items():
        row_data[key] = [value]

    if epoch_metrics is not None:
        for key, value in epoch_metrics.items():
            row_data[key] = [value]
    if mode == "Train":
        row_data["LR"] = lr
    df_result = pd.DataFrame(row_data)
    if os.path.exists(output_path):
        # Append without header
        df_result.to_csv(output_path, mode="a", header=False, index=False)
    else:
        # Write with header (first time)
        df_result.to_csv(output_path, mode="w", header=True, index=False)


def calculate_grad_norm(model: nn.Module, norm_type: int = 2) -> float:
    """Calculate gradient norm manually.

    Args:
        model (nn.Module): Model to calculate gradient norm.
        norm_type (int): Type of norm (1, 2, or 'inf')

    Returns:
        float: Gradient norm.
    """
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
