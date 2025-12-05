"""Trainer and Evaluator class."""

import logging
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.trainer_utils import (
    calculate_grad_norm,
    save_best_model,
    save_checkpoint,
    save_final_model,
    visulaize,
    write_to_tensorboard,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for training and evaluating the model."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        tensorboard_dir: str,
        result_dir: str,
        checkpoint_dir: str | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: optim.lr_scheduler = None,
        total_epochs: int | None = None,
        gradient_clip_value: float | None = None,
        steps_to_accumulate: int = 1,
        loss_fn: Callable = None,
        metrics_fn: Callable = None,
        input_fn: Callable = None,
        checkpoint_every_n_epochs: int = 1,
    ):
        """Initialize the Trainer class."""
        self.is_eval = True if test_loader is not None else False

        # DataLoader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Model
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Hyperparameters
        self.total_epochs = total_epochs
        self.gradient_clip_value = gradient_clip_value
        self.steps_to_accumulate = steps_to_accumulate

        # Other
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.result_dir = result_dir
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

        # Input, loss, and metrics functions
        self.input_fn = input_fn
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        self.sigmoid = nn.Sigmoid()
        self.model.to(self.device)

        # Tensorboard
        self.writer = None
        if self.tensorboard_dir:
            self.writer = SummaryWriter(self.tensorboard_dir)

        # Best model
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_model: nn.Module | None = None

    def _epoch(self, epoch: int = 0) -> tuple[dict[str, float], dict[str, float]]:
        """Perform one epoch of training or validation.

        Args:
            epoch (int): The current epoch. Defaults to 0.

        Returns:
            tuple[dict[str, float], dict[str, float]]:
                - avg_epoch_loss: The average epoch loss.
                - avg_epoch_metrics: The average epoch metrics.
        """
        # Get accumulators
        epoch_loss: dict[str, float] = {}
        epoch_metrics: dict[str, float] = {}

        # Get loader
        if self.is_eval:
            loader = self.test_loader
            mode = "Evaluation"
        else:
            loader = self.train_loader if self.model.training else self.val_loader
            mode = "Train" if self.model.training else "Validation"

        # Progress bar
        total_batches = len(loader)
        progress_bar = tqdm(loader, desc=mode, leave=True)

        # Iterate over loader
        for batch_idx, batch_data in enumerate(progress_bar):
            # Current step
            cur_step = batch_idx + (epoch - 1) * total_batches

            # Get input data
            input_data = self.input_fn(batch_data, device=self.device)

            # Forward Pass
            output = self.model(input_data)

            # Compute loss and update epoch_loss
            step_loss = self.loss_fn(model_output=output, batch_data=batch_data)
            for key, value in step_loss.items():
                if key not in epoch_loss:
                    epoch_loss[key] = 0.0
                epoch_loss[key] += value.item()

            # Calculate metrics and update epoch_metrics
            step_metrics = None
            if self.metrics_fn is not None:
                step_metrics = self.metrics_fn(
                    model_output=output, batch_data=batch_data
                )
                for key, value in step_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value.item()

            # Get total loss
            loss = step_loss["total_loss"]

            # Backward Pass (training only)
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                total_grad_norm = calculate_grad_norm(model=self.model, norm_type=2)
                self.optimizer.step()

            # Write to tensorboard
            if self.tensorboard_dir is not None:
                write_to_tensorboard(
                    epoch=cur_step,
                    epoch_loss=step_loss,
                    epoch_metrics=step_metrics,
                    grad_norm=total_grad_norm if self.model.training else None,
                    writer=self.writer,
                    mode=mode,
                    is_step=True,
                )

            # Update progress bar with current batch loss
            lr = self.optimizer.param_groups[0]["lr"] if self.model.training else None
            progress_bar.set_postfix({"loss": f"{loss:.4f}", "lr": lr})

        # Get average epoch loss and metrics
        avg_epoch_loss = {
            key: value / total_batches for key, value in epoch_loss.items()
        }
        avg_epoch_metrics = None
        if self.metrics_fn is not None:
            avg_epoch_metrics = {
                key: value / total_batches for key, value in epoch_metrics.items()
            }
        return avg_epoch_loss, avg_epoch_metrics

    def train_and_validate(self) -> None:
        """Run complete training and validation process."""
        logger.info(
            f"Starting training for {self.total_epochs} epochs on {self.device}\n"
        )

        # Initial validation and log result
        logger.info("Initial validation before training...")
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            init_epoch_loss, init_epoch_metrics = self._epoch(epoch=0)

            visulaize(
                epoch=0,
                epoch_loss=init_epoch_loss,
                epoch_metrics=init_epoch_metrics,
                mode="Validation",
                lr_scheduler=self.lr_scheduler,
                writer=self.writer,
                result_dir=self.result_dir,
                tensorboard_dir=self.tensorboard_dir,
            )

        # Train for total_epochs epochs and validate at each epoch
        for epoch in range(1, self.total_epochs + 1):
            logger.info(f"Epoch [{epoch}]")
            # Training phase
            # Set model to training mode
            self.model.train()
            train_epoch_loss, train_epoch_metrics = self._epoch(epoch=epoch)
            visulaize(
                epoch=epoch,
                epoch_loss=train_epoch_loss,
                epoch_metrics=train_epoch_metrics,
                mode="Train",
                lr_scheduler=self.lr_scheduler,
                writer=self.writer,
                result_dir=self.result_dir,
                tensorboard_dir=self.tensorboard_dir,
            )

            # Validation phase
            with torch.no_grad():
                # Set model to evaluation mode
                self.model.eval()
                val_epoch_loss, val_epoch_metrics = self._epoch(epoch=epoch)
                visulaize(
                    epoch=epoch,
                    epoch_loss=val_epoch_loss,
                    epoch_metrics=val_epoch_metrics,
                    mode="Validation",
                    lr_scheduler=self.lr_scheduler,
                    writer=self.writer,
                    result_dir=self.result_dir,
                    tensorboard_dir=self.tensorboard_dir,
                )

            # Update learning rate
            self.lr_scheduler.step()

            # Save model checkpoint
            if self.checkpoint_dir:
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir, epoch=epoch, model=self.model
                )

            # Update best model checkpoint
            if val_epoch_loss["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_epoch_loss["total_loss"]
                self.best_epoch = epoch
                self.best_model = self.model

        # Save final and best model
        if self.checkpoint_dir:
            save_final_model(checkpoint_dir=self.checkpoint_dir, model=self.model)
            save_best_model(
                checkpoint_dir=self.checkpoint_dir,
                best_model=self.best_model,
                best_epoch=self.best_epoch,
            )

        # Close tensorboard writer
        if self.tensorboard_dir:
            self.writer.close()
        return

    def evaluate(self) -> None:
        """Evaluate the model."""
        logger.info("Running Binding Affinity Prediction Model Evaluation Pipeline...")
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            eval_epoch_loss, eval_epoch_metrics = self._epoch(epoch=0)
            visulaize(
                epoch=0,
                epoch_loss=eval_epoch_loss,
                epoch_metrics=eval_epoch_metrics,
                mode="Evaluation",
                writer=self.writer,
                result_dir=self.result_dir,
                tensorboard_dir=self.tensorboard_dir,
            )
