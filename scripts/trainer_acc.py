"""Trainer and Evaluator class."""

import logging
from typing import Callable

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.accelerator_utils import (
    save_best_model,
    save_checkpoint,
    save_final_model,
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
        accelerator: Accelerator,
        model: nn.Module,
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
        checkpoint_dir: str | None = None,
        checkpoint_every_n_epochs: int = 1,
    ):
        """Initialize the Trainer class."""
        self.is_eval = True if test_loader is not None else False

        self.accelerator = accelerator
        self.device = self.accelerator.device

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

        # Input, loss, and metrics functions
        self.input_fn = input_fn
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        self.sigmoid = nn.Sigmoid()

        # Output directories
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

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
        progress_bar = tqdm(
            loader,
            desc=mode,
            leave=True,
            disable=not self.accelerator.is_local_main_process,
        )

        # Iterate over loader
        for batch_idx, batch_data in enumerate(progress_bar):
            # Get input data
            input_data = self.input_fn(batch_data)

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
                self.accelerator.backward(loss)
                self.optimizer.step()

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
        if self.accelerator.is_local_main_process:
            logger.info(
                f"Starting training for {self.total_epochs} epochs on {self.device}\n"
            )

        # Initial validation and log result
        if self.accelerator.is_local_main_process:
            logger.info("Initial validation before training...")

        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            init_epoch_loss, init_epoch_metrics = self._epoch(epoch=0)
            # Log initial validation results
            log_dict = {f"val/{k}": v for k, v in init_epoch_loss.items()}
            if init_epoch_metrics:
                log_dict.update({f"val/{k}": v for k, v in init_epoch_metrics.items()})
            self.accelerator.log(log_dict, step=0)

            if self.accelerator.is_local_main_process:
                logger.info(f"Initial Validation - Loss: {init_epoch_loss}")
                if init_epoch_metrics:
                    logger.info(f"Initial Validation - Metrics: {init_epoch_metrics}")

        # Train for total_epochs epochs and validate at each epoch
        for epoch in range(1, self.total_epochs + 1):
            if self.accelerator.is_local_main_process:
                logger.info(f"Epoch [{epoch}]")
            # Training phase
            # Set model to training mode
            self.model.train()

            with self.accelerator.accumulate(self.model):
                train_epoch_loss, train_epoch_metrics = self._epoch(epoch=epoch)

            # Log training results
            log_dict = {f"train/{k}": v for k, v in train_epoch_loss.items()}
            if train_epoch_metrics:
                log_dict.update(
                    {f"train/{k}": v for k, v in train_epoch_metrics.items()}
                )
            log_dict["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]

            # Validation phase
            with torch.no_grad():
                # Set model to evaluation mode
                self.model.eval()
                val_epoch_loss, val_epoch_metrics = self._epoch(epoch=epoch)

                # Add validation results to log dict
                log_dict.update({f"val/{k}": v for k, v in val_epoch_loss.items()})
                if val_epoch_metrics:
                    log_dict.update(
                        {f"val/{k}": v for k, v in val_epoch_metrics.items()}
                    )

            # Log to TensorBoard
            self.accelerator.log(log_dict, step=epoch)

            # Log to terminal
            if self.accelerator.is_local_main_process:
                logger.info(f"Train Loss: {train_epoch_loss}")
                if train_epoch_metrics:
                    logger.info(f"Train Metrics: {train_epoch_metrics}")
                logger.info(f"Val Loss: {val_epoch_loss}")
                if val_epoch_metrics:
                    logger.info(f"Val Metrics: {val_epoch_metrics}")

            # Update learning rate
            self.lr_scheduler.step()

            # Save model checkpoint
            self.accelerator.wait_for_everyone()

            # Save model checkpoint
            if (
                self.checkpoint_dir
                and epoch % self.checkpoint_every_n_epochs == 0
                and self.accelerator.is_local_main_process
            ):
                model = self.accelerator.unwrap_model(self.model)
                save_checkpoint(
                    accelerator=self.accelerator,
                    checkpoint_dir=self.checkpoint_dir,
                    epoch=epoch,
                    model=model,
                )

            # Update best model checkpoint
            if val_epoch_loss["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_epoch_loss["total_loss"]
                self.best_epoch = epoch
                self.best_model = self.model

        # Save final and best model
        if self.checkpoint_dir:
            model = self.accelerator.unwrap_model(self.model)
            best_model = self.accelerator.unwrap_model(self.best_model)
            save_final_model(
                accelerator=self.accelerator,
                checkpoint_dir=self.checkpoint_dir,
                model=model,
            )
            save_best_model(
                accelerator=self.accelerator,
                checkpoint_dir=self.checkpoint_dir,
                best_model=best_model,
                best_epoch=self.best_epoch,
            )

        self.accelerator.end_training()
        return

    def evaluate(self) -> None:
        """Evaluate the model."""
        logger.info("Running Binding Affinity Prediction Model Evaluation Pipeline...")
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            eval_epoch_loss, eval_epoch_metrics = self._epoch(epoch=0)
            # Log evaluation results
            log_dict = {f"eval/{k}": v for k, v in eval_epoch_loss.items()}
            if eval_epoch_metrics:
                log_dict.update({f"eval/{k}": v for k, v in eval_epoch_metrics.items()})
            self.accelerator.log(log_dict, step=0)

            if self.accelerator.is_local_main_process:
                logger.info(f"Evaluation - Loss: {eval_epoch_loss}")
                if eval_epoch_metrics:
                    logger.info(f"Evaluation - Metrics: {eval_epoch_metrics}")
