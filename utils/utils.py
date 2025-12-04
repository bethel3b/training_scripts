"""Utility functions for training and evaluation."""

import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def elapsed_time(start_time: float) -> str:
    """Calculate and format elapsed time.

    Args:
        start_time(float): Time when processing started (from time.time())

    Returns:
        str: Formatted time string (HH:MM:SS)
    """
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
