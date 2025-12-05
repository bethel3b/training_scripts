"""Wrapper script to launch training with accelerate launch command."""

import os
import subprocess
import sys


def main() -> None:
    """Launch training script with accelerate launch."""
    # Get number of processes from environment or default to 2
    num_processes = os.environ.get("ACCELERATE_NUM_PROCESSES", "2")
    num_machines = os.environ.get("ACCELERATE_NUM_MACHINES", "1")
    mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
    dynamo_backend = os.environ.get("ACCELERATE_DYNAMO_BACKEND", "no")
    debug = os.environ.get("ACCELERATE_DEBUG", "true")

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_acc.py")

    # Build accelerate launch command
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        str(num_machines),
        "--mixed_precision",
        mixed_precision,
        "--dynamo_backend",
        dynamo_backend,
        train_script,
        "-c",
        "models/custom_model/config.yaml",
    ]

    # Run the command
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
