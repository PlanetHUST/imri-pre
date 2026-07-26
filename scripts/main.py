"""Primary script for running the TDE-rate workflow."""

from src.workflows import *  # noqa: F403
from src.workflows import run_cli

if __name__ == "__main__":
    run_cli()
