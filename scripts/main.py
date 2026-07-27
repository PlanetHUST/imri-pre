"""Primary script for running the TDE-rate workflow."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (
    SOURCE_ROOT / "physical_process",
    SOURCE_ROOT / "disk_quantity",
    SOURCE_ROOT,
    PROJECT_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.workflows import *  # noqa: F403
from src.workflows import run_cli

if __name__ == "__main__":
    run_cli()
