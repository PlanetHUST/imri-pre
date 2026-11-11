"""Run one host: python scripts/main.py 1 outputs/single_host 1 --base-seed 777.

Only command-line wiring lives here; the portable model is in src/.
"""

import sys
from pathlib import Path

# Permit running directly from a checkout without relying on the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workflows.cli import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli()
