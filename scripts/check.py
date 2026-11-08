"""Run the small correctness suite and optionally validate saved host results.

    python scripts/check.py
    python scripts/check.py --run-dir outputs/single_host
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="also check a completed single-host run")
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    status = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "run_tests.py")], cwd=ROOT, env=env
    ).returncode
    if status or args.run_dir is None:
        return status
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_single_halo_workflow.py"), str(args.run_dir)],
        cwd=ROOT, env=env,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
