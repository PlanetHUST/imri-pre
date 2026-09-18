"""Primary entry point for the IMRI workflow: CLI dispatcher and programmatic API.

Running this file executes the command-line interface, which dispatches to
either the original single-halo mode or the merger-tree batch mode::

    python scripts/main.py 1 output/dyn_results 4 --smbh-masses 1e5 ...
    python scripts/main.py --tree galaxy_1e12.csv --output output/tree_run --workers 4

Importing this module exposes the same operations programmatically, so one
single-halo simulation can be executed without the CLI::

    from main import run_single_halo, IMRIDynConfig, HaloHost

    result = run_single_halo(config=base_config, host=host, seed=seed,
                             output_dir=output_dir)

``run_single_halo`` is the reusable single-halo interface: it resolves the
host-dependent inputs (redshift, SMBH mass, NSC mass, metallicity) onto the base
configuration, applies the deterministic seed and explicit output directory, and
calls the unchanged ``run_simulation`` kernel.  It contains no CSV-specific
code; merger-tree parsing lives in ``workflows.merger_tree`` only.
"""

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

from workflows.cli import run_cli  # noqa: E402
from workflows.config import IMRIDynConfig  # noqa: E402
from workflows.host_mapping import (  # noqa: E402
    HostMappingPolicy,
    ResolvedHostParameters,
    resolve_host_parameters,
)
from workflows.merger_tree import HaloHost, HostSelection, MergerTreeReader  # noqa: E402
from workflows.single_halo import (  # noqa: E402
    HaloSimulationResult,
    build_host_config,
    resolve_run_parameters,
    run_single_halo,
)
from workflows.tree_batch import (  # noqa: E402
    BatchSettings,
    HaloSimulationJob,
    run_jobs,
    run_tree_batch,
)
from workflows.workflow import run_simulation  # noqa: E402

__all__ = [
    "BatchSettings",
    "HaloHost",
    "HaloSimulationJob",
    "HaloSimulationResult",
    "HostMappingPolicy",
    "HostSelection",
    "IMRIDynConfig",
    "MergerTreeReader",
    "ResolvedHostParameters",
    "build_host_config",
    "resolve_host_parameters",
    "resolve_run_parameters",
    "run_cli",
    "run_jobs",
    "run_simulation",
    "run_single_halo",
    "run_tree_batch",
]

if __name__ == "__main__":
    run_cli()
