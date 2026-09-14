"""Plot production-parity pAGN disk and migration-torque diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (SOURCE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.disk_quantity.disk_plot_utils import (  # noqa: E402
    plot_agn_disk_parameter_profiles,
    plot_agn_disk_torque_profiles,
)
from src.physical_process.disk_migration import (  # noqa: E402
    DEFAULT_MIGRATION_MODEL,
    GAP_OPENING_CONDITIONS,
    MIGRATION_MODELS,
)
from src.workflows.config import (  # noqa: E402
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    PAGN_ALPHA,
)

DISK_OUTPUT_DIR = Path("diagnostics/migration_disk")
SMBH_MASSES_MSUN = (1.0e5, 1.0e6, 1.0e7)
MIGRATOR_MASS_MSUN = 10.0
PAGN_N_RESOLUTION = 1000
TORQUE_SMOOTHING = 0.005
TORQUE_MAX_RADIUS_RG = None
MIN_TRAP_SEPARATION_CELLS = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--migration-model",
        choices=MIGRATION_MODELS,
        default=DEFAULT_MIGRATION_MODEL,
    )
    parser.add_argument(
        "--gap-open-condition",
        choices=GAP_OPENING_CONDITIONS,
        default="kii",
        help="deep-gap diagnostic; selects Type II only in classical_type2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DISK_OUTPUT_DIR,
    )
    return parser.parse_args()


def main(
    gap_opening_condition: str,
    migration_model: str = DEFAULT_MIGRATION_MODEL,
    output_dir: Path = DISK_OUTPUT_DIR,
) -> None:
    """Render disk and torque diagnostics with production physical inputs."""
    plot_agn_disk_parameter_profiles(
        output_dir=str(output_dir),
        alpha=PAGN_ALPHA,
        smbh_masses=SMBH_MASSES_MSUN,
        accretion_rate_g_s_per_msun=DISK_ACCRETION_RATE_G_S_PER_MSUN,
        n_resolution=PAGN_N_RESOLUTION,
    )
    plot_agn_disk_torque_profiles(
        output_dir=str(output_dir),
        alpha=PAGN_ALPHA,
        smbh_masses=SMBH_MASSES_MSUN,
        migrator_mass_msun=MIGRATOR_MASS_MSUN,
        min_trap_separation_cells=MIN_TRAP_SEPARATION_CELLS,
        gap_opening_condition=gap_opening_condition,
        migration_model=migration_model,
        accretion_rate_g_s_per_msun=DISK_ACCRETION_RATE_G_S_PER_MSUN,
        n_resolution=PAGN_N_RESOLUTION,
        torque_smoothing=TORQUE_SMOOTHING,
        max_radius_rg=TORQUE_MAX_RADIUS_RG,
    )
    print("Migration-disk diagnostic")
    print(f"  migration_model: {migration_model}")
    print(f"  gap_opening_condition: {gap_opening_condition}")
    print(f"  alpha: {PAGN_ALPHA:.6g}")
    print(
        "  accretion_rate_g_s_per_msun: "
        f"{DISK_ACCRETION_RATE_G_S_PER_MSUN:.6e}"
    )
    print(f"  pagn_n_resolution: {PAGN_N_RESOLUTION}")
    print(f"  torque_smoothing: {TORQUE_SMOOTHING:.6g}")
    print(f"  output_dir: {output_dir}")


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        arguments.gap_open_condition,
        arguments.migration_model,
        arguments.output_dir,
    )
