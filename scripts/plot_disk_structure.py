"""Plot explicit pAGN disk-migration torque diagnostics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from viz.disk_plots import (
    plot_agn_disk_torque_profiles,
    plot_agn_disk_parameter_profiles,
)
DISK_OUTPUT_DIR = Path("figs")
DISK_ALPHA = 0.1
SMBH_MASSES = (1e3, 1e4, 1e5)
MIGRATOR_MASS_MSUN = 10.0


def main() -> None:
    """Plot the configured pAGN migration-torque profiles."""
    plot_agn_disk_torque_profiles(
        output_dir=str(DISK_OUTPUT_DIR),
        alpha=DISK_ALPHA,
        smbh_masses=SMBH_MASSES,
        migrator_mass_msun=MIGRATOR_MASS_MSUN,
    )
    plot_agn_disk_parameter_profiles(
        output_dir=str(DISK_OUTPUT_DIR), alpha=DISK_ALPHA, smbh_masses=SMBH_MASSES
    )
    print("migration_contract: jm17_grishin2024_continuous_kanagawa")


if __name__ == "__main__":
    main()
