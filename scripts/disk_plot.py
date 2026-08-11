import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.disk_quantity.disk_plot_utilts import (
    plot_agn_disk_method_comparison,
    plot_agn_disk_parameter_profiles,
    plot_agn_disk_torque_profiles,
)
from src.physical_process.disk_migration import GAP_OPENING_CONDITIONS

DISK_OUTPUT_DIR = Path("figs")
DISK_ALPHA = 0.1
SMBH_MASSES = (1e5, 1e6)
MIGRATOR_MASS_MSUN = 10.0
MIN_TRAP_SEPARATION_CELLS = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gap-open-condition",
        choices=GAP_OPENING_CONDITIONS,
        default="kii",
        help="kii: KII > 11; crida: P < 1 and R_H/H > 1",
    )
    return parser.parse_args()


def main(gap_opening_condition: str) -> None:
    # plot_agn_disk_parameter_profiles(
    #     output_dir=str(DISK_OUTPUT_DIR),
    #     alpha=DISK_ALPHA,
    #     smbh_masses=SMBH_MASSES,
    # )
    # plot_agn_disk_method_comparison(
    #     output_dir=str(DISK_OUTPUT_DIR),
    #     alpha=DISK_ALPHA,
    #     smbh_masses=SMBH_MASSES,
    # )
    plot_agn_disk_torque_profiles(
        output_dir=str(DISK_OUTPUT_DIR),
        alpha=DISK_ALPHA,
        smbh_masses=SMBH_MASSES,
        migrator_mass_msun=MIGRATOR_MASS_MSUN,
        min_trap_separation_cells=MIN_TRAP_SEPARATION_CELLS,
        gap_opening_condition=gap_opening_condition,
    )
    print(f"gap_opening_condition: {gap_opening_condition}")
    print("gap_open_torque: type_ii_only")


if __name__ == "__main__":
    main(_parse_args().gap_open_condition)
