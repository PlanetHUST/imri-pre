import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.disk_quantity.disk_plot_utilts import (
    plot_agn_disk_method_comparison,
    plot_agn_disk_parameter_profiles,
    plot_agn_disk_torque_profiles,
)

DISK_OUTPUT_DIR = Path("figs")
DISK_ALPHA = 0.1
SMBH_MASSES = (1e3, 1e4, 1e5, 1e6, 1e7)
MIGRATOR_MASS_MSUN = 10.0

plot_agn_disk_parameter_profiles(
    output_dir=str(DISK_OUTPUT_DIR),
    alpha=DISK_ALPHA,
    smbh_masses=SMBH_MASSES,
)
plot_agn_disk_method_comparison(
    output_dir=str(DISK_OUTPUT_DIR),
    alpha=DISK_ALPHA,
    smbh_masses=SMBH_MASSES,
)
plot_agn_disk_torque_profiles(
    output_dir=str(DISK_OUTPUT_DIR),
    alpha=DISK_ALPHA,
    smbh_masses=SMBH_MASSES,
    migrator_mass_msun=MIGRATOR_MASS_MSUN,
)
