import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.disk_quantity.disk_plot_utilts import (
    plot_agn_disk_method_comparison,
    plot_agn_disk_parameter_profiles,
    plot_agn_disk_torque_profiles,
)

disk_output_dir = Path("figs")

plot_agn_disk_parameter_profiles(
    output_dir=str(disk_output_dir),
)
plot_agn_disk_method_comparison(
    output_dir=str(disk_output_dir),
)
plot_agn_disk_torque_profiles(
    output_dir=str(disk_output_dir),
)
