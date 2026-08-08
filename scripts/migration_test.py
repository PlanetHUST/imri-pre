"""Compare semi-analytic and torque-based disk migration timescales."""

from __future__ import annotations

import argparse
import os
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.disk_quantity.migration_torque import (
    MigrationParameters,
    compute_torque_profile,
    find_stable_traps,
)
from src.disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from src.physical_process import constants as const
from src.physical_process.disk_migration import (
    integrated_semi_analytical_migration_timescale,
    t_migration_from_torque,
)
from src.physical_process.general_class import AGN_properties
from src.workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA

DEFAULT_MIGRATOR_MASSES_MSUN = (1.0, 5.0, 10.0)
DEFAULT_SMBH_MASS_MSUN = 1.0e4
DEFAULT_START_RADIUS_RG = 1.0e5
DEFAULT_TARGET_RADIUS_RG = 1.0e2
DEFAULT_N_RESOLUTION = 3000
DEFAULT_OUTPUT_DIR = Path("figs")
OUTPUT_STEM = "migration_timescale_comparison"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semi-analytic and torque-integrated migration timescales "
            "for an inward pAGN disk path."
        )
    )
    parser.add_argument(
        "--smbh-mass-msun",
        type=float,
        default=DEFAULT_SMBH_MASS_MSUN,
        help="central SMBH mass in solar masses",
    )
    parser.add_argument(
        "--migrator-masses-msun",
        type=float,
        nargs="+",
        default=list(DEFAULT_MIGRATOR_MASSES_MSUN),
        help="migrator masses in solar masses",
    )
    parser.add_argument(
        "--start-radius-rg",
        type=float,
        default=DEFAULT_START_RADIUS_RG,
        help="outer starting radius in gravitational radii",
    )
    parser.add_argument(
        "--target-radius-rg",
        type=float,
        default=DEFAULT_TARGET_RADIUS_RG,
        help="inner target radius in gravitational radii",
    )
    parser.add_argument(
        "--n-resolution",
        type=int,
        default=int(os.getenv("PAGN_DISK_N_RESOLUTION", DEFAULT_N_RESOLUTION)),
        help="pAGN radial grid resolution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the output figure",
    )
    return parser.parse_args()


def _positive_finite(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array > 0.0)


def _format_time(value_myr: float) -> str:
    if np.isfinite(value_myr):
        return f"{value_myr:.6e}"
    return "inf"


def main() -> None:
    args = _parse_args()
    if args.smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be positive")
    if args.start_radius_rg <= 0.0 or args.target_radius_rg <= 0.0:
        raise ValueError("radii must be positive")
    if args.start_radius_rg <= args.target_radius_rg:
        raise ValueError("start radius must be larger than target radius")
    if any(mass <= 0.0 for mass in args.migrator_masses_msun):
        raise ValueError("migrator masses must be positive")

    agn_properties = AGN_properties()
    agn_properties.init_AGN(args.smbh_mass_msun, duty_cycle_myr=1.0)
    rg_cm = agn_properties.gravitational_radius_cm
    target_radius_cm = args.target_radius_rg * rg_cm

    disk_profile = build_pagn_disk_profile(
        args.smbh_mass_msun,
        n_resolution=args.n_resolution,
        accretion_rate_g_s=(
            DISK_ACCRETION_RATE_G_S_PER_MSUN * args.smbh_mass_msun
        ),
        alpha=PAGN_ALPHA,
        eddington_ratio=0.1,
    )

    start_radii_rg = np.logspace(
        np.log10(args.target_radius_rg),
        np.log10(args.start_radius_rg),
        120,
    )
    start_radii_cm = start_radii_rg * rg_cm

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#1f77b4", "#BE9A62", "#2A7F62", "#8A5CA8", "#D55E00"]
    terminal_rows = []
    has_visible_trap = False

    for index, migrator_mass_msun in enumerate(args.migrator_masses_msun):
        color = colors[index % len(colors)]
        migrator_mass_g = float(migrator_mass_msun) * const.msun
        params = MigrationParameters(alpha=PAGN_ALPHA, include_type_ii=True)
        torque_profile = compute_torque_profile(
            disk_profile.disk,
            migrator_mass_msun=float(migrator_mass_msun),
            params=params,
        )
        torque_cgs = np.asarray(torque_profile.gamma_tot, dtype=float) * 1.0e7

        semi_times_myr = np.asarray(
            [
                integrated_semi_analytical_migration_timescale(
                    agn_properties,
                    migrator_mass_g,
                    float(start_radius_cm),
                    target_radius_cm,
                    aspect_ratio=disk_profile.aspect_ratio,
                    surface_density_g_cm2=disk_profile.surface_density_g_cm2,
                    radius_grid_cm=disk_profile.radius_cm,
                    consider_type_ii=True,
                )
                for start_radius_cm in start_radii_cm
            ],
            dtype=float,
        )
        torque_times_myr = np.asarray(
            [
                t_migration_from_torque(
                    np.asarray(torque_profile.radius_cm, dtype=float),
                    torque_cgs,
                    migrator_mass_g,
                    float(start_radius_cm),
                    target_radius_cm,
                    args.smbh_mass_msun,
                )
                for start_radius_cm in start_radii_cm
            ],
            dtype=float,
        )

        semi_mask = _positive_finite(semi_times_myr)
        torque_mask = _positive_finite(torque_times_myr)
        ax.plot(
            start_radii_rg[semi_mask],
            semi_times_myr[semi_mask],
            color=color,
            lw=2.8,
            ls="--",
            alpha=0.9,
        )
        ax.plot(
            start_radii_rg[torque_mask],
            torque_times_myr[torque_mask],
            color=color,
            lw=2.8,
            ls="-",
            alpha=0.9,
        )

        trap_radii_rg = find_stable_traps(torque_profile) / rg_cm
        trap_radii_rg = trap_radii_rg[
            np.isfinite(trap_radii_rg)
            & (trap_radii_rg >= args.target_radius_rg)
            & (trap_radii_rg <= args.start_radius_rg)
        ]
        has_visible_trap = has_visible_trap or trap_radii_rg.size > 0
        for trap_radius_rg in trap_radii_rg:
            ax.axvline(trap_radius_rg, color=color, lw=1.3, ls=":", alpha=0.5)

        terminal_rows.append(
            (
                migrator_mass_msun,
                semi_times_myr[-1],
                torque_times_myr[-1],
                int(np.count_nonzero(~np.isfinite(torque_times_myr))),
            )
        )

    ax.axvline(args.target_radius_rg, color="0.35", lw=1.5, ls=":", alpha=0.8)
    ax.axvline(args.start_radius_rg, color="0.35", lw=1.5, ls="-.", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Starting radius $R/r_g$")
    ax.set_ylabel(r"Migration time to target (Myr)")
    ax.set_title(
        rf"$M_\bullet={args.smbh_mass_msun:.0e}\,M_\odot$, "
        rf"$R_\mathrm{{target}}={args.target_radius_rg:.0e}\,r_g$"
    )
    ax.grid(True, alpha=0.25, linestyle="--", which="both")

    mass_handles = [
        Line2D(
            [0],
            [0],
            color=colors[index % len(colors)],
            lw=2.8,
            label=rf"$m={mass:g}\,M_\odot$",
        )
        for index, mass in enumerate(args.migrator_masses_msun)
    ]
    method_handles = [
        Line2D([0], [0], color="0.2", lw=2.8, ls="-", label="torque"),
        Line2D([0], [0], color="0.2", lw=2.8, ls="--", label="semi-analytic"),
        Line2D([0], [0], color="0.35", lw=1.5, ls=":", label="target radius"),
        Line2D([0], [0], color="0.35", lw=1.5, ls="-.", label="start radius"),
    ]
    if has_visible_trap:
        method_handles.append(
            Line2D([0], [0], color="0.2", lw=1.3, ls=":", label="stable trap")
        )
    ax.legend(handles=mass_handles + method_handles, loc="best", framealpha=0.88)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{OUTPUT_STEM}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Disk migration timescale comparison")
    print(f"  smbh_mass_msun: {args.smbh_mass_msun:.6e}")
    print(f"  rg_cm: {rg_cm:.6e}")
    print(
        "  disk_radius_rg: "
        f"{disk_profile.inner_radius_cm / rg_cm:.6e} to "
        f"{disk_profile.outer_radius_cm / rg_cm:.6e}"
    )
    print(
        "  path_rg: "
        f"{args.start_radius_rg:.6e} -> {args.target_radius_rg:.6e}"
    )
    print("  terminal_times_myr:")
    print("    mass_msun  semi_analytic  torque  nonfinite_torque_points")
    for mass_msun, semi_time_myr, torque_time_myr, nonfinite_count in terminal_rows:
        print(
            f"    {mass_msun:9.3g}  "
            f"{_format_time(float(semi_time_myr)):>13}  "
            f"{_format_time(float(torque_time_myr)):>13}  "
            f"{nonfinite_count:>23d}"
        )
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    main()
