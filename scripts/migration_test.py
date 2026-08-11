"""Compare semi-analytic and torque-based disk migration timescales."""

from __future__ import annotations

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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.disk_quantity.migration_torque import (
    MigrationParameters,
    classify_torque_traps,
    compute_torque_profile,
    torque_si_to_cgs,
)
from src.disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from src.physical_process import constants as const
from src.physical_process.disk_migration import (
    DISK_VISCOSITY_ALPHA,
    TYPE_II_GAP_PARAMETER_THRESHOLD,
    integrated_semi_analytical_migration_timescale,
    migration_time_from_torque,
    semi_analytical_migration_timescale,
)
from src.physical_process.general_class import AGN_properties
from src.workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA

MIGRATOR_MASSES_MSUN = (1.0, 5.0, 10.0)
SMBH_MASS_MSUN = 1.0e6
START_RADIUS_RG = 1.0e5
TARGET_RADIUS_RG = 1.0e2
N_RESOLUTION = 3000
OUTPUT_DIR = Path("figs")
PAGN_DISK_MODEL = "sirko"
PAGN_EDDINGTON_RATIO = 0.1
PAGN_B = 0
TORQUE_MAX_RADIUS_RG = None
TORQUE_SMOOTHING = 0.005
OUTPUT_STEM = "migration_timescale_comparison"
SEMI_LINESTYLE = (0, (3.0, 1.2, 1.0, 1.2))


def _format_time(value_myr: float) -> str:
    if np.isfinite(value_myr):
        return f"{value_myr:.6e}"
    return "inf"


def main() -> None:
    agn_properties = AGN_properties()
    agn_properties.init_AGN(SMBH_MASS_MSUN, duty_cycle_myr=1.0)
    rg_cm = agn_properties.gravitational_radius_cm
    target_radius_cm = TARGET_RADIUS_RG * rg_cm

    disk_profile = build_pagn_disk_profile(
        SMBH_MASS_MSUN,
        alpha=PAGN_ALPHA,
        model=PAGN_DISK_MODEL,
        n_resolution=N_RESOLUTION,
        eddington_ratio=PAGN_EDDINGTON_RATIO,
        b=PAGN_B,
        accretion_rate_g_s=(DISK_ACCRETION_RATE_G_S_PER_MSUN * SMBH_MASS_MSUN),
        suppress_pagn_output=True,
    )
    disk_radius_cm = np.asarray(disk_profile.radius_cm, dtype=float)
    disk_aspect_ratio = np.asarray(disk_profile.aspect_ratio, dtype=float)
    disk_order = np.argsort(disk_radius_cm)
    disk_radius_cm = disk_radius_cm[disk_order]
    disk_aspect_ratio = disk_aspect_ratio[disk_order]

    sample_radii_rg = np.logspace(
        np.log10(TARGET_RADIUS_RG),
        np.log10(START_RADIUS_RG),
        120,
    )
    sample_radii_cm = sample_radii_rg * rg_cm

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#1f77b4", "#BE9A62", "#2A7F62", "#8A5CA8", "#D55E00"]

    terminal_rows = []
    has_visible_trap = False
    params = MigrationParameters(
        alpha=PAGN_ALPHA,
        gamma_adiabat=5.0 / 3.0,
        hydrogen_fraction=0.7,
        kappa_t=20.0,
        kappa_norm=25.0,
    )

    for index, migrator_mass_msun in enumerate(MIGRATOR_MASSES_MSUN):
        color = colors[index % len(colors)]
        migrator_mass_g = float(migrator_mass_msun) * const.msun

        torque_profile = compute_torque_profile(
            disk_profile.disk,
            migrator_mass_msun=float(migrator_mass_msun),
            max_radius_rg=TORQUE_MAX_RADIUS_RG,
            smoothing=TORQUE_SMOOTHING,
            params=params,
        )

        torque_si = np.asarray(
            torque_profile.gamma_type_i_si
            + (1 - torque_profile.gap_attenuation_factor) * torque_profile.gamma_th_si,
            dtype=float,
        )
        torque_radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
        local_aspect_ratio = np.interp(
            torque_radius_cm,
            disk_radius_cm,
            disk_aspect_ratio,
        )
        type_ii_gap_parameter_dimensionless = migrator_mass_g**2 / (
            DISK_VISCOSITY_ALPHA
            * local_aspect_ratio**2
            * (SMBH_MASS_MSUN * const.msun) ** 2
        )
        type_ii_mask = np.isfinite(type_ii_gap_parameter_dimensionless) & (
            type_ii_gap_parameter_dimensionless > TYPE_II_GAP_PARAMETER_THRESHOLD
        )
        torque_si = torque_si + np.where(
            type_ii_mask,
            np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
            0.0,
        )
        torque_cgs = torque_si_to_cgs(torque_si)

        semi_times_myr = np.asarray(
            [
                semi_analytical_migration_timescale(
                    agn_properties,
                    migrator_mass_g,
                    float(start_radius_cm),
                    aspect_ratio=disk_profile.aspect_ratio,
                    surface_density_g_cm2=disk_profile.surface_density_g_cm2,
                    radius_grid_cm=disk_profile.radius_cm,
                    consider_type_ii=True,
                )
                for start_radius_cm in sample_radii_cm
            ],
            dtype=float,
        )

        int_semi_times_myr = np.asarray(
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
                for start_radius_cm in sample_radii_cm
            ],
            dtype=float,
        )

        torque_times_myr = np.asarray(
            [
                migration_time_from_torque(
                    torque_radius_cm,
                    torque_cgs,
                    migrator_mass_g,
                    float(start_radius_cm),
                    target_radius_cm,
                    SMBH_MASS_MSUN,
                )
                for start_radius_cm in sample_radii_cm
            ],
            dtype=float,
        )

        semi_mask = np.isfinite(semi_times_myr) & (semi_times_myr > 0.0)
        int_semi_mask = np.isfinite(int_semi_times_myr) & (int_semi_times_myr > 0.0)
        torque_mask = np.isfinite(torque_times_myr) & (torque_times_myr > 0.0)

        ax.plot(
            sample_radii_rg[semi_mask],
            semi_times_myr[semi_mask],
            color=color,
            lw=2.8,
            ls=SEMI_LINESTYLE,
            alpha=0.9,
        )
        ax.plot(
            sample_radii_rg[int_semi_mask],
            int_semi_times_myr[int_semi_mask],
            color=color,
            lw=2.8,
            ls="--",
            alpha=0.9,
        )
        ax.plot(
            sample_radii_rg[torque_mask],
            torque_times_myr[torque_mask],
            color=color,
            lw=2.8,
            ls="-",
            alpha=0.9,
        )

        torque_traps = classify_torque_traps(
            torque_radius_cm,
            torque_si,
        )
        trap_radii_rg = torque_traps.stable_radius_rg

        trap_radii_rg = trap_radii_rg[
            np.isfinite(trap_radii_rg)
            & (trap_radii_rg >= TARGET_RADIUS_RG)
            & (trap_radii_rg <= START_RADIUS_RG)
        ]
        has_visible_trap = has_visible_trap or trap_radii_rg.size > 0
        for trap_radius_rg in trap_radii_rg:
            ax.axvline(trap_radius_rg, color=color, lw=1.3, ls=":", alpha=0.5)

        terminal_rows.append(
            (
                migrator_mass_msun,
                semi_times_myr[-1],
                int_semi_times_myr[-1],
                torque_times_myr[-1],
                int(np.count_nonzero(~np.isfinite(torque_times_myr))),
            )
        )

    ax.axvline(TARGET_RADIUS_RG, color="0.35", lw=1.5, ls=":", alpha=0.8)
    ax.axvline(START_RADIUS_RG, color="0.35", lw=1.5, ls="-.", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Starting radius $R/r_g$")
    ax.set_ylabel(r"Migration timescale (Myr)")
    ax.set_title(
        rf"$M_\bullet={SMBH_MASS_MSUN:.0e}\,M_\odot$, "
        rf"$R_\mathrm{{target}}={TARGET_RADIUS_RG:.0e}\,r_g$"
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
        for index, mass in enumerate(MIGRATOR_MASSES_MSUN)
    ]
    method_handles = [
        Line2D([0], [0], color="0.2", lw=2.8, ls="-", label="torque"),
        Line2D([0], [0], color="0.2", lw=2.8, ls=SEMI_LINESTYLE, label="semi"),
        Line2D([0], [0], color="0.2", lw=2.8, ls="--", label="int-semi"),
        Line2D([0], [0], color="0.35", lw=1.5, ls=":", label="target radius"),
        Line2D([0], [0], color="0.35", lw=1.5, ls="-.", label="start radius"),
    ]
    if has_visible_trap:
        method_handles.append(
            Line2D([0], [0], color="0.2", lw=1.3, ls=":", label="stable trap")
        )
    ax.legend(handles=mass_handles + method_handles, loc="best", framealpha=0.88)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Disk migration timescale comparison")
    print(f"  smbh_mass_msun: {SMBH_MASS_MSUN:.6e}")
    print(f"  rg_cm: {rg_cm:.6e}")
    print(
        "  disk_radius_rg: "
        f"{disk_profile.inner_radius_cm / rg_cm:.6e} to "
        f"{disk_profile.outer_radius_cm / rg_cm:.6e}"
    )
    print("  path_rg: " f"{START_RADIUS_RG:.6e} -> {TARGET_RADIUS_RG:.6e}")
    print("  terminal_times_myr:")
    print("    mass_msun  semi  int-semi  torque  nonfinite_torque_points")
    for (
        mass_msun,
        semi_time_myr,
        int_semi_time_myr,
        torque_time_myr,
        nonfinite_count,
    ) in terminal_rows:
        print(
            f"    {mass_msun:9.3g}  "
            f"{_format_time(float(semi_time_myr)):>13}  "
            f"{_format_time(float(int_semi_time_myr)):>13}  "
            f"{_format_time(float(torque_time_myr)):>13}  "
            f"{nonfinite_count:>23d}"
        )
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    main()
