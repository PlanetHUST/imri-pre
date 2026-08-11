"""Compare integrated semi-analytic and torque-based migration timescales."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
from src.disk_quantity.pagn_disk_adapter import PagnDiskProfile, build_pagn_disk_profile
from src.physical_process import constants as const
from src.physical_process.disk_migration import (
    DISK_VISCOSITY_ALPHA,
    GAP_OPENING_CONDITIONS,
    TYPE_II_GAP_PARAMETER_THRESHOLD,
    gap_opening_mask,
    migration_time_from_torque,
)
from src.physical_process.general_class import AGN_properties
from src.workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA

MIGRATOR_MASSES_MSUN = (10.0, 30.0, 50.0)
SMBH_MASS_MSUN = 1.0e3
START_RADIUS_RG = 1.0e6
TARGET_RADIUS_RG = 5e1
N_RESOLUTION = 3000
OUTPUT_DIR = Path("figs")
OUTPUT_STEM = "migration_timescale_comparison"

PAGN_DISK_MODEL = "sirko"
PAGN_EDDINGTON_RATIO = 0.1
PAGN_B = 0
TORQUE_MAX_RADIUS_RG = None
TORQUE_SMOOTHING = 0.005
MIN_TRAP_SEPARATION_CELLS = 1.0

COLORS = ("#1f77b4", "#BE9A62", "#2A7F62", "#8A5CA8", "#D55E00")


@dataclass
class MigrationDisk:
    profile: PagnDiskProfile
    rg_cm: float
    radius_cm: np.ndarray
    aspect_ratio: np.ndarray
    surface_density_g_cm2: np.ndarray


@dataclass
class MigrationSeries:
    migrator_mass_msun: float
    integrated_semi_times_myr: np.ndarray
    torque_times_myr: np.ndarray
    gap_parameter: np.ndarray
    crida_parameter: np.ndarray
    hill_radius_over_scale_height: np.ndarray
    stable_trap_radii_rg: np.ndarray
    gap_open_path_points: int
    nonfinite_torque_points: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gap-open-condition",
        choices=GAP_OPENING_CONDITIONS,
        default="kii",
        help="kii: KII > 11; crida: P < 1 and R_H/H > 1",
    )
    return parser.parse_args()


def _build_disk() -> MigrationDisk:
    agn_properties = AGN_properties()
    agn_properties.init_AGN(SMBH_MASS_MSUN, duty_cycle_myr=1.0)
    profile = build_pagn_disk_profile(
        SMBH_MASS_MSUN,
        alpha=PAGN_ALPHA,
        model=PAGN_DISK_MODEL,
        n_resolution=N_RESOLUTION,
        eddington_ratio=PAGN_EDDINGTON_RATIO,
        b=PAGN_B,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * SMBH_MASS_MSUN,
        suppress_pagn_output=True,
    )
    radius_cm = np.asarray(profile.radius_cm, dtype=float)
    order = np.argsort(radius_cm)
    return MigrationDisk(
        profile=profile,
        rg_cm=agn_properties.gravitational_radius_cm,
        radius_cm=radius_cm[order],
        aspect_ratio=np.asarray(profile.aspect_ratio, dtype=float)[order],
        surface_density_g_cm2=np.asarray(profile.surface_density_g_cm2, dtype=float)[
            order
        ],
    )


def _migration_series(
    disk: MigrationDisk,
    migrator_mass_msun: float,
    sample_radii_cm: np.ndarray,
    gap_opening_condition: str,
    params: MigrationParameters,
) -> MigrationSeries:
    migrator_mass_g = migrator_mass_msun * const.msun
    smbh_mass_g = SMBH_MASS_MSUN * const.msun
    mass_ratio = migrator_mass_g / smbh_mass_g
    torque_profile = compute_torque_profile(
        disk.profile.disk,
        migrator_mass_msun=migrator_mass_msun,
        max_radius_rg=TORQUE_MAX_RADIUS_RG,
        smoothing=TORQUE_SMOOTHING,
        params=params,
    )
    torque_radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
    local_aspect_ratio = np.interp(torque_radius_cm, disk.radius_cm, disk.aspect_ratio)
    local_surface_density_g_cm2 = np.interp(
        torque_radius_cm, disk.radius_cm, disk.surface_density_g_cm2
    )

    gap_parameter = migrator_mass_g**2 / (
        DISK_VISCOSITY_ALPHA * local_aspect_ratio**2 * smbh_mass_g**2
    )
    hill_ratio = np.cbrt(mass_ratio / 3.0) / local_aspect_ratio
    crida_parameter = (
        0.75 / hill_ratio + 50.0 * PAGN_ALPHA * local_aspect_ratio**2 / mass_ratio
    )
    selected_gap_mask = gap_opening_mask(
        mass_ratio, local_aspect_ratio, PAGN_ALPHA, gap_opening_condition
    )

    type_i_thermal_torque_si = np.asarray(
        torque_profile.gamma_type_i_si
        + (1 - torque_profile.gap_attenuation_factor) * torque_profile.gamma_th_si,
        dtype=float,
    )
    # Gap-open cells use Type II alone; thermal torque stays gap-closed.
    torque_si = np.where(
        selected_gap_mask,
        np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
        type_i_thermal_torque_si,
    )
    torque_cgs = torque_si_to_cgs(torque_si)

    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / torque_radius_cm**3)
    semi_type_i_time_s = (
        smbh_mass_g**2
        * local_aspect_ratio**2
        / 2.0
        / migrator_mass_g
        / local_surface_density_g_cm2
        / torque_radius_cm**2
        / orbital_frequency_rad_s
    )
    semi_type_i_torque_cgs = (
        -0.5
        * migrator_mass_g
        * torque_radius_cm**2
        * orbital_frequency_rad_s
        / semi_type_i_time_s
    )
    semi_torque_cgs = np.where(
        selected_gap_mask,
        torque_si_to_cgs(torque_profile.gamma_type_ii_si),
        semi_type_i_torque_cgs,
    )

    target_radius_cm = TARGET_RADIUS_RG * disk.rg_cm
    integrated_semi_times_myr = np.asarray(
        [
            migration_time_from_torque(
                torque_radius_cm,
                semi_torque_cgs,
                migrator_mass_g,
                float(start_radius_cm),
                target_radius_cm,
                SMBH_MASS_MSUN,
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

    stable_trap_radii_rg = classify_torque_traps(
        torque_profile.radius_rg,
        torque_si,
        min_separation_cells=MIN_TRAP_SEPARATION_CELLS,
    ).stable_radius_rg
    stable_trap_radii_rg = stable_trap_radii_rg[
        np.isfinite(stable_trap_radii_rg)
        & (stable_trap_radii_rg >= TARGET_RADIUS_RG)
        & (stable_trap_radii_rg <= START_RADIUS_RG)
    ]

    sampled_aspect_ratio = np.interp(
        sample_radii_cm, torque_radius_cm, local_aspect_ratio
    )
    return MigrationSeries(
        migrator_mass_msun=migrator_mass_msun,
        integrated_semi_times_myr=integrated_semi_times_myr,
        torque_times_myr=torque_times_myr,
        gap_parameter=np.interp(sample_radii_cm, torque_radius_cm, gap_parameter),
        crida_parameter=np.interp(sample_radii_cm, torque_radius_cm, crida_parameter),
        hill_radius_over_scale_height=np.interp(
            sample_radii_cm, torque_radius_cm, hill_ratio
        ),
        stable_trap_radii_rg=stable_trap_radii_rg,
        gap_open_path_points=int(
            np.count_nonzero(
                gap_opening_mask(
                    mass_ratio,
                    sampled_aspect_ratio,
                    PAGN_ALPHA,
                    gap_opening_condition,
                )
            )
        ),
        nonfinite_torque_points=int(np.count_nonzero(~np.isfinite(torque_times_myr))),
    )


def _plot_results(
    sample_radii_rg: np.ndarray,
    series: list[MigrationSeries],
) -> Path:
    fig, (ax, gap_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (3.0, 1.0), "hspace": 0.06},
    )
    has_visible_trap = False
    for index, result in enumerate(series):
        color = COLORS[index % len(COLORS)]
        semi_mask = np.isfinite(result.integrated_semi_times_myr) & (
            result.integrated_semi_times_myr > 0.0
        )
        torque_mask = np.isfinite(result.torque_times_myr) & (
            result.torque_times_myr > 0.0
        )
        ax.plot(
            sample_radii_rg[semi_mask],
            result.integrated_semi_times_myr[semi_mask],
            color=color,
            lw=2.8,
            ls="--",
            alpha=0.9,
        )
        ax.plot(
            sample_radii_rg[torque_mask],
            result.torque_times_myr[torque_mask],
            color=color,
            lw=2.8,
            ls="-",
            alpha=0.9,
        )
        for trap_radius_rg in result.stable_trap_radii_rg:
            ax.axvline(trap_radius_rg, color=color, lw=1.3, ls=":", alpha=0.5)
        has_visible_trap |= result.stable_trap_radii_rg.size > 0

        gap_ax.plot(
            sample_radii_rg,
            result.gap_parameter / TYPE_II_GAP_PARAMETER_THRESHOLD,
            color=color,
            lw=2.2,
            ls="-",
        )
        gap_ax.plot(
            sample_radii_rg,
            result.crida_parameter,
            color=color,
            lw=2.2,
            ls="-.",
        )
        gap_ax.plot(
            sample_radii_rg,
            result.hill_radius_over_scale_height,
            color=color,
            lw=2.2,
            ls=":",
        )

    ax.axvline(TARGET_RADIUS_RG, color="0.35", lw=1.5, ls=":", alpha=0.8)
    ax.axvline(START_RADIUS_RG, color="0.35", lw=1.5, ls="-.", alpha=0.8)
    ax.set(xscale="log", yscale="log", ylabel="Migration timescale (Myr)")
    ax.set_title(
        rf"$M_\bullet={SMBH_MASS_MSUN:.0e}\,M_\odot$, "
        rf"$R_\mathrm{{target}}={TARGET_RADIUS_RG:.0e}\,r_g$"
    )
    ax.grid(True, alpha=0.25, linestyle="--", which="both")

    mass_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[index % len(COLORS)],
            lw=2.8,
            label=rf"$m={mass:g}\,M_\odot$",
        )
        for index, mass in enumerate(MIGRATOR_MASSES_MSUN)
    ]
    method_handles = [
        Line2D([0], [0], color="0.2", lw=2.8, ls="-", label="torque"),
        Line2D([0], [0], color="0.2", lw=2.8, ls="--", label="int-semi"),
        Line2D([0], [0], color="0.35", lw=1.5, ls=":", label="target radius"),
        Line2D([0], [0], color="0.35", lw=1.5, ls="-.", label="start radius"),
    ]
    if has_visible_trap:
        method_handles.append(
            Line2D([0], [0], color="0.2", lw=1.3, ls=":", label="stable trap")
        )
    ax.legend(handles=mass_handles + method_handles, loc="best", framealpha=0.88)

    gap_ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Starting radius $R/r_g$",
        ylabel="Dimensionless value",
    )
    gap_ax.grid(True, alpha=0.25, linestyle="--", which="both")
    gap_ax.legend(
        handles=[
            Line2D([0], [0], color="0.2", lw=2.2, ls="-", label=r"$K_\mathrm{II}/11$"),
            Line2D([0], [0], color="0.2", lw=2.2, ls="-.", label=r"$P_\mathrm{Crida}$"),
            Line2D([0], [0], color="0.2", lw=2.2, ls=":", label=r"$R_H/H$"),
        ],
        loc="best",
        framealpha=0.88,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _format_time(value_myr: float) -> str:
    return f"{value_myr:.6e}" if np.isfinite(value_myr) else "inf"


def _print_results(
    disk: MigrationDisk,
    series: list[MigrationSeries],
    gap_opening_condition: str,
    figure_path: Path,
) -> None:
    print("Disk migration timescale comparison")
    print(f"  smbh_mass_msun: {SMBH_MASS_MSUN:.6e}")
    print(f"  rg_cm: {disk.rg_cm:.6e}")
    print(
        "  disk_radius_rg: "
        f"{disk.profile.inner_radius_cm / disk.rg_cm:.6e} to "
        f"{disk.profile.outer_radius_cm / disk.rg_cm:.6e}"
    )
    print(f"  path_rg: {START_RADIUS_RG:.6e} -> {TARGET_RADIUS_RG:.6e}")
    print(f"  gap_opening_condition: {gap_opening_condition}")
    print("  gap_open_torque: type_ii_only")
    print("  terminal_times_myr:")
    print(
        "    mass_msun  int-semi  torque  max_gap_parameter  "
        "gap_open_path_points  nonfinite_torque_points"
    )
    for result in series:
        print(
            f"    {result.migrator_mass_msun:9.3g}  "
            f"{_format_time(float(result.integrated_semi_times_myr[-1])):>13}  "
            f"{_format_time(float(result.torque_times_myr[-1])):>13}  "
            f"{float(np.max(result.gap_parameter)):>17.6e}  "
            f"{result.gap_open_path_points:>13d}  "
            f"{result.nonfinite_torque_points:>23d}"
        )
    print(f"  figure: {figure_path}")


def main(gap_opening_condition: str) -> None:
    disk = _build_disk()
    sample_radii_rg = np.logspace(
        np.log10(TARGET_RADIUS_RG), np.log10(START_RADIUS_RG), 120
    )
    sample_radii_cm = sample_radii_rg * disk.rg_cm
    params = MigrationParameters(
        alpha=PAGN_ALPHA,
        gamma_adiabat=5.0 / 3.0,
        hydrogen_fraction=0.7,
        kappa_t=20.0,
        kappa_norm=25.0,
    )
    series = [
        _migration_series(
            disk,
            float(migrator_mass_msun),
            sample_radii_cm,
            gap_opening_condition,
            params,
        )
        for migrator_mass_msun in MIGRATOR_MASSES_MSUN
    ]
    figure_path = _plot_results(sample_radii_rg, series)
    _print_results(disk, series, gap_opening_condition, figure_path)


if __name__ == "__main__":
    main(_parse_args().gap_open_condition)
