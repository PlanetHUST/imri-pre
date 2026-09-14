"""Plot torque trap, anti-trap, disk outer, and LISA-entry radii."""

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
import numpy as np

from src.disk_quantity.migration_torque import (
    MigrationParameters,
    classify_torque_traps,
    comp_gap_parameter_dimensionless,
    compute_torque_profile,
)
from src.disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from src.physical_process import constants as const
from src.physical_process.disk_migration import (
    DEFAULT_MIGRATION_MODEL,
    GAP_OPENING_CONDITIONS,
    MIGRATION_MODELS,
    select_migration_torque_si,
)
from src.physical_process.gw_frequency_radius import lisa_entry_radius_cm
from src.workflows.config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ,
    PAGN_ALPHA,
)

SMBH_MIN_MSUN = 1.0e3
SMBH_MAX_MSUN = 1.0e7
SMBH_COUNT = 50
MIGRATOR_MASS_MSUN = 10.0
OUTPUT_DIR = Path("figs")
OUTPUT_STEM = "trap_radius_2d"

TORQUE_MAX_RADIUS_RG = None
MIN_TRAP_SEPARATION_CELLS = 1.0
LISA_MIN_GW_FREQUENCY_HZ = IMRI_LISA_MIN_GW_FREQUENCY_HZ
REDSHIFT = 10

plt.rcParams.update(
    {
        "font.size": 20,  # default text
        "axes.titlesize": 18,  # title
        "axes.labelsize": 18,  # x/y labels
        "xtick.labelsize": 16,  # x tick numbers
        "ytick.labelsize": 16,  # y tick numbers
        "legend.fontsize": 12,  # legend
        "figure.titlesize": 18,  # suptitle
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)


@dataclass
class TrapPoint:
    stable_radius_rg: np.ndarray
    anti_trap_radius_rg: np.ndarray
    lisa_entry_radius_rg: float
    disk_outer_radius_rg: float
    max_gap_parameter: float
    min_crida_parameter: float
    max_hill_ratio: float


@dataclass
class TrapResults:
    smbh_masses_msun: np.ndarray
    stable_radius_rg: list[np.ndarray]
    anti_trap_radius_rg: list[np.ndarray]
    lisa_entry_radius_rg: np.ndarray
    disk_outer_radius_rg: np.ndarray
    max_gap_parameter: np.ndarray
    min_crida_parameter: np.ndarray
    max_hill_ratio: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gap-open-condition",
        choices=GAP_OPENING_CONDITIONS,
        default="kii",
        help="kii: KII > 11; crida: P < 1 and R_H/H > 1",
    )
    parser.add_argument(
        "--migration-model",
        choices=MIGRATION_MODELS,
        default=DEFAULT_MIGRATION_MODEL,
    )
    return parser.parse_args()


def _positive_sorted(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.sort(array[np.isfinite(array) & (array > 0.0)])


def _trap_point(
    smbh_mass_msun: float,
    migrator_mass_msun: float,
    gap_opening_condition: str,
    params: MigrationParameters,
    migration_model: str,
) -> TrapPoint:
    smbh_mass_g = smbh_mass_msun * const.msun
    migrator_mass_g = migrator_mass_msun * const.msun
    mass_ratio = migrator_mass_g / smbh_mass_g
    rg_cm = const.G * smbh_mass_g / const.c**2

    disk_profile = build_pagn_disk_profile(
        smbh_mass_msun,
        PAGN_ALPHA,
        accretion_rate_g_s=(
            DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun
        ),
    )
    entry_radius_cm = lisa_entry_radius_cm(
        primary_mass_msun=smbh_mass_msun,
        secondary_mass_msun=migrator_mass_msun,
        min_gw_frequency_hz=LISA_MIN_GW_FREQUENCY_HZ,
        redshift=REDSHIFT,
    )
    lisa_entry_radius_rg = (
        entry_radius_cm / rg_cm
        if np.isfinite(entry_radius_cm) and entry_radius_cm > 0.0
        else np.nan
    )

    torque_profile = compute_torque_profile(
        disk_profile.disk,
        migrator_mass_msun=migrator_mass_msun,
        max_radius_rg=TORQUE_MAX_RADIUS_RG,
        params=params,
    )
    torque_radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
    solved_disk_radius_m = np.asarray(getattr(disk_profile.disk, "R"), dtype=float)
    solved_disk_aspect_ratio = (
        np.asarray(getattr(disk_profile.disk, "h"), dtype=float) / solved_disk_radius_m
    )
    disk_order = np.argsort(solved_disk_radius_m)
    local_aspect_ratio = np.interp(
        torque_radius_cm,
        solved_disk_radius_m[disk_order] * 100.0,
        solved_disk_aspect_ratio[disk_order],
    )

    gap_parameter = comp_gap_parameter_dimensionless(
        mass_ratio,
        local_aspect_ratio,
        PAGN_ALPHA,
    )
    hill_ratio = np.cbrt(mass_ratio / 3.0) / local_aspect_ratio
    crida_parameter = (
        0.75 / hill_ratio + 50.0 * PAGN_ALPHA * local_aspect_ratio**2 / mass_ratio
    )
    migration_torque_si = select_migration_torque_si(
        torque_profile,
        mass_ratio,
        local_aspect_ratio,
        PAGN_ALPHA,
        gap_opening_condition,
        consider_type_ii_migration=True,
        migration_model=migration_model,
    ).torque_si
    trap_radii = classify_torque_traps(
        torque_profile.radius_rg,
        migration_torque_si,
        min_separation_cells=MIN_TRAP_SEPARATION_CELLS,
    )

    positive_gap = _positive_sorted(gap_parameter)
    positive_crida = _positive_sorted(crida_parameter)
    positive_hill_ratio = _positive_sorted(hill_ratio)
    return TrapPoint(
        stable_radius_rg=_positive_sorted(trap_radii.stable_radius_rg),
        anti_trap_radius_rg=_positive_sorted(trap_radii.anti_trap_radius_rg),
        lisa_entry_radius_rg=float(lisa_entry_radius_rg),
        disk_outer_radius_rg=float(disk_profile.outer_radius_cm / rg_cm),
        max_gap_parameter=(
            float(np.max(positive_gap)) if positive_gap.size else np.nan
        ),
        min_crida_parameter=(
            float(np.min(positive_crida)) if positive_crida.size else np.nan
        ),
        max_hill_ratio=(
            float(np.max(positive_hill_ratio)) if positive_hill_ratio.size else np.nan
        ),
    )


def _compute_results(
    gap_opening_condition: str,
    migration_model: str,
) -> TrapResults:
    smbh_masses_msun = np.logspace(
        np.log10(SMBH_MIN_MSUN), np.log10(SMBH_MAX_MSUN), SMBH_COUNT
    )
    params = MigrationParameters(alpha=PAGN_ALPHA)
    points = [
        _trap_point(
            float(smbh_mass_msun),
            float(MIGRATOR_MASS_MSUN),
            gap_opening_condition,
            params,
            migration_model,
        )
        for smbh_mass_msun in smbh_masses_msun
    ]
    return TrapResults(
        smbh_masses_msun=smbh_masses_msun,
        stable_radius_rg=[point.stable_radius_rg for point in points],
        anti_trap_radius_rg=[point.anti_trap_radius_rg for point in points],
        lisa_entry_radius_rg=np.asarray(
            [point.lisa_entry_radius_rg for point in points]
        ),
        disk_outer_radius_rg=np.asarray(
            [point.disk_outer_radius_rg for point in points]
        ),
        max_gap_parameter=np.asarray([point.max_gap_parameter for point in points]),
        min_crida_parameter=np.asarray([point.min_crida_parameter for point in points]),
        max_hill_ratio=np.asarray([point.max_hill_ratio for point in points]),
    )


def _plot_branches(
    ax,
    smbh_masses_msun: np.ndarray,
    radius_grid: list[np.ndarray],
    color: str,
    marker: str,
    label: str,
) -> None:
    label_used = False
    max_branch_count = max((radii.size for radii in radius_grid), default=0)
    for branch_index in range(max_branch_count):
        y_values = np.full_like(smbh_masses_msun, np.nan, dtype=float)
        for smbh_index, radii_rg in enumerate(radius_grid):
            if branch_index < radii_rg.size:
                y_values[smbh_index] = radii_rg[branch_index]
        mask = np.isfinite(y_values) & (y_values > 0.0)
        if not np.any(mask):
            continue
        ax.scatter(
            smbh_masses_msun[mask],
            y_values[mask],
            color=color,
            marker=marker,
            s=28.0,
            linewidths=1.2,
            label=label if not label_used else None,
            zorder=3,
        )
        label_used = True


def _plot_results(results: TrapResults) -> Path:
    radius_values = [
        radii
        for radii in (*results.stable_radius_rg, *results.anti_trap_radius_rg)
        if radii.size
    ]
    radius_values.extend([results.lisa_entry_radius_rg, results.disk_outer_radius_rg])
    positive_radii = _positive_sorted(np.concatenate(radius_values))
    if positive_radii.size == 0:
        raise RuntimeError("no positive finite radii were computed")

    fig, (ax, gap_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 8.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (3.0, 1.0), "hspace": 0.06},
    )
    # fig, ax = plt.subplots(
    #     figsize=(10, 8.5),
    # )
    migrator_mass_msun = float(MIGRATOR_MASS_MSUN)
    _plot_branches(
        ax,
        results.smbh_masses_msun,
        results.stable_radius_rg,
        "#1f77b4",
        "o",
        rf"Stable trap, $m={migrator_mass_msun:g}\,M_\odot$",
    )
    _plot_branches(
        ax,
        results.smbh_masses_msun,
        results.anti_trap_radius_rg,
        "#7A3DB8",
        "x",
        rf"Anti-trap, $m={migrator_mass_msun:g}\,M_\odot$",
    )

    lisa_mask = np.isfinite(results.lisa_entry_radius_rg) & (
        results.lisa_entry_radius_rg > 0.0
    )
    if np.any(lisa_mask):
        ax.plot(
            results.smbh_masses_msun[lisa_mask],
            results.lisa_entry_radius_rg[lisa_mask],
            color="#D55E00",
            lw=2.8,
            ls="--",
            label=rf"LISA entry, $m={migrator_mass_msun:g}\,M_\odot$",
        )
    rout_mask = np.isfinite(results.disk_outer_radius_rg) & (
        results.disk_outer_radius_rg > 0.0
    )
    if np.any(rout_mask):
        ax.plot(
            results.smbh_masses_msun[rout_mask],
            results.disk_outer_radius_rg[rout_mask],
            color="#009E73",
            lw=2.4,
            ls=":",
            label=r"Disk outer radius $r_\mathrm{out}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelbottom=False)
    # ax.set_xlabel(r"$M_\bullet\;(M_\odot)$")
    ax.set_ylabel(r"Radius $R/r_g$")
    ax.set_ylim(float(positive_radii[0]) / 1.6, float(positive_radii[-1]) * 1.6)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="best", framealpha=0.88)
    ax.set_title(
        rf"Migration traps, disk outer radius, and LISA entry, "
        rf"$m={migrator_mass_msun:g}\,M_\odot$, $z={REDSHIFT:g}$"
    )

    for values, color, linestyle, label in (
        (results.max_gap_parameter, "#1f77b4", "-", r"$\max K_\mathrm{II}$"),
        (
            results.min_crida_parameter,
            "#D55E00",
            "-.",
            r"$\min P_\mathrm{Crida}$",
        ),
        (results.max_hill_ratio, "#2A7F62", ":", r"$\max(R_H/H)$"),
    ):
        mask = np.isfinite(values) & (values > 0.0)
        if np.any(mask):
            gap_ax.plot(
                results.smbh_masses_msun[mask],
                values[mask],
                color=color,
                lw=2.2,
                ls=linestyle,
                marker="." if linestyle == "-" else None,
                label=label,
            )
    gap_ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$M_\bullet\;(M_\odot)$",
        ylabel="Dimensionless value",
    )
    gap_ax.grid(True, which="both", alpha=0.25, linestyle="--")
    gap_ax.legend(loc="best", framealpha=0.88)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _print_range(label: str, values: np.ndarray) -> None:
    positive = _positive_sorted(values)
    if positive.size:
        print(f"  {label}: {positive[0]:.6e} to {positive[-1]:.6e}")


def _print_results(
    results: TrapResults,
    gap_opening_condition: str,
    migration_model: str,
    figure_path: Path,
) -> None:
    masses = results.smbh_masses_msun
    print("Trap radius 2D diagnostic")
    print(
        "  smbh_mass_range_msun: "
        f"{masses[0]:.6e} to {masses[-1]:.6e} ({masses.size} points)"
    )
    print(f"  migrator_mass_msun: {MIGRATOR_MASS_MSUN:.6g}")
    print(f"  gap_opening_condition: {gap_opening_condition}")
    print(f"  migration_model: {migration_model}")
    _print_range("disk_outer_radius_rg_range", results.disk_outer_radius_rg)
    _print_range("max_gap_parameter_K_range", results.max_gap_parameter)
    _print_range("min_crida_gap_parameter_range", results.min_crida_parameter)
    _print_range("max_hill_radius_over_scale_height_range", results.max_hill_ratio)
    print(
        "  stable_trap_points: "
        f"{sum(radii.size for radii in results.stable_radius_rg)}"
    )
    print(
        "  anti_trap_points: "
        f"{sum(radii.size for radii in results.anti_trap_radius_rg)}"
    )
    print(f"  torque_cells: {masses.size}")
    print(f"  figure: {figure_path}")


def main(
    gap_opening_condition: str,
    migration_model: str = DEFAULT_MIGRATION_MODEL,
) -> None:
    results = _compute_results(gap_opening_condition, migration_model)
    figure_path = _plot_results(results)
    _print_results(results, gap_opening_condition, migration_model, figure_path)


if __name__ == "__main__":
    arguments = _parse_args()
    main(arguments.gap_open_condition, arguments.migration_model)
