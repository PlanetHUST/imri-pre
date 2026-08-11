"""Plot torque trap, anti-trap, disk outer, and LISA-entry radii versus SMBH mass."""

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
import numpy as np

from src.disk_quantity.migration_torque import (
    MigrationParameters,
    classify_torque_traps,
    compute_torque_profile,
)
from src.disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from src.physical_process import constants as const
from src.physical_process.disk_migration import (
    DISK_VISCOSITY_ALPHA,
    TYPE_II_GAP_PARAMETER_THRESHOLD,
)
from src.physical_process.gw_frequency_radius import lisa_entry_radius_cm
from src.workflows.config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ,
    IMRI_REDSHIFT,
    PAGN_ALPHA,
)

SMBH_MIN_MSUN = 1.0e4
SMBH_MAX_MSUN = 1.0e8
SMBH_COUNT = 50
MIGRATOR_MASS_MSUN = 30.0
N_RESOLUTION = 2000
OUTPUT_DIR = Path("figs")
PAGN_DISK_MODEL = "sirko"
PAGN_EDDINGTON_RATIO = 0.1
PAGN_B = 0
TORQUE_MAX_RADIUS_RG = 1e7
TORQUE_SMOOTHING = 0.02
LISA_MIN_GW_FREQUENCY_HZ = IMRI_LISA_MIN_GW_FREQUENCY_HZ
REDSHIFT = IMRI_REDSHIFT
OUTPUT_STEM = "trap_radius_2d"


def _positive_sorted(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    return np.sort(array)


def main() -> None:
    smbh_masses_msun = np.logspace(
        np.log10(SMBH_MIN_MSUN),
        np.log10(SMBH_MAX_MSUN),
        SMBH_COUNT,
    )
    migrator_mass_msun = float(MIGRATOR_MASS_MSUN)
    migrator_mass_g = migrator_mass_msun * const.msun

    trap_radius_rg = [np.array([], dtype=float) for _ in smbh_masses_msun]
    anti_trap_radius_rg = [np.array([], dtype=float) for _ in smbh_masses_msun]
    lisa_entry_radius_rg = np.full(smbh_masses_msun.size, np.nan, dtype=float)
    disk_outer_radius_rg = np.full(smbh_masses_msun.size, np.nan, dtype=float)
    gap_open_parameter = np.full(smbh_masses_msun.size, np.nan, dtype=float)

    params = MigrationParameters(
        alpha=PAGN_ALPHA,
    )

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        smbh_mass_msun = float(smbh_mass_msun)
        smbh_mass_g = smbh_mass_msun * const.msun
        rg_cm = const.G * smbh_mass_g / const.c**2
        disk_profile = build_pagn_disk_profile(
            smbh_mass_msun,
            alpha=PAGN_ALPHA,
            model=PAGN_DISK_MODEL,
            n_resolution=N_RESOLUTION,
            eddington_ratio=PAGN_EDDINGTON_RATIO,
            b=PAGN_B,
            accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun,
            suppress_pagn_output=True,
        )
        disk_outer_radius_rg[smbh_index] = disk_profile.outer_radius_cm / rg_cm

        disk_radius_cm = np.asarray(disk_profile.radius_cm, dtype=float)
        disk_aspect_ratio = np.asarray(disk_profile.aspect_ratio, dtype=float)
        disk_order = np.argsort(disk_radius_cm)
        disk_radius_cm = disk_radius_cm[disk_order]
        disk_aspect_ratio = disk_aspect_ratio[disk_order]

        entry_radius_cm = lisa_entry_radius_cm(
            primary_mass_msun=smbh_mass_msun,
            secondary_mass_msun=migrator_mass_msun,
            min_gw_frequency_hz=LISA_MIN_GW_FREQUENCY_HZ,
            redshift=REDSHIFT,
        )
        if np.isfinite(entry_radius_cm) and entry_radius_cm > 0.0:
            lisa_entry_radius_rg[smbh_index] = entry_radius_cm / rg_cm

        torque_profile = compute_torque_profile(
            disk_profile.disk,
            migrator_mass_msun=migrator_mass_msun,
            max_radius_rg=TORQUE_MAX_RADIUS_RG,
            smoothing=TORQUE_SMOOTHING,
            params=params,
        )

        torque_radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
        torque_si = np.asarray(
            torque_profile.gamma_i_si
            + torque_profile.gap_attenuation_factor * torque_profile.gamma_th_si,
            dtype=float,
        )
        local_aspect_ratio = np.interp(
            torque_radius_cm,
            disk_radius_cm,
            disk_aspect_ratio,
        )
        type_ii_gap_parameter_dimensionless = migrator_mass_g**2 / (
            DISK_VISCOSITY_ALPHA * local_aspect_ratio**2 * smbh_mass_g**2
        )
        finite_gap_parameter = type_ii_gap_parameter_dimensionless[
            np.isfinite(type_ii_gap_parameter_dimensionless)
            & (type_ii_gap_parameter_dimensionless > 0.0)
        ]
        if finite_gap_parameter.size:
            gap_open_parameter[smbh_index] = float(np.max(finite_gap_parameter))
        type_ii_mask = np.isfinite(type_ii_gap_parameter_dimensionless) & (
            type_ii_gap_parameter_dimensionless > TYPE_II_GAP_PARAMETER_THRESHOLD
        )
        torque_si = torque_si + np.where(
            type_ii_mask,
            np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
            0.0,
        )

        trap_radii = classify_torque_traps(
            torque_radius_cm,
            torque_si,
            gravitational_radius_cm=rg_cm,
            stable_region_rg=100.0,
        )
        trap_radius_rg[smbh_index] = _positive_sorted(
            trap_radii.stable_radius_cm / rg_cm
        )
        anti_trap_radius_rg[smbh_index] = _positive_sorted(
            trap_radii.anti_trap_radius_cm / rg_cm
        )

    radius_values = [
        values for values in (*trap_radius_rg, *anti_trap_radius_rg) if values.size
    ]
    radius_values.extend([lisa_entry_radius_rg, disk_outer_radius_rg])
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

    for grid, color, marker, label in (
        (
            trap_radius_rg,
            "#1f77b4",
            "o",
            rf"Stable trap, $m={migrator_mass_msun:g}\,M_\odot$",
        ),
        (
            anti_trap_radius_rg,
            "#7A3DB8",
            "x",
            rf"Anti-trap, $m={migrator_mass_msun:g}\,M_\odot$",
        ),
    ):
        max_branch_count = max((radii_rg.size for radii_rg in grid), default=0)
        label_used = False
        for branch_index in range(max_branch_count):
            y_values = np.full_like(smbh_masses_msun, np.nan, dtype=float)
            for smbh_index, radii_rg in enumerate(grid):
                radii_rg = _positive_sorted(radii_rg)
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

    lisa_color = "#D55E00"
    lisa_mask = np.isfinite(lisa_entry_radius_rg) & (lisa_entry_radius_rg > 0.0)
    if np.any(lisa_mask):
        ax.plot(
            smbh_masses_msun[lisa_mask],
            lisa_entry_radius_rg[lisa_mask],
            color=lisa_color,
            lw=2.8,
            ls="--",
            label=rf"LISA entry, $m={migrator_mass_msun:g}\,M_\odot$",
        )

    rout_mask = np.isfinite(disk_outer_radius_rg) & (disk_outer_radius_rg > 0.0)
    if np.any(rout_mask):
        ax.plot(
            smbh_masses_msun[rout_mask],
            disk_outer_radius_rg[rout_mask],
            color="#009E73",
            lw=2.4,
            ls=":",
            label=r"Disk outer radius $r_\mathrm{out}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Radius $R/r_g$")
    ax.set_ylim(float(positive_radii[0]) / 1.6, float(positive_radii[-1]) * 1.6)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="best", framealpha=0.88)
    ax.set_title(
        rf"Migration traps, disk outer radius, and LISA entry, "
        rf"$m={migrator_mass_msun:g}\,M_\odot$, $z={REDSHIFT:g}$"
    )

    gap_mask = np.isfinite(gap_open_parameter) & (gap_open_parameter > 0.0)
    if np.any(gap_mask):
        gap_ax.plot(
            smbh_masses_msun[gap_mask],
            gap_open_parameter[gap_mask],
            color="0.15",
            lw=2.2,
            marker=".",
            label="Max gap-opening parameter",
        )
    gap_ax.axhline(
        TYPE_II_GAP_PARAMETER_THRESHOLD,
        color="0.45",
        lw=1.5,
        ls="--",
        label=r"Gap-opening threshold",
    )
    gap_ax.set_xscale("log")
    gap_ax.set_yscale("log")
    gap_ax.set_xlabel(r"$M_\bullet\;(M_\odot)$")
    gap_ax.set_ylabel(r"$m^2/[\alpha(H/R)^2M_\bullet^2]$")
    gap_ax.grid(True, which="both", alpha=0.25, linestyle="--")
    gap_ax.legend(loc="best", framealpha=0.88)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    stable_point_count = sum(radii_rg.size for radii_rg in trap_radius_rg)
    anti_point_count = sum(radii_rg.size for radii_rg in anti_trap_radius_rg)
    total_cells = int(smbh_masses_msun.size)
    positive_rout = _positive_sorted(disk_outer_radius_rg)
    positive_gap = _positive_sorted(gap_open_parameter)
    print("Trap radius 2D diagnostic")
    print(
        "  smbh_mass_range_msun: "
        f"{smbh_masses_msun[0]:.6e} to {smbh_masses_msun[-1]:.6e} "
        f"({smbh_masses_msun.size} points)"
    )
    print(f"  migrator_mass_msun: {migrator_mass_msun:.6g}")
    if positive_rout.size:
        print(
            "  disk_outer_radius_rg_range: "
            f"{positive_rout[0]:.6e} to {positive_rout[-1]:.6e}"
        )
    if positive_gap.size:
        print(
            "  max_gap_open_parameter_range: "
            f"{positive_gap[0]:.6e} to {positive_gap[-1]:.6e}"
        )
    print(f"  stable_trap_points: {stable_point_count}")
    print(f"  anti_trap_points: {anti_point_count}")
    print(f"  torque_cells: {total_cells}")
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    main()
