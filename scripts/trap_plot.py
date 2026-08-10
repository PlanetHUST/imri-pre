"""Plot torque trap, anti-trap, and LISA-entry radii versus SMBH mass."""

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

DEFAULT_SMBH_MIN_MSUN = 1.0e3
DEFAULT_SMBH_MAX_MSUN = 1.0e7
DEFAULT_SMBH_COUNT = 50
DEFAULT_MIGRATOR_MASSES_MSUN = (1.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_CENTER_MIGRATOR_MASS_MSUN = 15.0
DEFAULT_N_RESOLUTION = 2000
DEFAULT_OUTPUT_DIR = Path("figs")
OUTPUT_STEM = "trap_radius_2d"


def _parse_mass_vector(value: str) -> list[float]:
    items = value.strip().removeprefix("[").removesuffix("]").split(",")
    try:
        masses = [float(item.strip()) for item in items if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mass vector must contain comma-separated numbers"
        ) from error
    if not masses:
        raise argparse.ArgumentTypeError("mass vector must not be empty")
    if any(mass <= 0.0 for mass in masses):
        raise argparse.ArgumentTypeError("masses must be positive")
    return masses


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pAGN stable traps, anti-traps, and LISA entry radii."
    )
    parser.add_argument("--smbh-min-msun", type=float, default=DEFAULT_SMBH_MIN_MSUN)
    parser.add_argument("--smbh-max-msun", type=float, default=DEFAULT_SMBH_MAX_MSUN)
    parser.add_argument("--smbh-count", type=int, default=DEFAULT_SMBH_COUNT)
    parser.add_argument(
        "--migrator-masses-msun",
        type=_parse_mass_vector,
        default=list(DEFAULT_MIGRATOR_MASSES_MSUN),
        help="comma-separated migrator masses used for non-central samples",
    )
    parser.add_argument(
        "--center-migrator-mass-msun",
        type=float,
        default=DEFAULT_CENTER_MIGRATOR_MASS_MSUN,
        help="migrator mass used for central trap and LISA curves",
    )
    parser.add_argument(
        "--n-resolution",
        type=int,
        default=int(os.getenv("PAGN_DISK_N_RESOLUTION", DEFAULT_N_RESOLUTION)),
        help="pAGN radial grid resolution for each SMBH mass",
    )
    parser.add_argument(
        "--lisa-min-gw-frequency-hz",
        type=float,
        default=IMRI_LISA_MIN_GW_FREQUENCY_HZ,
        help="observed-frame lower GW frequency defining LISA-band entry",
    )
    parser.add_argument(
        "--redshift",
        type=float,
        default=IMRI_REDSHIFT,
        help="source redshift used by the LISA entry radius helper",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.02,
        help="log-slope smoothing passed to the pAGN torque profile",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the output figure",
    )
    return parser.parse_args()


def _positive_sorted(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    return np.sort(array)


def main() -> None:
    args = _parse_args()
    if args.smbh_min_msun <= 0.0 or args.smbh_max_msun <= 0.0:
        raise ValueError("SMBH mass grid bounds must be positive")
    if args.smbh_max_msun <= args.smbh_min_msun:
        raise ValueError("SMBH mass grid maximum must exceed minimum")
    if args.smbh_count < 2:
        raise ValueError("SMBH mass grid count must be at least 2")
    if args.center_migrator_mass_msun <= 0.0:
        raise ValueError("center migrator mass must be positive")

    smbh_masses_msun = np.logspace(
        np.log10(args.smbh_min_msun),
        np.log10(args.smbh_max_msun),
        int(args.smbh_count),
    )
    migrator_masses_msun = np.unique(
        np.asarray(
            [*args.migrator_masses_msun, args.center_migrator_mass_msun],
            dtype=float,
        )
    )
    center_index = int(
        np.argmin(np.abs(migrator_masses_msun - args.center_migrator_mass_msun))
    )
    center_mass_msun = float(migrator_masses_msun[center_index])

    trap_radius_rg = [
        [np.array([], dtype=float) for _ in smbh_masses_msun]
        for _ in migrator_masses_msun
    ]
    anti_trap_radius_rg = [
        [np.array([], dtype=float) for _ in smbh_masses_msun]
        for _ in migrator_masses_msun
    ]
    lisa_entry_radius_rg = np.full(
        (migrator_masses_msun.size, smbh_masses_msun.size),
        np.nan,
        dtype=float,
    )

    params = MigrationParameters(alpha=PAGN_ALPHA)
    failed_torque_cells = 0

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        smbh_mass_msun = float(smbh_mass_msun)
        smbh_mass_g = smbh_mass_msun * const.msun
        rg_cm = const.G * smbh_mass_g / const.c**2
        disk_profile = build_pagn_disk_profile(
            smbh_mass_msun,
            n_resolution=args.n_resolution,
            accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun,
            alpha=PAGN_ALPHA,
            eddington_ratio=0.1,
        )

        disk_radius_cm = np.asarray(disk_profile.radius_cm, dtype=float)
        disk_aspect_ratio = np.asarray(disk_profile.aspect_ratio, dtype=float)
        disk_order = np.argsort(disk_radius_cm)
        disk_radius_cm = disk_radius_cm[disk_order]
        disk_aspect_ratio = disk_aspect_ratio[disk_order]

        for migrator_index, migrator_mass_msun in enumerate(migrator_masses_msun):
            migrator_mass_msun = float(migrator_mass_msun)
            migrator_mass_g = migrator_mass_msun * const.msun

            entry_radius_cm = lisa_entry_radius_cm(
                smbh_mass_msun,
                migrator_mass_msun,
                args.lisa_min_gw_frequency_hz,
                args.redshift,
            )
            if np.isfinite(entry_radius_cm) and entry_radius_cm > 0.0:
                lisa_entry_radius_rg[migrator_index, smbh_index] = (
                    entry_radius_cm / rg_cm
                )

            try:
                torque_profile = compute_torque_profile(
                    disk_profile.disk,
                    migrator_mass_msun=migrator_mass_msun,
                    smoothing=args.smoothing,
                    params=params,
                )
            except Exception:
                failed_torque_cells += 1
                continue

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
            type_ii_mask = (
                np.isfinite(type_ii_gap_parameter_dimensionless)
                & (
                    type_ii_gap_parameter_dimensionless
                    > TYPE_II_GAP_PARAMETER_THRESHOLD
                )
            )
            torque_si = torque_si + np.where(
                type_ii_mask,
                np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
                0.0,
            )

            trap_radii = classify_torque_traps(torque_radius_cm, torque_si)
            trap_radius_rg[migrator_index][smbh_index] = _positive_sorted(
                trap_radii.stable_radius_cm / rg_cm
            )
            anti_trap_radius_rg[migrator_index][smbh_index] = _positive_sorted(
                trap_radii.anti_trap_radius_cm / rg_cm
            )

    lisa_lower = np.full(smbh_masses_msun.size, np.nan, dtype=float)
    lisa_upper = np.full(smbh_masses_msun.size, np.nan, dtype=float)
    for smbh_index in range(smbh_masses_msun.size):
        finite_lisa = _positive_sorted(lisa_entry_radius_rg[:, smbh_index])
        if finite_lisa.size:
            lisa_lower[smbh_index] = float(finite_lisa[0])
            lisa_upper[smbh_index] = float(finite_lisa[-1])
    lisa_center = lisa_entry_radius_rg[center_index]

    radius_values = [
        values
        for grid in (trap_radius_rg, anti_trap_radius_rg)
        for by_mass in grid
        for values in by_mass
        if values.size
    ]
    radius_values.extend([lisa_lower, lisa_upper, lisa_center])
    positive_radii = _positive_sorted(np.concatenate(radius_values))
    if positive_radii.size == 0:
        raise RuntimeError("no positive finite radii were computed")

    fig, ax = plt.subplots(figsize=(10, 7))

    stable_sample_points = 0
    anti_sample_points = 0
    for grid, color, marker, label in (
        (
            trap_radius_rg,
            "#1f77b4",
            "o",
            r"Stable trap samples, non-central $m$",
        ),
        (
            anti_trap_radius_rg,
            "#7A3DB8",
            "x",
            r"Anti-trap samples, non-central $m$",
        ),
    ):
        x_points = []
        y_points = []
        for migrator_index, by_smbh in enumerate(grid):
            if migrator_index == center_index:
                continue
            for smbh_index, radii_rg in enumerate(by_smbh):
                radii_rg = _positive_sorted(radii_rg)
                x_points.extend([smbh_masses_msun[smbh_index]] * radii_rg.size)
                y_points.extend(radii_rg.tolist())
        if x_points:
            ax.scatter(
                np.asarray(x_points, dtype=float),
                np.asarray(y_points, dtype=float),
                color=color,
                alpha=0.20,
                s=18.0,
                marker=marker,
                linewidths=0.8,
                label=label,
            )
        if grid is trap_radius_rg:
            stable_sample_points = len(x_points)
        else:
            anti_sample_points = len(x_points)

    stable_center_points = 0
    anti_center_points = 0
    for grid, color, marker, linestyle, label in (
        (
            trap_radius_rg,
            "#1f77b4",
            "o",
            "-",
            rf"Stable trap, $m={center_mass_msun:g}\,M_\odot$",
        ),
        (
            anti_trap_radius_rg,
            "#7A3DB8",
            "x",
            "-.",
            rf"Anti-trap, $m={center_mass_msun:g}\,M_\odot$",
        ),
    ):
        center_radii = grid[center_index]
        max_branch_count = max((radii_rg.size for radii_rg in center_radii), default=0)
        plotted_points = 0
        label_used = False
        for branch_index in range(max_branch_count):
            y_values = np.full_like(smbh_masses_msun, np.nan, dtype=float)
            for smbh_index, radii_rg in enumerate(center_radii):
                radii_rg = _positive_sorted(radii_rg)
                if branch_index < radii_rg.size:
                    y_values[smbh_index] = radii_rg[branch_index]
            mask = np.isfinite(y_values) & (y_values > 0.0)
            if not np.any(mask):
                continue
            plotted_points += int(np.count_nonzero(mask))
            ax.plot(
                smbh_masses_msun[mask],
                y_values[mask],
                color=color,
                lw=1.8,
                ls=linestyle,
                marker=marker,
                ms=5.0,
                label=label if not label_used else None,
            )
            label_used = True
        if grid is trap_radius_rg:
            stable_center_points = plotted_points
        else:
            anti_center_points = plotted_points

    lisa_color = "#D55E00"
    lisa_band = (
        np.isfinite(lisa_lower)
        & np.isfinite(lisa_upper)
        & (lisa_lower > 0.0)
        & (lisa_upper > 0.0)
    )
    if np.any(lisa_band):
        ax.fill_between(
            smbh_masses_msun,
            lisa_lower,
            lisa_upper,
            where=lisa_band.tolist(),
            color=lisa_color,
            alpha=0.16,
            linewidth=0.0,
        )
    lisa_mask = np.isfinite(lisa_center) & (lisa_center > 0.0)
    if np.any(lisa_mask):
        ax.plot(
            smbh_masses_msun[lisa_mask],
            lisa_center[lisa_mask],
            color=lisa_color,
            lw=2.8,
            ls="--",
            label=rf"LISA entry, $m={center_mass_msun:g}\,M_\odot$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_\bullet\;(M_\odot)$")
    ax.set_ylabel(r"Radius $R/r_g$")
    ax.set_ylim(float(positive_radii[0]) / 1.6, float(positive_radii[-1]) * 1.6)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="best", framealpha=0.88)
    ax.set_title(rf"Migration traps and LISA entry radii, $z={args.redshift:g}$")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{OUTPUT_STEM}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    stable_point_count = sum(
        radii_rg.size for by_mass in trap_radius_rg for radii_rg in by_mass
    )
    anti_point_count = sum(
        radii_rg.size for by_mass in anti_trap_radius_rg for radii_rg in by_mass
    )
    total_cells = int(migrator_masses_msun.size * smbh_masses_msun.size)
    print("Trap radius 2D diagnostic")
    print(
        "  smbh_mass_range_msun: "
        f"{smbh_masses_msun[0]:.6e} to {smbh_masses_msun[-1]:.6e} "
        f"({smbh_masses_msun.size} points)"
    )
    print(
        "  migrator_masses_msun: "
        + ", ".join(f"{mass:.6g}" for mass in migrator_masses_msun)
    )
    print(f"  center_migrator_mass_msun: {center_mass_msun:.6g}")
    print(f"  stable_trap_points: {stable_point_count}")
    print(f"  anti_trap_points: {anti_point_count}")
    print(f"  stable_sample_points: {stable_sample_points}")
    print(f"  anti_sample_points: {anti_sample_points}")
    print(f"  stable_center_points: {stable_center_points}")
    print(f"  anti_center_points: {anti_center_points}")
    print(f"  failed_torque_cells: {failed_torque_cells}/{total_cells}")
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    main()
