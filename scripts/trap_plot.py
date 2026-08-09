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
    DEFAULT_MIN_TRAP_SEPARATION_CELLS,
    MigrationParameters,
    find_traps_locations,
    compute_torque_profile,
)
from src.disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from src.physical_process import constants as const
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
        description=(
            "Plot pAGN stable traps, anti-traps, and LISA entry radii versus "
            "SMBH mass, using migrator mass samples for trap families and a "
        )
    )
    parser.add_argument("--smbh-min-msun", type=float, default=DEFAULT_SMBH_MIN_MSUN)
    parser.add_argument("--smbh-max-msun", type=float, default=DEFAULT_SMBH_MAX_MSUN)
    parser.add_argument("--smbh-count", type=int, default=DEFAULT_SMBH_COUNT)
    parser.add_argument(
        "--migrator-masses-msun",
        type=_parse_mass_vector,
        default=list(DEFAULT_MIGRATOR_MASSES_MSUN),
        help="comma-separated migrator masses used for the shaded bands",
    )
    parser.add_argument(
        "--center-migrator-mass-msun",
        type=float,
        default=DEFAULT_CENTER_MIGRATOR_MASS_MSUN,
        help="migrator mass used for the central trap markers and LISA curve",
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
        "--min-trap-separation-cells",
        type=float,
        default=DEFAULT_MIN_TRAP_SEPARATION_CELLS,
        help=(
            "discard adjacent stable/anti zero crossings closer than this many "
            "local radial grid intervals"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the output figure",
    )
    return parser.parse_args()


def _log_grid(min_value: float, max_value: float, count: int) -> np.ndarray:
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("SMBH mass grid bounds must be positive")
    if max_value <= min_value:
        raise ValueError("SMBH mass grid maximum must exceed minimum")
    if count < 2:
        raise ValueError("SMBH mass grid count must be at least 2")
    return np.logspace(np.log10(min_value), np.log10(max_value), int(count))


def _mass_samples_with_center(
    masses_msun: list[float],
    center_mass_msun: float,
) -> np.ndarray:
    if center_mass_msun <= 0.0:
        raise ValueError("center migrator mass must be positive")
    values = np.asarray([*masses_msun, center_mass_msun], dtype=float)
    return np.unique(values)


def _center_index(masses_msun: np.ndarray, center_mass_msun: float) -> int:
    return int(np.argmin(np.abs(masses_msun - center_mass_msun)))


def _empty_trap_grid(
    migrator_count: int,
    smbh_count: int,
) -> list[list[np.ndarray]]:
    return [
        [np.array([], dtype=float) for _ in range(smbh_count)]
        for _ in range(migrator_count)
    ]


def _positive_sorted(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    return np.sort(values)


def _flatten_trap_grid(trap_grid: list[list[np.ndarray]]) -> np.ndarray:
    values = [
        trap_values
        for by_mass in trap_grid
        for trap_values in by_mass
        if trap_values.size > 0
    ]
    if not values:
        return np.array([], dtype=float)
    return _positive_sorted(np.concatenate(values))


def _finite_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(values.shape[1], np.nan, dtype=float)
    upper = np.full(values.shape[1], np.nan, dtype=float)
    for index in range(values.shape[1]):
        finite = _positive_sorted(values[:, index])
        if finite.size == 0:
            continue
        lower[index] = float(finite[0])
        upper[index] = float(finite[-1])
    return lower, upper


# def _nearest_cross_family_separation(
#     stable_grid: list[list[np.ndarray]],
#     anti_grid: list[list[np.ndarray]],
#     smbh_masses_msun: np.ndarray,
#     migrator_masses_msun: np.ndarray,
# ) -> tuple[float, float, float, float, float, float] | None:
#     best: tuple[float, float, float, float, float, float] | None = None
#     for migrator_index, migrator_mass_msun in enumerate(migrator_masses_msun):
#         for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
#             stable_values = _positive_sorted(stable_grid[migrator_index][smbh_index])
#             anti_values = _positive_sorted(anti_grid[migrator_index][smbh_index])
#             if stable_values.size == 0 or anti_values.size == 0:
#                 continue
#             separation = np.abs(stable_values[:, None] - anti_values[None, :])
#             stable_index, anti_index = np.unravel_index(
#                 int(np.argmin(separation)),
#                 separation.shape,
#             )
#             separation_rg = float(separation[stable_index, anti_index])
#             stable_radius_rg = float(stable_values[stable_index])
#             anti_radius_rg = float(anti_values[anti_index])
#             fractional_separation = separation_rg / min(
#                 stable_radius_rg,
#                 anti_radius_rg,
#             )
#             candidate = (
#                 separation_rg,
#                 fractional_separation,
#                 float(smbh_mass_msun),
#                 float(migrator_mass_msun),
#                 stable_radius_rg,
#                 anti_radius_rg,
#             )
#             if best is None or candidate[0] < best[0]:
#                 best = candidate
#     return best


def _positive_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(array, dtype=float).ravel() for array in arrays]
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise RuntimeError("no positive finite radii were computed")
    return float(np.min(values)), float(np.max(values))


def _plot_band(
    ax,
    x_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> None:
    band_mask = np.isfinite(lower) & np.isfinite(upper) & (lower > 0.0) & (upper > 0.0)
    if not np.any(band_mask):
        return
    ax.fill_between(
        x_values,
        lower,
        upper,
        where=band_mask,
        color=color,
        alpha=alpha,
        linewidth=0.0,
    )


def _plot_lisa_entry(
    ax,
    x_values: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    center_mass_msun: float,
) -> None:
    color = "#D55E00"
    _plot_band(ax, x_values, lower, upper, color=color, alpha=0.16)
    # print(f"lower is {lower}, upper is {upper}")
    center_mask = np.isfinite(center) & (center > 0.0)
    if np.any(center_mask):
        ax.plot(
            x_values[center_mask],
            center[center_mask],
            color=color,
            lw=2.8,
            ls="--",
            label=rf"LISA entry, $m={center_mass_msun:g}\,M_\odot$",
        )


def _plot_trap_family(
    ax,
    x_values: np.ndarray,
    trap_grid: list[list[np.ndarray]],
    center_index: int,
    *,
    color: str,
    label: str,
    marker: str,
    linestyle: str,
) -> int:
    center_traps = trap_grid[center_index]
    max_count = max((trap_values.size for trap_values in center_traps), default=0)
    plotted_points = 0
    label_used = False
    for trap_rank in range(max_count):
        y_values = np.full_like(x_values, np.nan, dtype=float)
        for smbh_index, trap_values in enumerate(center_traps):
            sorted_values = _positive_sorted(trap_values)
            if trap_rank < sorted_values.size:
                y_values[smbh_index] = sorted_values[trap_rank]
        mask = np.isfinite(y_values) & (y_values > 0.0)
        if not np.any(mask):
            continue
        plotted_points += int(np.count_nonzero(mask))
        ax.plot(
            x_values,
            y_values,
            color=color,
            lw=1.8,
            ls=linestyle,
            marker=marker,
            ms=5.0,
            label=label if not label_used else None,
        )
        label_used = True
    return plotted_points


def _plot_trap_mass_samples(
    ax,
    x_values: np.ndarray,
    trap_grid: list[list[np.ndarray]],
    center_index: int,
    *,
    color: str,
    label: str,
    marker: str,
) -> int:
    x_points = []
    y_points = []
    for migrator_index, by_smbh in enumerate(trap_grid):
        if migrator_index == center_index:
            continue
        for smbh_index, trap_values in enumerate(by_smbh):
            sorted_values = _positive_sorted(trap_values)
            if sorted_values.size == 0:
                continue
            x_points.extend([x_values[smbh_index]] * int(sorted_values.size))
            y_points.extend(sorted_values.tolist())

    if not x_points:
        return 0

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
    return len(x_points)


def main() -> None:
    args = _parse_args()
    smbh_masses_msun = _log_grid(
        args.smbh_min_msun,
        args.smbh_max_msun,
        args.smbh_count,
    )
    migrator_masses_msun = _mass_samples_with_center(
        list(args.migrator_masses_msun),
        float(args.center_migrator_mass_msun),
    )
    center_index = _center_index(
        migrator_masses_msun,
        float(args.center_migrator_mass_msun),
    )
    center_mass_msun = float(migrator_masses_msun[center_index])

    trap_radius_rg = _empty_trap_grid(
        migrator_masses_msun.size,
        smbh_masses_msun.size,
    )
    anti_trap_radius_rg = _empty_trap_grid(
        migrator_masses_msun.size,
        smbh_masses_msun.size,
    )

    lisa_entry_radius_rg = np.full(
        (migrator_masses_msun.size, smbh_masses_msun.size),
        np.nan,
        dtype=float,
    )
    failed_torque_cells = 0

    params = MigrationParameters(alpha=PAGN_ALPHA, include_type_ii=True)

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        rg_cm = const.G * smbh_mass_msun * const.msun / const.c**2
        disk_profile = build_pagn_disk_profile(
            float(smbh_mass_msun),
            n_resolution=args.n_resolution,
            accretion_rate_g_s=(
                DISK_ACCRETION_RATE_G_S_PER_MSUN * float(smbh_mass_msun)
            ),
            alpha=PAGN_ALPHA,
            eddington_ratio=0.1,
        )

        for migrator_index, migrator_mass_msun in enumerate(migrator_masses_msun):
            entry_radius_cm = lisa_entry_radius_cm(
                float(smbh_mass_msun),
                float(migrator_mass_msun),
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
                    migrator_mass_msun=float(migrator_mass_msun),
                    smoothing=args.smoothing,
                    params=params,
                )
            except Exception:
                failed_torque_cells += 1
                continue

            trap_radius_cm, anti_trap_radius_cm = find_traps_locations(
                torque_profile,
            )

            trap_radius_rg[migrator_index][smbh_index] = _positive_sorted(
                trap_radius_cm / rg_cm
            )

            anti_trap_radius_rg[migrator_index][smbh_index] = _positive_sorted(
                anti_trap_radius_cm / rg_cm
            )

    lisa_lower, lisa_upper = _finite_band(lisa_entry_radius_rg)
    lisa_center = lisa_entry_radius_rg[center_index]

    y_min, y_max = _positive_limits(
        _flatten_trap_grid(trap_radius_rg),
        _flatten_trap_grid(anti_trap_radius_rg),
        lisa_lower,
        lisa_upper,
        lisa_center,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    stable_sample_points = _plot_trap_mass_samples(
        ax,
        smbh_masses_msun,
        trap_radius_rg,
        center_index,
        color="#1f77b4",
        label=r"Stable trap samples, $m=1$-$30\,M_\odot$",
        marker="o",
    )
    anti_sample_points = _plot_trap_mass_samples(
        ax,
        smbh_masses_msun,
        anti_trap_radius_rg,
        center_index,
        color="#7A3DB8",
        label=r"Anti-trap samples, $m=1$-$30\,M_\odot$",
        marker="x",
    )

    stable_center_points = _plot_trap_family(
        ax,
        smbh_masses_msun,
        trap_radius_rg,
        center_index,
        color="#1f77b4",
        label=rf"Stable trap, $m={center_mass_msun:g}\,M_\odot$",
        marker="o",
        linestyle="-",
    )

    anti_center_points = _plot_trap_family(
        ax,
        smbh_masses_msun,
        anti_trap_radius_rg,
        center_index,
        color="#7A3DB8",
        label=rf"Anti-trap, $m={center_mass_msun:g}\,M_\odot$",
        marker="x",
        linestyle="-.",
    )

    print(f"trap radius is {trap_radius_rg[10]}")
    print(f"anti-trap radius is {anti_trap_radius_rg[10]}")

    _plot_lisa_entry(
        ax,
        smbh_masses_msun,
        lisa_center,
        lisa_lower,
        lisa_upper,
        center_mass_msun=center_mass_msun,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_\bullet\;(M_\odot)$")
    ax.set_ylabel(r"Radius $R/r_g$")
    ax.set_ylim(y_min / 1.6, y_max * 1.6)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="best", framealpha=0.88)
    ax.set_title(rf"Migration traps and LISA entry radii, $z={args.redshift:g}$")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{OUTPUT_STEM}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    stable_point_count = sum(
        trap_values.size for by_mass in trap_radius_rg for trap_values in by_mass
    )
    anti_point_count = sum(
        trap_values.size for by_mass in anti_trap_radius_rg for trap_values in by_mass
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
    print(f"  min_trap_separation_cells: {args.min_trap_separation_cells:.6g}")
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
