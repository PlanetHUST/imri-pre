"""AGN disk and migration-torque diagnostic figures.

Consumes canonical physics only: the pAGN disk structure
(:mod:`disk_quantity.pagn_disk_adapter`), the in-repo comparison solver
(:mod:`disk_quantity.disk_solve`) and the migration package
(:mod:`physical_process.migration`).  This module defines no prescription,
threshold or criterion of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import physical_process.constants as const  # noqa: E402
from disk_quantity.disk_solve import radial_structure  # noqa: E402
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile  # noqa: E402
from physical_process.migration import (  # noqa: E402
    DEFAULT_GAP_CRITERION,
    MigrationParameters,
    classify_torque_traps,
    compute_torque_profile,
    select_migration_torque_si,
)

RADIAL_AXIS_LABELSIZE = 26
LEGEND_FONTSIZE = 18
LEGACY_DISK_MDOT_PER_SOLAR_MASS = 1.44e18
PAGN_DISK_MODEL = "sirko"
PAGN_EDDINGTON_RATIO = 0.1
PAGN_B = 0
PAGN_PROFILE_N_RESOLUTION = 1000
PAGN_TORQUE_N_RESOLUTION = 30000
TORQUE_MAX_RADIUS_RG = 1.0e7

TORQUE_SMOOTHING = 1.0
TORQUE_PLOT_LOG_SMOOTHING = 0.5
TORQUE_PLOT_MIN_SMOOTH_POINTS = 20
TORQUE_PLOT_MIN_BRANCH_WIDTH_RG = 100.0
TORQUE_PLOT_MIN_BRANCH_FRACTION = 0.001
PLOT_STYLE = {"xtick.labelsize": 20, "ytick.labelsize": 20}


COLORS_SERIS = ["#1f77b4", "#BE9A62", "#2A7F62", "#2B56a3", "#A89A11", "#3A7F11"]


def _save(fig: Figure, outdir: str | Path, stem: str, dpi: int = 300) -> None:
    """Save each figure in both PNG and PDF form for quick view and publication."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


def _disk_profile_payload(
    smbh_mass: float,
    alpha: float,
) -> Dict[str, np.ndarray | float | str]:

    disk_profile = build_pagn_disk_profile(
        smbh_mass,
        alpha=alpha,
        model=PAGN_DISK_MODEL,
        n_resolution=PAGN_PROFILE_N_RESOLUTION,
        eddington_ratio=PAGN_EDDINGTON_RATIO,
        b=PAGN_B,
        suppress_pagn_output=True,
    )

    r_over_rg = (
        np.asarray(disk_profile.radius_cm, dtype=float)
        / disk_profile.gravitational_radius_cm
    )

    q_values = np.asarray(disk_profile.toomre_q, dtype=float)
    r_out_over_rg = float(
        disk_profile.outer_radius_cm / disk_profile.gravitational_radius_cm
    )

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": disk_profile.model_name,
        "r_over_rg": r_over_rg,
        "r_out_over_rg": r_out_over_rg,
        "tc": np.asarray(disk_profile.temperature_k, dtype=float),
        "q": q_values,
        "sigma": np.asarray(disk_profile.surface_density_g_cm2, dtype=float),
        "tau": np.asarray(disk_profile.optical_depth, dtype=float),
        "h_over_r": np.asarray(disk_profile.aspect_ratio, dtype=float),
        "cs": np.asarray(disk_profile.sound_speed_cm_s, dtype=float),
        "rho": np.asarray(disk_profile.density_g_cm3, dtype=float),
    }


def _disk_solver_profile_payload(
    smbh_mass: float,
) -> Dict[str, np.ndarray | float | str]:
    """Compute the in-repo disk_solve.py profiles for one SMBH mass."""
    accretion_rate_g_s = float(smbh_mass) * LEGACY_DISK_MDOT_PER_SOLAR_MASS
    (
        outer_radius_cm,
        radius_cm,
        temperature_k,
        toomre_q,
        surface_density_g_cm2,
        optical_depth,
        aspect_ratio,
        sound_speed_cm_s,
        density_g_cm3,
    ) = radial_structure(float(smbh_mass), accretion_rate_g_s)
    rg_cm = const.G * float(smbh_mass) * const.msun / const.c**2

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": "disk_solve",
        "r_over_rg": np.asarray(radius_cm, dtype=float) / rg_cm,
        "r_out_over_rg": float(outer_radius_cm / rg_cm),
        "tc": np.asarray(temperature_k, dtype=float),
        "q": np.asarray(toomre_q, dtype=float),
        "sigma": np.asarray(surface_density_g_cm2, dtype=float),
        "tau": np.asarray(optical_depth, dtype=float),
        "h_over_r": np.asarray(aspect_ratio, dtype=float),
        "cs": np.asarray(sound_speed_cm_s, dtype=float),
        "rho": np.asarray(density_g_cm3, dtype=float),
    }


def _disk_torque_payload(
    smbh_mass: float,
    alpha: float,
    migrator_mass_msun: float = 10.0,
    min_trap_separation_cells: float = 1.0,
    gap_opening_condition: str = DEFAULT_GAP_CRITERION,
) -> Dict[str, np.ndarray | float | str]:
    """Compute a pAGN torque profile using the main simulation torque model."""
    disk_profile = build_pagn_disk_profile(
        smbh_mass,
        alpha=alpha,
        model=PAGN_DISK_MODEL,
        n_resolution=PAGN_TORQUE_N_RESOLUTION,
        eddington_ratio=PAGN_EDDINGTON_RATIO,
        b=PAGN_B,
        suppress_pagn_output=True,
    )
    params = MigrationParameters(
        alpha=alpha,
        gamma_adiabat=5.0 / 3.0,
        hydrogen_fraction=0.7,
        kappa_t=20.0,
        kappa_norm=25.0,
    )

    torque_profile = compute_torque_profile(
        disk_profile.disk,
        migrator_mass_msun=migrator_mass_msun,
        max_radius_rg=TORQUE_MAX_RADIUS_RG,
        smoothing=TORQUE_SMOOTHING,
        params=params,
    )

    solved_disk_radius_m = np.asarray(
        getattr(disk_profile.disk, "R"),
        dtype=float,
    )
    solved_disk_scale_height_m = np.asarray(
        getattr(disk_profile.disk, "h"),
        dtype=float,
    )
    solved_disk_aspect_ratio = solved_disk_scale_height_m / solved_disk_radius_m
    solved_order = np.argsort(solved_disk_radius_m)
    local_aspect_ratio = np.interp(
        np.asarray(torque_profile.radius_m, dtype=float),
        solved_disk_radius_m[solved_order],
        solved_disk_aspect_ratio[solved_order],
    )
    migration_torque_si = select_migration_torque_si(
        torque_profile,
        float(migrator_mass_msun) / float(smbh_mass),
        local_aspect_ratio,
        alpha,
        gap_opening_condition,
        consider_type_ii_migration=True,
    ).torque_si
    trap_radii = classify_torque_traps(
        torque_profile.radius_rg,
        migration_torque_si,
        min_separation_cells=min_trap_separation_cells,
    )

    torque_radius_over_rg = np.asarray(torque_profile.radius_rg, dtype=float)

    r_out_over_rg = float(
        disk_profile.outer_radius_cm / disk_profile.gravitational_radius_cm
    )

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": disk_profile.model_name,
        "r_over_rg": torque_radius_over_rg,
        "r_out_over_rg": r_out_over_rg,
        "gamma_i_si": np.asarray(torque_profile.gamma_type_i_si, dtype=float),
        "gamma_th_si": np.asarray(torque_profile.gamma_th_si, dtype=float),
        "gamma_migration_si": migration_torque_si,
        "gamma_ii_si": np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
        "trap_radii_over_rg": np.asarray(trap_radii.stable_radius_rg, dtype=float),
        "anti_trap_radii_over_rg": np.asarray(
            trap_radii.anti_trap_radius_rg,
            dtype=float,
        ),
    }


def _smooth_torque_magnitude_for_plot(
    radius_rg: np.ndarray,
    torque_magnitude: np.ndarray,
) -> np.ndarray:
    """Smooth visual torque magnitude in log-log space."""
    if radius_rg.size < TORQUE_PLOT_MIN_SMOOTH_POINTS:
        return torque_magnitude

    from scipy.interpolate import UnivariateSpline

    log_radius = np.log10(radius_rg)
    log_torque = np.log10(torque_magnitude)
    spline = UnivariateSpline(
        log_radius,
        log_torque,
        k=3,
        s=TORQUE_PLOT_LOG_SMOOTHING,
        ext=0,
    )
    return 10.0 ** np.asarray(spline(log_radius), dtype=float)


def _torque_magnitude_plot_values(
    radius_rg: np.ndarray,
    torque_magnitude: np.ndarray,
    branch_mask: np.ndarray,
) -> np.ndarray:

    plot_values = np.full_like(torque_magnitude, np.nan, dtype=float)
    branch_indices = np.flatnonzero(branch_mask)
    # split the branch based on it's diff
    split_indices = np.where(np.diff(branch_indices) > 1)[0] + 1
    for indices in np.split(branch_indices, split_indices):
        if indices.size == 0:
            continue
        branch_radius_rg = radius_rg[indices]
        branch_width_rg = branch_radius_rg[-1] - branch_radius_rg[0]
        local_width_rg = TORQUE_PLOT_MIN_BRANCH_FRACTION * np.sqrt(
            branch_radius_rg[0] * branch_radius_rg[-1]
        )
        min_branch_width_rg = max(TORQUE_PLOT_MIN_BRANCH_WIDTH_RG, local_width_rg)
        if branch_width_rg < min_branch_width_rg:
            continue
        edge_width_rg = 0.5 * min_branch_width_rg

        # select keep values
        keep = np.ones(indices.size, dtype=bool)
        if indices[0] > 0:
            keep &= branch_radius_rg - branch_radius_rg[0] >= edge_width_rg
        if indices[-1] < torque_magnitude.size - 1:
            keep &= branch_radius_rg[-1] - branch_radius_rg >= edge_width_rg

        plot_indices = indices[keep]
        if plot_indices.size == 0:
            continue

        plot_values[plot_indices] = _smooth_torque_magnitude_for_plot(
            radius_rg[plot_indices],
            torque_magnitude[plot_indices],
        )
    return plot_values


@mpl.rc_context(PLOT_STYLE)
def plot_agn_disk_parameter_profiles(
    output_dir: str,
    alpha: float,
    smbh_masses: Sequence[float],
) -> None:
    """Plot useful pAGN AGN disk parameters versus radius."""
    disk_profiles = [
        _disk_profile_payload(
            float(mass),
            alpha,
        )
        for mass in smbh_masses
    ]

    parameters = [
        # ("tc", r"$T_{\mathrm{c}}\;(\mathrm{K})$"),
        ("sigma", r"$\Sigma\;(\mathrm{g\,cm^{-2}})$"),
        # ("rho", r"$\rho\;(\mathrm{g\,cm^{-3}})$"),
        # ("q", r"$Q$"),
        # ("tau", r"$\tau$"),
        ("h_over_r", r"$H/R$"),
        # ("cs", r"$c_{\mathrm{s}}\;(\mathrm{cm\,s^{-1}})$"),
    ]

    colors = COLORS_SERIS

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (key, ylabel) in zip(axes, parameters):
        for profile, color in zip(disk_profiles, colors):
            radius = np.asarray(profile["r_over_rg"], dtype=float)
            values = np.asarray(profile[key], dtype=float)
            mask = (
                np.isfinite(radius) & np.isfinite(values) & (radius > 0) & (values > 0)
            )
            if not np.any(mask):
                continue

            mass_label = (
                rf"$M_\bullet=10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$"
            )
            ax.plot(
                radius[mask],
                values[mask],
                color=color,
                lw=3.8,
                label=mass_label,
                alpha=0.8,
            )
            ax.axvline(
                profile["r_out_over_rg"],
                color=color,
                lw=1.8,
                ls=":",
                alpha=0.75,
            )

        ax.set(xscale="log", yscale="log", xlim=(None, 1.0e7))
        ax.set_ylabel(ylabel, fontsize=RADIAL_AXIS_LABELSIZE)
        ax.grid(True, alpha=0.25, linestyle="--", which="both")

    for ax in axes:
        ax.set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)

    axes[0].legend(loc="best", framealpha=0.8, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, output_dir, "disk_parameter_profiles")
    plt.close(fig)


@mpl.rc_context(PLOT_STYLE)
def plot_agn_disk_method_comparison(
    output_dir: str,
    alpha: float,
    smbh_masses: Sequence[float],
) -> None:
    """Compare pAGN and disk_solve.py AGN disk profiles for the same inputs."""
    pagn_profiles = [
        _disk_profile_payload(
            float(mass),
            alpha=alpha,
        )
        for mass in smbh_masses
    ]
    solved_profiles = [
        _disk_solver_profile_payload(float(mass)) for mass in smbh_masses
    ]

    parameters = [
        ("sigma", r"$\Sigma\;(\mathrm{g\,cm^{-2}})$"),
        ("h_over_r", r"$H/R$"),
        ("q", r"$Q$"),
    ]
    colors = COLORS_SERIS

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (key, ylabel) in zip(axes, parameters):
        for pagn_profile, solved_profile, color in zip(
            pagn_profiles, solved_profiles, colors
        ):
            mass_label = (
                rf"$M_\bullet=10^{{{int(np.log10(pagn_profile['smbh_mass']))}}}"
                rf"\,M_\odot$"
            )
            for profile, linestyle, alpha_value in (
                (pagn_profile, "-", 0.9),
                (solved_profile, "--", 0.85),
            ):
                radius = np.asarray(profile["r_over_rg"], dtype=float)
                values = np.asarray(profile[key], dtype=float)
                mask = (
                    np.isfinite(radius)
                    & np.isfinite(values)
                    & (radius > 0)
                    & (values > 0)
                )
                if not np.any(mask):
                    continue
                ax.plot(
                    radius[mask],
                    values[mask],
                    color=color,
                    lw=3.2,
                    ls=linestyle,
                    alpha=alpha_value,
                    label=mass_label if linestyle == "-" else None,
                )

            for profile, linestyle in (
                (pagn_profile, ":"),
                (solved_profile, "-."),
            ):
                r_out = float(profile["r_out_over_rg"])
                if np.isfinite(r_out) and r_out > 0:
                    ax.axvline(r_out, color=color, lw=1.7, ls=linestyle, alpha=0.55)

        ax.set(xscale="log", yscale="log", xlim=(None, 1.0e7))
        ax.set_ylabel(ylabel, fontsize=RADIAL_AXIS_LABELSIZE)
        ax.grid(True, alpha=0.25, linestyle="--", which="both")

    axes[-1].set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)

    mass_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=3.2,
            label=rf"$M_\bullet=10^{{{int(np.log10(mass))}}}\,M_\odot$",
        )
        for mass, color in zip(smbh_masses, colors)
    ]
    method_handles = [
        Line2D([0], [0], color="0.2", lw=3.2, ls="-", label="pAGN"),
        Line2D([0], [0], color="0.2", lw=3.2, ls="--", label="disk_solve.py"),
        Line2D([0], [0], color="0.2", lw=1.7, ls=":", label="pAGN outer radius"),
        Line2D(
            [0],
            [0],
            color="0.2",
            lw=1.7,
            ls="-.",
            label="disk_solve.py outer radius",
        ),
    ]
    axes[0].legend(
        handles=mass_handles + method_handles,
        loc="best",
        framealpha=0.85,
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    _save(fig, output_dir, "disk_method_comparison")
    plt.close(fig)


def _plot_signed_torque_component(
    torque_profiles: Sequence[Mapping[str, np.ndarray | float | str]],
    colors: Sequence[str],
    *,
    output_dir: str | Path,
    component_key: str,
    ylabel: str,
    stem: str,
    show_traps: bool,
) -> None:

    fig, ax = plt.subplots(figsize=(14, 8))
    mass_handles = []
    has_inward = False
    has_outward = False
    has_traps = False
    has_anti_traps = False

    for torque_profile, color in zip(torque_profiles, colors):
        radius_rg = np.asarray(torque_profile["r_over_rg"], dtype=float)
        torque_values = np.asarray(torque_profile[component_key], dtype=float)

        # safe check
        valid = np.isfinite(radius_rg) & np.isfinite(torque_values) & (radius_rg > 0.0)
        if not np.any(valid):
            continue

        radius_rg = radius_rg[valid]
        torque_values = torque_values[valid]

        # radius sort
        radius_order = np.argsort(radius_rg)
        radius_rg = radius_rg[radius_order]
        torque_values = torque_values[radius_order]
        torque_magnitude = np.abs(torque_values)

        outward_mask = torque_values > 0.0
        inward_mask = torque_values < 0.0

        if np.any(outward_mask):
            outward_values = _torque_magnitude_plot_values(
                radius_rg,
                torque_magnitude,
                outward_mask,
            )

            ax.plot(
                radius_rg,
                outward_values,
                color=color,
                lw=3.2,
                ls="-.",
                alpha=0.9,
            )
            has_outward = True

        if np.any(inward_mask):
            inward_values = _torque_magnitude_plot_values(
                radius_rg,
                torque_magnitude,
                inward_mask,
            )

            ax.plot(
                radius_rg,
                inward_values,
                color=color,
                lw=3.2,
                ls="-",
                alpha=0.9,
            )
            has_inward = True

        mass_label = (
            rf"$M_\bullet=10^{{{int(np.log10(torque_profile['smbh_mass']))}}}\,M_\odot$"
        )
        mass_handles.append(Line2D([0], [0], color=color, lw=3.2, label=mass_label))

        r_out_over_rg = float(torque_profile["r_out_over_rg"])

        if np.isfinite(r_out_over_rg) and r_out_over_rg > 0.0:
            ax.axvline(r_out_over_rg, color=color, lw=3.0, ls=":", alpha=0.8)

        if show_traps:
            trap_radii_rg = np.asarray(
                torque_profile["trap_radii_over_rg"], dtype=float
            )
            anti_trap_radii_rg = np.asarray(
                torque_profile["anti_trap_radii_over_rg"], dtype=float
            )
            for radius in trap_radii_rg[
                np.isfinite(trap_radii_rg) & (trap_radii_rg > 0.0)
            ]:
                ax.axvline(radius, color=color, lw=2.5, ls="--", alpha=0.45)
                has_traps = True

            for radius in anti_trap_radii_rg[
                np.isfinite(anti_trap_radii_rg) & (anti_trap_radii_rg > 0.0)
            ]:
                ax.axvline(
                    radius,
                    color=color,
                    lw=2.5,
                    ls=(0, (4, 1, 1, 1)),
                    alpha=0.45,
                )
                has_anti_traps = True

    ax.set(xscale="log", yscale="log", xlim=(None, 1.0e7))
    ax.set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=RADIAL_AXIS_LABELSIZE)
    ax.grid(True, alpha=0.25, linestyle="--", which="both")

    plot_handles = []
    if has_inward:
        plot_handles.append(
            Line2D([0], [0], color="0.2", lw=3.2, ls="-", label="Inward migration")
        )
    if has_outward:
        plot_handles.append(
            Line2D([0], [0], color="0.2", lw=3.2, ls="-.", label="Outward migration")
        )

    if has_traps:
        plot_handles.append(
            Line2D([0], [0], color="0.2", lw=1.4, ls="--", label="Stable trap")
        )

    if has_anti_traps:
        plot_handles.append(
            Line2D(
                [0],
                [0],
                color="0.2",
                lw=1.4,
                ls=(0, (4, 1, 1, 1)),
                label="Anti-trap",
            )
        )

    ax.legend(
        handles=mass_handles + plot_handles,
        loc="best",
        framealpha=0.85,
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    _save(fig, output_dir, stem)
    plt.close(fig)


@mpl.rc_context(PLOT_STYLE)
def plot_agn_disk_torque_profiles(
    output_dir: str,
    alpha: float,
    smbh_masses: Sequence[float],
    migrator_mass_msun: float = 10.0,
    min_trap_separation_cells: float = 1.0,
    gap_opening_condition: str = DEFAULT_GAP_CRITERION,
) -> None:
    """Plot pAGN migration torque magnitudes using the main simulation model."""

    torque_profiles = [
        _disk_torque_payload(
            float(smbh_mass),
            alpha=alpha,
            migrator_mass_msun=migrator_mass_msun,
            min_trap_separation_cells=min_trap_separation_cells,
            gap_opening_condition=gap_opening_condition,
        )
        for smbh_mass in smbh_masses
    ]

    colors = COLORS_SERIS

    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_migration_si",
        ylabel=r"$|\Gamma_{\mathrm{migration}}|\;(\mathrm{N\,m})$",
        stem="tot_torque_profiles",
        show_traps=True,
    )
    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_ii_si",
        ylabel=r"$|\Gamma_{\mathrm{II}}|\;(\mathrm{N\,m})$",
        stem="type_ii_torque_profiles",
        show_traps=True,
    )
    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_i_si",
        ylabel=r"$|\Gamma_{\mathrm{I}}|\;(\mathrm{N\,m})$",
        stem="type_i_torque_profiles",
        show_traps=True,
    )
    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_th_si",
        ylabel=r"$|\Gamma_{\mathrm{th}}|\;(\mathrm{N\,m})$",
        stem="thermal_torque_profiles",
        show_traps=True,
    )
