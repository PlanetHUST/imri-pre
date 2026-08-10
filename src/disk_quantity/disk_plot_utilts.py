from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import src.physical_process.constants as const
from src.disk_quantity.migration_torque import (
    MigrationParameters,
    compute_torque_profile,
    find_traps_locations,
)
from src.disk_quantity.disk_solve import radial_structure
from src.disk_quantity.pagn_disk_adapter import build_pagn_disk_profile

RADIAL_AXIS_LABELSIZE = 26
LEGEND_FONTSIZE = 18
LEGACY_DISK_MDOT_PER_SOLAR_MASS = 1.44e18
PAGN_DISK_MODEL = "sirko"
PAGN_EDDINGTON_RATIO = 0.1
PAGN_B = 0
PAGN_PROFILE_N_RESOLUTION = 1000
PAGN_TORQUE_N_RESOLUTION = 30000
TORQUE_MAX_RADIUS_RG = 1.0e7
TORQUE_SMOOTHING = 0.005

plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20


COLORS_SERIS = ["#1f77b4", "#BE9A62", "#2A7F62", "#2f66a3", "#A89A11", "#3A7F11"]


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

    r_out_over_rg = Toomre_param_outer_radius(
        r_over_rg,
        q_values,
        fallback=float(
            disk_profile.outer_radius_cm / disk_profile.gravitational_radius_cm
        ),
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
    trap_radii_cm, anti_trap_radii_cm = find_traps_locations(torque_profile)

    torque_radius_over_rg = np.asarray(torque_profile.radius_rg, dtype=float)

    radius_over_rg = (
        np.asarray(disk_profile.radius_cm, dtype=float)
        / disk_profile.gravitational_radius_cm
    )

    r_out_over_rg = Toomre_param_outer_radius(
        radius_over_rg,
        np.asarray(disk_profile.toomre_q, dtype=float),
        fallback=float(
            disk_profile.outer_radius_cm / disk_profile.gravitational_radius_cm
        ),
    )

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": disk_profile.model_name,
        "r_over_rg": torque_radius_over_rg,
        "r_out_over_rg": r_out_over_rg,
        "gamma_i_si": np.asarray(torque_profile.gamma_i_si, dtype=float),
        "gamma_th_si": np.asarray(torque_profile.gamma_th_si, dtype=float),
        "gamma_i_th_si": np.asarray(
            torque_profile.gamma_i_si
            + torque_profile.gap_attenuation_factor * torque_profile.gamma_th_si,
            dtype=float,
        ),
        "gamma_ii_si": np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
        "trap_radii_over_rg": (
            np.asarray(trap_radii_cm, dtype=float)
            / disk_profile.gravitational_radius_cm
        ),
        "anti_trap_radii_over_rg": (
            np.asarray(anti_trap_radii_cm, dtype=float)
            / disk_profile.gravitational_radius_cm
        ),
    }


def Toomre_param_outer_radius(
    radius_over_rg: np.ndarray,
    q_values: np.ndarray,
    *,
    fallback: float,
) -> float:
    """Return the outer Q=1 radius in rg units, falling back when absent."""
    radius = np.asarray(radius_over_rg, dtype=float)
    q = np.asarray(q_values, dtype=float)
    n = min(radius.size, q.size)
    if n == 0:
        return float(fallback)

    radius = radius[:n]
    q = q[:n]
    valid = np.isfinite(radius) & np.isfinite(q) & (radius > 0.0) & (q > 0.0)
    radius = radius[valid]
    q = q[valid]
    if radius.size == 0:
        return float(fallback)

    order = np.argsort(radius)
    radius = radius[order]
    q = q[order]
    log_radius = np.log10(radius)
    log_q = np.log10(q)

    exact = np.where(np.isclose(log_q, 0.0, rtol=0.0, atol=1.0e-8))[0]
    if exact.size:
        return float(radius[exact[-1]])

    crossing_indices = np.where(np.sign(log_q[1:]) != np.sign(log_q[:-1]))[0]
    if crossing_indices.size == 0:
        return float(fallback)

    roots = []
    for idx in crossing_indices:
        x0, x1 = log_radius[idx], log_radius[idx + 1]
        y0, y1 = log_q[idx], log_q[idx + 1]
        if y1 == y0:
            continue
        roots.append(10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0)))

    if not roots:
        return float(fallback)
    return float(np.max(roots))


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
    profiles: Sequence[Mapping[str, np.ndarray | float | str]],
    colors: Sequence[str],
    *,
    output_dir: str | Path,
    component_key: str,
    ylabel: str,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    mass_handles = []
    has_inward = False
    has_outward = False
    has_traps = False
    has_anti_traps = False

    for profile, color in zip(profiles, colors):
        radius_rg = np.asarray(profile["r_over_rg"], dtype=float)
        torque_values = np.asarray(profile[component_key], dtype=float)
        valid = (
            np.isfinite(radius_rg)
            & np.isfinite(torque_values)
            & (radius_rg > 0.0)
            & (np.abs(torque_values) > 0.0)
        )
        if not np.any(valid):
            continue

        radius_rg = radius_rg[valid]
        torque_values = torque_values[valid]
        torque_magnitude = np.abs(torque_values)
        outward = torque_values > 0.0
        inward = torque_values < 0.0

        if np.any(outward):
            ax.plot(
                radius_rg,
                np.where(outward, torque_magnitude, np.nan),
                color=color,
                lw=3.2,
                ls="-.",
                alpha=0.9,
            )
            has_outward = True

        if np.any(inward):
            ax.plot(
                radius_rg,
                np.where(inward, torque_magnitude, np.nan),
                color=color,
                lw=3.2,
                ls="-",
                alpha=0.9,
            )
            has_inward = True

        mass_label = (
            rf"$M_\bullet=10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$"
        )
        mass_handles.append(Line2D([0], [0], color=color, lw=3.2, label=mass_label))

        r_out_over_rg = float(profile["r_out_over_rg"])
        if np.isfinite(r_out_over_rg) and r_out_over_rg > 0.0:
            ax.axvline(r_out_over_rg, color=color, lw=1.7, ls=":", alpha=0.55)

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


def plot_agn_disk_torque_profiles(
    output_dir: str,
    alpha: float,
    smbh_masses: Sequence[float],
    migrator_mass_msun: float = 10.0,
) -> None:
    """Plot pAGN migration torque magnitudes using the main simulation model."""

    torque_profiles = [
        _disk_torque_payload(
            float(smbh_mass),
            alpha=alpha,
            migrator_mass_msun=migrator_mass_msun,
        )
        for smbh_mass in smbh_masses
    ]

    colors = COLORS_SERIS

    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_i_th_si",
        ylabel=(
            r"$|\Gamma_{\mathrm{I}}+e^{-K/K_t}\Gamma_{\mathrm{th}}|"
            r"\;(\mathrm{N\,m})$"
        ),
        stem="tot_torque_profiles",
    )
    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_ii_si",
        ylabel=r"$|\Gamma_{\mathrm{II}}|\;(\mathrm{N\,m})$",
        stem="type_ii_torque_profiles",
    )
    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_i_si",
        ylabel=r"$|\Gamma_{\mathrm{I}}|\;(\mathrm{N\,m})$",
        stem="type_i_torque_profiles",
    )
    _plot_signed_torque_component(
        torque_profiles,
        colors,
        output_dir=output_dir,
        component_key="gamma_th_si",
        ylabel=r"$|\Gamma_{\mathrm{th}}|\;(\mathrm{N\,m})$",
        stem="thermal_torque_profiles",
    )
