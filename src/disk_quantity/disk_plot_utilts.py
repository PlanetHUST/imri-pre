from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Any

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

plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

SMBH_MASSES = (1e3, 1e4, 1e5)


def _save(fig: Figure, outdir: str | Path, stem: str, dpi: int = 300) -> None:
    """Save each figure in both PNG and PDF form for quick view and publication."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


def _disk_profile_payload(
    smbh_mass: float,
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float,
    alpha: float,
    n_resolution: int | None = None,
) -> Dict[str, np.ndarray | float | str]:
    """Compute pAGN disk profiles for one SMBH mass using the main model path."""

    if n_resolution is None:
        n_resolution = int(os.getenv("PAGN_DISK_N_RESOLUTION", "10000"))

    accretion_rate_g_s = None
    if mdot_per_solar_mass is not None:
        accretion_rate_g_s = float(smbh_mass) * float(mdot_per_solar_mass)

    profile = build_pagn_disk_profile(
        smbh_mass,
        model=disk_model,
        n_resolution=n_resolution,
        eddington_ratio=eddington_ratio,
        alpha=alpha,
        b=0,
        accretion_rate_g_s=accretion_rate_g_s,
    )

    r_over_rg = (
        np.asarray(profile.radius_cm, dtype=float) / profile.gravitational_radius_cm
    )

    q_values = np.asarray(profile.toomre_q, dtype=float)

    r_out_over_rg = Toomre_param_outer_radius(
        r_over_rg,
        q_values,
        fallback=float(profile.outer_radius_cm / profile.gravitational_radius_cm),
    )

    return {
        "smbh_mass": float(smbh_mass),
        "disk_model": profile.model_name,
        "r_over_rg": r_over_rg,
        "r_out_over_rg": r_out_over_rg,
        "tc": np.asarray(profile.temperature_k, dtype=float),
        "q": q_values,
        "sigma": np.asarray(profile.surface_density_g_cm2, dtype=float),
        "tau": np.asarray(profile.optical_depth, dtype=float),
        "h_over_r": np.asarray(profile.aspect_ratio, dtype=float),
        "cs": np.asarray(profile.sound_speed_cm_s, dtype=float),
        "rho": np.asarray(profile.density_g_cm3, dtype=float),
    }


def _disk_solver_profile_payload(
    smbh_mass: float,
    mdot_per_solar_mass: float | None = None,
) -> Dict[str, np.ndarray | float | str]:
    """Compute the in-repo disk_solve.py profiles for one SMBH mass."""
    if mdot_per_solar_mass is None:
        mdot_per_solar_mass = LEGACY_DISK_MDOT_PER_SOLAR_MASS

    accretion_rate_g_s = float(smbh_mass) * float(mdot_per_solar_mass)
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
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
    migrator_mass_msun: float = 10.0,
    smoothing: float = 0.005,
) -> Dict[str, np.ndarray | float | str]:
    """Compute a pAGN torque profile using the main simulation torque model."""
    if n_resolution is None:
        n_resolution = int(os.getenv("PAGN_DISK_N_RESOLUTION", "30000"))

    accretion_rate_g_s = None
    if mdot_per_solar_mass is not None:
        accretion_rate_g_s = float(smbh_mass) * float(mdot_per_solar_mass)

    disk_profile = build_pagn_disk_profile(
        smbh_mass,
        model=disk_model,
        n_resolution=n_resolution,
        eddington_ratio=eddington_ratio,
        alpha=alpha,
        b=0,
        accretion_rate_g_s=accretion_rate_g_s,
    )
    params = MigrationParameters(alpha=alpha)
    torque_profile = compute_torque_profile(
        disk_profile.disk,
        migrator_mass_msun=migrator_mass_msun,
        smoothing=smoothing,
        params=params,
    )
    trap_radii_cm, anti_trap_radii_cm = find_traps_locations(torque_profile)

    torque_radius_over_rg = np.asarray(torque_profile.radius_rg, dtype=float)
    q_radius_over_rg = (
        np.asarray(disk_profile.radius_cm, dtype=float)
        / disk_profile.gravitational_radius_cm
    )
    r_out_over_rg = Toomre_param_outer_radius(
        q_radius_over_rg,
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
        "gamma_tot_cgs": np.asarray(torque_profile.gamma_tot, dtype=float),
        "gamma_i_cgs": np.asarray(torque_profile.gamma_i, dtype=float),
        "gamma_th_cgs": np.asarray(torque_profile.gamma_th, dtype=float),
        "trap_radii_over_rg": (
            np.asarray(trap_radii_cm, dtype=float)
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


def _disk_plot_radius_mask(profile: Mapping[str, np.ndarray | float | str], radius):
    """Limit one plotted line to its Q=1 outer radius."""
    radius = np.asarray(radius, dtype=float)
    r_out = float(profile.get("r_out_over_rg", np.nan))
    if not np.isfinite(r_out) or r_out <= 0.0:
        return np.ones_like(radius, dtype=bool)
    return radius <= r_out


def _disk_radius_bounds(
    profiles: Sequence[Mapping[str, np.ndarray | float | str]],
) -> tuple[float, float] | None:
    """Return finite plot limits from solved profile radii and Q=1 outer edges."""
    radii = []
    r_out_values = []
    for profile in profiles:
        radius = np.asarray(profile["r_over_rg"], dtype=float)
        valid_radius = radius[
            np.isfinite(radius) & (radius > 0) & _disk_plot_radius_mask(profile, radius)
        ]
        if valid_radius.size:
            radii.append(valid_radius)

        r_out = float(profile["r_out_over_rg"])
        if np.isfinite(r_out) and r_out > 0:
            r_out_values.append(r_out)

    if not radii:
        return None

    all_radii = np.concatenate(radii)
    r_min = float(np.nanmin(all_radii))
    r_max = (
        float(np.nanmax(r_out_values)) if r_out_values else float(np.nanmax(all_radii))
    )
    if r_min <= 0 or r_max <= r_min:
        return None
    return r_min, r_max


def plot_agn_disk_parameter_profiles(
    output_dir: str = "figs",
    smbh_masses: Sequence[float] = SMBH_MASSES,
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
) -> None:
    """Plot useful pAGN AGN disk parameters versus radius."""
    profiles = [
        _disk_profile_payload(
            float(mass),
            mdot_per_solar_mass,
            disk_model=disk_model,
            eddington_ratio=eddington_ratio,
            alpha=alpha,
            n_resolution=n_resolution,
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
    colors = ["#1f77b4", "#BE9A62", "#2A7F62"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    radius_bounds = _disk_radius_bounds(profiles)

    for ax, (key, ylabel) in zip(axes, parameters):
        for profile, color in zip(profiles, colors):
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

        ax.set_xscale("log")
        ax.set_yscale("log")
        if radius_bounds is not None:
            ax.set_xlim(radius_bounds)
        ax.set_ylabel(ylabel, fontsize=RADIAL_AXIS_LABELSIZE)
        ax.grid(True, alpha=0.25, linestyle="--", which="both")

    for ax in axes:
        ax.set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)

    axes[0].legend(loc="best", framealpha=0.8, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, output_dir, "agn_disk_parameter_profiles")
    plt.close(fig)


def plot_agn_disk_method_comparison(
    output_dir: str = "figs",
    smbh_masses: Sequence[float] = SMBH_MASSES,
    mdot_per_solar_mass: float | None = LEGACY_DISK_MDOT_PER_SOLAR_MASS,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
) -> None:
    """Compare pAGN and disk_solve.py AGN disk profiles for the same inputs."""
    pagn_profiles = [
        _disk_profile_payload(
            float(mass),
            mdot_per_solar_mass,
            disk_model=disk_model,
            eddington_ratio=eddington_ratio,
            alpha=alpha,
            n_resolution=n_resolution,
        )
        for mass in smbh_masses
    ]
    solved_profiles = [
        _disk_solver_profile_payload(float(mass), mdot_per_solar_mass)
        for mass in smbh_masses
    ]

    parameters = [
        ("sigma", r"$\Sigma\;(\mathrm{g\,cm^{-2}})$"),
        ("h_over_r", r"$H/R$"),
        ("q", r"$Q$"),
    ]
    colors = ["#1f77b4", "#BE9A62", "#2A7F62"]
    radius_bounds = _disk_radius_bounds([*pagn_profiles, *solved_profiles])

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
                    & _disk_plot_radius_mask(profile, radius)
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

        ax.set_xscale("log")
        ax.set_yscale("log")
        if radius_bounds is not None:
            ax.set_xlim(radius_bounds)
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
    _save(fig, output_dir, "agn_disk_method_comparison")
    plt.close(fig)


def plot_agn_disk_torque_profiles(
    output_dir: str = "figs",
    smbh_masses: Sequence[float] = SMBH_MASSES,
    mdot_per_solar_mass: float | None = None,
    *,
    disk_model: str = "sirko",
    eddington_ratio: float = 0.1,
    alpha: float = 0.01,
    n_resolution: int | None = None,
    migrator_mass_msun: float = 10.0,
    smoothing: float = 0.005,
) -> None:
    """Plot pAGN migration torque magnitudes using the main simulation model."""
    profiles = [
        _disk_torque_payload(
            float(mass),
            mdot_per_solar_mass,
            disk_model=disk_model,
            eddington_ratio=eddington_ratio,
            alpha=alpha,
            n_resolution=n_resolution,
            migrator_mass_msun=migrator_mass_msun,
            smoothing=smoothing,
        )
        for mass in smbh_masses
    ]

    colors = ["#1f77b4", "#BE9A62", "#2A7F62"]
    radius_bounds = _disk_radius_bounds(profiles)

    fig, ax = plt.subplots(figsize=(14, 8))
    mass_handles = []

    for profile, color in zip(profiles, colors):
        radius = np.asarray(profile["r_over_rg"], dtype=float)
        radius_mask = _disk_plot_radius_mask(profile, radius)
        gamma_tot_cgs = np.asarray(profile["gamma_tot_cgs"], dtype=float)
        gamma_abs_cgs = np.abs(gamma_tot_cgs)
        mask = (
            np.isfinite(radius)
            & np.isfinite(gamma_abs_cgs)
            & (radius > 0)
            & (gamma_abs_cgs > 0)
            & radius_mask
        )
        if not np.any(mask):
            continue

        mass_label = (
            rf"$M_\bullet=10^{{{int(np.log10(profile['smbh_mass']))}}}\,M_\odot$"
        )
        mass_handles.append(Line2D([0], [0], color=color, lw=3.2, label=mass_label))

        gamma_outward = np.where(mask & (gamma_tot_cgs > 0.0), gamma_abs_cgs, np.nan)
        gamma_inward = np.where(mask & (gamma_tot_cgs < 0.0), gamma_abs_cgs, np.nan)
        ax.plot(
            radius,
            gamma_outward,
            color=color,
            lw=3.2,
            ls="-.",
            alpha=0.9,
        )
        ax.plot(
            radius,
            gamma_inward,
            color=color,
            lw=3.2,
            ls="-",
            alpha=0.9,
        )

        r_out = float(profile["r_out_over_rg"])
        trap_radii = np.asarray(profile["trap_radii_over_rg"], dtype=float)
        trap_mask = np.isfinite(trap_radii) & (trap_radii > 0)

        if np.isfinite(r_out) and r_out > 0:
            trap_mask &= trap_radii <= r_out
        trap_radii = trap_radii[trap_mask]

        # if trap_radii.size:
        #     for trap_radius in trap_radii:
        #         ax.axvline(trap_radius, color=color, lw=3.5, ls="--", alpha=0.8)

        if np.isfinite(r_out) and r_out > 0:
            ax.axvline(r_out, color=color, lw=3.5, ls=":", alpha=0.65)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if radius_bounds is not None:
        ax.set_xlim(radius_bounds)
    ax.set_xlabel(r"$R/r_g$", fontsize=RADIAL_AXIS_LABELSIZE)
    ax.set_ylabel(
        r"$|\Gamma_{\rm tot}|\;(\mathrm{g\,cm^2\,s^{-2}})$",
        fontsize=RADIAL_AXIS_LABELSIZE,
    )
    ax.grid(True, alpha=0.25, linestyle="--", which="both")

    direction_handles = [
        Line2D([0], [0], color="0.2", lw=3.2, ls="-", label="Inward migration"),
        Line2D([0], [0], color="0.2", lw=3.2, ls="-.", label="Outward migration"),
        # Line2D([0], [0], color="0.2", lw=1.4, ls="--", label="Stable trap"),
    ]
    ax.legend(
        handles=mass_handles + direction_handles,
        loc="best",
        framealpha=0.85,
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    _save(fig, output_dir, "agn_disk_torque_profiles")
    plt.close(fig)
