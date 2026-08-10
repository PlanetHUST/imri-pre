"""Adapter for pAGN disk profiles used by the merger simulation.

The simulation code uses CGS arrays, while pAGN solves and stores disk profiles
in SI units. This module keeps that conversion boundary explicit.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

METER_TO_CM = 100.0
KG_M2_TO_G_CM2 = 0.1
KG_M3_TO_G_CM3 = 1.0e-3
M2_KG_TO_CM2_G = 10.0
TOOMRE_Q_OUTER_BOUNDARY = 1.0
SIRKO_Q_SEARCH_MAX_RADIUS_RG = 1.0e12
SIRKO_Q_SEARCH_RADIUS_FACTOR = 10.0
SIRKO_Q_SEARCH_RESOLUTION = 600
SIRKO_Q_BOUNDARY_PADDING = 1.05


@dataclass
class PagnDiskProfile:
    model_name: str
    disk: object
    radius_cm: np.ndarray
    temperature_k: np.ndarray
    toomre_q: np.ndarray
    surface_density_g_cm2: np.ndarray
    optical_depth: np.ndarray
    aspect_ratio: np.ndarray
    sound_speed_cm_s: np.ndarray
    density_g_cm3: np.ndarray
    opacity_cm2_g: np.ndarray
    inner_radius_cm: float
    outer_radius_cm: float
    gravitational_radius_cm: float
    schwarzschild_radius_cm: float


def _require_pagn():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from pagn import Sirko, Thompson
        import pagn.constants as ct
    except ImportError as exc:
        raise ImportError(
            "pAGN is required for physical disk profiles. Install with "
            "`pip install pagn`."
        ) from exc
    return Sirko, Thompson, ct


def _sort_disk_profile_arrays(disk) -> None:
    """Sort one-dimensional pAGN profile arrays by increasing radius in-place."""
    radius_m = np.asarray(disk.R, dtype=float)
    order = np.argsort(radius_m)
    if np.all(order == np.arange(radius_m.size)):
        return

    for name in [
        "R",
        "rho",
        "T",
        "h",
        "cs",
        "Omega",
        "tauV",
        "kappa",
        "Teff4",
        "Q",
    ]:
        if not hasattr(disk, name):
            continue
        values = getattr(disk, name)
        arr = np.asarray(values)
        if arr.shape == radius_m.shape:
            setattr(disk, name, arr[order])


def _profile_q(disk, ct) -> np.ndarray:
    """Return the dimensionless Toomre Q profile from pAGN SI arrays."""
    radius_m = np.asarray(disk.R, dtype=float)
    if hasattr(disk, "Q"):
        toomre_q = np.asarray(disk.Q, dtype=float)
        if toomre_q.shape == radius_m.shape:
            return toomre_q

    sound_speed_m_s = np.asarray(disk.cs, dtype=float)
    orbital_frequency_rad_s = np.asarray(disk.Omega, dtype=float)
    surface_density_kg_m2 = (
        2.0 * np.asarray(disk.rho, dtype=float) * np.asarray(disk.h, dtype=float)
    )
    return (
        sound_speed_m_s
        * orbital_frequency_rad_s
        / (np.pi * ct.G * surface_density_kg_m2)
    )


def _disk_radius_attr_cm(disk, fallback_cm: float, *names: str) -> float:
    """Return a pAGN scalar radius attribute in cm, or a supplied CGS fallback."""
    for name in names:
        if not hasattr(disk, name):
            continue
        value = np.asarray(getattr(disk, name), dtype=float)
        if value.shape != ():
            continue
        radius_m = float(value)
        radius_cm = radius_m * METER_TO_CM
        if np.isfinite(radius_cm) and radius_cm > 0.0:
            return radius_cm
    return float(fallback_cm)


def _toomre_q_outer_radius_cm(
    radius_cm: np.ndarray,
    toomre_q: np.ndarray,
    *,
    fallback_cm: float,
) -> float:
    """Return the outer disk radius [cm] where the Toomre parameter reaches Q=1."""
    outer_radius_cm = _find_toomre_q_outer_radius_cm(radius_cm, toomre_q)
    if outer_radius_cm is None:
        return float(fallback_cm)
    return outer_radius_cm


def _find_toomre_q_outer_radius_cm(
    radius_cm: np.ndarray,
    toomre_q: np.ndarray,
) -> float | None:
    """Return the first outward Q=1 self-gravity boundary [cm], if present."""
    radius_values_cm = np.asarray(radius_cm, dtype=float)
    q_values = np.asarray(toomre_q, dtype=float)
    n_values = min(radius_values_cm.size, q_values.size)
    if n_values == 0:
        return None

    radius_values_cm = radius_values_cm[:n_values]
    q_values = q_values[:n_values]
    valid = (
        np.isfinite(radius_values_cm)
        & np.isfinite(q_values)
        & (radius_values_cm > 0.0)
        & (q_values > 0.0)
    )
    radius_values_cm = radius_values_cm[valid]
    q_values = q_values[valid]
    if radius_values_cm.size == 0:
        return None

    order = np.argsort(radius_values_cm)
    radius_values_cm = radius_values_cm[order]
    q_values = q_values[order]
    log_radius_cm = np.log10(radius_values_cm)
    log_q_ratio = np.log10(q_values / TOOMRE_Q_OUTER_BOUNDARY)
    boundary_tolerance = 1.0e-8

    seen_stable_disk = False
    outermost_exact_radius_cm = None
    for idx, q_ratio_value in enumerate(log_q_ratio):
        if q_ratio_value > boundary_tolerance:
            seen_stable_disk = True
            continue
        if np.isclose(q_ratio_value, 0.0, rtol=0.0, atol=boundary_tolerance):
            outermost_exact_radius_cm = float(radius_values_cm[idx])
            if seen_stable_disk:
                return outermost_exact_radius_cm
            continue
        if seen_stable_disk and idx > 0 and q_ratio_value < -boundary_tolerance:
            previous_q_ratio = log_q_ratio[idx - 1]
            if previous_q_ratio <= boundary_tolerance:
                return float(radius_values_cm[idx])
            x0, x1 = log_radius_cm[idx - 1], log_radius_cm[idx]
            y0, y1 = previous_q_ratio, q_ratio_value
            if y1 == y0:
                return float(radius_values_cm[idx])
            return float(10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0)))

    if outermost_exact_radius_cm is not None:
        return outermost_exact_radius_cm
    return None


def _reset_sirko_solve_state(disk) -> None:
    disk.i = 0
    disk.isf = -1
    disk.Mdisk = 0.0


def _solve_sirko_disk_with_q_outer_radius(disk, n_resolution: int, ct) -> None:
    """Solve a Sirko pAGN disk with its outer edge set by the first Q=1 radius."""
    initial_outer_radius_m = float(disk.Rmax)
    gravitational_radius_m = ct.G * float(disk.Mbh) / ct.c**2
    max_search_radius_m = SIRKO_Q_SEARCH_MAX_RADIUS_RG * gravitational_radius_m
    search_radius_m = initial_outer_radius_m
    search_resolution = min(int(n_resolution), SIRKO_Q_SEARCH_RESOLUTION)
    q_outer_radius_cm = None

    while search_radius_m <= max_search_radius_m:
        disk.Rmax = search_radius_m
        _reset_sirko_solve_state(disk)
        disk.solve_disk(N=search_resolution)
        search_profile_radius_cm = np.asarray(disk.R, dtype=float) * METER_TO_CM
        search_profile_toomre_q = np.asarray(disk.Q, dtype=float)
        q_outer_radius_cm = _find_toomre_q_outer_radius_cm(
            search_profile_radius_cm,
            search_profile_toomre_q,
        )
        if q_outer_radius_cm is not None:
            break
        search_radius_m *= SIRKO_Q_SEARCH_RADIUS_FACTOR

    if q_outer_radius_cm is None:
        disk.Rmax = initial_outer_radius_m
    else:
        disk.Rmax = q_outer_radius_cm / METER_TO_CM * SIRKO_Q_BOUNDARY_PADDING
    _reset_sirko_solve_state(disk)
    disk.solve_disk(N=n_resolution)


def _set_disk_outer_radius_cm(disk, outer_radius_cm: float) -> None:
    """Mirror the Q=1 boundary onto common pAGN outer-radius attributes."""
    outer_radius_m = float(outer_radius_cm) / METER_TO_CM
    for name in ("Rout", "R_out"):
        try:
            setattr(disk, name, outer_radius_m)
        except AttributeError:
            continue


def build_pagn_disk_profile(
    smbh_mass_msun: float,
    alpha: float,
    *,
    model: str = "sriko",
    n_resolution: int = 1000,
    eddington_ratio: float = 0.1,
    b: int = 0,
    accretion_rate_g_s: Optional[float] = None,
    thompson_rout_rg: float = 1.0e7,
    thompson_outer_accretion_rate_msun_per_yr: Optional[float] = None,
    suppress_pagn_output: bool = True,
) -> PagnDiskProfile:
    """Build a pAGN disk and return CGS arrays for the existing pipeline."""
    Sirko, Thompson, ct = _require_pagn()

    model_key = model.lower()
    smbh_mass_kg = float(smbh_mass_msun) * ct.MSun

    with contextlib.ExitStack() as output_context:
        if suppress_pagn_output:
            devnull = output_context.enter_context(open(os.devnull, "w"))
            output_context.enter_context(contextlib.redirect_stdout(devnull))

        if model_key == "sirko":
            kwargs = dict(Mbh=smbh_mass_kg, alpha=float(alpha), b=int(b))
            if accretion_rate_g_s is not None:
                kwargs["Mdot"] = float(accretion_rate_g_s) * 1.0e-3
            else:
                kwargs["le"] = float(eddington_ratio)
            disk = Sirko.SirkoAGN(**kwargs)
            _solve_sirko_disk_with_q_outer_radius(disk, int(n_resolution), ct)
        elif model_key == "thompson":
            thompson_outer_radius_m = float(thompson_rout_rg) * (
                ct.G * smbh_mass_kg / ct.c**2
            )
            kwargs = dict(Mbh=smbh_mass_kg, Rout=thompson_outer_radius_m)
            if thompson_outer_accretion_rate_msun_per_yr is not None:
                kwargs["Mdot_out"] = (
                    float(thompson_outer_accretion_rate_msun_per_yr) * ct.MSun / ct.yr
                )
            disk = Thompson.ThompsonAGN(**kwargs)
        else:
            raise ValueError("model must be 'sirko' or 'thompson'")

        if model_key != "sirko":
            disk.solve_disk(N=int(n_resolution))

    _sort_disk_profile_arrays(disk)

    radius_m = np.asarray(disk.R, dtype=float)
    scale_height_m = np.asarray(disk.h, dtype=float)
    density_kg_m3 = np.asarray(disk.rho, dtype=float)

    radius_cm = radius_m * METER_TO_CM
    temperature_k = np.asarray(disk.T, dtype=float)
    surface_density_g_cm2 = 2.0 * density_kg_m3 * scale_height_m * KG_M2_TO_G_CM2
    optical_depth = np.asarray(disk.tauV, dtype=float)
    aspect_ratio = scale_height_m / radius_m
    sound_speed_cm_s = np.asarray(disk.cs, dtype=float) * METER_TO_CM
    density_g_cm3 = density_kg_m3 * KG_M3_TO_G_CM3
    opacity_cm2_g = np.asarray(disk.kappa, dtype=float) * M2_KG_TO_CM2_G
    toomre_q = _profile_q(disk, ct)

    schwarzschild_radius_cm = float(np.asarray(disk.Rs, dtype=float)) * METER_TO_CM
    gravitational_radius_cm = 0.5 * schwarzschild_radius_cm
    inner_radius_cm = _disk_radius_attr_cm(
        disk,
        float(np.nanmin(radius_cm)),
        "Rin",
        "R_in",
    )
    fallback_outer_radius_cm = _disk_radius_attr_cm(
        disk,
        float(np.nanmax(radius_cm)),
        "Rout",
        "R_out",
    )
    outer_radius_cm = _toomre_q_outer_radius_cm(
        radius_cm,
        toomre_q,
        fallback_cm=fallback_outer_radius_cm,
    )
    _set_disk_outer_radius_cm(disk, outer_radius_cm)

    return PagnDiskProfile(
        model_name=model_key,
        disk=disk,
        radius_cm=radius_cm,
        temperature_k=temperature_k,
        toomre_q=toomre_q,
        surface_density_g_cm2=surface_density_g_cm2,
        optical_depth=optical_depth,
        aspect_ratio=aspect_ratio,
        sound_speed_cm_s=sound_speed_cm_s,
        density_g_cm3=density_g_cm3,
        opacity_cm2_g=opacity_cm2_g,
        inner_radius_cm=inner_radius_cm,
        outer_radius_cm=outer_radius_cm,
        gravitational_radius_cm=gravitational_radius_cm,
        schwarzschild_radius_cm=schwarzschild_radius_cm,
    )
