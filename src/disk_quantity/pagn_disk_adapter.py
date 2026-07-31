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
    radius = np.asarray(disk.R, dtype=float)
    order = np.argsort(radius)
    if np.all(order == np.arange(radius.size)):
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
        if arr.shape == radius.shape:
            setattr(disk, name, arr[order])


def _profile_q(disk, ct) -> np.ndarray:
    # stability parameter Q
    if hasattr(disk, "Q"):
        toomre_q = np.asarray(disk.Q, dtype=float)
        if toomre_q.shape == np.asarray(disk.R).shape:
            return toomre_q

    surface_density_kg_m2 = (
        2.0 * np.asarray(disk.rho, dtype=float) * np.asarray(disk.h, dtype=float)
    )
    return (
        np.asarray(disk.cs, dtype=float)
        * np.asarray(disk.Omega, dtype=float)
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
        radius_cm = float(value) * METER_TO_CM
        if np.isfinite(radius_cm) and radius_cm > 0.0:
            return radius_cm
    return float(fallback_cm)


def build_pagn_disk_profile(
    smbh_mass_msun: float,
    *,
    model: str = "sirko",
    n_resolution: int = 10000,
    eddington_ratio: float = 0.5,
    alpha: float = 0.01,
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
        elif model_key == "thompson":
            thompson_outer_radius_m = (
                float(thompson_rout_rg) * (ct.G * smbh_mass_kg / ct.c**2)
            )
            kwargs = dict(Mbh=smbh_mass_kg, Rout=thompson_outer_radius_m)
            if thompson_outer_accretion_rate_msun_per_yr is not None:
                kwargs["Mdot_out"] = (
                    float(thompson_outer_accretion_rate_msun_per_yr)
                    * ct.MSun
                    / ct.yr
                )
            disk = Thompson.ThompsonAGN(**kwargs)
        else:
            raise ValueError("model must be 'sirko' or 'thompson'")

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
    outer_radius_cm = _disk_radius_attr_cm(
        disk,
        float(np.nanmax(radius_cm)),
        "Rout",
        "R_out",
    )

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
