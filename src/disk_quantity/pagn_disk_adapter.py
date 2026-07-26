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
    r_list: np.ndarray
    Tc_list: np.ndarray
    Q_list: np.ndarray
    sigma_list: np.ndarray
    tau_list: np.ndarray
    h_list: np.ndarray
    cs_list: np.ndarray
    rho_list: np.ndarray
    kappa_list: np.ndarray
    r_in: float
    r_out: float
    rg: float
    rs: float


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
        q = np.asarray(disk.Q, dtype=float)
        if q.shape == np.asarray(disk.R).shape:
            return q

    sigma_si = 2.0 * np.asarray(disk.rho, dtype=float) * np.asarray(disk.h, dtype=float)
    return (
        np.asarray(disk.cs, dtype=float)
        * np.asarray(disk.Omega, dtype=float)
        / (np.pi * ct.G * sigma_si)
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
    M_smbh_msun: float,
    *,
    model: str = "sirko",
    n_resolution: int = 10000,
    eddington_ratio: float = 0.5,
    alpha: float = 0.01,
    b: int = 0,
    mdot_cgs: Optional[float] = None,
    thompson_rout_rg: float = 1.0e7,
    thompson_mdot_out_msun_yr: Optional[float] = None,
    suppress_pagn_output: bool = True,
) -> PagnDiskProfile:
    """Build a pAGN disk and return CGS arrays for the existing pipeline."""
    Sirko, Thompson, ct = _require_pagn()

    model_key = model.lower()
    M_si = float(M_smbh_msun) * ct.MSun

    with contextlib.ExitStack() as output_context:
        if suppress_pagn_output:
            devnull = output_context.enter_context(open(os.devnull, "w"))
            output_context.enter_context(contextlib.redirect_stdout(devnull))

        if model_key == "sirko":
            kwargs = dict(Mbh=M_si, alpha=float(alpha), b=int(b))
            if mdot_cgs is not None:
                kwargs["Mdot"] = float(mdot_cgs) * 1.0e-3
            else:
                kwargs["le"] = float(eddington_ratio)
            disk = Sirko.SirkoAGN(**kwargs)
        elif model_key == "thompson":
            Rout_si = float(thompson_rout_rg) * (ct.G * M_si / ct.c**2)
            kwargs = dict(Mbh=M_si, Rout=Rout_si)
            if thompson_mdot_out_msun_yr is not None:
                kwargs["Mdot_out"] = float(thompson_mdot_out_msun_yr) * ct.MSun / ct.yr
            disk = Thompson.ThompsonAGN(**kwargs)
        else:
            raise ValueError("model must be 'sirko' or 'thompson'")

        disk.solve_disk(N=int(n_resolution))

    _sort_disk_profile_arrays(disk)

    radius_si = np.asarray(disk.R, dtype=float)
    scale_height_si = np.asarray(disk.h, dtype=float)
    rho_si = np.asarray(disk.rho, dtype=float)

    r_list = radius_si * METER_TO_CM
    Tc_list = np.asarray(disk.T, dtype=float)
    sigma_list = 2.0 * rho_si * scale_height_si * KG_M2_TO_G_CM2
    tau_list = np.asarray(disk.tauV, dtype=float)
    h_list = scale_height_si / radius_si
    cs_list = np.asarray(disk.cs, dtype=float) * METER_TO_CM
    rho_list = rho_si * KG_M3_TO_G_CM3
    kappa_list = np.asarray(disk.kappa, dtype=float) * M2_KG_TO_CM2_G
    Q_list = _profile_q(disk, ct)

    rs = float(np.asarray(disk.Rs, dtype=float)) * METER_TO_CM
    rg = 0.5 * rs
    r_in = _disk_radius_attr_cm(disk, float(np.nanmin(r_list)), "Rin", "R_in")
    r_out = _disk_radius_attr_cm(disk, float(np.nanmax(r_list)), "Rout", "R_out")

    return PagnDiskProfile(
        model_name=model_key,
        disk=disk,
        r_list=r_list,
        Tc_list=Tc_list,
        Q_list=Q_list,
        sigma_list=sigma_list,
        tau_list=tau_list,
        h_list=h_list,
        cs_list=cs_list,
        rho_list=rho_list,
        kappa_list=kappa_list,
        r_in=r_in,
        r_out=r_out,
        rg=rg,
        rs=rs,
    )
