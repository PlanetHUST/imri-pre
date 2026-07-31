"""Reusable physics resources for dynamical IMRI simulations."""

import numpy as np

import disk_quantity.disk_solve as disk
import physical_process.constants as const
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile

from .config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    DISK_INNER_RADIUS_RG,
    IMRIDynConfig,
)
from .models import DiskProfiles
from .runtime import vprint


class IMRIDynEngine:
    """Shared IMRI engine resources used by single-realization workflows."""

    def __init__(self, config: IMRIDynConfig) -> None:
        self.config = config
        self.duty_cycle_myr = config.duty_cycle_myr
        self.disk_cache: dict[float, DiskProfiles] = {}

    def estimate_nsc_radius_cm(self, m_nsc_msun: float) -> float:
        """Estimate NSC radius [cm] from NSC mass [Msun]."""
        nsc_radius_pc = (
            self.config.nsc_radius_pc_at_reference_mass
            * (m_nsc_msun / self.config.nsc_radius_reference_mass_msun)
            ** self.config.nsc_radius_mass_slope
        )
        return float(nsc_radius_pc * const.parsec)

    def get_disk_profiles(self, m_smbh_msun: float) -> DiskProfiles:
        """Return cached disk profiles for an SMBH mass [Msun]."""
        if m_smbh_msun not in self.disk_cache:
            accretion_rate_g_s = DISK_ACCRETION_RATE_G_S_PER_MSUN * m_smbh_msun

            (
                outer_radius_cm,
                radius_cm,
                _temperature_k,
                _toomre_q,
                surface_density_g_cm2,
                _optical_depth,
                aspect_ratio,
                _sound_speed_cm_s,
                _density_g_cm3,
            ) = disk.radial_structure(m_smbh_msun, accretion_rate_g_s)

            gravitational_radius_cm = (
                const.G * m_smbh_msun * const.msun / const.c**2
            )
            self.disk_cache[m_smbh_msun] = {
                "inner_radius_cm": float(
                    DISK_INNER_RADIUS_RG * gravitational_radius_cm
                ),
                "outer_radius_cm": float(outer_radius_cm),
                "radius_cm": np.asarray(radius_cm, dtype=float),
                "surface_density_g_cm2": np.asarray(
                    surface_density_g_cm2,
                    dtype=float,
                ),
                "aspect_ratio": np.asarray(aspect_ratio, dtype=float),
                "gravitational_radius_cm": float(gravitational_radius_cm),
            }
            try:
                pagn_alpha = 0.01
                pagn_disk = build_pagn_disk_profile(
                    m_smbh_msun,
                    mdot_cgs=accretion_rate_g_s,
                    alpha=pagn_alpha,
                    eddington_ratio=0.1,
                )
                self.disk_cache[m_smbh_msun]["torque_disk"] = pagn_disk.disk
                self.disk_cache[m_smbh_msun]["torque_alpha"] = pagn_alpha
                vprint(
                    f"  cached pAGN disk for future torque work at "
                    f"M_SMBH={m_smbh_msun:.3e} Msun"
                )
            except Exception:
                vprint(
                    f"  pAGN disk unavailable for M_SMBH={m_smbh_msun:.3e} Msun; "
                    f"using disk_solve migration only"
                )
            vprint(f"  cached disk for M_SMBH={m_smbh_msun:.3e} Msun")
        return self.disk_cache[m_smbh_msun]
