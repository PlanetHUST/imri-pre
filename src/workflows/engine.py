"""Reusable physics resources for dynamical IMRI simulations."""

import numpy as np

import disk_quantity.disk_solve as disk
import physical_process.constants as const
from disk_quantity.migration_torque import (
    MigrationParameters,
    TorqueProfile,
    compute_torque_profile,
)
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile

from .config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    DISK_INNER_RADIUS_RG,
    PAGN_ALPHA,
    IMRIDynConfig,
)
from .models import DiskProfiles
from .runtime import vprint


class TorqueProfileError(RuntimeError):
    """Raised when the required pAGN torque profile cannot be constructed."""


class IMRIDynEngine:
    """Shared IMRI engine resources used by single-realization workflows."""

    def __init__(self, config: IMRIDynConfig) -> None:
        self.config = config
        self.duty_cycle_myr = config.duty_cycle_myr
        self.disk_cache: dict[float, DiskProfiles] = {}
        self.torque_profile_cache: dict[tuple[float, float, float], TorqueProfile] = {}

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
            try:
                pagn_alpha = PAGN_ALPHA
                pagn_disk = build_pagn_disk_profile(
                    m_smbh_msun,
                    pagn_alpha,
                    accretion_rate_g_s=accretion_rate_g_s,
                )
                self.disk_cache[m_smbh_msun] = {
                    "inner_radius_cm": pagn_disk.inner_radius_cm,
                    "outer_radius_cm": pagn_disk.outer_radius_cm,
                    "radius_cm": np.asarray(pagn_disk.radius_cm, dtype=float),
                    "surface_density_g_cm2": np.asarray(
                        pagn_disk.surface_density_g_cm2,
                        dtype=float,
                    ),
                    "aspect_ratio": np.asarray(pagn_disk.aspect_ratio, dtype=float),
                    "gravitational_radius_cm": pagn_disk.gravitational_radius_cm,
                }
                self.disk_cache[m_smbh_msun]["torque_disk"] = pagn_disk.disk
                self.disk_cache[m_smbh_msun]["torque_alpha"] = pagn_alpha
                vprint(
                    f"  cached pAGN disk profile for damping and torque work at "
                    f"smbh_mass_msun={m_smbh_msun:.3e}"
                )
            except Exception:
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
                vprint(
                    f"  pAGN disk unavailable for smbh_mass_msun={m_smbh_msun:.3e}; "
                    f"using disk_solve migration only"
                )
            vprint(f"  cached disk for smbh_mass_msun={m_smbh_msun:.3e}")
        return self.disk_cache[m_smbh_msun]

    def get_torque_profile(
        self,
        disk_data: DiskProfiles,
        smbh_mass_msun: float,
        migrator_mass_msun: float,
    ) -> TorqueProfile:
        """Return a cached mass-dependent pAGN torque profile."""
        if "torque_disk" not in disk_data:
            raise TorqueProfileError("cached disk has no pAGN torque model")

        torque_alpha = (
            float(disk_data["torque_alpha"]) if "torque_alpha" in disk_data else 0.01
        )
        cache_key = (
            float(smbh_mass_msun),
            float(migrator_mass_msun),
            torque_alpha,
        )
        if cache_key not in self.torque_profile_cache:
            params = MigrationParameters(alpha=torque_alpha)
            try:
                self.torque_profile_cache[cache_key] = compute_torque_profile(
                    disk_data["torque_disk"],
                    migrator_mass_msun=migrator_mass_msun,
                    max_radius_rg=None,
                    params=params,
                )
            except Exception as exc:
                raise TorqueProfileError(
                    "torque profile construction failed for "
                    f"smbh_mass_msun={smbh_mass_msun:.3e}, "
                    f"migrator_mass_msun={migrator_mass_msun:.3e}: {exc}"
                ) from exc
        return self.torque_profile_cache[cache_key]
