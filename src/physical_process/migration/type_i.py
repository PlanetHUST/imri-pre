"""Default inward Type-I migration on a sampled disk profile.

The local timescale is M**2 h**2 / (2 m Sigma r**2 Omega); the coupled
orbit integrator consumes dr/dt = -r/t_I alongside GW drift.
"""

import numpy as np

from physical_process import constants as const


def type_i_local_time_s(
    radius_cm: np.ndarray,
    surface_density_g_cm2: np.ndarray,
    aspect_ratio: np.ndarray,
    migrator_mass_g: float,
    smbh_mass_msun: float,
) -> np.ndarray:
    """Return the Type-I local radial time [s], with no gap or trap switch."""
    radius = np.asarray(radius_cm, dtype=float)
    sigma = np.asarray(surface_density_g_cm2, dtype=float)
    h = np.asarray(aspect_ratio, dtype=float)
    if not (radius.ndim == sigma.ndim == h.ndim == 1
            and radius.shape == sigma.shape == h.shape and radius.size >= 2):
        raise ValueError("Type-I disk profile arrays must align and have at least two cells")
    if not all(np.all(np.isfinite(x) & (x > 0)) for x in (radius, sigma, h)):
        raise ValueError("Type-I disk profile must be finite and positive")
    if not (np.isfinite(migrator_mass_g) and migrator_mass_g > 0
            and np.isfinite(smbh_mass_msun) and smbh_mass_msun > 0):
        raise ValueError("migrator and SMBH masses must be finite and positive")
    primary_mass_g = smbh_mass_msun * const.msun
    omega = np.sqrt(const.G * primary_mass_g / radius**3)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        time_s = primary_mass_g**2 * h**2 / (
            2 * migrator_mass_g * sigma * radius**2 * omega
        )
    if not np.all(np.isfinite(time_s) & (time_s > 0)):
        raise ValueError("Type-I migration time must be finite and positive")
    return np.asarray(time_s, dtype=float)


def type_i_drift_cm_s(
    radius_cm: np.ndarray,
    surface_density_g_cm2: np.ndarray,
    aspect_ratio: np.ndarray,
    migrator_mass_g: float,
    smbh_mass_msun: float,
) -> np.ndarray:
    """Inward gas drift [cm/s] for the coupled circular orbit."""
    return -np.asarray(radius_cm, dtype=float) / type_i_local_time_s(
        radius_cm, surface_density_g_cm2, aspect_ratio,
        migrator_mass_g, smbh_mass_msun,
    )
