"""Disk migration timescale helpers."""

import numpy as np
import numpy.typing as npt

from disk_quantity.disk_solve import DISK_VISCOSITY_ALPHA

try:
    import physical_process.constants as const
    from physical_process.general_class import AGN_properties
except ImportError:
    import constants as const
    from general_class import AGN_properties

FloatArray = npt.NDArray[np.float64]

TYPE_II_GAP_PARAMETER_THRESHOLD = 11.0


def find_closest_index(r_list: FloatArray, r: float) -> int:
    """Return the index of the radius [cm] nearest to ``r`` [cm]."""
    radius_cm = np.asarray(r_list, dtype=float)
    if radius_cm.size == 0:
        raise ValueError("radius grid must not be empty")
    insertion_index = int(np.searchsorted(radius_cm, r))
    if insertion_index == 0:
        return 0
    if insertion_index == len(radius_cm):
        return len(radius_cm) - 1
    left_index = insertion_index - 1
    if abs(radius_cm[left_index] - r) <= abs(radius_cm[insertion_index] - r):
        return left_index
    return insertion_index


def h_gas(r: float, h_list: FloatArray, r_list: FloatArray) -> float:
    """Return dimensionless disk aspect ratio at radius ``r`` [cm]."""
    return float(h_list[find_closest_index(r_list, r)])


def sigma_gas(r: float, sigma_list: FloatArray, r_list: FloatArray) -> float:
    """Return disk surface density [g/cm^2] at radius ``r`` [cm]."""
    return float(sigma_list[find_closest_index(r_list, r)])


def type_ii_gap_parameter(
    AGN: AGN_properties,
    m_g: float,
    R: float,
    h_list: FloatArray,
    r_list: FloatArray,
) -> float:
    """Return the gap-opening parameter for type-II migration."""
    if m_g <= 0.0 or R <= 0.0:
        raise ValueError("object mass and orbital radius must be positive")
    smbh_mass_g = AGN.M_SMBH * const.msun
    aspect_ratio = h_gas(R, h_list, r_list)
    return float(m_g**2 / (DISK_VISCOSITY_ALPHA * aspect_ratio**2 * smbh_mass_g**2))


def type_ii_threshold_mass_msun(
    AGN: AGN_properties,
    R: float,
    h_list: FloatArray,
    r_list: FloatArray,
) -> float:
    """Return the local stellar mass threshold [Msun] for K > 11."""
    if R <= 0.0:
        raise ValueError("orbital radius must be positive")
    aspect_ratio = h_gas(R, h_list, r_list)
    return float(
        np.sqrt(TYPE_II_GAP_PARAMETER_THRESHOLD * DISK_VISCOSITY_ALPHA)
        * aspect_ratio
        * AGN.M_SMBH
    )


def t_migration(
    AGN: AGN_properties,
    m_g: float,
    R: float,
    h_list: FloatArray,
    sigma_list: FloatArray,
    r_list: FloatArray,
    consider_type_ii: bool = True,
) -> float:
    """Return type-I or gap-opening type-II migration time [Myr].

    Low-mass stars use ``t = M^2 h^2 / (m Sigma R^2 Omega)``.
    Stars satisfying ``K = m^2 / (alpha h^2 M^2) > 11`` use the viscous
    type-II timescale ``t = 1 / (alpha h^2 Omega)`` when
    ``consider_type_ii`` is enabled.
    """
    if m_g <= 0.0 or R <= 0.0:
        raise ValueError("object mass and orbital radius must be positive")
    smbh_mass_g = AGN.M_SMBH * const.msun
    aspect_ratio = h_gas(R, h_list, r_list)
    surface_density_g_cm2 = sigma_gas(R, sigma_list, r_list)
    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / R**3)
    gap_parameter = m_g**2 / (DISK_VISCOSITY_ALPHA * aspect_ratio**2 * smbh_mass_g**2)
    if consider_type_ii and gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD:
        migration_time_s = 1.0 / (
            DISK_VISCOSITY_ALPHA * aspect_ratio**2 * orbital_frequency_rad_s
        )
    else:
        migration_time_s = (
            smbh_mass_g**2
            * aspect_ratio**2
            / m_g
            / surface_density_g_cm2
            / R**2
            / orbital_frequency_rad_s
        )
    return float(migration_time_s / const.Myr)


def t_migration_from_torque(
    r_grid_cm: FloatArray,
    gamma_tot_cgs: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
) -> float:
    """Return migration time [Myr] by integrating the torque profile.

    Uses the formula from prompt.md:
        t_mig = integral(R_ini -> R_fin) m * R^2 * Omega / Gamma_tot dR

    Args:
        r_grid_cm: radial grid [cm] from the torque computation.
        gamma_tot_cgs: total torque [g cm^2 s^-2] at each radius, including
            Type-I, thermal, and Type-II contributions.
        migrator_mass_g: migrator (stellar) mass [g].
        r_ini_cm: starting orbital radius [cm].
        r_fin_cm: target orbital radius [cm].
        smbh_mass_msun: SMBH mass [Msun].

    Returns:
        Migration timescale [Myr]. Returns ``np.inf`` if the integrand
        is not finite over the integration range, or if a torque zero-crossing
        (migration trap) lies between ``r_ini_cm`` and ``r_fin_cm``.
    """
    if migrator_mass_g <= 0.0:
        raise ValueError("migrator mass must be positive")
    if r_ini_cm <= 0.0 or r_fin_cm <= 0.0:
        raise ValueError("radii must be positive")

    smbh_mass_g = smbh_mass_msun * const.msun

    r_grid = np.asarray(r_grid_cm, dtype=float)
    gamma = np.asarray(gamma_tot_cgs, dtype=float)
    if r_grid.shape != gamma.shape:
        raise ValueError("radius and torque grids must have the same shape")

    finite = np.isfinite(r_grid) & np.isfinite(gamma)
    r_grid = r_grid[finite]
    gamma = gamma[finite]
    if r_grid.size < 2:
        return np.inf

    order = np.argsort(r_grid)
    r_grid = r_grid[order]
    gamma = gamma[order]
    unique_r, unique_indices = np.unique(r_grid, return_index=True)
    r_grid = unique_r
    gamma = gamma[unique_indices]
    if r_grid.size < 2:
        return np.inf

    r_min = min(r_ini_cm, r_fin_cm)
    r_max = max(r_ini_cm, r_fin_cm)
    if r_min < r_grid[0] or r_max > r_grid[-1]:
        return np.inf

    mask = (r_grid >= r_min) & (r_grid <= r_max)
    r_seg = np.concatenate(
        ([r_min], r_grid[mask & (r_grid > r_min) & (r_grid < r_max)], [r_max])
    )
    gamma_seg = np.interp(r_seg, r_grid, gamma)

    if r_seg.size < 2:
        return np.inf

    if np.any(gamma_seg == 0.0):
        return np.inf
    sign_changes = np.where(np.sign(gamma_seg[1:]) != np.sign(gamma_seg[:-1]))[0]
    if sign_changes.size > 0:
        return np.inf

    omega = np.sqrt(const.G * smbh_mass_g / r_seg**3)
    integrand = migrator_mass_g * r_seg**2 * omega / gamma_seg

    if r_ini_cm > r_fin_cm:
        r_integral = r_seg[::-1]
        integrand_integral = integrand[::-1]
    else:
        r_integral = r_seg
        integrand_integral = integrand

    trapezoid = getattr(np, "trapezoid", np.trapz)
    t_mig_s = float(trapezoid(integrand_integral, r_integral))

    if not np.isfinite(t_mig_s) or t_mig_s <= 0.0:
        return np.inf

    return float(t_mig_s / const.Myr)
