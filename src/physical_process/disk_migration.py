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


def find_closest_index(radius_grid_cm: FloatArray, radius_cm: float) -> int:
    """Return the index of the radius [cm] nearest to ``r`` [cm]."""
    radius_grid = np.asarray(radius_grid_cm, dtype=float)
    if radius_grid.size == 0:
        raise ValueError("radius grid must not be empty")
    insertion_index = int(np.searchsorted(radius_grid, radius_cm))
    if insertion_index == 0:
        return 0
    if insertion_index == len(radius_grid):
        return len(radius_grid) - 1
    left_index = insertion_index - 1
    if abs(radius_grid[left_index] - radius_cm) <= abs(
        radius_grid[insertion_index] - radius_cm
    ):
        return left_index
    return insertion_index


def h_gas(
    radius_cm: float,
    aspect_ratio: FloatArray,
    radius_grid_cm: FloatArray,
) -> float:
    """Return dimensionless disk aspect ratio at radius ``radius_cm``."""
    return float(aspect_ratio[find_closest_index(radius_grid_cm, radius_cm)])


def sigma_gas(
    radius_cm: float,
    surface_density_g_cm2: FloatArray,
    radius_grid_cm: FloatArray,
) -> float:
    """Return disk surface density [g/cm^2] at radius ``radius_cm``."""
    return float(surface_density_g_cm2[find_closest_index(radius_grid_cm, radius_cm)])


def type_ii_gap_parameter(
    agn_properties: AGN_properties,
    object_mass_g: float,
    orbital_radius_cm: float,
    aspect_ratio: FloatArray,
    radius_grid_cm: FloatArray,
) -> float:
    """Return the gap-opening parameter for type-II migration."""
    if object_mass_g <= 0.0 or orbital_radius_cm <= 0.0:
        raise ValueError("object mass and orbital radius must be positive")
    smbh_mass_g = agn_properties.smbh_mass_msun * const.msun
    local_aspect_ratio = h_gas(orbital_radius_cm, aspect_ratio, radius_grid_cm)
    return float(
        object_mass_g**2
        / (DISK_VISCOSITY_ALPHA * local_aspect_ratio**2 * smbh_mass_g**2)
    )


def type_ii_threshold_mass_msun(
    agn_properties: AGN_properties,
    orbital_radius_cm: float,
    aspect_ratio: FloatArray,
    radius_grid_cm: FloatArray,
) -> float:
    """Return the local stellar mass threshold [Msun] for K > 11."""
    if orbital_radius_cm <= 0.0:
        raise ValueError("orbital radius must be positive")
    local_aspect_ratio = h_gas(
        orbital_radius_cm,
        aspect_ratio,
        radius_grid_cm,
    )
    return float(
        np.sqrt(TYPE_II_GAP_PARAMETER_THRESHOLD * DISK_VISCOSITY_ALPHA)
        * local_aspect_ratio
        * agn_properties.smbh_mass_msun
    )


def semi_analytical_migration_timescale(
    agn_properties: AGN_properties,
    object_mass_g: float,
    orbital_radius_cm: float,
    aspect_ratio: FloatArray,
    surface_density_g_cm2: FloatArray,
    radius_grid_cm: FloatArray,
    consider_type_ii: bool = True,
) -> float:
    """Return type-I or gap-opening type-II migration time [Myr].

    Low-mass stars use ``t = M^2 h^2 / (m Sigma R^2 Omega)``.
    Stars satisfying ``K = m^2 / (alpha h^2 M^2) > 11`` use the viscous
    type-II timescale ``t = 1 / (alpha h^2 Omega)`` when
    ``consider_type_ii`` is enabled.
    """
    if object_mass_g <= 0.0 or orbital_radius_cm <= 0.0:
        raise ValueError("object mass and orbital radius must be positive")
    smbh_mass_g = agn_properties.smbh_mass_msun * const.msun
    local_aspect_ratio = h_gas(orbital_radius_cm, aspect_ratio, radius_grid_cm)
    local_surface_density_g_cm2 = sigma_gas(
        orbital_radius_cm,
        surface_density_g_cm2,
        radius_grid_cm,
    )
    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / orbital_radius_cm**3)
    gap_parameter = object_mass_g**2 / (
        DISK_VISCOSITY_ALPHA * local_aspect_ratio**2 * smbh_mass_g**2
    )
    if consider_type_ii and gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD:
        migration_time_s = 1.0 / (
            DISK_VISCOSITY_ALPHA * local_aspect_ratio**2 * orbital_frequency_rad_s
        )
    else:
        migration_time_s = (
            smbh_mass_g**2
            * local_aspect_ratio**2
            / object_mass_g
            / local_surface_density_g_cm2
            / orbital_radius_cm**2
            / orbital_frequency_rad_s
        )
    return float(migration_time_s / const.Myr)


def t_migration_from_torque(
    radius_grid_cm: FloatArray,
    gamma_tot_cgs: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
) -> float:
    """Return migration time [Myr] by integrating the torque profile.

    Uses ``dL/dt = Gamma_tot`` for circular angular momentum
    ``L = m R^2 Omega``:
        t_mig = integral(R_ini -> R_fin) m * R * Omega / (2 Gamma_tot) dR

    Args:
        radius_grid_cm: radial grid [cm] from the torque computation.
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

    radius_grid = np.asarray(radius_grid_cm, dtype=float)
    torque_cgs = np.asarray(gamma_tot_cgs, dtype=float)
    if radius_grid.shape != torque_cgs.shape:
        raise ValueError("radius and torque grids must have the same shape")

    finite = np.isfinite(radius_grid) & np.isfinite(torque_cgs)
    radius_grid = radius_grid[finite]
    torque_cgs = torque_cgs[finite]
    if radius_grid.size < 2:
        return np.inf

    order = np.argsort(radius_grid)
    radius_grid = radius_grid[order]
    torque_cgs = torque_cgs[order]
    unique_radius_cm, unique_indices = np.unique(radius_grid, return_index=True)
    radius_grid = unique_radius_cm
    torque_cgs = torque_cgs[unique_indices]
    if radius_grid.size < 2:
        return np.inf

    min_radius_cm = min(r_ini_cm, r_fin_cm)
    max_radius_cm = max(r_ini_cm, r_fin_cm)
    if min_radius_cm < radius_grid[0] or max_radius_cm > radius_grid[-1]:
        return np.inf

    mask = (radius_grid >= min_radius_cm) & (radius_grid <= max_radius_cm)
    segment_radius_cm = np.concatenate(
        (
            [min_radius_cm],
            radius_grid[
                mask & (radius_grid > min_radius_cm) & (radius_grid < max_radius_cm)
            ],
            [max_radius_cm],
        )
    )
    segment_torque_cgs = np.interp(segment_radius_cm, radius_grid, torque_cgs)

    if segment_radius_cm.size < 2:
        return np.inf

    if np.any(segment_torque_cgs == 0.0):
        return np.inf
    sign_changes = np.where(
        np.sign(segment_torque_cgs[1:]) != np.sign(segment_torque_cgs[:-1])
    )[0]
    if sign_changes.size > 0:
        return np.inf

    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / segment_radius_cm**3)
    integrand = (
        0.5
        * migrator_mass_g
        * segment_radius_cm
        * orbital_frequency_rad_s
        / segment_torque_cgs
    )

    if r_ini_cm > r_fin_cm:
        integration_radius_cm = segment_radius_cm[::-1]
        integrand_integral = integrand[::-1]
    else:
        integration_radius_cm = segment_radius_cm
        integrand_integral = integrand

    migration_time_s = float(np.trapezoid(integrand_integral, integration_radius_cm))

    if not np.isfinite(migration_time_s) or migration_time_s <= 0.0:
        return np.inf

    return float(migration_time_s / const.Myr)
