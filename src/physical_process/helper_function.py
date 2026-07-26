"""Shared sampling and timescale helpers.

The module retains historical function names for compatibility. Parameters
and local variables explicitly state their physical units.
"""

import multiprocessing

import numpy as np
import numpy.typing as npt

import constants as const
from disk_quantity.disk_solve import DISK_VISCOSITY_ALPHA
from general_class import AGN_properties

FloatArray = npt.NDArray[np.float64]

DISK_INNER_RADIUS_RG = 6.0
NSC_TO_SMBH_MASS_RATIO = 10.0
MIN_INCLINATION_RAD = np.pi / 18.0
MAX_INCLINATION_RAD = 17.0 * np.pi / 18.0
VELOCITY_DISPERSION_NORMALIZATION_CM_S = 2.3e5
VELOCITY_DISPERSION_MASS_EXPONENT = 1.0 / 4.38
RELAXATION_COEFFICIENT = 0.34
COULOMB_LOGARITHM = 10.0
RELAXATION_REFERENCE_STELLAR_MASS_G = const.msun
TYPE_II_GAP_PARAMETER_THRESHOLD = 11.0


def get_cpu_count() -> int:
    """Return available CPU count while reserving one core when possible."""
    try:
        return max(1, multiprocessing.cpu_count() - 1)
    except NotImplementedError:
        return 2


def initial_mass(N: int, min: float = 1e4, max: float = 1e5) -> FloatArray:
    """Return ``N`` logarithmically spaced masses [Msun].

    ``min`` and ``max`` retain their historical parameter names for API
    compatibility; both are measured in solar masses.
    """
    if N <= 0:
        raise ValueError("number of mass samples must be positive")
    if not 0.0 < min <= max:
        raise ValueError("mass bounds must satisfy 0 < min <= max")
    return np.logspace(np.log10(min), np.log10(max), N)


def get_NSC_mass(M_SMBH: float | FloatArray) -> float | FloatArray:
    """Estimate NSC mass [Msun] from SMBH mass [Msun]."""
    m_smbh_msun = np.asarray(M_SMBH)
    m_nsc_msun = NSC_TO_SMBH_MASS_RATIO * m_smbh_msun
    return float(m_nsc_msun) if m_nsc_msun.ndim == 0 else m_nsc_msun


def initial_star(
    num: int,
    M_SMBH: float,
    rmax: float,
    gamma: float = 7.0 / 4.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sample orbital elements.

    Args:
        num: Number of stellar orbits.
        M_SMBH: SMBH mass [Msun].
        rmax: Maximum semi-major axis [cm].
        gamma: Number-density power-law index.

    Returns:
        Semi-major axis [cm], eccentricity, inclination [rad], and
        ``cos(omega)``. The historical angular distributions are preserved.
    """
    if num < 0:
        raise ValueError("number of stars must be nonnegative")
    if M_SMBH <= 0.0 or rmax <= 0.0:
        raise ValueError("SMBH mass and maximum radius must be positive")

    volume_integral_exponent = 2.0 - gamma
    inverse_cdf_exponent = volume_integral_exponent + 1.0
    gravitational_radius_cm = const.G * M_SMBH * const.msun / const.c**2
    minimum_radius_cm = DISK_INNER_RADIUS_RG * gravitational_radius_cm
    if rmax < minimum_radius_cm:
        raise ValueError("maximum radius must exceed the inner orbital radius")

    uniform_radius = np.random.random(num)
    semi_major_axis_cm = (
        minimum_radius_cm**inverse_cdf_exponent
        + uniform_radius
        * (rmax**inverse_cdf_exponent - minimum_radius_cm**inverse_cdf_exponent)
    ) ** (1.0 / inverse_cdf_exponent)
    eccentricity = np.sqrt(np.random.random(num))
    inclination_rad = np.random.uniform(MIN_INCLINATION_RAD, MAX_INCLINATION_RAD, num)
    cos_omega = np.random.random(num)
    return semi_major_axis_cm, eccentricity, inclination_rad, cos_omega


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


def velocity_disp(M_SMBH: float) -> float:
    """Return NSC velocity dispersion [cm/s] for SMBH mass [Msun]."""
    if M_SMBH <= 0.0:
        raise ValueError("SMBH mass must be positive")
    return float(
        VELOCITY_DISPERSION_NORMALIZATION_CM_S
        * M_SMBH**VELOCITY_DISPERSION_MASS_EXPONENT
    )


def number_R_MSC(M_SMBH: float, m: float, R: float, gamma: float = 1.3) -> float:
    """Return NSC stellar number density [cm^-3].

    ``M_SMBH`` is in solar masses; stellar mass ``m`` is in grams and radius
    ``R`` is in centimeters.
    """
    if M_SMBH <= 0.0 or m <= 0.0 or R <= 0.0:
        raise ValueError("masses and radius must be positive")
    smbh_mass_g = M_SMBH * const.msun
    influence_radius_cm = const.G * smbh_mass_g / velocity_disp(M_SMBH) ** 2
    return float(
        (3.0 - gamma)
        * smbh_mass_g
        / (2.0 * np.pi * m)
        * influence_radius_cm ** (-3.0)
        * (R / influence_radius_cm) ** (-gamma)
    )


def t_relax(M_SMBH: float, R: float) -> float:
    """Return the two-body relaxation time [s] at radius ``R`` [cm]."""
    number_density_cm3 = number_R_MSC(M_SMBH, RELAXATION_REFERENCE_STELLAR_MASS_G, R)
    relaxation_time_s = (
        RELAXATION_COEFFICIENT
        * velocity_disp(M_SMBH) ** 3
        / (
            const.G**2
            * RELAXATION_REFERENCE_STELLAR_MASS_G**2
            * number_density_cm3
            * COULOMB_LOGARITHM
        )
    )
    return float(relaxation_time_s)
