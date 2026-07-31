"""NSC orbital sampling and relaxation helpers."""

import multiprocessing

import numpy as np
import numpy.typing as npt

try:
    import physical_process.constants as const
except ImportError:
    import constants as const

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
