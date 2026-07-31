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


def initial_mass(
    sample_count: int,
    min_mass_msun: float = 1e4,
    max_mass_msun: float = 1e5,
) -> FloatArray:
    """Return ``N`` logarithmically spaced masses [Msun].

    Both mass bounds are measured in solar masses.
    """
    if sample_count <= 0:
        raise ValueError("number of mass samples must be positive")
    if not 0.0 < min_mass_msun <= max_mass_msun:
        raise ValueError("mass bounds must satisfy 0 < min <= max")
    return np.logspace(
        np.log10(min_mass_msun),
        np.log10(max_mass_msun),
        sample_count,
    )


def get_NSC_mass(smbh_mass_msun: float | FloatArray) -> float | FloatArray:
    """Estimate NSC mass [Msun] from SMBH mass [Msun]."""
    nsc_mass_msun = NSC_TO_SMBH_MASS_RATIO * np.asarray(smbh_mass_msun)
    return float(nsc_mass_msun) if nsc_mass_msun.ndim == 0 else nsc_mass_msun


def initial_star(
    star_count: int,
    smbh_mass_msun: float,
    max_radius_cm: float,
    gamma: float = 7.0 / 4.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sample orbital elements.

    Args:
        star_count: Number of stellar orbits.
        smbh_mass_msun: SMBH mass [Msun].
        max_radius_cm: Maximum semi-major axis [cm].
        gamma: Number-density power-law index.

    Returns:
        Semi-major axis [cm], eccentricity, inclination [rad], and
        ``cos(omega)``. The historical angular distributions are preserved.
    """
    if star_count < 0:
        raise ValueError("number of stars must be nonnegative")
    if smbh_mass_msun <= 0.0 or max_radius_cm <= 0.0:
        raise ValueError("SMBH mass and maximum radius must be positive")

    volume_integral_exponent = 2.0 - gamma
    inverse_cdf_exponent = volume_integral_exponent + 1.0
    gravitational_radius_cm = const.G * smbh_mass_msun * const.msun / const.c**2
    minimum_radius_cm = DISK_INNER_RADIUS_RG * gravitational_radius_cm
    if max_radius_cm < minimum_radius_cm:
        raise ValueError("maximum radius must exceed the inner orbital radius")

    uniform_radius = np.random.random(star_count)
    semi_major_axis_cm = (
        minimum_radius_cm**inverse_cdf_exponent
        + uniform_radius
        * (
            max_radius_cm**inverse_cdf_exponent
            - minimum_radius_cm**inverse_cdf_exponent
        )
    ) ** (1.0 / inverse_cdf_exponent)
    eccentricity = np.sqrt(np.random.random(star_count))
    inclination_rad = np.random.uniform(
        MIN_INCLINATION_RAD,
        MAX_INCLINATION_RAD,
        star_count,
    )
    cos_omega = np.random.random(star_count)
    return semi_major_axis_cm, eccentricity, inclination_rad, cos_omega


def velocity_disp(smbh_mass_msun: float) -> float:
    """Return NSC velocity dispersion [cm/s] for SMBH mass [Msun]."""
    if smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be positive")
    return float(
        VELOCITY_DISPERSION_NORMALIZATION_CM_S
        * smbh_mass_msun**VELOCITY_DISPERSION_MASS_EXPONENT
    )


def number_R_MSC(
    smbh_mass_msun: float,
    stellar_mass_g: float,
    radius_cm: float,
    gamma: float = 1.3,
) -> float:
    """Return NSC stellar number density [cm^-3].

    ``smbh_mass_msun`` is in solar masses, stellar mass is in grams, and
    radius is in centimeters.
    """
    if smbh_mass_msun <= 0.0 or stellar_mass_g <= 0.0 or radius_cm <= 0.0:
        raise ValueError("masses and radius must be positive")
    smbh_mass_g = smbh_mass_msun * const.msun
    influence_radius_cm = (
        const.G * smbh_mass_g / velocity_disp(smbh_mass_msun) ** 2
    )
    return float(
        (3.0 - gamma)
        * smbh_mass_g
        / (2.0 * np.pi * stellar_mass_g)
        * influence_radius_cm ** (-3.0)
        * (radius_cm / influence_radius_cm) ** (-gamma)
    )


def t_relax(smbh_mass_msun: float, radius_cm: float) -> float:
    """Return the two-body relaxation time [s] at radius ``radius_cm``."""
    number_density_cm3 = number_R_MSC(
        smbh_mass_msun,
        RELAXATION_REFERENCE_STELLAR_MASS_G,
        radius_cm,
    )
    relaxation_time_s = (
        RELAXATION_COEFFICIENT
        * velocity_disp(smbh_mass_msun) ** 3
        / (
            const.G**2
            * RELAXATION_REFERENCE_STELLAR_MASS_G**2
            * number_density_cm3
            * COULOMB_LOGARITHM
        )
    )
    return float(relaxation_time_s)
