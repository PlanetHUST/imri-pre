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
PETERS_ECCENTRICITY_SMALL = 1.0e-8
PETERS_ECCENTRICITY_MAX = 1.0 - 1.0e-8
PETERS_ECCENTRICITY_INTEGRATION_POINTS = 4096
ISCO_RADIUS_RG = 6.0


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


###############################################


def source_frame_gw_frequency_hz(
    observed_gw_frequency_hz: float,
    redshift: float,
) -> float:
    """Return source-frame GW frequency [Hz] from an observed frequency."""
    if (
        not np.isfinite(observed_gw_frequency_hz)
        or observed_gw_frequency_hz <= 0.0
    ):
        raise ValueError("observed GW frequency must be positive")
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    return float(observed_gw_frequency_hz / (1.0 + redshift))


def observed_frame_time_myr(source_time_myr: float, redshift: float) -> float:
    """Return observed-frame time [Myr] from a source-frame duration."""
    if np.isnan(source_time_myr) or source_time_myr < 0.0:
        raise ValueError("source-frame time must be nonnegative")
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    return float(source_time_myr * (1.0 + redshift))


def redshifted_mass_msun(source_mass_msun: float, redshift: float) -> float:
    """Return observed-frame redshifted mass [Msun]."""
    if not np.isfinite(source_mass_msun) or source_mass_msun <= 0.0:
        raise ValueError("source mass must be positive")
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    return float(source_mass_msun * (1.0 + redshift))


def redshift_transport_time_myr(
    redshift: float,
    hubble_constant_per_myr: float,
) -> float:
    """Return the prompt's linear light-transport time ``z / H0`` [Myr]."""
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    if (
        not np.isfinite(hubble_constant_per_myr)
        or hubble_constant_per_myr <= 0.0
    ):
        raise ValueError("Hubble constant must be positive")
    return float(redshift / hubble_constant_per_myr)


def redshift_source_time_budget_myr(
    redshift: float,
    hubble_constant_per_myr: float,
    zero_redshift_limit_myr: float,
) -> float:
    """Return the source-frame GW time budget for an IMRI at redshift ``z``.

    For nonzero redshift this applies the prompt's ``t_src = t_obs / (1 + z)``
    to the transport time. At ``z = 0``, use the existing local Hubble-time
    budget to keep local-universe runs backward compatible.
    """
    if (
        not np.isfinite(zero_redshift_limit_myr)
        or zero_redshift_limit_myr <= 0.0
    ):
        raise ValueError("zero-redshift GW time limit must be positive")
    if redshift == 0.0:
        return float(zero_redshift_limit_myr)
    return float(
        redshift_transport_time_myr(redshift, hubble_constant_per_myr)
        / (1.0 + redshift)
    )


def gw_frequency_hz(
    total_mass_msun: float,
    semi_major_axis_cm: float,
    redshift: float = 0.0,
) -> float:
    """Return the dominant circular GW frequency [Hz] at separation ``a`` [cm].

    When ``redshift`` is nonzero, the frequency relation uses the redshifted
    observed mass ``M_z = M * (1 + z)`` requested by the prompt.
    """
    if (
        not np.isfinite(total_mass_msun)
        or total_mass_msun <= 0.0
        or not np.isfinite(semi_major_axis_cm)
        or semi_major_axis_cm <= 0.0
    ):
        raise ValueError("binary mass and semi-major axis must be positive")
    total_mass_g = redshifted_mass_msun(total_mass_msun, redshift) * const.msun
    orbital_frequency_hz = np.sqrt(const.G * total_mass_g / semi_major_axis_cm**3) / (
        2.0 * np.pi
    )
    return float(2.0 * orbital_frequency_hz)


def gw_radius_for_frequency_cm(
    total_mass_msun: float,
    gw_frequency_hz: float,
    redshift: float = 0.0,
) -> float:
    """Return the circular binary separation [cm] for a GW frequency [Hz]."""
    if (
        not np.isfinite(total_mass_msun)
        or total_mass_msun <= 0.0
        or not np.isfinite(gw_frequency_hz)
        or gw_frequency_hz <= 0.0
    ):
        raise ValueError("binary mass and GW frequency must be positive")
    total_mass_g = redshifted_mass_msun(total_mass_msun, redshift) * const.msun
    return float(
        (const.G * total_mass_g / (np.pi * gw_frequency_hz) ** 2) ** (1.0 / 3.0)
    )


def peters_merger_time_myr(
    primary_mass_msun: float,
    secondary_mass_msun: float,
    semi_major_axis_cm: float,
    eccentricity: float = 0.0,
) -> float:
    """Return the Peters (1964) GW coalescence time [Myr].

    The circular limit uses the analytic ``a^4 / (4 beta)`` expression.
    For eccentric binaries, the standard Peters integral is evaluated
    numerically without requiring SciPy.
    """
    if primary_mass_msun <= 0.0 or secondary_mass_msun <= 0.0:
        raise ValueError("binary component masses must be positive")
    if semi_major_axis_cm <= 0.0:
        raise ValueError("semi-major axis must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")

    primary_mass_g = primary_mass_msun * const.msun
    secondary_mass_g = secondary_mass_msun * const.msun
    beta_cm4_s = (
        64.0
        / 5.0
        * const.G**3
        * primary_mass_g
        * secondary_mass_g
        * (primary_mass_g + secondary_mass_g)
        / const.c**5
    )
    if eccentricity < PETERS_ECCENTRICITY_SMALL:
        return float(semi_major_axis_cm**4 / (4.0 * beta_cm4_s) / const.Myr)

    eccentricity = min(float(eccentricity), PETERS_ECCENTRICITY_MAX)
    grid = np.linspace(
        PETERS_ECCENTRICITY_SMALL,
        eccentricity,
        PETERS_ECCENTRICITY_INTEGRATION_POINTS,
    )
    integrand = (
        grid ** (29.0 / 19.0)
        * (1.0 + 121.0 / 304.0 * grid**2) ** (1181.0 / 2299.0)
        / (1.0 - grid**2) ** 1.5
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    integral = float(trapezoid(integrand, grid))
    c0_cm = (
        semi_major_axis_cm
        * (1.0 - eccentricity**2)
        * eccentricity ** (-12.0 / 19.0)
        * (1.0 + 121.0 / 304.0 * eccentricity**2) ** (-870.0 / 2299.0)
    )
    return float((12.0 / 19.0) * c0_cm**4 * integral / beta_cm4_s / const.Myr)


def lisa_entry_radius_cm(
    primary_mass_msun: float,
    secondary_mass_msun: float,
    min_gw_frequency_hz: float,
    redshift: float = 0.0,
) -> float:
    """Return the separation [cm] where a circular binary reaches LISA's lower band."""

    total_mass_msun = primary_mass_msun + secondary_mass_msun
    source_min_gw_frequency_hz = source_frame_gw_frequency_hz(
        min_gw_frequency_hz,
        redshift,
    )
    entry_radius_cm = gw_radius_for_frequency_cm(
        total_mass_msun,
        source_min_gw_frequency_hz,
        redshift,
    )
    isco_radius_cm = (
        ISCO_RADIUS_RG * const.G * total_mass_msun * const.msun / const.c**2
    )
    if entry_radius_cm <= isco_radius_cm:
        return np.inf
    return entry_radius_cm


def peters_time_to_lisa_band_myr(
    primary_mass_msun: float,
    secondary_mass_msun: float,
    semi_major_axis_cm: float,
    min_gw_frequency_hz: float,
    eccentricity: float = 0.0,
    redshift: float = 0.0,
) -> float:
    """Return the GW time [Myr] from ``a`` to LISA lower-band entry."""
    entry_radius_cm = lisa_entry_radius_cm(
        primary_mass_msun,
        secondary_mass_msun,
        min_gw_frequency_hz,
        redshift,
    )
    if not np.isfinite(entry_radius_cm):
        return np.inf
    if semi_major_axis_cm <= entry_radius_cm:
        return 0.0
    start_time_myr = peters_merger_time_myr(
        primary_mass_msun,
        secondary_mass_msun,
        semi_major_axis_cm,
        eccentricity,
    )
    entry_time_myr = peters_merger_time_myr(
        primary_mass_msun,
        secondary_mass_msun,
        entry_radius_cm,
        eccentricity,
    )
    return float(max(0.0, start_time_myr - entry_time_myr))


def imri_lisa_hubble_threshold_radius_cm(
    primary_mass_msun: float,
    secondary_mass_msun: float,
    max_gw_time_myr: float,
    min_gw_frequency_hz: float,
    eccentricity: float = 0.0,
    redshift: float = 0.0,
) -> float:
    """Return the outermost radius [cm] that reaches LISA within ``max_gw_time``.

    This simplified demo keeps a fixed eccentricity when solving the threshold.
    With the default ``e=0``, it is exactly the circular Peters solution.
    """
    if max_gw_time_myr <= 0.0:
        raise ValueError("maximum GW time must be positive")
    entry_radius_cm = lisa_entry_radius_cm(
        primary_mass_msun,
        secondary_mass_msun,
        min_gw_frequency_hz,
        redshift,
    )
    if not np.isfinite(entry_radius_cm):
        return np.inf

    coefficient_myr_per_cm4 = peters_merger_time_myr(
        primary_mass_msun,
        secondary_mass_msun,
        1.0,
        eccentricity,
    )
    entry_time_myr = peters_merger_time_myr(
        primary_mass_msun,
        secondary_mass_msun,
        entry_radius_cm,
        eccentricity,
    )
    threshold_time_myr = max_gw_time_myr + entry_time_myr
    return float((threshold_time_myr / coefficient_myr_per_cm4) ** 0.25)


######################################################


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
