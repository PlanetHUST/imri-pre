"""GW frequency, radius, Peters-time, and redshift helpers."""

import numpy as np

try:
    import physical_process.constants as const
except ImportError:
    import constants as const

PETERS_ECCENTRICITY_SMALL = 1.0e-8
PETERS_ECCENTRICITY_MAX = 1.0 - 1.0e-8
PETERS_ECCENTRICITY_INTEGRATION_POINTS = 4096
ISCO_RADIUS_RG = 6.0


def source_frame_gw_frequency_hz(
    observed_gw_frequency_hz: float,
    redshift: float,
) -> float:
    """Return source-frame GW frequency [Hz] from an observed frequency."""
    if not np.isfinite(observed_gw_frequency_hz) or observed_gw_frequency_hz <= 0.0:
        raise ValueError("observed GW frequency must be positive")
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    return float(observed_gw_frequency_hz / (1.0 + redshift))


def observed_frame_gw_frequency_hz(
    source_gw_frequency_hz: float,
    redshift: float,
) -> float:
    """Return observed-frame GW frequency [Hz] from a source-frame frequency."""
    if not np.isfinite(source_gw_frequency_hz) or source_gw_frequency_hz <= 0.0:
        raise ValueError("source GW frequency must be positive")
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    return float(source_gw_frequency_hz * (1.0 + redshift))


def observed_frame_time_myr(source_time_myr: float, redshift: float) -> float:
    """Return observed-frame time [Myr] from a source-frame duration."""
    if np.isnan(source_time_myr) or source_time_myr < 0.0:
        raise ValueError("source-frame time must be nonnegative")
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    return float(source_time_myr * (1.0 + redshift))


def redshift_transport_time_myr(
    redshift: float,
    hubble_constant_per_myr: float,
) -> float:
    """Return the prompt's linear light-transport time ``z / H0`` [Myr]."""
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be nonnegative")
    if not np.isfinite(hubble_constant_per_myr) or hubble_constant_per_myr <= 0.0:
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
    if not np.isfinite(zero_redshift_limit_myr) or zero_redshift_limit_myr <= 0.0:
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
) -> float:
    """Return the dominant circular GW frequency [Hz] at separation ``a`` [cm].

    The frequency relation uses the original source mass only.
    """
    if (
        not np.isfinite(total_mass_msun)
        or total_mass_msun <= 0.0
        or not np.isfinite(semi_major_axis_cm)
        or semi_major_axis_cm <= 0.0
    ):
        raise ValueError("binary mass and semi-major axis must be positive")
    total_mass_g = total_mass_msun * const.msun
    orbital_frequency_hz = np.sqrt(const.G * total_mass_g / semi_major_axis_cm**3) / (
        2.0 * np.pi
    )
    return float(2.0 * orbital_frequency_hz)


def gw_radius_for_frequency_cm(
    total_mass_msun: float,
    gw_frequency_hz: float,
) -> float:
    """Return the circular binary separation [cm] for a GW frequency [Hz]."""
    if (
        not np.isfinite(total_mass_msun)
        or total_mass_msun <= 0.0
        or not np.isfinite(gw_frequency_hz)
        or gw_frequency_hz <= 0.0
    ):
        raise ValueError("binary mass and GW frequency must be positive")
    total_mass_g = total_mass_msun * const.msun
    return float(
        (const.G * total_mass_g / (np.pi * gw_frequency_hz) ** 2) ** (1.0 / 3.0)
    )


def peters_merger_time_myr(
    primary_mass_msun: float,
    secondary_mass_msun: float,
    semi_major_axis_cm: float,
    eccentricity: float,
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
    integral = float(np.trapezoid(integrand, grid))
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
    redshift: float,
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
    eccentricity: float,
    redshift: float,
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


def imri_lisa_time_buget_radius_cm(
    primary_mass_msun: float,
    secondary_mass_msun: float,
    max_gw_time_myr: float,
    min_gw_frequency_hz: float,
    eccentricity: float,
    redshift: float,
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
