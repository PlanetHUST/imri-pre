"""Stellar initial-mass-function and lifetime sampling.

All masses are measured in solar masses. Lifetime functions return years.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

SOLAR_METALLICITY = 0.0145
POPULATION_III_THRESHOLD_SOLAR = 1.0e-5
POPULATION_II_MIN_RETAINED_MASS_MSUN = 0.1
SAMPLING_SAFETY_FACTOR = 1.2
POPULATION_II_APPROX_MEAN_MASS_MSUN = 0.35
POPULATION_III_APPROX_MEAN_MASS_MSUN = 50.0
MIN_POPULATION_II_BATCH_SIZE = 1000
MIN_POPULATION_III_BATCH_SIZE = 100

# Backward-compatible public name.
Zsun = SOLAR_METALLICITY


def sample_salpeter(
    n: int,
    m_min: float = 0.01,
    m_max: float = 250.0,
    alpha: float = 2.35,
) -> FloatArray:
    """Draw ``n`` stellar masses [Msun] from a power-law Salpeter IMF."""
    if n < 0:
        raise ValueError("number of samples must be nonnegative")
    if not 0.0 < m_min < m_max:
        raise ValueError("mass bounds must satisfy 0 < m_min < m_max")
    exponent = 1.0 - alpha
    uniform_samples = np.random.rand(n)
    return (
        uniform_samples * (m_max**exponent - m_min**exponent) + m_min**exponent
    ) ** (1.0 / exponent)


def sample_topheavy(
    n: int,
    m_min: float = 13.6,
    m_max: float = 197.0,
    alpha: float = 1.77,
) -> FloatArray:
    """Draw ``n`` stellar masses [Msun] from the top-heavy IMF."""
    return sample_salpeter(n, m_min=m_min, m_max=m_max, alpha=alpha)


def estimate_stars_needed(total_mass_msun: float, average_mass_msun: float) -> int:
    """Estimate a batch size for reaching ``total_mass_msun``."""
    if total_mass_msun <= 0.0 or average_mass_msun <= 0.0:
        raise ValueError("total and average stellar masses must be positive")
    return int(total_mass_msun / average_mass_msun * SAMPLING_SAFETY_FACTOR)


def sample_popii_efficient(
    total_mass_msun: float,
    m_min: float = 0.01,
    m_max: float = 250.0,
    alpha: float = 2.35,
) -> FloatArray:
    """Sample retained Pop II stars until their mass reaches the NSC mass.

    The historical model discards sampled stars below 1 Msun before summing
    the population mass. That scientific assumption is preserved here.
    """
    batch_size = max(
        MIN_POPULATION_II_BATCH_SIZE,
        estimate_stars_needed(total_mass_msun, POPULATION_II_APPROX_MEAN_MASS_MSUN),
    )
    mass_batches: list[float] = []
    sampled_mass_msun = 0.0

    while sampled_mass_msun < total_mass_msun:
        batch_msun = sample_salpeter(batch_size, m_min, m_max, alpha)
        # for efficiency
        batch_msun = batch_msun[batch_msun > POPULATION_II_MIN_RETAINED_MASS_MSUN]
        if batch_msun.size == 0:
            continue
        mass_batches.extend(batch_msun.tolist())
        sampled_mass_msun += float(np.sum(batch_msun))

    masses_msun = np.asarray(mass_batches, dtype=float)
    cumulative_mass_msun = np.cumsum(masses_msun)
    final_index = int(np.searchsorted(cumulative_mass_msun, total_mass_msun))
    return masses_msun[: final_index + 1]


def sample_popiii_efficient(
    total_mass_msun: float,
    m_min: float = 13.6,
    m_max: float = 197.0,
    alpha: float = 1.77,
) -> FloatArray:
    """Sample Pop III stars [Msun] until their mass reaches the target."""
    batch_size = max(
        MIN_POPULATION_III_BATCH_SIZE,
        estimate_stars_needed(total_mass_msun, POPULATION_III_APPROX_MEAN_MASS_MSUN),
    )
    mass_batches: list[float] = []
    sampled_mass_msun = 0.0

    while sampled_mass_msun < total_mass_msun:
        batch_msun = sample_topheavy(batch_size, m_min, m_max, alpha)
        mass_batches.extend(batch_msun.tolist())
        sampled_mass_msun += float(np.sum(batch_msun))

    masses_msun = np.asarray(mass_batches, dtype=float)
    cumulative_mass_msun = np.cumsum(masses_msun)
    final_index = int(np.searchsorted(cumulative_mass_msun, total_mass_msun))
    return masses_msun[: final_index + 1]


def population_iii_lifetime_yr(mass_msun: FloatArray) -> FloatArray:
    """Return Pop III stellar lifetimes [yr] for masses in solar masses."""
    log_mass = np.log10(mass_msun)
    return 10 ** (9.785 - 3.759 * log_mass + 1.413 * log_mass**2 - 0.186 * log_mass**3)


def population_ii_lifetime_yr(mass_msun: FloatArray) -> FloatArray:
    """Return Pop II stellar lifetimes [yr] for masses in solar masses."""
    log_mass = np.log10(mass_msun)
    return 10 ** (10.0 - 3.68 * log_mass + 1.413 * log_mass**2 - 0.186 * log_mass**3)


def TIII(m: FloatArray) -> FloatArray:
    """Backward-compatible alias for :func:`population_iii_lifetime_yr`."""
    return population_iii_lifetime_yr(np.asarray(m, dtype=float))


def TII(m: FloatArray) -> FloatArray:
    """Backward-compatible alias for :func:`population_ii_lifetime_yr`."""
    return population_ii_lifetime_yr(np.asarray(m, dtype=float))


def sample_efficient(mtot: float, Z: float) -> tuple[FloatArray, FloatArray]:
    """Return stellar masses [Msun] and lifetimes [yr]."""

    Z = Z * SOLAR_METALLICITY

    if mtot <= 0.0:
        raise ValueError("total stellar population mass must be positive")
    if Z < 0.0:
        raise ValueError("metallicity must be nonnegative")

    if Z < POPULATION_III_THRESHOLD_SOLAR * SOLAR_METALLICITY:
        masses_msun = sample_popiii_efficient(mtot)
        lifetime_yr = population_iii_lifetime_yr(masses_msun)
    else:
        masses_msun = sample_popii_efficient(mtot)
        lifetime_yr = population_ii_lifetime_yr(masses_msun)
    return masses_msun, lifetime_yr
